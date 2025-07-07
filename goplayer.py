import tkinter as tk
import tkinter.messagebox as messagebox
import copy
import random
import openai
import os
from dotenv import load_dotenv
from goruler import is_capture_move, is_valid_move, has_liberty


# AI棋手或人类棋手的抽象类
class GoPlayer:
    def __init__(self, color):
        self.color = color  # 玩家的颜色 ('black' 或 'white')
        self.move = None
    # 把下棋的记录放在这里，
    def my_move(self):
        
        return [self.move,self.color]
    # 抽象方法，子类需要实现具体的下棋逻辑。
    def make_move(self, board):

        raise NotImplementedError("This method should be implemented by subclasses.")

class HumanPlayer(GoPlayer):
    def make_move(self, board, event=None):
        if event is None:
            return False
        row = round((event.x - board.margin) / board.cell_size)
        col = round((event.y - board.margin) / board.cell_size)
        if board.place_stone(row, col, self.color):
            self.move = (row, col)
            self.my_move()
            board.moves_history_store(row, col, self.color)  # 存储下棋记录

            return True
        return False

def get_prompt(board_state,board,color="white"):
        # 构造提示信息
        prompt = (
            f"当前棋盘中，black为黑子，white为白子，empty为可落子的空点\n"
            f"棋盘从左到右从上到下分别是1到{board.size}\n"
            f"当前棋盘状态如下：\n{board_state}\n"
            f"请注意，棋盘大小为 {board.size}x{board.size}。\n"
            f"你是一个围棋 AI，当前执子颜色为 {color}。\n"
            f"请根据当前棋盘状态和你的颜色，选择一个合法的落子位置。\n"
            f"合法的落子位置应该从棋盘上的空点中选择。\n"
            f"如果没有合法的落子位置，请返回 '-1,-1'。\n"
            f"请以 'row,col' 的格式返回下一步棋的位置（例如：5,7）。"
            f"请不要返回多余的解释，返回格式中只有 row 和 col 的数字。\n"
        )
        return prompt

def get_modfied_prompt(board_state, board,last_move,color="white"):
    # 构造提示信息
    prompt = (
        f"当前棋盘中，black为黑子，white为白子，empty为可落子的空点\n"
        f"棋盘从左到右从上到下分别是1到{board.size}\n"
        f"当前棋盘状态如下：\n{board_state}\n"
        f"请注意，棋盘大小为 {board.size}x{board.size}。\n"
        f"你是一个围棋 AI，当前执子颜色为 {color}。\n"
        f"请根据当前棋盘状态和你的颜色，选择一个合法的落子位置。\n"
        f"合法的落子位置应该从棋盘上的空点中选择。\n"
        f"如果没有合法的落子位置，请返回 '-1,-1'。\n"
        f"请以 'row,col' 的格式返回下一步棋的位置（例如：5,7）。\n"
        f"上一步棋是 {last_move}，这步棋是无效的，不要这样下，请重新考虑。\n"
        f"请不要返回多余的解释，返回格式中只有 row 和 col 的数字。\n"
    )
    return prompt


class AIPlayer(GoPlayer):
    def __init__(self, color):
        super().__init__(color)
        # 初始化 OpenAI API 客户端
        model=os.getenv("OPENAI_MODEL")
        api_key=os.getenv("OPENAI_API_KEY")
        api_base=os.getenv("OPENAI_BASE_URL")

        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self.tourance=3  # 连续三次不合法的落子后，随机选择一个空位


    def make_move(self, board,event=None):

        # 先尝试使用 GPT 获取下一步棋的位置
        move = self.get_move_from_gpt(board)
        if move is None:
            move = self.make_random_move(board, event)
            print("AI 随机选择位置：", move)
        else:
            print("AI 选择位置：", move)
        
        row, col = move
        if board.place_stone(row, col, self.color):
            self.move = (row, col)
            self.my_move()
            board.moves_history_store(row, col, self.color)  # 存储下棋记录
            return True
        return False
    def make_random_move(self, board, event=None):
        empty_positions = [
            (r, c) for r in range(board.size) for c in range(board.size) if board.grid[r][c] is None
        ]
        if not empty_positions:
            return False
        row, col = random.choice(empty_positions)
        move = (row, col)
        return move

    def get_move_from_gpt(self, board):
        tourance=self.tourance

        # 将棋盘状态转换为字符串表示
        # black为黑子，white为白子，empty为可落子的空点
        board_state = "\n".join(
            [" ".join([str(cell) if cell is not None else "empty" for cell in row]) for row in board.grid]
        )
        # print(f"当前棋盘状态：\n{board_state}")

        prompt= get_prompt(board_state,board)
        messages=[
            {"role": "system", "content": "你是一个厉害的围棋选手,现在和我下棋。"},
            {"role": "user", "content": prompt}
        ]
        try:
            # 调用 OpenAI GPT API
            # 读取配置

            response = self.client.chat.completions.create(
                model=self.model,  
                messages=messages,
                temperature=0.1,
            )
            # print(f"Raw API Response: {response}")
            # 解析返回的下一步棋位置
            move_str = response.choices[0].message.content.strip()
            row, col = map(int, move_str.split(","))
            # print(f"AI 回答：{move_str}")
            print(f"AI 玩家下棋位置：{row}, {col}")
            move = (row, col)
            # 检查返回的坐标是否合法
            isLegelMove = self.isLegelMove(board, row, col)
            while not isLegelMove:
                print(f"AI 玩家下棋位置不合法：{row}, {col}")
                tourance-=1
                if tourance>0:

                    prompt_modify= get_modfied_prompt(board_state, board, move_str, self.color)
                    messages=[
                        {"role": "system", "content": "你是一个厉害的围棋选手,现在和我下棋。"},
                        {"role": "user", "content": prompt_modify}
                    ]
                    response = self.client.chat.completions.create(
                        model=self.model,  
                        messages=messages,
                        temperature=0.1,
                    )
                    move_str = response.choices[0].message.content.strip()
                    row, col = map(int, move_str.split(","))
                    isLegelMove = self.isLegelMove(board, row, col)
                elif tourance<=0:
                    print("AI 玩家连续下棋位置不合法，随机选择一个空位")

                    move=self.make_random_move(board, event=None)
                    isLegelMove = self.isLegelMove(board, row, col)
                    break
            return move
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # 如果 API 调用失败，随机选择一个空位
            # empty_positions = [
            #     (r, c) for r in range(board.size) for c in range(board.size) if board.grid[r][c] is None
            # ]
            # return random.choice(empty_positions) if empty_positions else None
            return self.make_random_move(board, event=None)

    def isLegelMove(self, board, row, col):
        # 检查落子是否合法
        return is_valid_move(board, row, col, self.color)

# 随机生成可选点下棋
class RandomPlayer(GoPlayer):
    def make_move(self, board, event=None):
        move = self.get_random_move(board)
        if move is None:
            return False
        if board.place_stone(move[0], move[1], self.color):
            self.move = move
            self.my_move()
            board.moves_history_store(move[0], move[1], self.color)  # 存储下棋记录
            return True
        return False

    def get_random_move(self, board):
        empty_positions = [
            (r, c) for r in range(board.size) for c in range(board.size) if board.grid[r][c] is None
        ]
        if not empty_positions:
            return False
        row, col = random.choice(empty_positions)
        is_legal_move = self.isLegelMove(board, row, col)
        while not is_legal_move:
            empty_positions.remove((row, col))
            if not empty_positions:
                return False
            row, col = random.choice(empty_positions)
        move = (row, col)
        return move
    def isLegelMove(self, board, row, col):
        # 检查落子是否合法
        return is_valid_move(board, row, col, self.color)