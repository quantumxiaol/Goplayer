import tkinter as tk
import tkinter.messagebox as messagebox
import copy
import random
import openai
import os
from readConfig import get_openai_config


# AI棋手或人类棋手的抽象类
class GoPlayer:
    def __init__(self, color):
        self.color = color  # 玩家的颜色 ('black' 或 'white')
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
            return True
        return False
    
class AIPlayer(GoPlayer):
    def __init__(self, color):
        super().__init__(color)
        # 初始化 OpenAI API 客户端
        model, api_key, api_base = get_openai_config()
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base)

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

        # 将棋盘状态转换为字符串表示
        board_state = "\n".join(
            "".join(cell or "." for cell in row) for row in board.grid
        )

        # 构造提示信息
        prompt = (
            f"当前棋盘状态如下：\n{board_state}\n"
            f"请注意，棋盘大小为 {board.size}x{board.size}。\n"
            f"你是一个围棋 AI，当前执子颜色为 {self.color}。\n"
            f"请以 'row,col' 的格式返回下一步棋的位置（例如：5,7）。"
            f"请不要返回多余的解释，返回格式中只有 row 和 col 的数字。\n"
        )
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
            print(f"AI 回答：{move_str}")
            print(f"AI 玩家下棋位置：{row}, {col}")
            return (row, col)
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # 如果 API 调用失败，随机选择一个空位
            empty_positions = [
                (r, c) for r in range(board.size) for c in range(board.size) if board.grid[r][c] is None
            ]
            return random.choice(empty_positions) if empty_positions else None


