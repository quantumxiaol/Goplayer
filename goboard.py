import tkinter as tk
import tkinter.messagebox as messagebox
import copy
import random
import openai
import os
from readConfig import get_openai_config
from goplayer import HumanPlayer, AIPlayer,RandomPlayer

class GoBoard(tk.Frame):
    def __init__(self, parent, size=19):
        super().__init__(parent)
        self.size = size
        self.reduce = self.komi()
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.current_player = None  # 当前玩家对象
        self.players = {}  # 玩家字典 {'black': Player, 'white': Player}
        self.score = {'black': 0, 'white': 0}
        self.mode = "human_vs_human"  # 默认对弈模式
        self.canvas = tk.Canvas(self, bg='peach puff')
        self.canvas.bind('<Configure>', self.draw_board)
        self.canvas.bind('<Button-1>', self.handle_click)
        self.canvas.pack(fill=tk.BOTH, expand=1)
        self.moves_history = []  # 用于存储棋盘状态的历史记录{[row, col, color], ...}
        self.setup_players()

    # 根据棋盘大小设置贴目
    def komi(self):
        if self.size == 19:
            komi = 7.5
        elif self.size == 13:
            komi = 2.0  # 13x13 的贴目可以是 2 目
        elif self.size == 9:
            komi = 5.5  # 9x9 的贴目可以是 5.5 目
        else:
            komi = 0.0  # 默认不贴目
        return komi

    def reset(self):
        """
        重置棋盘状态。
        """
        self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.current_player = 'black'
        self.score = {'black': 0, 'white': 0}
        self.draw_board()
        self.setup_players()  # 重新设置玩家
    def draw_board(self, event=None):
        self.canvas.delete('all')
        self.margin = 25
        self.width, self.height = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.cell_size = (min(self.width, self.height) - 2 * self.margin) / (self.size - 1)

        # 加粗最外边框
        border_width = 3  # 边框宽度
        self.canvas.create_rectangle(
            self.margin, self.margin,
            self.width - self.margin, self.height - self.margin,
            outline='black', width=border_width
        )

        # 绘制网格线
        for i in range(self.size ):  
            x = i * self.cell_size + self.margin
            y = i * self.cell_size + self.margin
            # 水平线
            self.canvas.create_line(self.margin, y, self.width - self.margin, y, fill='black')
            # 竖直线
            self.canvas.create_line(x, self.margin, x, self.height - self.margin, fill='black')

        # 绘制星位
        self.draw_star_points()

        # 绘制棋子
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] is not None:
                    self.draw_stone(i, j, self.grid[i][j])
    # def place_stone(self, event):
    #     row = round((event.x - self.margin) / self.cell_size)
    #     col = round((event.y - self.margin) / self.cell_size)
    #     if 0 <= row < self.size and 0 <= col < self.size and self.grid[row][col] is None:
    #         # 保留一个棋盘状态的副本
    #         old_grid = copy.deepcopy(self.grid)
    #         self.grid[row][col] = self.current_player
    #         self.draw_stone(row, col, self.current_player)
    #         self.remove_captured_stones('white' if self.current_player == 'black' else 'black')
            
    #         # 检查是否形成禁入点
    #         if self.grid[row][col] is not None and self.is_captured(self.flood_fill(row, col, self.current_player)):
    #             # 如果形成禁入点，恢复到先前的棋盘状态并阻止落子
    #             self.grid = old_grid
    #             self.draw_board()
    #         else:
    #             self.current_player = 'white' if self.current_player == 'black' else 'black'
    #     print(self.score)

    def place_stone(self, row, col, color):
        if not (0 <= row < self.size and 0 <= col < self.size) or self.grid[row][col] is not None:
            return False
        old_grid = copy.deepcopy(self.grid)
        self.grid[row][col] = color
        self.draw_stone(row, col, color)
        opponent_color = 'white' if color == 'black' else 'black'
        self.remove_captured_stones(opponent_color)
        if self.is_captured(self.flood_fill(row, col, color)):
            self.grid = old_grid
            self.draw_board()
            return False
        return True

    def setup_players(self):
        """
        根据对弈模式和颜色选择设置玩家。
        """
        if self.mode == "human_vs_human":
            self.players['black'] = HumanPlayer('black')
            self.players['white'] = HumanPlayer('white')
        elif self.mode == "human_vs_ai":
            human_color = 'black'  # 默认人类玩家执黑子
            ai_color = 'white'
            self.players[human_color] = HumanPlayer(human_color)
            self.players[ai_color] = AIPlayer(ai_color)
        elif self.mode == "random_vs_random":
            self.players['black'] = RandomPlayer('black')
            self.players['white'] = RandomPlayer('white')
        self.current_player = self.players['black']  # 初始玩家为黑方
        print(f"Players initialized: {self.players}")
        print(f"Current player: {self.current_player}")
        
    # 设置对弈模式并重新初始化玩家
    def setup_mode(self, mode):
        self.mode = mode
        self.reset() 
    def draw_star_points(self):
        """
        在棋盘上绘制星位标记。
        """
        if self.size == 19:  # 标准 19x19 棋盘
            star_points = [(3, 3), (9, 3), (15, 3),
                           (3, 9), (9, 9), (15, 9),
                           (3, 15), (9, 15), (15, 15)]
        elif self.size == 13:  # 13x13 棋盘
            star_points = [(3, 3), (6, 3), (9, 3),
                           (3, 6), (6, 6), (9, 6),
                           (3, 9), (6, 9), (9, 9)]
        elif self.size == 9:  # 9x9 棋盘
            star_points = [(2, 2), (4, 2), (6, 2),
                           (2, 4), (4, 4), (6, 4),
                           (2, 6), (4, 6), (6, 6)]
        else:
            return  # 其他尺寸棋盘不绘制星位

        # 绘制星位
        for row, col in star_points:
            x = col * self.cell_size + self.margin
            y = row * self.cell_size + self.margin
            radius = self.cell_size / 10  # 星位半径
            self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill='black'
            )

    def draw_stone(self, row, col, color):
        x = row * self.cell_size + self.margin
        y = col * self.cell_size + self.margin
        radius = self.cell_size / 2
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color)

    def get_neighbors(self, row, col):
        neighbors = []
        if row > 0: neighbors.append((row - 1, col))
        if col > 0: neighbors.append((row, col - 1))
        if row < self.size - 1: neighbors.append((row + 1, col))
        if col < self.size - 1: neighbors.append((row, col + 1))
        return neighbors

    def flood_fill(self, row, col, color):
        group = [(row, col)]
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            for nr, nc in self.get_neighbors(r, c):
                if (nr, nc) not in group and self.grid[nr][nc] == color:
                    group.append((nr, nc))
                    stack.append((nr, nc))
        return group

    def is_captured(self, group):
        for r, c in group:
            for nr, nc in self.get_neighbors(r, c):
                if self.grid[nr][nc] is None:
                    return False
        return True

    def remove_captured_stones(self, color):
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == color:
                    group = self.flood_fill(i, j, color)
                    if self.is_captured(group):
                        for r, c in group:
                            self.grid[r][c] = None
                            # self.score[self.current_player] += 1
                            self.score[color] += 1
                        self.draw_board()

    def handle_click(self, event):
        if isinstance(self.current_player, HumanPlayer):
        # 确保 event 不为 None
            if event is None:
                print("Error: Event is None in handle_click.")
                return
        if self.current_player.make_move(self, event):
            self.switch_player()
        if isinstance(self.current_player, HumanPlayer):
            if self.current_player.make_move(self, event):
                self.switch_player()
        # if isinstance(self.current_player, RandomPlayer):
        #     if self.current_player.make_move(self, event):
        #         self.switch_player()

    def switch_player(self):
        """
        切换到下一个玩家。
        """
        next_color = 'white' if self.current_player.color == 'black' else 'black'
        self.current_player = self.players[next_color]
        if isinstance(self.current_player, AIPlayer):
            self.after(100, self.make_ai_move)

    def make_ai_move(self):
        """
        让 AI 玩家下棋。
        """
        if self.current_player.make_move(self):
            self.switch_player()

    # 悔棋功能
    def undo_move(self):
        # last_move = self.grid.pop()
        last_move = self.moves_history.pop()
        row, col ,color = last_move
        self.grid[row][col] = None
        self.draw_board()
        self.switch_player()
        print(f"Undo move: {last_move}")
        print(f"Current player: {self.current_player}")
        self.remove_captured_stones(self.current_player)


    def moves_history_store(self, row, col, color):
        self.moves_history.append((row, col, color))


    # 提示功能

# 根据棋盘状态判断胜负，子多的获胜，其中黑子因为先手贴目要-7.5(19路)，白子不变
# 判断胜负的并输出结果
    def judge_winner(self):

        black_score = sum(row.count('black') for row in self.grid)- self.reduce
        white_score = sum(row.count('white') for row in self.grid)
        
        if black_score > white_score:
            result= 'black'
        elif white_score >= black_score:
            result= 'white'
        # 输出结果
        print(f"黑子得分：{black_score}，白子得分：{white_score}")
        print(f"胜者：{result}")

        messagebox.showinfo("比赛结果", f"黑子得分：{black_score}，白子得分：{white_score}\n胜者：{result}")
        # messagebox.showinfo("比赛结果", result)

        return result
