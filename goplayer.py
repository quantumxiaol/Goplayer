import tkinter as tk
import tkinter.messagebox as messagebox
import copy
class GoBoard(tk.Frame):
    def __init__(self, parent, size=19):
        super().__init__(parent)
        self.size = size
        self.reduce = self.komi()
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.current_player = 'black'
        self.score = {'black': 0, 'white': 0}
        self.canvas = tk.Canvas(self, bg='peach puff')
        self.canvas.bind('<Configure>', self.draw_board)
        self.canvas.bind('<Button-1>', self.place_stone)
        self.canvas.pack(fill=tk.BOTH, expand=1)

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
    def place_stone(self, event):
        row = round((event.x - self.margin) / self.cell_size)
        col = round((event.y - self.margin) / self.cell_size)
        if 0 <= row < self.size and 0 <= col < self.size and self.grid[row][col] is None:
            # 保留一个棋盘状态的副本
            old_grid = copy.deepcopy(self.grid)
            self.grid[row][col] = self.current_player
            self.draw_stone(row, col, self.current_player)
            self.remove_captured_stones('white' if self.current_player == 'black' else 'black')
            
            # 检查是否形成禁入点
            if self.grid[row][col] is not None and self.is_captured(self.flood_fill(row, col, self.current_player)):
                # 如果形成禁入点，恢复到先前的棋盘状态并阻止落子
                self.grid = old_grid
                self.draw_board()
            else:
                self.current_player = 'white' if self.current_player == 'black' else 'black'
        print(self.score)

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
                            self.score[self.current_player] += 1
                        self.draw_board()

# 根据棋盘状态判断胜负，子多的获胜，其中黑子因为先手贴目要-7.5(19路)，白子不变
# 关闭棋盘是判断胜负的并输出结果
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

        messagebox.showinfo("比赛结果", result)

        return result




if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('700x700')  # You can set the geometry to be the size you want
    board = GoBoard(root, size=19)
    board.pack(fill=tk.BOTH, expand=1)
    # 创建菜单栏
    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="New Game", command=board.reset)  # 使用 reset 方法重置棋盘
    file_menu.add_command(label="Judge Winner", command=lambda: print(board.judge_winner()))
    menubar.add_cascade(label="option", menu=file_menu)

    # 将菜单栏绑定到主窗口
    root.config(menu=menubar)

    root.mainloop()
