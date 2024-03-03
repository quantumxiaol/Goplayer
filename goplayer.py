import tkinter as tk
import copy
class GoBoard(tk.Frame):
    def __init__(self, parent, size=19):
        super().__init__(parent)
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.current_player = 'black'
        self.score = {'black': 0, 'white': 0}
        self.canvas = tk.Canvas(self, bg='peach puff')
        self.canvas.bind('<Configure>', self.draw_board)
        self.canvas.bind('<Button-1>', self.place_stone)
        self.canvas.pack(fill=tk.BOTH, expand=1)

    def draw_board(self, event=None):
        self.canvas.delete('all')
        self.margin = 20
        self.width, self.height = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.cell_size = (min(self.width, self.height) - 2 * self.margin) / (self.size - 1)
        for i in range(self.size):
            x = i * self.cell_size + self.margin
            self.canvas.create_line(x, self.margin, x, self.height - self.margin)
            self.canvas.create_line(self.margin, x, self.width - self.margin, x)
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

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('800x800')  # You can set the geometry to be the size you want
    board = GoBoard(root, size=19)
    board.pack(fill=tk.BOTH, expand=1)
    root.mainloop()
