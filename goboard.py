from PyQt6.QtWidgets import QWidget, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
import copy
from goplayer import HumanPlayer, AIPlayer, RandomPlayer

class GoBoard(QWidget):
    def __init__(self, parent=None, size=19):
        super().__init__(parent)
        self.size = size
        self.reduce = self.komi()
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.current_player = None  # 当前玩家对象
        self.players = {}  # 玩家字典 {'black': Player, 'white': Player}
        self.score = {'black': 0, 'white': 0}
        self.mode = "human_vs_human"  # 默认对弈模式
        self.margin = 25
        self.cell_size = 0
        self.moves_history = []  # 用于存储棋盘状态的历史记录{[row, col, color], ...}
        self.consecutive_passes = 0  # 连续 pass 的次数（两个玩家都 pass 算一次）
        self.game_over = False  # 游戏是否结束
        self.setMinimumSize(700, 700)
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
        self.score = {'black': 0, 'white': 0}
        self.moves_history = []
        self.consecutive_passes = 0  # 重置 pass 计数
        self.game_over = False  # 重置游戏状态
        self.update()  # 触发重绘
        self.setup_players()  # 重新设置玩家（会设置 current_player）

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 设置背景色
        painter.fillRect(self.rect(), QColor(255, 218, 185))  # peach puff
        
        width = self.width()
        height = self.height()
        self.cell_size = (min(width, height) - 2 * self.margin) / (self.size - 1)

        # 加粗最外边框
        border_width = 3
        pen = QPen(QColor(0, 0, 0), border_width)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(
            int(self.margin), int(self.margin),
            int(width - 2 * self.margin), int(height - 2 * self.margin)
        )

        # 绘制网格线
        pen = QPen(QColor(0, 0, 0), 1)
        painter.setPen(pen)
        for i in range(self.size):
            x = int(i * self.cell_size + self.margin)
            y = int(i * self.cell_size + self.margin)
            # 水平线
            painter.drawLine(int(self.margin), y, int(width - self.margin), y)
            # 竖直线
            painter.drawLine(x, int(self.margin), x, int(height - self.margin))

        # 绘制星位
        self.draw_star_points(painter)

        # 绘制棋子
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] is not None:
                    self.draw_stone(painter, i, j, self.grid[i][j])

    def place_stone(self, row, col, color):
        # 如果游戏已结束，不允许下棋
        if self.game_over:
            return False
            
        # 基本边界和位置检查
        if not (0 <= row < self.size and 0 <= col < self.size) or self.grid[row][col] is not None:
            return False
        
        # 保存旧棋盘状态，用于回滚
        old_grid = copy.deepcopy(self.grid)
        
        # 放置棋子
        self.grid[row][col] = color
        opponent_color = 'white' if color == 'black' else 'black'
        
        # 移除对方被吃的棋子（这一步可能会释放出新的气）
        self.remove_captured_stones(opponent_color)
        
        # 检查自己放置的棋子所在的连通块是否被吃
        # 注意：在移除对方棋子后检查，因为吃掉对方可能会给自己带来气
        my_group = self.flood_fill(row, col, color)
        if self.is_captured(my_group):
            # 如果自己的棋子被吃，回滚棋盘状态
            self.grid = old_grid
            self.update()
            return False
        
        # 走法合法，重置连续 pass 计数（因为成功下棋了）
        self.consecutive_passes = 0
        # 更新显示
        self.update()
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
        
        # 如果是随机对弈模式，自动触发第一个玩家的下棋
        if self.mode == "random_vs_random" and isinstance(self.current_player, RandomPlayer):
            QTimer.singleShot(100, self.make_ai_move)
        
    # 设置对弈模式并重新初始化玩家
    def setup_mode(self, mode):
        self.mode = mode
        self.reset() 

    def draw_star_points(self, painter):
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
        brush = QBrush(QColor(0, 0, 0))
        painter.setBrush(brush)
        painter.setPen(Qt.PenStyle.NoPen)
        for row, col in star_points:
            x = int(col * self.cell_size + self.margin)
            y = int(row * self.cell_size + self.margin)
            radius = int(self.cell_size / 10)  # 星位半径
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)

    def draw_stone(self, painter, row, col, color):
        x = int(row * self.cell_size + self.margin)
        y = int(col * self.cell_size + self.margin)
        radius = int(self.cell_size / 2)
        
        stone_color = QColor(0, 0, 0) if color == 'black' else QColor(255, 255, 255)
        brush = QBrush(stone_color)
        painter.setBrush(brush)
        pen = QPen(QColor(0, 0, 0), 1)
        painter.setPen(pen)
        painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)

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
        visited = set()  # 用于避免重复处理同一个连通块
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == color and (i, j) not in visited:
                    group = self.flood_fill(i, j, color)
                    # 标记这个连通块的所有棋子为已访问
                    for r, c in group:
                        visited.add((r, c))
                    if self.is_captured(group):
                        for r, c in group:
                            self.grid[r][c] = None
                            self.score[color] += 1
                        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # 确保 event 不为 None
            if event is None:
                print("Error: Event is None in mousePressEvent.")
                return
            # 只有人类玩家才响应点击事件
            if isinstance(self.current_player, HumanPlayer):
                if self.current_player.make_move(self, event):
                    self.switch_player()

    def switch_player(self):
        """
        切换到下一个玩家。
        """
        next_color = 'white' if self.current_player.color == 'black' else 'black'
        self.current_player = self.players[next_color]
        if isinstance(self.current_player, AIPlayer) or isinstance(self.current_player, RandomPlayer):
            QTimer.singleShot(100, self.make_ai_move)

    def make_ai_move(self):
        """
        让 AI 玩家下棋。
        """
        # 如果游戏已结束，不再下棋
        if self.game_over:
            return
            
        if isinstance(self.current_player, AIPlayer) or isinstance(self.current_player, RandomPlayer):
            if self.current_player.make_move(self):
                # 成功下棋，重置 pass 计数
                self.consecutive_passes = 0
                self.switch_player()
            else:
                # 如果下棋失败（比如没有合法走法），检查是否应该 pass
                print(f"玩家 {self.current_player.color} 无法下棋，可能没有合法走法")
                # 检查是否真的没有合法走法
                has_legal_move = False
                for r in range(self.size):
                    for c in range(self.size):
                        if self.grid[r][c] is None:
                            from goruler import is_valid_move
                            if is_valid_move(self, r, c, self.current_player.color):
                                has_legal_move = True
                                break
                    if has_legal_move:
                        break
                
                if not has_legal_move:
                    # 真的没有合法走法，记录为 pass
                    print(f"玩家 {self.current_player.color} 没有合法走法，pass")
                    self.consecutive_passes += 1
                    
                    # 如果连续两次 pass（两个玩家都 pass），结束游戏
                    if self.consecutive_passes >= 2:
                        print("连续两次 pass，游戏结束")
                        self.game_over = True
                        # 自动判断胜负
                        self.judge_winner()
                    else:
                        # 只有一次 pass，切换到对方玩家
                        self.switch_player()

    # 悔棋功能
    def undo_move(self):
        if not self.moves_history:
            QMessageBox.information(self, "提示", "没有可以悔棋的步骤")
            return
        last_move = self.moves_history.pop()
        row, col, color = last_move
        self.grid[row][col] = None
        self.update()
        # 切换到上一步的玩家
        self.switch_player()
        print(f"Undo move: {last_move}")
        print(f"Current player: {self.current_player}")

    def moves_history_store(self, row, col, color):
        self.moves_history.append((row, col, color))

    # 根据棋盘状态判断胜负，子多的获胜，其中黑子因为先手贴目要-7.5(19路)，白子不变
    # 判断胜负的并输出结果
    def judge_winner(self):
        black_score = sum(row.count('black') for row in self.grid) - self.reduce
        white_score = sum(row.count('white') for row in self.grid)
        
        if black_score > white_score:
            result = 'black'
        elif white_score >= black_score:
            result = 'white'
        # 输出结果
        print(f"黑子得分：{black_score}，白子得分：{white_score}")
        print(f"胜者：{result}")

        QMessageBox.information(self, "比赛结果", f"黑子得分：{black_score}，白子得分：{white_score}\n胜者：{result}")

        return result
