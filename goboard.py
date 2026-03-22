import random

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QMessageBox, QWidget

from goruler import is_valid_move, serialize_grid
from goplayer import AIPlayer, HumanPlayer, RandomPlayer


class BoardSnapshot:
    def __init__(self, size, grid, position_history):
        self.size = size
        self.grid = grid
        self.position_history = position_history


class AIMoveWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, player, board_snapshot):
        super().__init__()
        self.player = player
        self.board_snapshot = board_snapshot

    @pyqtSlot()
    def run(self):
        try:
            move = self.player.get_move_from_gpt(self.board_snapshot)
            self.finished.emit(move)
        except Exception as exc:
            self.failed.emit(str(exc))


class GoBoard(QWidget):
    def __init__(self, parent=None, size=19):
        super().__init__(parent)
        self.size = size
        self.reduce = self.komi()
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.current_player = None
        self.players = {}
        self.score = {"black": 0, "white": 0}
        self.mode = "human_vs_human"
        self.margin = 25
        self.cell_size = 0
        self.moves_history = []
        self.undo_stack = []
        self.position_history = {self.board_signature()}
        self.consecutive_passes = 0
        self.game_over = False
        self.ai_thread = None
        self.ai_worker = None
        self.ai_thinking = False
        self.setMinimumSize(700, 700)
        self.setup_players()

    # 根据棋盘大小设置贴目
    def komi(self):
        if self.size == 19:
            komi = 7.5
        elif self.size == 13:
            komi = 2.0
        elif self.size == 9:
            komi = 5.5
        else:
            komi = 0.0
        return komi

    def board_signature(self, grid=None):
        return serialize_grid(self.grid if grid is None else grid)

    def _stop_ai_worker(self):
        self.ai_thinking = False
        if self.ai_thread is None:
            return
        self.ai_thread.quit()
        self.ai_thread.wait(1000)
        if self.ai_worker is not None:
            self.ai_worker.deleteLater()
        self.ai_thread.deleteLater()
        self.ai_worker = None
        self.ai_thread = None

    def reset(self):
        self._stop_ai_worker()
        self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.score = {"black": 0, "white": 0}
        self.moves_history = []
        self.undo_stack = []
        self.position_history = {self.board_signature()}
        self.consecutive_passes = 0
        self.game_over = False
        self.update()
        self.setup_players()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QColor(255, 218, 185))  # peach puff

        width = self.width()
        height = self.height()
        self.cell_size = (min(width, height) - 2 * self.margin) / (self.size - 1)

        border_width = 3
        pen = QPen(QColor(0, 0, 0), border_width)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(
            int(self.margin),
            int(self.margin),
            int(width - 2 * self.margin),
            int(height - 2 * self.margin),
        )

        pen = QPen(QColor(0, 0, 0), 1)
        painter.setPen(pen)
        for i in range(self.size):
            x = int(i * self.cell_size + self.margin)
            y = int(i * self.cell_size + self.margin)
            painter.drawLine(int(self.margin), y, int(width - self.margin), y)
            painter.drawLine(x, int(self.margin), x, int(height - self.margin))

        self.draw_star_points(painter)

        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] is not None:
                    self.draw_stone(painter, i, j, self.grid[i][j])

        self.draw_last_move_marker(painter)
        self.draw_turn_indicator(painter)

    def place_stone(self, row, col, color):
        if self.game_over:
            return False
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self.grid[row][col] is not None:
            return False

        old_grid = [line[:] for line in self.grid]
        old_score = self.score.copy()
        old_history = set(self.position_history)
        old_consecutive_passes = self.consecutive_passes
        old_game_over = self.game_over

        self.grid[row][col] = color
        opponent_color = "white" if color == "black" else "black"
        self.remove_captured_stones_around(row, col, opponent_color)

        _, liberties = self.get_group_and_liberties(row, col)
        if not liberties:
            self.grid = old_grid
            self.score = old_score
            return False

        next_signature = self.board_signature()
        if next_signature in self.position_history:
            self.grid = old_grid
            self.score = old_score
            return False

        self.undo_stack.append(
            {
                "grid": old_grid,
                "score": old_score,
                "position_history": old_history,
                "consecutive_passes": old_consecutive_passes,
                "game_over": old_game_over,
            }
        )
        self.position_history.add(next_signature)
        self.moves_history.append((row, col, color))
        self.consecutive_passes = 0
        self.update()
        return True

    def setup_players(self):
        if self.mode == "human_vs_human":
            self.players["black"] = HumanPlayer("black")
            self.players["white"] = HumanPlayer("white")
        elif self.mode == "human_vs_ai":
            human_color = "black"
            ai_color = "white"
            self.players[human_color] = HumanPlayer(human_color)
            self.players[ai_color] = AIPlayer(ai_color)
        elif self.mode == "random_vs_random":
            self.players["black"] = RandomPlayer("black")
            self.players["white"] = RandomPlayer("white")

        self.current_player = self.players["black"]
        self.update()

        if self.mode == "random_vs_random" and isinstance(self.current_player, RandomPlayer):
            QTimer.singleShot(100, self.make_ai_move)

    # 设置对弈模式并重新初始化玩家
    def setup_mode(self, mode):
        self.mode = mode
        self.reset()

    def draw_star_points(self, painter):
        if self.size == 19:
            star_points = [
                (3, 3),
                (9, 3),
                (15, 3),
                (3, 9),
                (9, 9),
                (15, 9),
                (3, 15),
                (9, 15),
                (15, 15),
            ]
        elif self.size == 13:
            star_points = [
                (3, 3),
                (6, 3),
                (9, 3),
                (3, 6),
                (6, 6),
                (9, 6),
                (3, 9),
                (6, 9),
                (9, 9),
            ]
        elif self.size == 9:
            star_points = [
                (2, 2),
                (4, 2),
                (6, 2),
                (2, 4),
                (4, 4),
                (6, 4),
                (2, 6),
                (4, 6),
                (6, 6),
            ]
        else:
            return

        brush = QBrush(QColor(0, 0, 0))
        painter.setBrush(brush)
        painter.setPen(Qt.PenStyle.NoPen)
        for row, col in star_points:
            x = int(col * self.cell_size + self.margin)
            y = int(row * self.cell_size + self.margin)
            radius = int(self.cell_size / 10)
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)

    def draw_stone(self, painter, row, col, color):
        x = int(row * self.cell_size + self.margin)
        y = int(col * self.cell_size + self.margin)
        radius = int(self.cell_size / 2)

        stone_color = QColor(0, 0, 0) if color == "black" else QColor(255, 255, 255)
        brush = QBrush(stone_color)
        painter.setBrush(brush)
        pen = QPen(QColor(0, 0, 0), 1)
        painter.setPen(pen)
        painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)

    def draw_last_move_marker(self, painter):
        if not self.moves_history:
            return
        row, col, _ = self.moves_history[-1]
        x = int(row * self.cell_size + self.margin)
        y = int(col * self.cell_size + self.margin)
        radius = max(3, int(self.cell_size / 8))
        painter.setBrush(QBrush(QColor(255, 64, 64)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)

    def draw_turn_indicator(self, painter):
        if self.current_player is None:
            return
        color_cn = "黑方" if self.current_player.color == "black" else "白方"
        status = "（AI 思考中）" if self.ai_thinking else ""
        text = f"当前回合：{color_cn}{status}"
        painter.setPen(QPen(QColor(20, 20, 20), 1))
        font = painter.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(int(self.margin), int(self.margin - 8), text)

    def get_neighbors(self, row, col):
        neighbors = []
        if row > 0:
            neighbors.append((row - 1, col))
        if col > 0:
            neighbors.append((row, col - 1))
        if row < self.size - 1:
            neighbors.append((row + 1, col))
        if col < self.size - 1:
            neighbors.append((row, col + 1))
        return neighbors

    def get_group_and_liberties(self, row, col):
        color = self.grid[row][col]
        if color is None:
            return set(), set()

        group = set()
        liberties = set()
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if (r, c) in group:
                continue
            group.add((r, c))
            for nr, nc in self.get_neighbors(r, c):
                stone = self.grid[nr][nc]
                if stone is None:
                    liberties.add((nr, nc))
                elif stone == color and (nr, nc) not in group:
                    stack.append((nr, nc))
        return group, liberties

    def remove_captured_stones_around(self, row, col, color):
        checked = set()
        captured_count = 0
        for nr, nc in self.get_neighbors(row, col):
            if self.grid[nr][nc] != color or (nr, nc) in checked:
                continue
            group, liberties = self.get_group_and_liberties(nr, nc)
            checked.update(group)
            if liberties:
                continue
            for gr, gc in group:
                self.grid[gr][gc] = None
            captured_count += len(group)

        if captured_count > 0:
            self.score[color] += captured_count
        return captured_count

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if isinstance(self.current_player, HumanPlayer):
            if self.current_player.make_move(self, event):
                self.switch_player()

    def switch_player(self):
        if self.game_over:
            return
        next_color = "white" if self.current_player.color == "black" else "black"
        self.current_player = self.players[next_color]
        self.update()
        if isinstance(self.current_player, (AIPlayer, RandomPlayer)):
            QTimer.singleShot(100, self.make_ai_move)

    def _all_legal_moves(self, color):
        legal_moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] is None and is_valid_move(self, r, c, color):
                    legal_moves.append((r, c))
        return legal_moves

    def _register_pass(self):
        if self.game_over:
            return
        self.consecutive_passes += 1
        print(f"玩家 {self.current_player.color} pass（连续pass={self.consecutive_passes}）")
        if self.consecutive_passes >= 2:
            print("连续两次 pass，游戏结束")
            self.game_over = True
            self.judge_winner()
            return
        self.switch_player()

    def _start_ai_move_async(self):
        if self.ai_thread is not None:
            return
        snapshot = BoardSnapshot(
            self.size,
            [row[:] for row in self.grid],
            set(self.position_history),
        )
        self.ai_thread = QThread(self)
        self.ai_worker = AIMoveWorker(self.current_player, snapshot)
        self.ai_worker.moveToThread(self.ai_thread)
        self.ai_thread.started.connect(self.ai_worker.run)
        self.ai_worker.finished.connect(self._on_ai_move_ready)
        self.ai_worker.failed.connect(self._on_ai_move_failed)
        self.ai_worker.finished.connect(self._cleanup_ai_worker)
        self.ai_worker.failed.connect(self._cleanup_ai_worker)
        self.ai_thinking = True
        self.update()
        self.ai_thread.start()

    def _cleanup_ai_worker(self, *_):
        self.ai_thinking = False
        self._stop_ai_worker()
        self.update()

    def _on_ai_move_ready(self, move):
        if self.game_over or not isinstance(self.current_player, AIPlayer):
            return

        if isinstance(move, (tuple, list)) and len(move) == 2:
            row, col = int(move[0]), int(move[1])
            if (row, col) != (-1, -1) and self.place_stone(row, col, self.current_player.color):
                self.switch_player()
                return

        legal_moves = self._all_legal_moves(self.current_player.color)
        if legal_moves:
            row, col = random.choice(legal_moves)
            if self.place_stone(row, col, self.current_player.color):
                self.switch_player()
                return
        self._register_pass()

    def _on_ai_move_failed(self, error_message):
        print(f"AI 线程调用失败：{error_message}")
        if self.game_over:
            return
        legal_moves = self._all_legal_moves(self.current_player.color)
        if legal_moves:
            row, col = random.choice(legal_moves)
            if self.place_stone(row, col, self.current_player.color):
                self.switch_player()
                return
        self._register_pass()

    def make_ai_move(self):
        if self.game_over:
            return
        if isinstance(self.current_player, AIPlayer):
            self._start_ai_move_async()
            return
        if isinstance(self.current_player, RandomPlayer):
            pass_streak_before_move = self.consecutive_passes
            if self.current_player.make_move(self):
                if pass_streak_before_move > 0:
                    print(
                        f"玩家 {self.current_player.color} 落子 {self.current_player.move}，已打断连续pass"
                    )
                self.consecutive_passes = 0
                self.switch_player()
                return
            self._register_pass()

    # 悔棋功能
    def undo_move(self):
        if not self.undo_stack or not self.moves_history:
            QMessageBox.information(self, "提示", "没有可以悔棋的步骤")
            return

        self._stop_ai_worker()
        last_move = self.moves_history.pop()
        snapshot = self.undo_stack.pop()
        self.grid = snapshot["grid"]
        self.score = snapshot["score"]
        self.position_history = snapshot["position_history"]
        self.consecutive_passes = snapshot["consecutive_passes"]
        self.game_over = snapshot["game_over"]
        self.current_player = self.players[last_move[2]]
        self.update()

    # 兼容旧接口，避免历史调用报错
    def moves_history_store(self, row, col, color):
        if not self.moves_history or self.moves_history[-1] != (row, col, color):
            self.moves_history.append((row, col, color))

    def _explore_empty_region(self, row, col, visited):
        stack = [(row, col)]
        region = set()
        border_colors = set()

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            if self.grid[r][c] is not None:
                continue
            region.add((r, c))

            for nr, nc in self.get_neighbors(r, c):
                stone = self.grid[nr][nc]
                if stone is None and (nr, nc) not in visited:
                    stack.append((nr, nc))
                elif stone in ("black", "white"):
                    border_colors.add(stone)
        return region, border_colors

    def calculate_area_score(self):
        black_stones = sum(row.count("black") for row in self.grid)
        white_stones = sum(row.count("white") for row in self.grid)
        black_territory = 0
        white_territory = 0

        visited = set()
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] is None and (r, c) not in visited:
                    region, border_colors = self._explore_empty_region(r, c, visited)
                    if len(border_colors) == 1:
                        owner = next(iter(border_colors))
                        if owner == "black":
                            black_territory += len(region)
                        elif owner == "white":
                            white_territory += len(region)

        black_score = float(black_stones + black_territory)
        white_score = float(white_stones + white_territory + self.reduce)
        return black_score, white_score

    # 根据棋盘状态判断胜负（面积计分：棋子 + 地盘，白方加贴目）
    def judge_winner(self):
        # 统一将“判断胜负”视作终局动作，避免终局后继续调度随机/AI落子
        self.game_over = True
        self._stop_ai_worker()

        black_score, white_score = self.calculate_area_score()

        if black_score > white_score:
            result = "black"
        else:
            result = "white"

        print(f"黑子得分：{black_score:.1f}，白子得分：{white_score:.1f}")
        print(f"胜者：{result}")
        QMessageBox.information(
            self,
            "比赛结果",
            f"黑子得分：{black_score:.1f}，白子得分：{white_score:.1f}\n胜者：{result}",
        )
        return result
