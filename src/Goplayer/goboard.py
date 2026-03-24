import random

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QMessageBox, QWidget

from .goenv import GoEnv
from .goplayer import AIPlayer, AlphaZeroPlayer, HumanPlayer, RandomPlayer


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
        self.env = GoEnv(size=size)
        self.size = self.env.size
        self.current_player = None
        self.players = {}
        self.mode = "human_vs_human"
        self.margin = 25
        self.cell_size = 0
        self.ai_thread = None
        self.ai_worker = None
        self.ai_thinking = False
        self.setMinimumSize(700, 700)
        self.setup_players()

    # 兼容旧调用：将环境状态暴露为 Board 属性
    @property
    def grid(self):
        return self.env.grid

    @grid.setter
    def grid(self, value):
        self.env.grid = value

    @property
    def position_history(self):
        return self.env.position_history

    @position_history.setter
    def position_history(self, value):
        self.env.position_history = value

    @property
    def moves_history(self):
        return self.env.moves_history

    @property
    def consecutive_passes(self):
        return self.env.consecutive_passes

    @consecutive_passes.setter
    def consecutive_passes(self, value):
        self.env.consecutive_passes = value

    @property
    def game_over(self):
        return self.env.game_over

    @game_over.setter
    def game_over(self, value):
        self.env.game_over = value

    @property
    def score(self):
        return self.env.captures

    @score.setter
    def score(self, value):
        self.env.captures = value

    def board_signature(self, grid=None):
        return self.env.board_signature(grid)

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
        self.env.reset()
        self.update()
        self.setup_players()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(255, 218, 185))

        width = self.width()
        height = self.height()
        self.cell_size = (min(width, height) - 2 * self.margin) / (self.size - 1)

        pen = QPen(QColor(0, 0, 0), 3)
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
        placed = self.env.place_stone(row, col, color)
        if placed:
            self.update()
        return placed

    def setup_players(self):
        if self.mode == "human_vs_human":
            self.players["black"] = HumanPlayer("black")
            self.players["white"] = HumanPlayer("white")
        elif self.mode == "human_vs_ai":
            self.players["black"] = HumanPlayer("black")
            self.players["white"] = AIPlayer("white")
        elif self.mode == "human_vs_alphazero":
            self.players["black"] = HumanPlayer("black")
            self.players["white"] = AlphaZeroPlayer("white")
        elif self.mode == "random_vs_random":
            self.players["black"] = RandomPlayer("black")
            self.players["white"] = RandomPlayer("white")

        self.current_player = self.players["black"]
        self.update()

        if self.mode == "random_vs_random" and isinstance(self.current_player, RandomPlayer):
            QTimer.singleShot(100, self.make_ai_move)

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

        painter.setBrush(QBrush(QColor(0, 0, 0)))
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
        painter.setBrush(QBrush(stone_color))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
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
        if isinstance(self.current_player, (AIPlayer, RandomPlayer, AlphaZeroPlayer)):
            QTimer.singleShot(100, self.make_ai_move)

    def _all_legal_moves(self, color):
        return self.env.legal_moves(color)

    def _register_pass(self):
        if self.game_over:
            return
        count, ended = self.env.register_pass()
        print(f"玩家 {self.current_player.color} pass（连续pass={count}）")
        if ended:
            print("连续两次 pass，游戏结束")
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
        if isinstance(self.current_player, (RandomPlayer, AlphaZeroPlayer)):
            pass_streak_before_move = self.consecutive_passes
            if self.current_player.make_move(self):
                if pass_streak_before_move > 0:
                    print(
                        f"玩家 {self.current_player.color} 落子 {self.current_player.move}，已打断连续pass"
                    )
                self.switch_player()
                return
            self._register_pass()

    def undo_move(self):
        undone_move = self.env.undo()
        if undone_move is None:
            QMessageBox.information(self, "提示", "没有可以悔棋的步骤")
            return
        self._stop_ai_worker()
        self.current_player = self.players[undone_move[2]]
        self.update()

    # 兼容旧接口
    def moves_history_store(self, row, col, color):
        return

    def calculate_area_score(self):
        return self.env.calculate_area_score()

    def judge_winner(self):
        self._stop_ai_worker()
        result = self.env.judge_winner()
        black_score = result["black_score"]
        white_score = result["white_score"]
        winner = result["winner"]

        print(f"黑子得分：{black_score:.1f}，白子得分：{white_score:.1f}")
        print(f"胜者：{winner}")
        QMessageBox.information(
            self,
            "比赛结果",
            f"黑子得分：{black_score:.1f}，白子得分：{white_score:.1f}\n胜者：{winner}",
        )
        return winner
