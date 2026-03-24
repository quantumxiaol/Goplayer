import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow

# 允许 `python ItisMyGo.py` 直接从项目根目录启动
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Goplayer.goboard import GoBoard


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_window = QMainWindow()
    main_window.setWindowTitle("围棋游戏")
    main_window.setGeometry(100, 100, 700, 700)

    board = GoBoard(size=19)
    state = {"board": board}
    main_window.setCentralWidget(board)

    def current_board():
        return state["board"]

    def set_window_title(size):
        main_window.setWindowTitle(f"围棋游戏（{size}路）")

    def switch_board_size(new_size):
        old_board = current_board()
        if old_board.size == new_size:
            return
        current_mode = old_board.mode
        old_board.reset()  # 停止潜在 AI 线程

        new_board = GoBoard(size=new_size)
        state["board"] = new_board
        main_window.setCentralWidget(new_board)
        new_board.setup_mode(current_mode)
        set_window_title(new_size)

    set_window_title(board.size)

    # 创建菜单栏 - 强制在窗口内显示（不使用系统菜单栏）
    menubar = main_window.menuBar()
    # menubar.setNativeMenuBar(False)  # 在窗口内显示菜单栏，而不是系统菜单栏

    # 对弈模式菜单
    mode_menu = menubar.addMenu("对弈模式")
    mode_menu.addAction("人人对弈", lambda: current_board().setup_mode("human_vs_human"))
    mode_menu.addAction("人机对弈", lambda: current_board().setup_mode("human_vs_ai"))
    mode_menu.addAction("挑战 AlphaZero", lambda: current_board().setup_mode("human_vs_alphazero"))
    mode_menu.addAction("AlphaZero vs AlphaZero", lambda: current_board().setup_mode("alphazero_vs_alphazero"))
    mode_menu.addAction("随机对弈", lambda: current_board().setup_mode("random_vs_random"))

    # 棋盘大小菜单
    size_menu = menubar.addMenu("棋盘大小")
    size_menu.addAction("9路", lambda: switch_board_size(9))
    size_menu.addAction("13路", lambda: switch_board_size(13))
    size_menu.addAction("19路", lambda: switch_board_size(19))

    # 选项菜单
    file_menu = menubar.addMenu("选项")
    file_menu.addAction("新游戏", lambda: current_board().reset())
    file_menu.addAction("停一手", lambda: current_board().pass_turn())
    file_menu.addAction("认输", lambda: current_board().resign_current_player())
    file_menu.addAction("判断胜负", lambda: current_board().judge_winner())
    file_menu.addAction("悔棋", lambda: current_board().undo_move())

    main_window.show()
    sys.exit(app.exec())
