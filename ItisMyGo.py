import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QMenuBar, QMenu
from goboard import GoBoard


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    main_window = QMainWindow()
    main_window.setWindowTitle("围棋游戏")
    main_window.setGeometry(100, 100, 700, 700)
    
    board = GoBoard(size=19)
    main_window.setCentralWidget(board)
    
    # 创建菜单栏 - 强制在窗口内显示（不使用系统菜单栏）
    menubar = main_window.menuBar()
    # menubar.setNativeMenuBar(False)  # 在窗口内显示菜单栏，而不是系统菜单栏
    
    # 对弈模式菜单
    mode_menu = menubar.addMenu("对弈模式")
    mode_menu.addAction("人人对弈", lambda: board.setup_mode("human_vs_human"))
    mode_menu.addAction("人机对弈", lambda: board.setup_mode("human_vs_ai"))
    mode_menu.addAction("随机对弈", lambda: board.setup_mode("random_vs_random"))
    
    # 选项菜单
    file_menu = menubar.addMenu("选项")
    file_menu.addAction("新游戏", board.reset)
    file_menu.addAction("判断胜负", board.judge_winner)
    file_menu.addAction("悔棋", board.undo_move)
    
    main_window.show()
    sys.exit(app.exec())
