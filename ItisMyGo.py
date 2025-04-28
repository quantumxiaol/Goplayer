import tkinter as tk
import tkinter.messagebox as messagebox
import copy
import random
import openai
import os
from readConfig import get_openai_config
from goboard import GoBoard
from goplayer import HumanPlayer, AIPlayer


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('700x700')  # You can set the geometry to be the size you want
    board = GoBoard(root, size=19)
    board.pack(fill=tk.BOTH, expand=1)
    # 创建菜单栏
    menubar = tk.Menu(root)
    mode_menu = tk.Menu(menubar, tearoff=0)
    mode_menu.add_command(label="人人对弈", command=lambda: board.setup_mode("human_vs_human"))
    mode_menu.add_command(label="人机对弈", command=lambda: board.setup_mode("human_vs_ai"))
    mode_menu.add_command(label="随机对弈", command=lambda: board.setup_mode("random_vs_random"))
    menubar.add_cascade(label="对弈模式", menu=mode_menu)

    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="新游戏", command=board.reset)
    file_menu.add_command(label="判断胜负", command=board.judge_winner)
    file_menu.add_command(label="悔棋", command=board.undo_move)
    menubar.add_cascade(label="选项", menu=file_menu)

    # 将菜单栏绑定到主窗口
    root.config(menu=menubar)

    root.mainloop()