# 判断棋盘状态、这一步是否合法

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# 简单的测试棋盘类，用于模拟下棋
class TestBoard:
    def __init__(self, size, grid):
        self.size = size
        self.grid = grid


def clone_grid(grid):
    return [row[:] for row in grid]


def serialize_grid(grid):
    # 用紧凑字符串作为棋盘签名，便于做打劫/同形判重
    return "|".join(
        "".join("B" if cell == "black" else "W" if cell == "white" else "." for cell in row)
        for row in grid
    )


def get_group_and_liberties(board, row, col):
    color = board.grid[row][col]
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

        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board.size and 0 <= nc < board.size:
                neighbor = board.grid[nr][nc]
                if neighbor is None:
                    liberties.add((nr, nc))
                elif neighbor == color and (nr, nc) not in group:
                    stack.append((nr, nc))

    return group, liberties


# 检查棋子是否有气
def has_liberty(board, row, col, color):
    if not (0 <= row < board.size and 0 <= col < board.size):
        return False
    if board.grid[row][col] != color:
        return False
    _, liberties = get_group_and_liberties(board, row, col)
    return bool(liberties)


def simulate_move(board, row, col, color):
    """
    模拟落子：
    1. 先落子
    2. 仅检查四邻敌方连通块是否被提
    3. 判断己方新连通块是否有气
    返回：模拟后的棋盘（合法）或 None（非法）
    """
    if board.grid[row][col] is not None:
        return None

    test_grid = clone_grid(board.grid)
    test_grid[row][col] = color
    test_board = TestBoard(board.size, test_grid)

    opponent_color = "white" if color == "black" else "black"
    checked = set()

    for dr, dc in DIRS4:
        nr, nc = row + dr, col + dc
        if not (0 <= nr < board.size and 0 <= nc < board.size):
            continue
        if test_grid[nr][nc] != opponent_color or (nr, nc) in checked:
            continue

        group, liberties = get_group_and_liberties(test_board, nr, nc)
        checked.update(group)
        if not liberties:
            for gr, gc in group:
                test_grid[gr][gc] = None

    # 重新构造一次 test_board，确保提子后的局面用于气判断
    test_board = TestBoard(board.size, test_grid)
    if not has_liberty(test_board, row, col, color):
        return None
    return test_grid


# 判断是否是虽然无气但可以提子（需要传入已经模拟放置棋子的棋盘）
def is_capture_move(board, row, col, color, test_board=None):
    if board.grid[row][col] is not None:
        return False  # 落子位置已经有棋子

    if test_board is None:
        test_grid = clone_grid(board.grid)
        test_grid[row][col] = color
        test_board = TestBoard(board.size, test_grid)

    opponent_color = "white" if color == "black" else "black"
    checked = set()
    for dr, dc in DIRS4:
        nr, nc = row + dr, col + dc
        if 0 <= nr < board.size and 0 <= nc < board.size:
            if test_board.grid[nr][nc] == opponent_color and (nr, nc) not in checked:
                group, liberties = get_group_and_liberties(test_board, nr, nc)
                checked.update(group)
                if not liberties:
                    return True  # 可以吃掉对方棋子
    return False


# 判断落子是否合法
def is_valid_move(board, row, col, color):
    size = board.size
    if row < 0 or row >= size or col < 0 or col >= size:
        return False
    if board.grid[row][col] is not None:
        return False

    # 合法性仅由“非自杀（提子例外）+ 打劫/同形禁着”决定
    simulated_grid = simulate_move(board, row, col, color)
    if simulated_grid is None:
        return False

    # 若棋盘提供历史签名集合，则做同形禁着检查（Superko）
    history = getattr(board, "position_history", None)
    if history is not None:
        next_signature = serialize_grid(simulated_grid)
        if next_signature in history:
            return False

    return True
