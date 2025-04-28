# 判断棋盘状态、这一步是否合法

# 检查棋子是否有气
def has_liberty(board, row, col, color):
    size = board.size
    visited = set()
    stack = [(row, col)]

    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))

        # 检查四个方向
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                if board.grid[nr][nc] is None:
                    return True  # 找到气
                elif board.grid[nr][nc] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))

    return False  # 没有气
# 判断是否是虽然无气但可以提子
def is_capture_move(board, row, col, color):
    if board.grid[row][col] is not None:
        return False  # 落子位置已经有棋子
    if not has_liberty(board, row, col, color):
        # 检查是否有气
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < board.size and 0 <= nc < board.size:
                if board.grid[nr][nc] is not None and board.grid[nr][nc] != color:
                    # 检查对方棋子是否被提掉
                    if not has_liberty(board, nr, nc, board.grid[nr][nc]):
                        return True
    return False  # 没有提子

# 判断落子是否合法
def is_valid_move(board, row, col, color):
    size = board.size
    if row < 0 or row >= size or col < 0 or col >= size:
        return False
    if board.grid[row][col] is not None:
        return False
    # 检查是否有气
    if has_liberty(board, row, col, color):
        return True
    # 检查是否是提子
    if is_capture_move(board, row, col, color):
        return True
    return True

