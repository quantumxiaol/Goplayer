# 判断棋盘状态、这一步是否合法

# 简单的测试棋盘类，用于模拟下棋
class TestBoard:
    def __init__(self, size, grid):
        self.size = size
        self.grid = grid

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

# 检查是否填眼（填自己的眼是不合法的）
def is_filling_eye(board, row, col, color):
    """
    检查是否填眼。
    规则：
    - 中间位置：8个相邻位置中，有7个是自己的棋子（剩下1个是空点，就是下棋的位置）
    - 边位置：5个相邻位置中，有4个是自己的棋子
    - 角位置：3个相邻位置中，有2个是自己的棋子
    - 特殊情况：如果只有上下左右4个方向是自己的，不算填眼（因为还有对角线方向不是自己的）
    """
    size = board.size
    
    # 获取所有8个方向的相邻位置
    neighbors_8 = [
        (-1, -1), (-1, 0), (-1, 1),  # 上排
        (0, -1),           (0, 1),   # 中排（左右）
        (1, -1),  (1, 0),  (1, 1)    # 下排
    ]
    
    # 获取4个基本方向（上下左右）
    neighbors_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # 获取4个对角线方向
    neighbors_diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # 统计有效的相邻位置
    valid_neighbors = []
    for dr, dc in neighbors_8:
        nr, nc = row + dr, col + dc
        if 0 <= nr < size and 0 <= nc < size:
            valid_neighbors.append((nr, nc))
    
    # 判断位置类型
    num_neighbors = len(valid_neighbors)
    
    # 统计自己的棋子数量
    own_count = sum(1 for nr, nc in valid_neighbors if board.grid[nr][nc] == color)
    
    # 如果是中间位置（8个相邻位置）
    if num_neighbors == 8:
        # 需要7个是自己的棋子（8个中有7个是自己的）
        if own_count >= 7:
            # 检查是否只有上下左右4个方向是自己的（这种情况不算填眼）
            own_4_directions = sum(1 for dr, dc in neighbors_4 
                                  if board.grid[row + dr][col + dc] == color)
            own_diagonal = sum(1 for dr, dc in neighbors_diagonal 
                              if board.grid[row + dr][col + dc] == color)
            
            # 如果只有上下左右4个方向是自己的，且对角线方向都不是自己的，不算填眼
            if own_4_directions == 4 and own_diagonal == 0:
                return False  # 只有上下左右四个方向是自己的，不算填眼
            
            return True  # 填眼
    
    # 如果是边位置（5个相邻位置）
    elif num_neighbors == 5:
        # 需要4个是自己的棋子
        if own_count >= 4:
            return True  # 填眼
    
    # 如果是角位置（3个相邻位置）
    elif num_neighbors == 3:
        # 需要2个是自己的棋子
        if own_count >= 2:
            return True  # 填眼
    
    return False  # 不是填眼

# 判断是否是虽然无气但可以提子（需要传入已经模拟放置棋子的棋盘）
def is_capture_move(board, row, col, color, test_board=None):
    if board.grid[row][col] is not None:
        return False  # 落子位置已经有棋子
    
    # 如果传入了测试棋盘，使用它；否则创建一个
    import copy
    if test_board is None:
        # 只复制 grid，而不是整个 board 对象
        test_grid = copy.deepcopy(board.grid)
        test_board = TestBoard(board.size, test_grid)
        test_board.grid[row][col] = color
    
    # 检查是否能吃掉对方棋子
    opponent_color = 'white' if color == 'black' else 'black'
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < board.size and 0 <= nc < board.size:
            if board.grid[nr][nc] == opponent_color:
                # 在模拟棋盘上检查对方棋子是否被提掉
                if not has_liberty(test_board, nr, nc, opponent_color):
                    return True  # 可以吃掉对方棋子
    return False  # 没有提子

# 判断落子是否合法
def is_valid_move(board, row, col, color):
    size = board.size
    if row < 0 or row >= size or col < 0 or col >= size:
        return False
    if board.grid[row][col] is not None:
        return False
    
    # 检查是否填眼（填自己的眼是不合法的）
    if is_filling_eye(board, row, col, color):
        return False  # 填眼不合法
    
    # 先模拟放置棋子检查是否有气或能提子
    # 只复制 grid，而不是整个 board 对象（避免 pickle 错误）
    import copy
    test_grid = copy.deepcopy(board.grid)
    test_board = TestBoard(board.size, test_grid)
    test_board.grid[row][col] = color
    
    # 检查放置后自己是否有气
    if has_liberty(test_board, row, col, color):
        return True  # 自己有气，合法
    
    # 检查是否能吃掉对方棋子（如果能吃掉对方，即使自己暂时无气也是合法的）
    # 传入 test_board 避免重复深拷贝
    if is_capture_move(board, row, col, color, test_board):
        return True
    
    # 既没有气，也不能提子，不合法
    return False

