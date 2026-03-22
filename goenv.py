from goruler import serialize_grid, is_valid_move


class GoEnv:
    """
    纯围棋环境：
    - 不依赖任何 UI 框架
    - 只管理棋盘状态、规则判定、终局与计分
    """

    def __init__(self, size=19):
        self.size = size
        self.komi = self._default_komi(size)
        self.reset()

    @staticmethod
    def _default_komi(size):
        if size == 19:
            return 7.5
        if size == 13:
            return 2.0
        if size == 9:
            return 5.5
        return 0.0

    def reset(self):
        self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        # 历史兼容：记录被提掉的颜色数量（而非提子方数量）
        self.captures = {"black": 0, "white": 0}
        self.moves_history = []
        self.undo_stack = []
        self.position_history = {self.board_signature()}
        self.consecutive_passes = 0
        self.game_over = False

    def board_signature(self, grid=None):
        return serialize_grid(self.grid if grid is None else grid)

    def is_valid_move(self, row, col, color):
        return is_valid_move(self, row, col, color)

    def legal_moves(self, color):
        legal = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] is None and self.is_valid_move(r, c, color):
                    legal.append((r, c))
        return legal

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

    def _remove_captured_stones_around(self, row, col, color):
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
            self.captures[color] += captured_count
        return captured_count

    def place_stone(self, row, col, color):
        if self.game_over:
            return False
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self.grid[row][col] is not None:
            return False

        old_grid = [line[:] for line in self.grid]
        old_captures = self.captures.copy()
        old_history = set(self.position_history)
        old_consecutive_passes = self.consecutive_passes
        old_game_over = self.game_over

        self.grid[row][col] = color
        opponent_color = "white" if color == "black" else "black"
        self._remove_captured_stones_around(row, col, opponent_color)

        _, liberties = self.get_group_and_liberties(row, col)
        if not liberties:
            self.grid = old_grid
            self.captures = old_captures
            return False

        next_signature = self.board_signature()
        if next_signature in self.position_history:
            self.grid = old_grid
            self.captures = old_captures
            return False

        self.undo_stack.append(
            {
                "grid": old_grid,
                "captures": old_captures,
                "position_history": old_history,
                "consecutive_passes": old_consecutive_passes,
                "game_over": old_game_over,
            }
        )
        self.position_history.add(next_signature)
        self.moves_history.append((row, col, color))
        self.consecutive_passes = 0
        return True

    def register_pass(self):
        if self.game_over:
            return self.consecutive_passes, True
        self.consecutive_passes += 1
        if self.consecutive_passes >= 2:
            self.game_over = True
            return self.consecutive_passes, True
        return self.consecutive_passes, False

    def undo(self):
        if not self.undo_stack or not self.moves_history:
            return None
        last_move = self.moves_history.pop()
        snapshot = self.undo_stack.pop()
        self.grid = snapshot["grid"]
        self.captures = snapshot["captures"]
        self.position_history = snapshot["position_history"]
        self.consecutive_passes = snapshot["consecutive_passes"]
        self.game_over = snapshot["game_over"]
        return last_move

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
        white_score = float(white_stones + white_territory + self.komi)
        return black_score, white_score

    def judge_winner(self):
        self.game_over = True
        black_score, white_score = self.calculate_area_score()
        winner = "black" if black_score > white_score else "white"
        return {
            "black_score": black_score,
            "white_score": white_score,
            "winner": winner,
        }
