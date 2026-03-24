import os
import random
import math
from pathlib import Path

import openai
from dotenv import load_dotenv

from .goruler import is_valid_move

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


class GoPlayer:
    # AI棋手或人类棋手的抽象基类
    def __init__(self, color):
        self.color = color  # 'black' 或 'white'
        self.move = None

    def my_move(self):
        return [self.move, self.color]

    def make_move(self, board):
        raise NotImplementedError("This method should be implemented by subclasses.")


class HumanPlayer(GoPlayer):
    def make_move(self, board, event=None):
        if event is None:
            return False
        pos = event.pos()
        row = round((pos.x() - board.margin) / board.cell_size)
        col = round((pos.y() - board.margin) / board.cell_size)
        if board.place_stone(row, col, self.color):
            self.move = (row, col)
            self.my_move()
            return True
        return False


def get_prompt(board_state, board, color="white"):
    prompt = (
        f"当前棋盘中，black为黑子，white为白子，empty为可落子的空点\n"
        f"棋盘从左到右从上到下分别是1到{board.size}\n"
        f"当前棋盘状态如下：\n{board_state}\n"
        f"请注意，棋盘大小为 {board.size}x{board.size}。\n"
        f"你是一个围棋 AI，当前执子颜色为 {color}。\n"
        f"请根据当前棋盘状态和你的颜色，选择一个合法的落子位置。\n"
        f"合法的落子位置应该从棋盘上的空点中选择。\n"
        f"如果没有合法的落子位置，请返回 '-1,-1'。\n"
        f"请以 'row,col' 的格式返回下一步棋的位置（例如：5,7）。"
        f"请不要返回多余的解释，返回格式中只有 row 和 col 的数字。\n"
    )
    return prompt


def get_modfied_prompt(board_state, board, last_move, color="white"):
    prompt = (
        f"当前棋盘中，black为黑子，white为白子，empty为可落子的空点\n"
        f"棋盘从左到右从上到下分别是1到{board.size}\n"
        f"当前棋盘状态如下：\n{board_state}\n"
        f"请注意，棋盘大小为 {board.size}x{board.size}。\n"
        f"你是一个围棋 AI，当前执子颜色为 {color}。\n"
        f"请根据当前棋盘状态和你的颜色，选择一个合法的落子位置。\n"
        f"合法的落子位置应该从棋盘上的空点中选择。\n"
        f"如果没有合法的落子位置，请返回 '-1,-1'。\n"
        f"请以 'row,col' 的格式返回下一步棋的位置（例如：5,7）。\n"
        f"上一步棋是 {last_move}，这步棋是无效的，不要这样下，请重新考虑。\n"
        f"请不要返回多余的解释，返回格式中只有 row 和 col 的数字。\n"
    )
    return prompt


class AIPlayer(GoPlayer):
    def __init__(self, color):
        super().__init__(color)
        load_dotenv()
        model = os.getenv("OPENAI_MODEL")
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_BASE_URL")
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self.tourance = 3  # 连续三次不合法后，随机选空位

    def make_move(self, board, event=None):
        move = self.get_move_from_gpt(board)
        if move is None:
            move = self.make_random_move(board, event)
            print("AI 随机选择位置：", move)
        else:
            print("AI 选择位置：", move)

        row, col = move
        if board.place_stone(row, col, self.color):
            self.move = (row, col)
            self.my_move()
            return True
        return False

    def make_random_move(self, board, event=None):
        empty_positions = [
            (r, c) for r in range(board.size) for c in range(board.size) if board.grid[r][c] is None
        ]
        if not empty_positions:
            return False
        row, col = random.choice(empty_positions)
        return (row, col)

    def get_move_from_gpt(self, board):
        tourance = self.tourance
        board_state = "\n".join(
            [" ".join([str(cell) if cell is not None else "empty" for cell in row]) for row in board.grid]
        )

        prompt = get_prompt(board_state, board)
        messages = [
            {"role": "system", "content": "你是一个厉害的围棋选手,现在和我下棋。"},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
            )
            move_str = response.choices[0].message.content.strip()
            row, col = map(int, move_str.split(","))
            print(f"AI 玩家下棋位置：{row}, {col}")
            move = (row, col)
            is_legal_move = self.isLegelMove(board, row, col)

            while not is_legal_move:
                print(f"AI 玩家下棋位置不合法：{row}, {col}")
                tourance -= 1
                if tourance > 0:
                    prompt_modify = get_modfied_prompt(board_state, board, move_str, self.color)
                    messages = [
                        {"role": "system", "content": "你是一个厉害的围棋选手,现在和我下棋。"},
                        {"role": "user", "content": prompt_modify},
                    ]
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.1,
                    )
                    move_str = response.choices[0].message.content.strip()
                    row, col = map(int, move_str.split(","))
                    move = (row, col)
                    is_legal_move = self.isLegelMove(board, row, col)
                else:
                    print("AI 玩家连续下棋位置不合法，随机选择一个空位")
                    move = self.make_random_move(board, event=None)
                    if move:
                        row, col = move
                        print(f"AI 随机选择位置：{row}, {col}")
                    break
            return move
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self.make_random_move(board, event=None)

    def isLegelMove(self, board, row, col):
        return is_valid_move(board, row, col, self.color)


class AlphaZeroPlayer(GoPlayer):
    """
    Local AlphaZero-style player.
    Falls back to random legal moves when RL dependencies/checkpoint are unavailable.
    """

    def __init__(
        self,
        color,
        checkpoint_path=None,
        num_simulations=160,
        c_puct=1.5,
    ):
        super().__init__(color)
        load_dotenv()
        default_checkpoint = os.getenv("ALPHAZERO_CHECKPOINT_PATH", "checkpoints/best_model.pth")
        self.checkpoint_path = Path(checkpoint_path or default_checkpoint)
        raw_min_moves = os.getenv("ALPHAZERO_MIN_MOVES_BEFORE_PASS")
        self.min_moves_before_pass = None
        if raw_min_moves is not None and raw_min_moves.strip():
            try:
                self.min_moves_before_pass = max(0, int(raw_min_moves.strip()))
            except ValueError:
                self.min_moves_before_pass = None
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = None
        self.model = None
        self.mcts = None
        self._init_error = None

    def _ensure_initialized(self, board):
        if self.mcts is not None:
            return
        if self._init_error is not None:
            raise RuntimeError(self._init_error)
        if torch is None:
            raise ImportError("PyTorch not found. Install optional deps: uv sync --extra rl")

        from rl.mcts import MCTS
        from rl.net import GoNet
        from rl.utils import get_default_device, load_checkpoint

        self.device = get_default_device()
        board_size = board.size
        self.model = GoNet(size=board_size).to(self.device)

        checkpoint_file = self.checkpoint_path
        if not checkpoint_file.is_absolute():
            checkpoint_file = Path.cwd() / checkpoint_file
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        payload = load_checkpoint(checkpoint_file, map_location=self.device)
        state_dict, metadata = self._parse_checkpoint_payload(payload)
        checkpoint_board_size = metadata.get("board_size")
        if checkpoint_board_size is None:
            checkpoint_board_size = self._infer_board_size_from_state_dict(state_dict)
        if checkpoint_board_size is not None and int(checkpoint_board_size) != int(board_size):
            raise ValueError(
                f"Checkpoint board_size={checkpoint_board_size} does not match current board size={board_size}."
            )

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.mcts = MCTS(self.model, c_puct=self.c_puct, num_simulations=self.num_simulations)
        print(f"AlphaZeroPlayer loaded from {checkpoint_file} on {self.device}")

    def _parse_checkpoint_payload(self, payload):
        if not isinstance(payload, dict):
            return payload, {}

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        state_dict = payload.get("model_state_dict") or payload.get("state_dict")
        if state_dict is None:
            # Backward compatibility: older checkpoints may store raw state_dict directly.
            state_dict = payload
        return state_dict, metadata

    def _infer_board_size_from_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            return None
        policy_fc_weight = state_dict.get("policy_fc.weight")
        if policy_fc_weight is None or not hasattr(policy_fc_weight, "shape"):
            return None
        if len(policy_fc_weight.shape) != 2:
            return None

        action_size = int(policy_fc_weight.shape[0])
        flattened_features = int(policy_fc_weight.shape[1])
        board_area = action_size - 1  # pass action
        if board_area <= 0:
            return None

        side = math.isqrt(board_area)
        if side * side != board_area:
            return None
        if flattened_features != 2 * side * side:
            return None
        return side

    def _random_legal_move(self, board):
        legal_moves = board.env.legal_moves(self.color)
        if not legal_moves:
            return None
        return random.choice(legal_moves)

    def make_move(self, board, event=None):
        try:
            self._ensure_initialized(board)
        except Exception as exc:
            if self._init_error is None:
                self._init_error = str(exc)
                print(f"AlphaZero unavailable ({exc}), fallback to random.")
            move = self._random_legal_move(board)
            if move is None:
                return False
            if board.place_stone(move[0], move[1], self.color):
                self.move = move
                self.my_move()
                return True
            return False

        legal_moves = board.env.legal_moves(self.color)
        min_moves_before_pass = (
            self.min_moves_before_pass
            if self.min_moves_before_pass is not None
            else max(0, board.size * 2)
        )
        allow_pass = bool(legal_moves) and len(board.moves_history) >= min_moves_before_pass
        if not legal_moves:
            allow_pass = True

        action = self.mcts.get_action(
            board.env,
            self.color,
            temperature=1e-6,
            device=self.device,
            deterministic=True,
            allow_pass=allow_pass,
        )

        if action == (-1, -1) and not allow_pass and legal_moves:
            action = random.choice(legal_moves)

        if action == (-1, -1):
            return False

        row, col = action
        if board.place_stone(row, col, self.color):
            self.move = (row, col)
            self.my_move()
            return True

        # Rare fallback if something changed between search and apply.
        move = self._random_legal_move(board)
        if move and board.place_stone(move[0], move[1], self.color):
            self.move = move
            self.my_move()
            return True
        return False


class RandomPlayer(GoPlayer):
    # 随机落子
    def make_move(self, board, event=None):
        move = self.get_random_move(board)
        if move is None:
            return False
        if board.place_stone(move[0], move[1], self.color):
            self.move = move
            self.my_move()
            return True
        return False

    def get_random_move(self, board):
        empty_positions = [
            (r, c) for r in range(board.size) for c in range(board.size) if board.grid[r][c] is None
        ]
        if not empty_positions:
            return None
        row, col = random.choice(empty_positions)
        is_legal_move = self.isLegelMove(board, row, col)
        while not is_legal_move:
            empty_positions.remove((row, col))
            if not empty_positions:
                return None
            row, col = random.choice(empty_positions)
            is_legal_move = self.isLegelMove(board, row, col)
        return (row, col)

    def isLegelMove(self, board, row, col):
        return is_valid_move(board, row, col, self.color)
