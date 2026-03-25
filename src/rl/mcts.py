from __future__ import annotations

import copy
import math
from typing import Dict, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - depends on optional rl deps
    torch = None

from .encoder import encode_state

PASS_MOVE = (-1, -1)


def other_color(color: str) -> str:
    return "white" if color == "black" else "black"


def action_to_index(action: Tuple[int, int], board_size: int) -> int:
    if action == PASS_MOVE:
        return board_size * board_size
    row, col = action
    return row * board_size + col


def index_to_action(index: int, board_size: int) -> Tuple[int, int]:
    if index == board_size * board_size:
        return PASS_MOVE
    return index // board_size, index % board_size


class MCTSNode:
    def __init__(self, parent: "MCTSNode | None" = None, prior_prob: float = 1.0):
        self.parent = parent
        self.children: Dict[Tuple[int, int], MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = float(prior_prob)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visit_count: int, c_puct: float) -> float:
        exploration = c_puct * self.prior_prob * math.sqrt(max(parent_visit_count, 1)) / (
            1 + self.visit_count
        )
        # q_value is stored from this node's current player's perspective.
        # During selection at the parent, this value must be negated so the parent
        # compares actions from the parent's perspective.
        return -self.q_value + exploration

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    """
    AlphaZero-style Monte Carlo Tree Search guided by a policy-value network.
    """

    def __init__(self, model, c_puct: float = 1.5, num_simulations: int = 200):
        if torch is None:
            raise ImportError("PyTorch is required for MCTS. Install with: uv sync --extra rl")
        self.model = model
        self.c_puct = float(c_puct)
        self.num_simulations = int(num_simulations)

    def _evaluate_policy_value(self, env, current_color: str, device):
        state = encode_state(env, current_color).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(state)
            policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value_scalar = float(value.item())
        return policy, value_scalar

    def _legal_actions(self, env, current_color: str, allow_pass: bool = True):
        legal_moves = env.legal_moves(current_color)
        if allow_pass or not legal_moves:
            return legal_moves + [PASS_MOVE]
        return legal_moves

    def _expand(self, node: MCTSNode, env, current_color: str, device, allow_pass: bool = True):
        legal_actions = self._legal_actions(env, current_color, allow_pass=allow_pass)
        policy, value = self._evaluate_policy_value(env, current_color, device)

        action_size = env.size * env.size + 1
        mask = np.zeros(action_size, dtype=np.float32)
        for action in legal_actions:
            mask[action_to_index(action, env.size)] = 1.0

        masked_policy = policy * mask
        total = float(masked_policy.sum())
        if total <= 1e-12:
            masked_policy = mask / max(mask.sum(), 1.0)
        else:
            masked_policy /= total

        for action in legal_actions:
            idx = action_to_index(action, env.size)
            node.children[action] = MCTSNode(parent=node, prior_prob=float(masked_policy[idx]))

        return value

    def _select_child(self, node: MCTSNode):
        parent_visits = max(node.visit_count, 1)
        best_action = None
        best_child = None
        best_score = -float("inf")
        for action, child in node.children.items():
            score = child.ucb_score(parent_visits, self.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _apply_action(self, env, action: Tuple[int, int], current_color: str):
        if action == PASS_MOVE:
            env.register_pass()
            return
        row, col = action
        if not env.place_stone(row, col, current_color):
            # Should be rare because actions are legal by construction.
            env.register_pass()

    def _terminal_value(self, env, current_color: str) -> float:
        black_score, white_score = env.calculate_area_score()
        winner = "black" if black_score > white_score else "white"
        return 1.0 if winner == current_color else -1.0

    def _backpropagate(self, search_path, leaf_value: float):
        value = leaf_value
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def get_action_probs(
        self,
        root_env,
        current_color: str,
        temperature: float = 1.0,
        device="cpu",
        allow_pass: bool = True,
    ):
        root = MCTSNode()

        for _ in range(self.num_simulations):
            node = root
            env = copy.deepcopy(root_env)
            player = current_color
            search_path = [node]

            while node.is_expanded():
                action, child = self._select_child(node)
                if child is None:
                    break
                self._apply_action(env, action, player)
                node = child
                search_path.append(node)
                player = other_color(player)
                if env.game_over:
                    break

            if env.game_over:
                leaf_value = self._terminal_value(env, player)
            else:
                leaf_value = self._expand(node, env, player, device, allow_pass=allow_pass)
            self._backpropagate(search_path, leaf_value)

        action_size = root_env.size * root_env.size + 1
        visit_counts = np.zeros(action_size, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action_to_index(action, root_env.size)] = float(child.visit_count)

        if visit_counts.sum() <= 0:
            return np.ones(action_size, dtype=np.float32) / action_size

        def _argmax_one_hot():
            probs = np.zeros_like(visit_counts, dtype=np.float32)
            probs[int(np.argmax(visit_counts))] = 1.0
            return probs

        if temperature <= 1e-6:
            return _argmax_one_hot()

        positive_mask = visit_counts > 0
        if not np.any(positive_mask):
            return np.ones(action_size, dtype=np.float32) / action_size

        # Use log-space temperature scaling to avoid overflow when temperature is very small.
        logits = np.full(action_size, -np.inf, dtype=np.float64)
        logits[positive_mask] = np.log(visit_counts[positive_mask]) / float(temperature)

        max_logit = np.max(logits[positive_mask])
        adjusted = np.zeros(action_size, dtype=np.float64)
        adjusted[positive_mask] = np.exp(logits[positive_mask] - max_logit)

        adjusted_sum = float(adjusted.sum())
        if not np.isfinite(adjusted_sum) or adjusted_sum <= 1e-12:
            return _argmax_one_hot()

        probs = (adjusted / adjusted_sum).astype(np.float32)
        if not np.all(np.isfinite(probs)):
            return _argmax_one_hot()
        return probs

    def get_action(
        self,
        root_env,
        current_color: str,
        temperature: float = 1.0,
        device="cpu",
        deterministic: bool = True,
        allow_pass: bool = True,
    ):
        probs = self.get_action_probs(
            root_env,
            current_color,
            temperature=temperature,
            device=device,
            allow_pass=allow_pass,
        )
        if deterministic:
            idx = int(np.argmax(probs))
        else:
            idx = int(np.random.choice(len(probs), p=probs))
        return index_to_action(idx, root_env.size)
