from __future__ import annotations

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


def temperature_policy(policy, temperature):
    """Transform a copy for move sampling; never mutate the training target."""
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    policy = np.asarray(policy, dtype=np.float64)
    if not np.isfinite(policy).all() or np.any(policy < 0) or policy.sum() <= 0:
        raise ValueError("policy must be a finite, nonnegative distribution")
    if temperature <= 1e-6:
        result = np.zeros_like(policy)
        result[np.argmax(policy)] = 1.0
    else:
        positive = policy > 0
        logits = np.log(policy[positive]) / temperature
        result = np.zeros_like(policy)
        result[positive] = np.exp(logits - logits.max())
        result /= result.sum()
    return result.astype(np.float32)


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
        if self.num_simulations < 1:
            raise ValueError("num_simulations must be at least 1")

    def _evaluate_batch(self, environments, colors, device):
        features = getattr(self.model, "input_features", "stones-v1")
        states = torch.stack([encode_state(env, color, features) for env, color in zip(environments, colors)]).to(device)
        self.model.eval()
        with torch.inference_mode():
            logits, values = self.model(states)
            policies = torch.softmax(logits, dim=1).cpu().numpy()
            values = values.flatten().cpu().numpy()
        return list(zip(policies, values))

    def _legal_actions(self, env, current_color: str, allow_pass: bool = True):
        legal_moves = env.legal_moves(current_color)
        if allow_pass or not legal_moves:
            return legal_moves + [PASS_MOVE]
        return legal_moves

    def _expand(self, node: MCTSNode, env, current_color: str, prediction, allow_pass: bool = True):
        legal_actions = self._legal_actions(env, current_color, allow_pass=allow_pass)
        policy, value = prediction

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
            env.register_pass(current_color)
            return
        row, col = action
        if not env.place_stone(row, col, current_color):
            raise RuntimeError(f"MCTS selected an illegal move: {action}")

    def _terminal_value(self, env, current_color: str) -> float:
        black_score, white_score = env.calculate_area_score()
        if black_score == white_score:
            return 0.0
        winner = "black" if black_score > white_score else "white"
        return 1.0 if winner == current_color else -1.0

    def _backpropagate(self, search_path, leaf_value: float):
        value = leaf_value
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    @staticmethod
    def _add_root_noise(root, alpha, fraction, rng):
        if not 0 <= fraction <= 1 or alpha <= 0:
            raise ValueError("Root noise requires alpha > 0 and fraction in [0, 1]")
        if fraction == 0:
            return
        noise = rng.dirichlet(np.full(len(root.children), alpha))
        for child, sample in zip(root.children.values(), noise):
            child.prior_prob = (1 - fraction) * child.prior_prob + fraction * float(sample)

    @staticmethod
    def _root_policy(root, size):
        counts = np.zeros(size * size + 1, dtype=np.float32)
        for action, child in root.children.items():
            counts[action_to_index(action, size)] = child.visit_count
        if counts.sum() <= 0:
            for action, child in root.children.items():
                counts[action_to_index(action, size)] = child.prior_prob
        if counts.sum() <= 0 or not np.isfinite(counts).all():
            raise RuntimeError("Search produced no finite legal policy")
        return counts / counts.sum()

    def get_action_probs_batch(
        self, root_envs, current_colors, temperature=1.0, device="cpu",
        allow_pass=True, add_root_noise=False, dirichlet_alpha=0.1,
        noise_fraction=0.25, rng=None, min_moves_before_pass=0,
    ):
        """Search independent games together; one GPU forward per simulation round.

        Temperature=1 preserves normalized visit counts for policy supervision.
        Exploration noise is opt-in, so GUI/evaluation searches remain unchanged.
        """
        if not root_envs or len(root_envs) != len(current_colors):
            raise ValueError("Provide equally sized, non-empty positions and colors")
        if any(env.game_over for env in root_envs):
            raise ValueError("Cannot search a finished game")
        if len({env.size for env in root_envs}) != 1:
            raise ValueError("Batched positions must have the same board size")
        rng = rng if rng is not None else np.random.default_rng()
        roots = [MCTSNode() for _ in root_envs]
        for _ in range(self.num_simulations):
            pending = []
            for root, root_env, color in zip(roots, root_envs, current_colors):
                env = root_env.clone_for_search()
                node, player, path = root, color, [root]
                while node.is_expanded():
                    action, child = self._select_child(node)
                    self._apply_action(env, action, player)
                    node = child
                    path.append(node)
                    player = other_color(player)
                    if env.game_over:
                        break
                if env.game_over:
                    self._backpropagate(path, self._terminal_value(env, player))
                else:
                    pending.append((root, node, env, player, path))
            if not pending:
                continue
            predictions = self._evaluate_batch(
                [item[2] for item in pending], [item[3] for item in pending], device,
            )
            for (root, node, env, player, path), prediction in zip(pending, predictions):
                leaf_allow_pass = allow_pass and len(env.moves_history) >= min_moves_before_pass
                value = self._expand(node, env, player, prediction, leaf_allow_pass)
                if node is root and add_root_noise:
                    self._add_root_noise(root, dirichlet_alpha, noise_fraction, rng)
                self._backpropagate(path, float(value))
        self.last_root_values = [root.q_value for root in roots]
        return [temperature_policy(self._root_policy(root, env.size), temperature)
                for root, env in zip(roots, root_envs)]

    def get_action_probs(
        self, root_env, current_color, temperature=1.0, device="cpu",
        allow_pass=True, **search_options,
    ):
        return self.get_action_probs_batch(
            [root_env], [current_color], temperature=temperature, device=device,
            allow_pass=allow_pass, **search_options,
        )[0]

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
