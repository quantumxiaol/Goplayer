from __future__ import annotations

from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - depends on optional rl deps
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - depends on optional rl deps
    torch = None


def _require_rl_array_runtime() -> None:
    if np is None or torch is None:
        raise ImportError("RL dependencies are missing. Install with: uv sync --extra rl")


def encode_state(env: Any, current_color: str):
    """
    Encode GoEnv state as a tensor with shape [3, board_size, board_size].

    Channel layout:
    - 0: current player's stones
    - 1: opponent stones
    - 2: color indicator plane (all ones for black, all zeros for white)
    """
    _require_rl_array_runtime()

    size = env.size
    state = np.zeros((3, size, size), dtype=np.float32)
    opponent = "white" if current_color == "black" else "black"

    for row in range(size):
        for col in range(size):
            cell = env.grid[row][col]
            if cell == current_color:
                state[0, row, col] = 1.0
            elif cell == opponent:
                state[1, row, col] = 1.0

    if current_color == "black":
        state[2, :, :] = 1.0

    return torch.from_numpy(state)
