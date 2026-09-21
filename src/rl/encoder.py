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


FEATURE_CHANNELS = {"stones-v1": 3, "pass-v2": 4}


def encode_state(env: Any, current_color: str, input_features="stones-v1"):
    """
    Encode GoEnv as [channels, board_size, board_size]; default preserves v1.

    Channel layout:
    - 0: current player's stones
    - 1: opponent stones
    - 2: color indicator plane (all ones for black, all zeros for white)
    - 3 (pass-v2 only): last move was Pass (all ones), otherwise all zeros
    """
    _require_rl_array_runtime()

    size = env.size
    state = np.zeros((FEATURE_CHANNELS[input_features], size, size), dtype=np.float32)
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
    if input_features == "pass-v2" and env.consecutive_passes > 0:
        state[3, :, :] = 1.0

    return torch.from_numpy(state)
