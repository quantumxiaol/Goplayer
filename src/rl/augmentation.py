"""Go board symmetries: transform inputs and policy together, preserving Pass."""
import numpy as np
import torch


def transform_sample(state, policy, symmetry):
    if symmetry not in range(8):
        raise ValueError("symmetry must be in 0..7")
    size = state.shape[-1]
    board_policy = np.asarray(policy[:-1]).reshape(size, size)
    state = torch.rot90(state, symmetry % 4, dims=(-2, -1))
    board_policy = np.rot90(board_policy, symmetry % 4)
    if symmetry >= 4:
        state = torch.flip(state, dims=(-1,))
        board_policy = np.flip(board_policy, axis=1)
    transformed = np.concatenate([board_policy.reshape(-1), np.asarray(policy[-1:])])
    return state.contiguous(), transformed.astype(np.float32)
