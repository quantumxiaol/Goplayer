from __future__ import annotations

import random
from collections import deque


class ReplayBuffer:
    """
    Stores AlphaZero training tuples: (state_tensor, target_policy, target_value).
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, policy, value: float):
        self.buffer.append((state, policy, float(value)))

    def save_game(self, game_history):
        for item in game_history:
            self.buffer.append(item)

    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            return [], [], []
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        return list(states), list(policies), list(values)

    def __len__(self) -> int:
        return len(self.buffer)
