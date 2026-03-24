from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - depends on optional rl deps
    torch = None
    nn = None
    F = None


if nn is None:  # pragma: no cover - executed only without optional deps
    class GoNet:
        def __init__(self, *_args, **_kwargs):
            raise ImportError("PyTorch is required. Install with: uv sync --extra rl")


else:
    class ResBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + residual
            return F.relu(out)


    class GoNet(nn.Module):
        """
        AlphaZero-style dual-head residual network.
        Outputs policy logits and value in [-1, 1].
        """

        def __init__(self, size: int = 9, num_channels: int = 64, num_res_blocks: int = 3):
            super().__init__()
            self.size = size
            self.action_size = size * size + 1  # +1 for pass

            self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])

            self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
            self.policy_bn = nn.BatchNorm2d(2)
            self.policy_fc = nn.Linear(2 * size * size, self.action_size)

            self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_fc1 = nn.Linear(size * size, 64)
            self.value_fc2 = nn.Linear(64, 1)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            for block in self.res_blocks:
                x = block(x)

            policy = F.relu(self.policy_bn(self.policy_conv(x)))
            policy = policy.flatten(1)
            policy_logits = self.policy_fc(policy)

            value = F.relu(self.value_bn(self.value_conv(x)))
            value = value.flatten(1)
            value = F.relu(self.value_fc1(value))
            value = torch.tanh(self.value_fc2(value))

            return policy_logits, value
