"""Q-network for Atari, following the Nature DQN architecture.

Implementation: issue I-02.
Architecture reference: Nature DQN (2015), SB3 NatureCNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """conv 32@8x8/s4 -> 64@4x4/s2 -> 64@3x3/s1 -> FC512 -> n_actions."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
    ) -> None:
        super().__init__()

        channels, height, width = obs_shape

        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            feature_dim = self.features(dummy).shape[1]

        self.q_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32) / 255.0
        return self.q_head(self.features(x))
