"""Q-network for Atari, following the paper architecture.

Implementation: issue I-02.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """conv 32@8x8/s4 -> 64@4x4/s2 -> 64@3x3/s1 -> FC512 -> n_actions."""

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int) -> None:
        super().__init__()
        raise NotImplementedError("I-02: QNetwork.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("I-02: QNetwork.forward")
