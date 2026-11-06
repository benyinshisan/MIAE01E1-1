"""Replay buffer. numpy only - this module must not import torch.

Implementation: issue I-04.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np


class TransitionBatch(NamedTuple):
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray


class ReplayBuffer(Protocol):
    """Single-array ring buffer of uint8 frames. Implementation: I-04."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, int, int],
        *,
        seed: int | None = None,
    ) -> None: ...

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None: ...
    def sample(self, batch_size: int) -> TransitionBatch: ...
    def __len__(self) -> int: ...
    @property
    def capacity(self) -> int: ...
