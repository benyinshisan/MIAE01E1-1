"""DQN agent: action selection, gradient update, target sync.

Implementation: issue I-05.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np

from miae01_dqn.buffer import ReplayBuffer
from miae01_dqn.config import AlgoSpec, OptimSpec
from miae01_dqn.network import QNetwork


class Agent(Protocol):
    def __init__(
        self,
        q_network: QNetwork,
        buffer: ReplayBuffer,
        algo: AlgoSpec,
        optim: OptimSpec,
        device: str,
        *,
        seed: int | None = None,
    ) -> None: ...

    def act(self, obs: np.ndarray, epsilon: float) -> int: ...
    def act_greedy(self, obs: np.ndarray) -> int: ...
    def q_values(self, obs: np.ndarray) -> np.ndarray: ...
    def epsilon_at(self, agent_step: int, spec: AlgoSpec) -> float: ...
    def update(self) -> float | None: ...
    def sync_target(self) -> None: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
