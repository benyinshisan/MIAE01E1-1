"""Step-driven training loop. Implementation: issue I-06."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

from miae01_dqn.agent import Agent
from miae01_dqn.buffer import ReplayBuffer
from miae01_dqn.config import RunConfig
from miae01_dqn.envs import EnvFactory

if TYPE_CHECKING:
    from miae01_dqn.runner import Logger


class TrainResult(NamedTuple):
    agent_steps: int
    frames: int
    updates: int
    episodes: int
    wall_clock_s: float
    final_return_mean: float


class Trainer(Protocol):
    def __init__(
        self,
        cfg: RunConfig,
        agent: Agent,
        env_factory: EnvFactory,
        buffer: ReplayBuffer,
        logger: "Logger",
    ) -> None: ...

    def run(self) -> TrainResult: ...
