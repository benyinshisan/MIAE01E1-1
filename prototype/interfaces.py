"""MIAE01 DQN —— 接口契约。

设计说明、不变量与理由见 docs/design/dqn-architecture.md。
本文件只有签名与数据结构，函数体均为 NotImplementedError。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- config

@dataclass(frozen=True)
class EnvSpec:
    env_id: str = "ALE/Pong-v5"
    frame_size: int = 84
    frame_stack: int = 4
    action_repeat: int = 4
    noop_max: int = 30
    frameskip: int = 1
    repeat_action_probability: float = 0.0
    episodic_life: bool = True
    clip_reward: bool = True
    grayscale: bool = True
    max_and_skip: bool = True

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        raise NotImplementedError


@dataclass(frozen=True)
class OptimSpec:
    kind: str = "rmsprop"
    lr: float = 2.5e-4
    momentum: float = 0.95
    alpha: float = 0.95
    eps: float = 0.01
    td_error_clip: float = 1.0
    grad_clip_norm: float | None = None


@dataclass(frozen=True)
class AlgoSpec:
    gamma: float = 0.99
    batch_size: int = 32
    replay_capacity: int = 200_000
    replay_start_frames: int = 50_000
    learn_every_steps: int = 4
    target_update_every_updates: int = 10_000
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_frames: int = 1_000_000
    total_agent_steps: int = 2_500_000
    action_repeat: int = 4

    @property
    def total_frames(self) -> int:
        raise NotImplementedError

    @property
    def replay_start_steps(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class EvalSpec:
    every_frames: int = 1_000_000
    episodes: int = 30
    epsilon: float = 0.05
    max_episode_frames: int = 18_000


@dataclass(frozen=True)
class RunSpec:
    seed: int = 1
    device: str = "cpu"
    run_name: str = ""
    log_dir: Path = field(default_factory=lambda: Path("runs"))
    save_checkpoints: bool = False

    @property
    def run_dir(self) -> Path:
        raise NotImplementedError


@dataclass(frozen=True)
class RunConfig:
    env: EnvSpec = field(default_factory=EnvSpec)
    optim: OptimSpec = field(default_factory=OptimSpec)
    algo: AlgoSpec = field(default_factory=AlgoSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)
    run: RunSpec = field(default_factory=RunSpec)

    def with_overrides(self, **flat: Any) -> "RunConfig":
        raise NotImplementedError


# ---------------------------------------------------------------- envs

class EnvHandle(Protocol):
    observation_space: Any
    action_space: Any

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]: ...
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]: ...
    def close(self) -> None: ...


class EnvFactory(Protocol):
    def __init__(self, spec: EnvSpec) -> None: ...
    def make_train(self, seed: int) -> EnvHandle: ...
    def make_eval(self, seed: int) -> EnvHandle: ...
    def n_actions(self) -> int: ...
    def obs_shape(self) -> tuple[int, int, int]: ...


# ---------------------------------------------------------------- buffer

class TransitionBatch(NamedTuple):
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray


class ReplayBuffer(Protocol):
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


# ---------------------------------------------------------------- network

class QNetwork(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------- agent

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


# ---------------------------------------------------------------- trainer

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


# ---------------------------------------------------------------- evaluator

class EvalResult(NamedTuple):
    at_frame: int
    scores: Sequence[float]
    mean: float
    std: float

    @property
    def normalized(self) -> float:
        raise NotImplementedError


class Evaluator(Protocol):
    def __init__(
        self,
        agent: Agent,
        env_factory: EnvFactory,
        spec: EvalSpec,
        *,
        seed: int | None = None,
    ) -> None: ...

    def evaluate(self, at_frame: int) -> EvalResult: ...


# ---------------------------------------------------------------- runner

class Logger(Protocol):
    def scalar(self, tag: str, value: float, step: int) -> None: ...
    def text(self, tag: str, text: str) -> None: ...
    def close(self) -> None: ...


def build_config(cli_or_yaml: Mapping[str, Any]) -> RunConfig:
    raise NotImplementedError


def run_experiment(cfg: RunConfig) -> dict[str, Any]:
    raise NotImplementedError
