"""Configuration dataclasses. Implementation: issue I-01."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
        raise NotImplementedError("I-01: EnvSpec.obs_shape")


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
        raise NotImplementedError("I-01: AlgoSpec.total_frames")

    @property
    def replay_start_steps(self) -> int:
        raise NotImplementedError("I-01: AlgoSpec.replay_start_steps")


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
        raise NotImplementedError("I-01: RunSpec.run_dir")


@dataclass(frozen=True)
class RunConfig:
    env: EnvSpec = field(default_factory=EnvSpec)
    optim: OptimSpec = field(default_factory=OptimSpec)
    algo: AlgoSpec = field(default_factory=AlgoSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)
    run: RunSpec = field(default_factory=RunSpec)

    def with_overrides(self, **flat: Any) -> "RunConfig":
        raise NotImplementedError("I-01: RunConfig.with_overrides")
