"""MIAE01 DQN replication package.

Paper: Mnih et al., Human-level control through deep reinforcement learning,
Nature 518:529-533 (2015). Design notes: docs/architecture.md.
"""

from miae01_dqn.agent import Agent
from miae01_dqn.buffer import ReplayBuffer, TransitionBatch
from miae01_dqn.config import (
    AlgoSpec,
    EnvSpec,
    EvalSpec,
    OptimSpec,
    RunConfig,
    RunSpec,
)
from miae01_dqn.envs import EnvFactory, EnvHandle
from miae01_dqn.evaluator import EvalResult, Evaluator
from miae01_dqn.network import QNetwork
from miae01_dqn.trainer import Trainer, TrainResult

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AlgoSpec",
    "EnvFactory",
    "EnvHandle",
    "EnvSpec",
    "EvalResult",
    "EvalSpec",
    "Evaluator",
    "OptimSpec",
    "QNetwork",
    "ReplayBuffer",
    "RunConfig",
    "RunSpec",
    "TrainResult",
    "Trainer",
    "TransitionBatch",
    "__version__",
]
