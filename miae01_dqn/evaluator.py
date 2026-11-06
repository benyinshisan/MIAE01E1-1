"""Independent evaluation on a separate environment. Implementation: issue I-07."""

from __future__ import annotations

from typing import NamedTuple, Protocol, Sequence

from miae01_dqn.agent import Agent
from miae01_dqn.config import EvalSpec
from miae01_dqn.envs import EnvFactory


class EvalResult(NamedTuple):
    at_frame: int
    scores: Sequence[float]
    mean: float
    std: float

    @property
    def normalized(self) -> float:
        raise NotImplementedError("I-07: EvalResult.normalized")


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
