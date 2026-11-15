"""Issue #7 acceptance on the real ALE/Pong-v5 ROM (no trained agent needed)."""

import numpy as np

from miae01_dqn.config import EnvSpec, EvalSpec
from miae01_dqn.envs import EnvFactory
from miae01_dqn.evaluator import Evaluator


class UnusedAgent:
    def act_greedy(self, obs):
        raise AssertionError("epsilon=1 must select only random actions")


def test_random_policy_30_pong_games():
    result = Evaluator(
        UnusedAgent(), EnvFactory(EnvSpec()), EvalSpec(epsilon=1.0), seed=2260031814,
        random_score=-20.7, human_score=9.3,
    ).evaluate(0)
    assert len(result.scores) == 30
    assert np.isfinite(result.scores).all()
    assert -23.7 <= result.mean <= -17.7
    assert result.std == np.std(result.scores)
    print(f"\nRandom Pong acceptance: {result._asdict()}")
