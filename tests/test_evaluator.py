"""Issue #7: evaluation protocol, isolation, raw returns and frame budgets."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium import spaces

from miae01_dqn.config import EvalSpec
from miae01_dqn.evaluator import EvalResult, Evaluator


class EvaluationEnv:
    def __init__(self, *, reward=2.5, end_after=3, truncate=False, reset_frames=0):
        self.action_space = spaces.Discrete(6)
        self.reward = reward
        self.end_after = end_after
        self.truncate = truncate
        self.reset_frames = reset_frames
        self.closed = False
        self.actions = []
        self.reset_count = 0
        self.frames = 0
        self.unwrapped = SimpleNamespace(ale=SimpleNamespace(
            getEpisodeFrameNumber=lambda: self.frames))

    def reset(self):
        self.reset_count += 1
        self.steps = 0
        self.frames = self.reset_frames
        return np.zeros((4, 84, 84), dtype=np.uint8), {}

    def step(self, action):
        self.actions.append(action)
        self.steps += 1
        self.frames += 4
        done = self.steps == self.end_after
        return np.zeros((4, 84, 84), dtype=np.uint8), self.reward, (
            done and not self.truncate), (done and self.truncate), {}

    def close(self):
        self.closed = True


class Factory:
    def __init__(self, **options):
        self.options = options
        self.environments = []
        self.seeds = []

    def make_eval(self, seed):
        self.seeds.append(seed)
        env = EvaluationEnv(**self.options)
        self.environments.append(env)
        return env

    def make_train(self, seed):
        raise AssertionError("evaluation must not create a training environment")


class GreedyAgent:
    def __init__(self):
        self.epsilon = 0.6
        self.rng = np.random.default_rng(101)
        self.calls = 0

    def act_greedy(self, obs):
        assert not torch.is_grad_enabled()
        self.calls += 1
        return 2

    def act(self, *args):
        raise AssertionError("do not consume training exploration state")

    def update(self):
        raise AssertionError("evaluation must not update the agent or buffer")


@pytest.mark.parametrize("truncate", [False, True])
def test_raw_scores_statistics_and_injected_baselines(truncate):
    factory = Factory(truncate=truncate)
    evaluator = Evaluator(GreedyAgent(), factory, EvalSpec(), seed=1,
                          random_score=-10.0, human_score=10.0)
    result = evaluator.evaluate(1234)
    assert result.at_frame == 1234
    assert result.scores == (7.5,) * 30
    assert result.mean == 7.5
    assert result.std == 0.0
    assert result.normalized == 87.5
    assert factory.environments[0].reset_count == 30
    assert factory.environments[0].closed


def test_evaluation_is_repeatable_and_preserves_training_random_state():
    a, b = GreedyAgent(), GreedyAgent()
    fa, fb = Factory(end_after=500), Factory(end_after=500)
    before = a.rng.bit_generator.state
    torch_before = torch.random.get_rng_state().clone()
    ea = Evaluator(a, fa, EvalSpec(episodes=2), seed=123)
    eb = Evaluator(b, fb, EvalSpec(episodes=2), seed=123)
    assert ea.evaluate(0) == eb.evaluate(0)
    assert fa.seeds == fb.seeds
    assert fa.environments[0].actions == fb.environments[0].actions
    assert 20 < sum(x != 2 for x in fa.environments[0].actions) < 70
    assert a.epsilon == 0.6 and a.rng.bit_generator.state == before
    assert torch.equal(torch_before, torch.random.get_rng_state())
    ea.evaluate(100)
    assert fa.environments[0] is not fa.environments[1]
    assert fa.seeds[0] != fa.seeds[1]
    assert all(e.closed for e in fa.environments)


@pytest.mark.parametrize("epsilon,expected_calls", [(0.0, 30), (1.0, 0)])
def test_exploration_endpoints(epsilon, expected_calls):
    agent, factory = GreedyAgent(), Factory(end_after=30)
    Evaluator(agent, factory, EvalSpec(episodes=1, epsilon=epsilon), seed=1).evaluate(0)
    assert agent.calls == expected_calls
    assert all(0 <= a < 6 for a in factory.environments[0].actions)


def test_frame_limit_includes_ale_reset_frames_and_never_rounds_up():
    factory = Factory(end_after=100, reset_frames=3)
    result = Evaluator(GreedyAgent(), factory,
                       EvalSpec(episodes=1, max_episode_frames=18), seed=1).evaluate(0)
    assert factory.environments[0].frames == 15
    assert len(factory.environments[0].actions) == 3
    assert result.scores == (7.5,)


def test_18000_frames_are_4500_decisions_not_18000_decisions():
    factory = Factory(end_after=99999)
    Evaluator(GreedyAgent(), factory, EvalSpec(episodes=1), seed=1).evaluate(0)
    assert len(factory.environments[0].actions) == 4500
    assert factory.environments[0].frames == 18000


def test_close_on_agent_failure_and_restore_grad_mode():
    class FailingAgent:
        def act_greedy(self, obs):
            raise RuntimeError("policy failure")

    factory = Factory()
    with pytest.raises(RuntimeError, match="policy failure"):
        Evaluator(FailingAgent(), factory, EvalSpec(epsilon=0.0)).evaluate(0)
    assert factory.environments[0].closed
    assert torch.is_grad_enabled()


def test_mixed_scores_use_population_standard_deviation():
    class VaryingEnv(EvaluationEnv):
        def step(self, action):
            obs, _, term, trunc, info = super().step(action)
            return obs, float(self.reset_count), term, trunc, info

    factory = SimpleNamespace(make_eval=lambda seed: VaryingEnv(end_after=1))
    result = Evaluator(GreedyAgent(), factory, EvalSpec(episodes=3)).evaluate(0)
    assert result.scores == (1.0, 2.0, 3.0)
    assert result.mean == 2.0
    assert result.std == pytest.approx(np.sqrt(2 / 3))


@pytest.mark.parametrize("kwargs", [
    {"random_score": 0}, {"human_score": 1},
    {"random_score": 1, "human_score": 1},
    {"random_score": float("nan"), "human_score": 1},
    {"action_repeat": 0},
])
def test_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Evaluator(GreedyAgent(), Factory(), EvalSpec(), **kwargs)


def test_optional_normalization_is_explicit():
    result = EvalResult(0, (1.0,), 1.0, 0.0)
    with pytest.raises(ValueError, match="requires"):
        _ = result.normalized


def test_frame_budget_exhausted_during_reset_closes_env():
    factory = Factory(reset_frames=30)
    with pytest.raises(ValueError, match="reset"):
        Evaluator(GreedyAgent(), factory,
                  EvalSpec(episodes=1, max_episode_frames=12)).evaluate(0)
    assert factory.environments[0].closed


def test_wrong_repeat_is_detected():
    factory = Factory()
    with pytest.raises(ValueError, match="frame advance"):
        Evaluator(GreedyAgent(), factory, EvalSpec(), action_repeat=1).evaluate(0)
    assert factory.environments[0].closed


def test_factory_clips_training_rewards_but_keeps_raw_evaluation(monkeypatch):
    from miae01_dqn.config import EnvSpec
    from miae01_dqn.envs import EnvFactory

    class RewardEnv(gym.Env):
        observation_space = spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8)
        action_space = spaces.Discrete(6)
        ale = SimpleNamespace(lives=lambda: 3)

        def get_action_meanings(self):
            return ["NOOP", "UP", "DOWN", "LEFT", "RIGHT", "WAIT"]

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros((8, 8, 3), dtype=np.uint8), {}

        def step(self, action):
            return np.zeros((8, 8, 3), dtype=np.uint8), 2.5, False, False, {}

    monkeypatch.setattr(gym, "make", lambda *a, **kw: RewardEnv())
    factory = EnvFactory(EnvSpec())
    train, evaluation = factory.make_train(1), factory.make_eval(1)
    try:
        train.reset()
        evaluation.reset()
        assert train.step(0)[1] == 1.0
        assert evaluation.step(0)[1] == 10.0
        assert train.unwrapped is not evaluation.unwrapped
    finally:
        train.close()
        evaluation.close()
