"""Issue #10: independent checks of the merged Issue #3 environment module."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from miae01_dqn.config import EnvSpec
from miae01_dqn.envs import EnvFactory, MaxAndSkip


def assert_four_ale_frames(env):
    before = env.unwrapped.ale.getFrameNumber()
    _, _, terminated, truncated, _ = env.step(0)
    after = env.unwrapped.ale.getFrameNumber()
    # Use a nonterminal opening action; a terminal block may legitimately stop early.
    assert not (terminated or truncated), "frame probe must be a nonterminal step"
    assert after - before == 4, f"expected 4 ALE frames, got {after - before}"


@pytest.mark.parametrize("kind", ["make_train", "make_eval"])
def test_real_pong_observation_actions_and_frame_advance(kind):
    factory = EnvFactory(EnvSpec())
    env = getattr(factory, kind)(1)
    try:
        obs, _ = env.reset()
        assert obs.shape == (4, 84, 84)
        assert obs.dtype == np.uint8
        assert env.observation_space.contains(obs)
        assert env.action_space.n == 6
        assert factory.n_actions() == 6
        assert factory.obs_shape() == (4, 84, 84)
        for _ in range(5):
            assert_four_ale_frames(env)
    finally:
        env.close()


@pytest.mark.parametrize("kind", ["make_train", "make_eval"])
def test_real_pong_seeded_starts(kind):
    factory = EnvFactory(EnvSpec())

    def starting_trace(seed):
        env = getattr(factory, kind)(seed)
        try:
            obs, _ = env.reset()
            reset_frames = env.unwrapped.ale.getEpisodeFrameNumber()
            trace = [obs.copy()]
            for _ in range(20):
                obs, _, terminated, truncated, _ = env.step(0)
                assert not (terminated or truncated)
                trace.append(obs.copy())
            return np.stack(trace), reset_frames
        finally:
            env.close()

    first, frames = starting_trace(1)
    again, again_frames = starting_trace(1)
    np.testing.assert_array_equal(first, again)
    assert frames == again_frames
    other_starts = [starting_trace(seed) for seed in (2, 3, 4, 5)]
    # Pong's reset screen can be identical despite different no-op counts;
    # compare 20 subsequent fixed-action observations as well as the ALE clock.
    # Finite no-op choices can also collide, so use several fixed seeds.
    assert any(not np.array_equal(first, obs) for obs, _ in other_starts)
    assert any(count != frames for _, count in other_starts)


def test_frame_check_catches_double_skip_mutation(monkeypatch):
    real_make = gym.make

    def incorrect_make(*args, **kwargs):
        kwargs["frameskip"] = 4  # Simulate accidentally retaining ALE's default.
        return real_make(*args, **kwargs)

    monkeypatch.setattr(gym, "make", incorrect_make)
    env = EnvFactory(EnvSpec()).make_eval(1)
    try:
        env.reset()
        with pytest.raises(AssertionError, match="expected 4 ALE frames, got 16"):
            assert_four_ale_frames(env)
    finally:
        env.close()


class LifeEnv(gym.Env):
    """Deterministic life counter: Pong has no lives, so it cannot cover this rule."""

    observation_space = spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8)
    action_space = spaces.Discrete(6)

    def __init__(self):
        self.lives_left = 3
        self.drop_life = False
        self.game_over = False
        self.closed = False
        self.ale = SimpleNamespace(lives=lambda: self.lives_left)

    def get_action_meanings(self):
        return ["NOOP", "UP", "DOWN", "LEFT", "RIGHT", "WAIT"]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.lives_left = 3
        self.game_over = False
        return np.zeros((8, 8, 3), dtype=np.uint8), {}

    def step(self, action):
        if self.drop_life:
            self.lives_left -= 1
            self.drop_life = False
        return np.zeros((8, 8, 3), dtype=np.uint8), 0.0, self.game_over, False, {}

    def close(self):
        self.closed = True


def test_life_loss_terminates_training_but_not_evaluation(monkeypatch):
    kwargs_seen = []

    def make(*args, **kwargs):
        kwargs_seen.append(kwargs)
        return LifeEnv()

    monkeypatch.setattr(gym, "make", make)
    factory = EnvFactory(EnvSpec())
    train, evaluation = factory.make_train(1), factory.make_eval(1)
    try:
        assert train.unwrapped is not evaluation.unwrapped
        train.reset()
        evaluation.reset()
        train.unwrapped.drop_life = True
        evaluation.unwrapped.drop_life = True
        assert train.step(0)[2] is True
        assert evaluation.step(0)[2] is False
        evaluation.unwrapped.game_over = True
        assert evaluation.step(0)[2] is True
        assert all(k["frameskip"] == 1 for k in kwargs_seen)
        assert all(k["repeat_action_probability"] == 0.0 for k in kwargs_seen)
    finally:
        train.close()
        evaluation.close()


def test_max_pool_uses_last_two_frames_and_sums_rewards():
    class FlickerEnv(LifeEnv):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def step(self, action):
            self.calls += 1
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            if self.calls == 1:
                frame[:] = 255  # Must not enter the last-two-frame maximum.
            elif self.calls == 3:
                frame[0, 0] = 90
            elif self.calls == 4:
                frame[0, 1] = 120
            return frame, float(self.calls), False, False, {}

    raw = FlickerEnv()
    env = MaxAndSkip(raw, skip=4)
    try:
        frame, reward, term, trunc, _ = env.step(0)
        assert raw.calls == 4 and reward == 10.0
        np.testing.assert_array_equal(frame[0, 0], [90, 90, 90])
        np.testing.assert_array_equal(frame[0, 1], [120, 120, 120])
        assert frame[1:].max() == 0
        assert not term and not trunc
    finally:
        env.close()
