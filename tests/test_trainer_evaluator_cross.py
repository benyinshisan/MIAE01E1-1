"""Issue #12 (X-04): independent, non-author verification of Trainer and Evaluator.

Author: 2260031685 沈坤龙. Trainer (I-06, #6) was authored by 2260032112 黄俊皓
in PR #19 and Evaluator (I-07, #7) was authored by 2260031814 陈博文 in PR #20.
Neither module was owned by me (I authored I-01 config and I-03 envs, plus a
review-fix commit ``6b3f9dc`` on PR #19 before merge). No production code is
modified by this branch; findings go to ``docs/cross-tests-trainer-evaluator.md``.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from miae01_dqn.agent import Agent
from miae01_dqn.buffer import ReplayBuffer
from miae01_dqn.config import EnvSpec, EvalSpec, RunConfig
from miae01_dqn.envs import EnvFactory
from miae01_dqn.evaluator import Evaluator
from miae01_dqn.network import QNetwork
from miae01_dqn.trainer import Trainer, TrainResult

OBS_SHAPE_TINY = (1, 1, 1)


def _tiny_frame(value: int) -> np.ndarray:
    return np.full(OBS_SHAPE_TINY, value, dtype=np.uint8)


class _MemoryLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, float, int]] = []

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.records.append((tag, float(value), int(step)))

    def values_for(self, tag: str) -> list[tuple[int, float]]:
        return [(step, val) for t, val, step in self.records if t == tag]


class _SpyBuffer:
    """Records every ``push`` call while optionally forwarding to a real buffer."""

    def __init__(self, backing: ReplayBuffer | None = None) -> None:
        self.backing = backing
        self.pushes: list[dict] = []

    def __len__(self) -> int:
        return len(self.backing) if self.backing is not None else len(self.pushes)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        truncated: bool = False,
    ) -> None:
        self.pushes.append(
            {
                "obs": np.array(obs, copy=True),
                "action": int(action),
                "reward": float(reward),
                "next_obs": np.array(next_obs, copy=True),
                "done": bool(done),
                "truncated": bool(truncated),
            }
        )
        if self.backing is not None:
            self.backing.push(obs, action, reward, next_obs, done, truncated=truncated)


class _SpyAgent:
    """Deterministic stub Agent that records update and sync_target events."""

    def __init__(self, *, none_updates: int = 0) -> None:
        self.none_updates = none_updates
        self.update_attempts = 0
        self.successful_updates = 0
        self.sync_at_updates: list[int] = []
        self.seen_obs: list[np.ndarray] = []

    def epsilon_at(self, step: int, spec) -> float:
        return Agent.epsilon_at(step, spec)

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        self.seen_obs.append(np.array(obs, copy=True))
        return 0

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return np.array([0.5, -0.5], dtype=np.float32)

    def update(self) -> float | None:
        self.update_attempts += 1
        if self.update_attempts <= self.none_updates:
            return None
        self.successful_updates += 1
        return 0.25

    def sync_target(self) -> None:
        self.sync_at_updates.append(self.successful_updates)


class _AutoResetTruncEnv:
    """Fake env simulating Gymnasium same-step autoreset on truncation."""

    action_space = spaces.Discrete(2)

    def __init__(
        self,
        *,
        trunc_every: int = 3,
        final_marker: int = 77,
        reset_marker: int = 99,
        include_final_obs: bool = True,
    ) -> None:
        self.trunc_every = trunc_every
        self.final_marker = final_marker
        self.reset_marker = reset_marker
        self.include_final_obs = include_final_obs
        self.reset_count = 0
        self.closed = False
        self._ep_step = 0
        self._global_step = 0

    def reset(self):
        self.reset_count += 1
        self._ep_step = 0
        return _tiny_frame(self.reset_marker), {}

    def step(self, action: int):
        self._ep_step += 1
        self._global_step += 1
        truncated = self._ep_step == self.trunc_every
        if truncated:
            self._ep_step = 0
            if self.include_final_obs:
                info = {
                    "final_observation": _tiny_frame(self.final_marker),
                    "raw_reward": 5.0,
                }
                return _tiny_frame(self.reset_marker), 1.0, False, True, info
            return _tiny_frame(self.final_marker), 1.0, False, True, {"raw_reward": 5.0}
        return _tiny_frame(10 + self._ep_step), 1.0, False, False, {"raw_reward": 5.0}

    def close(self) -> None:
        self.closed = True


class _CountingEnv:
    """Minimal env that terminates every ``episode_len`` steps."""

    action_space = spaces.Discrete(2)

    def __init__(self, episode_len: int = 10) -> None:
        self.episode_len = episode_len
        self.step_in_ep = 0
        self.closed = False

    def reset(self):
        self.step_in_ep = 0
        return _tiny_frame(1), {}

    def step(self, action: int):
        self.step_in_ep += 1
        terminated = self.step_in_ep >= self.episode_len
        return _tiny_frame(2), 1.0, terminated, False, {"raw_reward": 1.0}

    def close(self) -> None:
        self.closed = True


def assert_truncation_contract(
    pushes: list[dict],
    *,
    trunc_indices: list[int],
    expected_final: int,
    forbidden_reset: int,
) -> None:
    """Shared assertion: truncated steps store ``final_observation``, ``done=False``."""
    assert trunc_indices, "must have at least one truncated step"
    for idx in trunc_indices:
        rec = pushes[idx]
        assert rec["truncated"] is True, f"step {idx} missing truncated=True"
        assert rec["done"] is False, f"step {idx} must not mark truncation as done=True"
        actual = int(rec["next_obs"].item())
        assert actual != forbidden_reset, (
            f"step {idx} stored autoreset frame {forbidden_reset} instead of final_observation"
        )
        assert actual == expected_final, (
            f"step {idx} expected final_observation {expected_final}, got {actual}"
        )


def assert_update_and_sync_counts(
    result: TrainResult,
    sync_at_updates: list[int],
    cfg: RunConfig,
) -> None:
    """Shared assertion: update count and target sync cadence match architecture.md."""
    algo = cfg.algo
    expected_updates = sum(
        1
        for step in range(algo.total_agent_steps)
        if step >= algo.replay_start_steps and step % algo.learn_every_steps == 0
    )
    expected_syncs = expected_updates // algo.target_update_every_updates
    expected_schedule = [
        k * algo.target_update_every_updates for k in range(1, expected_syncs + 1)
    ]
    assert result.updates == expected_updates, (
        f"expected {expected_updates} updates, got {result.updates}"
    )
    assert len(sync_at_updates) == expected_syncs, (
        f"expected {expected_syncs} target syncs, got {len(sync_at_updates)}"
    )
    assert sync_at_updates == expected_schedule, (
        f"expected target syncs at updates {expected_schedule}, got {sync_at_updates}"
    )


def _snapshot_agent(agent: Agent) -> dict:
    return deepcopy(
        {
            "policy": agent.policy.state_dict(),
            "target": agent.target.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "rng": agent.rng.bit_generator.state,
        }
    )


def _assert_agent_unchanged(agent: Agent, before: dict) -> None:
    after = _snapshot_agent(agent)
    assert after["rng"] == before["rng"]
    for group in ("policy", "target"):
        assert set(after[group].keys()) == set(before[group].keys())
        for key, tensor in before[group].items():
            torch.testing.assert_close(after[group][key], tensor, rtol=0, atol=0)
    assert after["optimizer"]["param_groups"] == before["optimizer"]["param_groups"]
    assert set(after["optimizer"]["state"].keys()) == set(
        before["optimizer"]["state"].keys()
    )
    for param_id, state_before in before["optimizer"]["state"].items():
        state_after = after["optimizer"]["state"][param_id]
        assert set(state_after.keys()) == set(state_before.keys())
        for key, value in state_before.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(state_after[key], value, rtol=0, atol=0)
            else:
                assert state_after[key] == value


# ---------------------------------------------------------------------------
# 1. Trainer: 5k real Pong smoke run
# ---------------------------------------------------------------------------
def test_trainer_real_pong_5k_smoke_and_logging():
    """Issue #12 item 1: real 5,000-step Pong smoke finishes cleanly with episodes > 0."""
    torch.set_num_threads(1)
    torch.manual_seed(1)
    cfg = RunConfig().with_overrides(
        total_agent_steps=5000,
        replay_start_frames=1000,
        replay_capacity=5000,
        target_update_every_updates=500,
        seed=1,
    )
    factory = EnvFactory(cfg.env)
    buffer = ReplayBuffer(cfg.algo.replay_capacity, factory.obs_shape(), seed=1)
    network = QNetwork(factory.obs_shape(), factory.n_actions())
    agent = Agent(network, buffer, cfg.algo, cfg.optim, "cpu", seed=1)
    logger = _MemoryLogger()

    result = Trainer(cfg, agent, factory, buffer, logger, log_every_steps=1000).run()

    assert result.agent_steps == 5000
    assert result.frames == 20000
    assert result.episodes > 0
    assert result.updates == 1187  # steps in [250, 4999] divisible by 4
    assert len(buffer) == 5000
    assert result.wall_clock_s > 0
    assert np.isfinite(result.final_return_mean)

    tags = {tag for tag, _, _ in logger.records}
    assert tags == {"episodic_return", "epsilon", "td_loss", "q_values", "SPS"}
    assert len(logger.values_for("episodic_return")) == result.episodes
    assert all(np.isfinite(val) for _, val, _ in logger.records)


# ---------------------------------------------------------------------------
# 2. Trainer: truncation replaces next_obs with final_observation + negative controls
# ---------------------------------------------------------------------------
def test_trainer_truncation_uses_final_observation_and_preserves_in_real_buffer():
    """Issue #12 item 2: autoreset truncation stores final_observation with done=False."""
    cfg = RunConfig().with_overrides(
        total_agent_steps=12,
        replay_start_frames=48,  # warmup_steps=12 == total_agent_steps: skip updates
        replay_capacity=32,
        batch_size=2,
    )
    env = _AutoResetTruncEnv(
        trunc_every=3, final_marker=77, reset_marker=99, include_final_obs=True
    )
    real_buf = ReplayBuffer(32, OBS_SHAPE_TINY, seed=5)
    spy_buf = _SpyBuffer(real_buf)
    spy_agent = _SpyAgent()
    logger = _MemoryLogger()

    result = Trainer(
        cfg,
        spy_agent,
        SimpleNamespace(make_train=lambda seed: env),
        spy_buf,
        logger,
        log_every_steps=6,
    ).run()

    assert result.episodes == 4
    assert env.reset_count == 1  # same-step autoreset must not call reset() again
    assert env.closed
    assert len(spy_buf.pushes) == 12
    assert_truncation_contract(
        spy_buf.pushes,
        trunc_indices=[2, 5, 8, 11],
        expected_final=77,
        forbidden_reset=99,
    )
    # Next step after each truncation receives the reset observation (99).
    for step_after_trunc in (3, 6, 9):
        assert int(spy_agent.seen_obs[step_after_trunc].item()) == 99

    # Raw episode returns (3 * 5.0 = 15.0) are logged while buffer stores clipped 1.0.
    assert [v for _, v in logger.values_for("episodic_return")] == [15.0] * 4
    assert all(rec["reward"] == 1.0 for rec in spy_buf.pushes)

    # Real ReplayBuffer preserves final_observation (77) for truncated slots even
    # after subsequent episodes overwrite the ring's next slot with reset frame 99.
    saw_trunc_sample = False
    for _ in range(50):
        batch = real_buf.sample(4)
        for obs, nxt, done in zip(batch.obs, batch.next_obs, batch.dones):
            if int(obs.item()) == 12:  # step 3 of every episode is truncated
                saw_trunc_sample = True
                assert int(nxt.item()) == 77
                assert bool(done) is False
    assert saw_trunc_sample


def test_trainer_plain_truncation_without_final_observation_resets_env():
    """Plain single-env truncation falls back to next_obs, sets done=False, and resets."""
    cfg = RunConfig().with_overrides(
        total_agent_steps=9, replay_start_frames=36, replay_capacity=16, batch_size=2
    )
    env = _AutoResetTruncEnv(
        trunc_every=3, final_marker=77, reset_marker=99, include_final_obs=False
    )
    spy_buf = _SpyBuffer()
    logger = _MemoryLogger()

    result = Trainer(
        cfg,
        _SpyAgent(),
        SimpleNamespace(make_train=lambda seed: env),
        spy_buf,
        logger,
        log_every_steps=9,
    ).run()

    assert result.episodes == 3
    assert env.reset_count == 4  # initial reset + 3 post-truncation resets
    assert env.closed
    assert_truncation_contract(
        spy_buf.pushes,
        trunc_indices=[2, 5, 8],
        expected_final=77,
        forbidden_reset=99,
    )


def test_truncation_assertion_catches_final_obs_and_done_bugs():
    """Acceptance check: ignoring final_observation or setting done=True is caught."""
    good_pushes = [
        {
            "obs": _tiny_frame(12),
            "action": 0,
            "reward": 1.0,
            "next_obs": _tiny_frame(77),
            "done": False,
            "truncated": True,
        }
    ]
    assert_truncation_contract(
        good_pushes, trunc_indices=[0], expected_final=77, forbidden_reset=99
    )

    # Bug A: forgot to replace next_obs with info["final_observation"].
    bug_wrong_next = [dict(good_pushes[0], next_obs=_tiny_frame(99))]
    with pytest.raises(AssertionError, match="autoreset frame"):
        assert_truncation_contract(
            bug_wrong_next, trunc_indices=[0], expected_final=77, forbidden_reset=99
        )

    # Bug B: collapsed done = terminated or truncated.
    bug_done_true = [dict(good_pushes[0], done=True)]
    with pytest.raises(AssertionError, match="done=True"):
        assert_truncation_contract(
            bug_done_true, trunc_indices=[0], expected_final=77, forbidden_reset=99
        )

    # Bug C: dropped truncated=True flag.
    bug_not_truncated = [dict(good_pushes[0], truncated=False)]
    with pytest.raises(AssertionError, match="truncated=True"):
        assert_truncation_contract(
            bug_not_truncated, trunc_indices=[0], expected_final=77, forbidden_reset=99
        )


# ---------------------------------------------------------------------------
# 3 & 4. Trainer: updates vs learn_every_steps & target sync counts + negative controls
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "total_steps,start_frames,repeat,learn_every,target_every",
    [
        (100, 40, 4, 4, 5),   # start_step=10 (non-multiple of 4): 22 updates, 4 syncs
        (40, 16, 4, 2, 3),    # start_step=4 (exact multiple): 18 updates, 6 syncs
        (30, 80, 4, 4, 10),   # below one target sync interval: 3 updates, 0 syncs
        (50, 20, 2, 3, 4),    # custom action_repeat=2: 13 updates, 3 syncs
    ],
)
def test_trainer_updates_and_target_sync_counts(
    total_steps: int,
    start_frames: int,
    repeat: int,
    learn_every: int,
    target_every: int,
):
    """Issue #12 items 3-4: update count and sync_target cadence match contract."""
    cfg = RunConfig().with_overrides(
        total_agent_steps=total_steps,
        replay_start_frames=start_frames,
        learn_every_steps=learn_every,
        target_update_every_updates=target_every,
        replay_capacity=200,
        batch_size=2,
        **{"env.action_repeat": repeat, "algo.action_repeat": repeat},
    )
    spy_buf = _SpyBuffer()
    spy_agent = _SpyAgent()
    env = _CountingEnv(episode_len=10)

    result = Trainer(
        cfg,
        spy_agent,
        SimpleNamespace(make_train=lambda seed: env),
        spy_buf,
        _MemoryLogger(),
        log_every_steps=total_steps,
    ).run()

    assert result.agent_steps == total_steps
    assert result.frames == total_steps * repeat
    assert len(spy_buf.pushes) == total_steps  # warmup transitions are never dropped
    assert_update_and_sync_counts(result, spy_agent.sync_at_updates, cfg)


def test_trainer_none_updates_not_counted_and_on_eval_uses_frames():
    """None returns from agent.update() do not increment updates or trigger syncs."""
    cfg = RunConfig().with_overrides(
        total_agent_steps=24,
        replay_start_frames=16,  # warmup_steps=4 -> 10 update attempts at steps 4,6..22
        learn_every_steps=2,
        target_update_every_updates=3,
        every_frames=24,  # 24 frames / 4 = every 6 agent steps -> steps 6, 12, 18, 24
        replay_capacity=64,
        batch_size=2,
    )
    spy_agent = _SpyAgent(none_updates=3)  # 10 update attempts -> 7 real updates -> 2 syncs
    eval_steps: list[int] = []
    result = Trainer(
        cfg,
        spy_agent,
        SimpleNamespace(make_train=lambda seed: _CountingEnv(episode_len=10)),
        _SpyBuffer(),
        _MemoryLogger(),
        log_every_steps=10,
        on_eval=eval_steps.append,
    ).run()

    assert spy_agent.update_attempts == 10
    assert result.updates == 7
    assert spy_agent.sync_at_updates == [3, 6]
    assert eval_steps == [6, 12, 18, 24]

    with pytest.raises(ValueError):
        Trainer(
            cfg,
            spy_agent,
            SimpleNamespace(make_train=lambda seed: _CountingEnv()),
            _SpyBuffer(),
            _MemoryLogger(),
            log_every_steps=0,
        )


def test_counting_assertion_catches_frame_step_and_sync_unit_bugs():
    """Acceptance check: update/sync counting bugs must be caught by the assertion."""
    cfg = RunConfig().with_overrides(
        total_agent_steps=100,
        replay_start_frames=40,
        learn_every_steps=4,
        target_update_every_updates=5,
        replay_capacity=200,
        batch_size=2,
    )
    # Correct baseline: 22 updates, 4 syncs at [5, 10, 15, 20].
    good = TrainResult(100, 400, 22, 10, 0.1, 10.0)
    assert_update_and_sync_counts(good, [5, 10, 15, 20], cfg)

    # Bug A: waited replay_start_frames (40) steps instead of replay_start_steps (10).
    bug_start_unit = TrainResult(100, 400, 15, 10, 0.1, 10.0)
    with pytest.raises(AssertionError, match="expected 22 updates, got 15"):
        assert_update_and_sync_counts(bug_start_unit, [5, 10, 15], cfg)

    # Bug B: synced target every 5 agent steps instead of every 5 gradient updates.
    with pytest.raises(AssertionError, match="expected 4 target syncs"):
        assert_update_and_sync_counts(good, list(range(1, 21)), cfg)

    # Bug C: updated every step instead of every learn_every_steps=4 steps.
    bug_every_step = TrainResult(100, 400, 90, 10, 0.1, 10.0)
    with pytest.raises(AssertionError, match="expected 22 updates, got 90"):
        assert_update_and_sync_counts(bug_every_step, list(range(5, 91, 5)), cfg)


# ---------------------------------------------------------------------------
# 5 & 6. Evaluator: exact 30 episodes, consistent return stats & state invariance
# ---------------------------------------------------------------------------
def test_evaluator_exact_30_episodes_and_consistent_statistics():
    """Issue #12 item 5: default EvalSpec runs exactly 30 games with consistent stats."""

    class VaryingEvalEnv:
        action_space = spaces.Discrete(6)

        def __init__(self) -> None:
            self.reset_count = 0
            self.closed = False
            self.steps_in_ep = 0

        def reset(self):
            self.reset_count += 1
            self.steps_in_ep = 0
            return np.zeros((4, 84, 84), dtype=np.uint8), {}

        def step(self, action: int):
            self.steps_in_ep += 1
            # Episode k (1..30) finishes in 2 steps with total return k - 15.5.
            reward = (self.reset_count - 15.5) / 2.0
            return np.zeros((4, 84, 84), dtype=np.uint8), reward, self.steps_in_ep == 2, False, {}

        def close(self) -> None:
            self.closed = True

    created: list[VaryingEvalEnv] = []

    class Factory:
        def make_eval(self, seed: int):
            env = VaryingEvalEnv()
            created.append(env)
            return env

        def make_train(self, seed: int):
            raise AssertionError("Evaluator must not call make_train")

    greedy = SimpleNamespace(act_greedy=lambda obs: 0)
    result = Evaluator(
        greedy, Factory(), EvalSpec(), seed=12, random_score=-20.7, human_score=9.3
    ).evaluate(1_000_000)

    expected_scores = tuple(k - 15.5 for k in range(1, 31))
    assert result.at_frame == 1_000_000
    assert len(result.scores) == 30
    assert result.scores == expected_scores
    assert result.mean == pytest.approx(float(np.mean(expected_scores)))
    assert result.std == pytest.approx(float(np.std(expected_scores)))
    assert result.normalized == pytest.approx(
        100.0 * (result.mean - (-20.7)) / (9.3 - (-20.7))
    )
    assert len(created) == 1
    assert created[0].reset_count == 30
    assert created[0].closed


def test_evaluator_30_games_preserves_real_cnn_agent_and_buffer():
    """Issue #12 item 6: 30-game evaluation leaves weights, optimizer, RNG, and buffer intact."""
    torch.set_num_threads(1)
    torch.manual_seed(12)
    obs_shape = (4, 84, 84)
    real_buf = ReplayBuffer(16, obs_shape, seed=12)
    rng = np.random.default_rng(12)
    for i in range(6):
        obs = rng.integers(0, 256, size=obs_shape, dtype=np.uint8)
        nxt = rng.integers(0, 256, size=obs_shape, dtype=np.uint8)
        real_buf.push(obs, i % 6, 1.0 if i % 2 == 0 else -1.0, nxt, i == 5)

    # Wrap buffer to verify zero pushes occur during evaluation.
    push_calls = 0
    original_push = real_buf.push

    def guarded_push(*args, **kwargs):
        nonlocal push_calls
        push_calls += 1
        return original_push(*args, **kwargs)

    real_buf.push = guarded_push  # type: ignore[method-assign]

    cfg = RunConfig().with_overrides(batch_size=4, replay_capacity=16)
    agent = Agent(QNetwork(obs_shape, 6), real_buf, cfg.algo, cfg.optim, "cpu", seed=12)
    # Run 2 real updates so optimizer momentum/square states are populated and non-zero.
    assert agent.update() is not None
    assert agent.update() is not None
    assert agent.optimizer.state

    before = _snapshot_agent(agent)
    buffer_len_before = len(real_buf)
    buffer_full_before = real_buf.is_full

    class FastImageEvalEnv:
        action_space = spaces.Discrete(6)

        def __init__(self) -> None:
            self.reset_count = 0
            self.closed = False
            self._step = 0

        def reset(self):
            self.reset_count += 1
            self._step = 0
            return np.full(obs_shape, self.reset_count % 256, dtype=np.uint8), {}

        def step(self, action: int):
            self._step += 1
            return (
                np.full(obs_shape, (self.reset_count + self._step) % 256, dtype=np.uint8),
                float(action) - 2.0,
                self._step == 3,
                False,
                {},
            )

        def close(self) -> None:
            self.closed = True

    eval_env = FastImageEvalEnv()
    result = Evaluator(
        agent,
        SimpleNamespace(make_eval=lambda seed: eval_env),
        EvalSpec(),
        seed=99,
    ).evaluate(500_000)

    assert len(result.scores) == 30
    assert eval_env.reset_count == 30
    assert eval_env.closed
    assert push_calls == 0
    assert len(real_buf) == buffer_len_before
    assert real_buf.is_full == buffer_full_before
    assert torch.is_grad_enabled()
    _assert_agent_unchanged(agent, before)


def test_evaluator_real_pong_30_games_preserves_agent_and_buffer():
    """End-to-end Evaluator check on real ALE/Pong-v5 with a real QNetwork Agent."""
    torch.set_num_threads(1)
    torch.manual_seed(7)
    factory = EnvFactory(EnvSpec())
    obs_shape = factory.obs_shape()
    buffer = ReplayBuffer(16, obs_shape, seed=7)
    rng = np.random.default_rng(7)
    for i in range(4):
        obs = rng.integers(0, 256, size=obs_shape, dtype=np.uint8)
        nxt = rng.integers(0, 256, size=obs_shape, dtype=np.uint8)
        buffer.push(obs, i % 6, 1.0, nxt, i == 3)

    cfg = RunConfig().with_overrides(batch_size=4, replay_capacity=16)
    agent = Agent(
        QNetwork(obs_shape, factory.n_actions()), buffer, cfg.algo, cfg.optim, "cpu", seed=7
    )
    assert agent.update() is not None
    before = _snapshot_agent(agent)
    buffer_len_before = len(buffer)

    result = Evaluator(
        agent,
        factory,
        EvalSpec(),
        seed=2260031685,
        random_score=-20.7,
        human_score=9.3,
    ).evaluate(20_000)

    assert result.at_frame == 20_000
    assert len(result.scores) == 30
    assert all(np.isfinite(s) and -21.0 <= s <= 21.0 for s in result.scores)
    assert result.mean == pytest.approx(float(np.mean(result.scores)))
    assert result.std == pytest.approx(float(np.std(result.scores)))
    assert result.normalized == pytest.approx(
        100.0 * (result.mean - (-20.7)) / (9.3 - (-20.7))
    )
    assert len(buffer) == buffer_len_before
    assert torch.is_grad_enabled()
    _assert_agent_unchanged(agent, before)
