"""Issue #11 (X-03): independent, non-author verification of the merged Agent.

Author: 2260031685 沈坤龙. The merged ``miae01_dqn/agent.py`` on ``main`` was
implemented by 2260031814 陈博文 in PR #23 (commits ``0a39511`` and
``595f8cd``, which closed #5; my earlier parallel PR #22 was closed unmerged).
This file therefore verifies the merged Agent independently without modifying
any production code; findings are recorded in ``docs/cross-tests-agent.md``.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from miae01_dqn.agent import Agent, NatureRMSprop
from miae01_dqn.buffer import ReplayBuffer
from miae01_dqn.config import AlgoSpec, OptimSpec
from miae01_dqn.network import QNetwork


class LinearQNet(nn.Module):
    """Minimal deterministic Q-network with ``q_head`` matching ``QNetwork``."""

    def __init__(self, n_actions: int = 6) -> None:
        super().__init__()
        self.q_head = nn.Sequential(nn.Linear(1, n_actions, bias=False))
        nn.init.zeros_(self.q_head[0].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_head(x.float().reshape(-1, 1))


def _make_linear_agent(
    *,
    n_actions: int = 6,
    batch_size: int = 1,
    capacity: int = 16,
    algo: AlgoSpec | None = None,
    optim: OptimSpec | None = None,
    seed: int = 11,
) -> Agent:
    buf = ReplayBuffer(capacity, (1, 1, 1), seed=seed)
    return Agent(
        LinearQNet(n_actions),
        buf,
        algo or AlgoSpec(batch_size=batch_size),
        optim or OptimSpec(),
        "cpu",
        seed=seed,
    )


def _make_cnn_agent(
    *,
    n_actions: int = 6,
    batch_size: int = 4,
    capacity: int = 16,
    optim: OptimSpec | None = None,
    seed: int = 11,
) -> Agent:
    torch.manual_seed(seed)
    obs_shape = (4, 84, 84)
    buf = ReplayBuffer(capacity, obs_shape, seed=seed)
    net = QNetwork(obs_shape, n_actions)
    return Agent(
        net,
        buf,
        AlgoSpec(batch_size=batch_size, replay_capacity=capacity),
        optim or OptimSpec(),
        "cpu",
        seed=seed,
    )


def assert_frame_based_epsilon_schedule(epsilon_fn, spec: AlgoSpec) -> None:
    """Shared schedule invariant: ``agent_step`` converts to frames via ``action_repeat``."""
    assert spec.eps_decay_frames > 0
    assert spec.eps_decay_frames % (4 * spec.action_repeat) == 0

    end_step = spec.eps_decay_frames // spec.action_repeat
    mid_step = end_step // 2
    quarter_step = end_step // 4

    expected_mid = 0.5 * (spec.eps_start + spec.eps_end)
    expected_quarter = spec.eps_start + 0.25 * (spec.eps_end - spec.eps_start)

    assert epsilon_fn(0, spec) == pytest.approx(spec.eps_start, abs=1e-12)
    assert epsilon_fn(quarter_step, spec) == pytest.approx(expected_quarter, abs=1e-12)
    assert epsilon_fn(mid_step, spec) == pytest.approx(expected_mid, abs=1e-12)
    assert epsilon_fn(end_step, spec) == pytest.approx(spec.eps_end, abs=1e-12)
    assert epsilon_fn(end_step + 1, spec) == pytest.approx(spec.eps_end, abs=1e-12)
    assert epsilon_fn(end_step * 4, spec) == pytest.approx(spec.eps_end, abs=1e-12)


# ---------------------------------------------------------------------------
# 1 & 2. Epsilon schedule: frame conversion, post-decay clamp & unit mutation
# ---------------------------------------------------------------------------
def test_epsilon_frame_schedule_default_and_custom_repeat():
    """Issue #11 items 1-2: step=0 / midpoint / endpoint and post-decay floor."""
    default_spec = AlgoSpec()
    assert_frame_based_epsilon_schedule(Agent.epsilon_at, default_spec)

    # Explicit paper-table checkpoints: 0, 500k, 1M, >1M frames (action_repeat=4).
    assert Agent.epsilon_at(0, default_spec) == pytest.approx(1.0)
    assert Agent.epsilon_at(125_000, default_spec) == pytest.approx(0.55)
    assert Agent.epsilon_at(250_000, default_spec) == pytest.approx(0.1)
    for post_step in (250_001, 300_000, 1_000_000, 2_500_000, 10_000_000):
        assert Agent.epsilon_at(post_step, default_spec) == pytest.approx(0.1)

    custom_spec = AlgoSpec(
        action_repeat=2,
        eps_start=0.8,
        eps_end=0.2,
        eps_decay_frames=200,
    )
    assert_frame_based_epsilon_schedule(Agent.epsilon_at, custom_spec)

    # Edge cases: zero decay horizon locks immediately; negative steps fail loudly.
    zero_decay = AlgoSpec(eps_decay_frames=0)
    assert Agent.epsilon_at(0, zero_decay) == pytest.approx(zero_decay.eps_end)
    with pytest.raises(ValueError):
        Agent.epsilon_at(-1, default_spec)


def test_epsilon_schedule_catches_frame_vs_step_unit_errors():
    """Acceptance check: confusing frames with agent steps must fail loudly."""
    spec = AlgoSpec()

    def buggy_step_as_frame(agent_step: int, s: AlgoSpec) -> float:
        # Bug: forgets ``* s.action_repeat``, decaying 4x too slowly.
        frac = min(agent_step / s.eps_decay_frames, 1.0)
        return float(s.eps_start + frac * (s.eps_end - s.eps_start))

    def buggy_divided_repeat(agent_step: int, s: AlgoSpec) -> float:
        # Bug: divides by ``action_repeat`` instead of multiplying.
        frac = min((agent_step / s.action_repeat) / s.eps_decay_frames, 1.0)
        return float(s.eps_start + frac * (s.eps_end - s.eps_start))

    with pytest.raises(AssertionError):
        assert_frame_based_epsilon_schedule(buggy_step_as_frame, spec)
    with pytest.raises(AssertionError):
        assert_frame_based_epsilon_schedule(buggy_divided_repeat, spec)


# ---------------------------------------------------------------------------
# 3 & 4. sync_target() exact copy & update() gating on real QNetwork
# ---------------------------------------------------------------------------
def test_update_returns_none_when_buffer_insufficient_and_updates_when_ready():
    """Issue #11 item 4: buffer < batch_size is a no-op; >= batch_size updates policy."""
    agent = _make_cnn_agent(batch_size=4, capacity=16, seed=17)
    policy_before = deepcopy(agent.policy.state_dict())
    target_before = deepcopy(agent.target.state_dict())
    rng = np.random.default_rng(17)

    # 0, 1, 2, 3 stored transitions: all strictly below batch_size=4.
    for count in range(4):
        assert len(agent.buffer) == count
        assert agent.update() is None
        assert not agent.optimizer.state
        for key, tensor in agent.policy.state_dict().items():
            torch.testing.assert_close(tensor, policy_before[key], rtol=0, atol=0)
        for key, tensor in agent.target.state_dict().items():
            torch.testing.assert_close(tensor, target_before[key], rtol=0, atol=0)
        if count < 3:
            obs = rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
            nxt = rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
            agent.buffer.push(obs, count % 6, 1.0, nxt, False)

    # Fourth transition reaches batch_size=4; update must run and mutate policy only.
    obs = rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
    nxt = rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
    agent.buffer.push(obs, 3, 1.0, nxt, True)
    loss = agent.update()
    assert isinstance(loss, float)
    assert np.isfinite(loss) and loss > 0.0

    changed = any(
        not torch.equal(tensor, policy_before[key])
        for key, tensor in agent.policy.state_dict().items()
    )
    assert changed, "policy parameters did not change after update()"
    for key, tensor in agent.target.state_dict().items():
        torch.testing.assert_close(tensor, target_before[key], rtol=0, atol=0)


def test_sync_target_copies_all_cnn_parameters_without_aliasing():
    """Issue #11 item 3: sync_target() hard-copies policy -> target across all layers."""
    agent = _make_cnn_agent(batch_size=4, capacity=16, seed=23)
    rng = np.random.default_rng(23)
    for i in range(6):
        obs = rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
        nxt = rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
        agent.buffer.push(obs, i % 6, 1.0 if i % 2 == 0 else -1.0, nxt, i == 5)

    for _ in range(3):
        assert agent.update() is not None

    # Confirm policy genuinely diverged from target before syncing.
    assert any(
        not torch.equal(p, t)
        for p, t in zip(agent.policy.parameters(), agent.target.parameters())
    )

    agent.sync_target()
    for (p_name, p), (t_name, t) in zip(
        agent.policy.named_parameters(), agent.target.named_parameters()
    ):
        assert p_name == t_name
        torch.testing.assert_close(t, p, rtol=0, atol=0)
        assert not t.requires_grad
        assert t.grad is None
    assert not agent.target.training

    # Verify deep copy (no shared tensor storage): another update mutates policy only.
    synced_snapshot = deepcopy(agent.target.state_dict())
    assert agent.update() is not None
    assert any(
        not torch.equal(p, synced_snapshot[k])
        for k, p in agent.policy.state_dict().items()
    )
    for k, t in agent.target.state_dict().items():
        torch.testing.assert_close(t, synced_snapshot[k], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 5. TD error clipping (|TD| > 1 gradient contract), bootstrap & optimizers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "reward,expected_loss,expected_grad",
    [
        (10.0, 9.5, -1.0),    # delta = -10.0 (< -1): clipped derivative -1
        (-10.0, 9.5, +1.0),   # delta = +10.0 (> +1): clipped derivative +1
        (500.0, 499.5, -1.0), # much larger |delta| still saturates at -1
        (0.4, 0.08, -0.4),    # |delta| = 0.4 <= 1: quadratic region, unclipped
    ],
)
def test_td_error_clipping_gradient_and_loss_contract(
    reward: float, expected_loss: float, expected_grad: float
):
    """Issue #11 item 5: |TD| > 1 clips dL/dQ to +-1 while |TD| <= 1 stays linear."""
    agent = _make_linear_agent(n_actions=6, batch_size=1, seed=31)
    obs = np.ones((1, 1, 1), dtype=np.uint8)
    chosen = 2
    agent.buffer.push(obs, chosen, reward, obs, True)

    loss = agent.update()
    assert loss == pytest.approx(expected_loss, rel=1e-6, abs=1e-6)

    grad = agent.policy.q_head[0].weight.grad
    assert grad is not None
    assert grad[chosen, 0].item() == pytest.approx(expected_grad, abs=1e-6)
    # Non-selected actions must receive identically zero gradient.
    for action in range(6):
        if action != chosen:
            assert grad[action, 0].item() == 0.0


def test_td_clipping_distinguishes_clipped_from_unclipped_and_batch_sum():
    """Verify clipping actually caps large TD gradients and batch backward sums samples."""
    obs = np.ones((1, 1, 1), dtype=np.uint8)

    clipped = _make_linear_agent(
        n_actions=4, batch_size=4, optim=OptimSpec(td_error_clip=1.0), seed=37
    )
    unclipped = _make_linear_agent(
        n_actions=4, batch_size=4, optim=OptimSpec(td_error_clip=0.0), seed=37
    )
    for a in (clipped, unclipped):
        for _ in range(4):
            a.buffer.push(obs, 1, 50.0, obs, True)

    clipped_loss = clipped.update()
    unclipped_loss = unclipped.update()

    # Documented contract: returned loss is batch mean; backward sums per-sample Huber.
    assert clipped_loss == pytest.approx(49.5, abs=1e-5)
    assert clipped.policy.q_head[0].weight.grad[1, 0].item() == pytest.approx(-4.0, abs=1e-6)

    # Unclipped MSE scales linearly with delta: grad = 4 * (-50) = -200.
    assert unclipped_loss == pytest.approx(0.5 * 50.0**2, rel=1e-5)
    assert unclipped.policy.q_head[0].weight.grad[1, 0].item() == pytest.approx(
        -200.0, rel=1e-5
    )


def test_grad_clip_norm_disabled_by_default_and_active_when_set():
    """OptimSpec.grad_clip_norm=None leaves norm untouched; positive value clips norm."""
    obs = np.ones((1, 1, 1), dtype=np.uint8)
    no_clip = _make_linear_agent(
        n_actions=2, batch_size=4, optim=OptimSpec(grad_clip_norm=None), seed=41
    )
    with_clip = _make_linear_agent(
        n_actions=2, batch_size=4, optim=OptimSpec(grad_clip_norm=0.5), seed=41
    )
    for a in (no_clip, with_clip):
        for _ in range(4):
            a.buffer.push(obs, 0, 10.0, obs, True)

    no_clip.update()
    with_clip.update()
    assert no_clip.policy.q_head[0].weight.grad.norm().item() == pytest.approx(4.0, abs=1e-6)
    assert with_clip.policy.q_head[0].weight.grad.norm().item() == pytest.approx(0.5, abs=1e-6)


def test_target_bootstrap_for_nonterminal_and_mask_for_terminal():
    """Target Q bootstraps iff done=False; terminal transitions mask target Q to 0."""
    obs = np.ones((1, 1, 1), dtype=np.uint8)
    nonterm = _make_linear_agent(n_actions=3, batch_size=1, seed=43)
    term = _make_linear_agent(n_actions=3, batch_size=1, seed=43)
    for a in (nonterm, term):
        with torch.no_grad():
            a.target.q_head[0].weight[:, 0] = torch.tensor([-2.0, 0.8, 0.3])

    # max(target(next_obs)) = 0.8; expected = r + 0.99 * 0.8 = 0.792 when done=False.
    nonterm.buffer.push(obs, 0, 0.0, obs, False)
    term.buffer.push(obs, 0, 0.0, obs, True)

    assert nonterm.update() == pytest.approx(0.5 * (0.99 * 0.8) ** 2, abs=1e-6)
    assert nonterm.policy.q_head[0].weight.grad[0, 0].item() == pytest.approx(
        -0.99 * 0.8, abs=1e-6
    )
    assert term.update() == pytest.approx(0.0, abs=1e-12)
    assert term.policy.q_head[0].weight.grad[0, 0].item() == pytest.approx(0.0, abs=1e-12)


def test_nature_rmsprop_and_adam_optimizer_contracts():
    """Verify centered RMSProp equations independently and ensure Adam option works."""
    param = nn.Parameter(torch.tensor([0.5, -0.25], dtype=torch.float32))
    opt = NatureRMSprop([param], lr=2.5e-4, momentum=0.95, alpha=0.95, eps=0.01)

    m = np.zeros(2, dtype=np.float64)
    v = np.zeros(2, dtype=np.float64)
    expected = np.array([0.5, -0.25], dtype=np.float64)
    for g in (
        np.array([1.5, -0.5]),
        np.array([-2.0, 3.0]),
        np.array([0.0, -1.0]),
        np.array([0.25, 0.25]),
    ):
        m = 0.95 * m + 0.05 * g
        v = 0.95 * v + 0.05 * (g * g)
        denom = np.sqrt(np.maximum(v - m * m, 0.0) + 0.01)
        expected = expected - 2.5e-4 * g / denom
        param.grad = torch.tensor(g, dtype=torch.float32)
        opt.step()
        np.testing.assert_allclose(param.detach().numpy(), expected, atol=1e-7, rtol=0)

    adam_agent = _make_linear_agent(
        n_actions=2, batch_size=1, optim=OptimSpec(kind="adam", lr=1e-3), seed=47
    )
    obs = np.ones((1, 1, 1), dtype=np.uint8)
    adam_agent.buffer.push(obs, 0, 2.0, obs, True)
    loss = adam_agent.update()
    assert loss == pytest.approx(1.5, abs=1e-6)
    assert adam_agent.policy.q_head[0].weight[0, 0].item() > 0.0


# ---------------------------------------------------------------------------
# 6. Action selection: eps=0 greedy argmax, eps=1 approximately uniform
# ---------------------------------------------------------------------------
def test_action_selection_eps0_greedy_and_eps1_uniform_over_six_actions():
    """Issue #11 item 6: eps=0 always picks argmax; eps=1 is ~uniform across 6 actions."""
    agent = _make_linear_agent(n_actions=6, batch_size=1, seed=53)
    with torch.no_grad():
        agent.policy.q_head[0].weight[:, 0] = torch.tensor(
            [-1.2, 0.3, 1.5, -0.4, 2.7, 0.9]
        )
    obs = np.ones((1, 1, 1), dtype=np.uint8)

    # eps=0 must always return argmax (action 4) and match act_greedy.
    rng_before = deepcopy(agent.rng.bit_generator.state)
    greedy_actions = [agent.act_greedy(obs) for _ in range(50)]
    assert all(type(a) is int and a == 4 for a in greedy_actions)
    assert agent.rng.bit_generator.state == rng_before
    assert torch.is_grad_enabled()

    eps0_actions = [agent.act(obs, 0.0) for _ in range(200)]
    assert all(type(a) is int and a == 4 for a in eps0_actions)

    # eps=1 over 6,000 draws across 6 actions (expected 1,000 each):
    # chi-square (df=5, critical value at p=1e-4 is ~25.7) + per-bin sanity band.
    n_draws = 6_000
    counts = Counter(agent.act(obs, 1.0) for _ in range(n_draws))
    assert set(counts.keys()) == set(range(6))
    expected_per_action = n_draws / 6
    chi2 = sum((counts[a] - expected_per_action) ** 2 / expected_per_action for a in range(6))
    assert chi2 < 25.0, f"eps=1 action counts far from uniform: {dict(counts)} (chi2={chi2:.2f})"
    for a in range(6):
        assert 850 <= counts[a] <= 1150, f"action {a} count out of range: {counts[a]}"

    # Invalid epsilon values are rejected.
    with pytest.raises(ValueError):
        agent.act(obs, -0.01)
    with pytest.raises(ValueError):
        agent.act(obs, 1.01)


def test_real_qnetwork_greedy_action_and_state_dict_isolation():
    """Real (4, 84, 84) uint8 observations work end to end and state_dict deep-copies."""
    agent = _make_cnn_agent(n_actions=6, batch_size=2, capacity=8, seed=59)
    obs = np.random.default_rng(59).integers(0, 256, size=(4, 84, 84), dtype=np.uint8)

    q_vals = agent.q_values(obs)
    assert q_vals.shape == (6,)
    assert np.isfinite(q_vals).all()
    assert agent.act(obs, 0.0) == int(np.argmax(q_vals))
    assert agent.act_greedy(obs) == int(np.argmax(q_vals))

    # Mutating a saved state_dict must not corrupt live agent state.
    saved = agent.state_dict()
    for tensor in saved["policy"].values():
        tensor.zero_()
    assert np.allclose(agent.q_values(obs), q_vals, rtol=0, atol=0)

    # Roundtrip onto a fresh agent reproduces greedy Q-values and exploration draws.
    fresh = _make_cnn_agent(n_actions=6, batch_size=2, capacity=8, seed=99)
    fresh.load_state_dict(agent.state_dict())
    np.testing.assert_array_equal(fresh.q_values(obs), agent.q_values(obs))
    assert [fresh.act(obs, 1.0) for _ in range(25)] == [
        agent.act(obs, 1.0) for _ in range(25)
    ]
