"""Regression tests for the DQN update and Agent state."""

import io

import numpy as np
import pytest
import torch
from torch import nn

from miae01_dqn import Agent, AlgoSpec, OptimSpec, ReplayBuffer


class TableNetwork(nn.Module):
    n_actions = 3

    def __init__(self):
        super().__init__()
        self.values = nn.Parameter(torch.tensor([[0.0, 1.0, -1.0], [4.0, 2.0, 0.0]]))

    def forward(self, obs):
        assert obs.dtype == torch.uint8
        return self.values[obs[:, 0, 0, 0].long()]


OBS = np.zeros((1, 1, 1), dtype=np.uint8)
NEXT_OBS = np.ones_like(OBS)


def make_agent(*, batch_size=1, optim=None, seed=7):
    online = TableNetwork()
    buffer = ReplayBuffer(8, OBS.shape, seed=1)
    agent = Agent(online, buffer, AlgoSpec(batch_size=batch_size),
                  optim or OptimSpec(), "cpu", seed=seed)
    return agent, online, buffer


def test_epsilon_uses_frames_and_supports_zero_decay():
    agent, _, _ = make_agent()
    assert agent.epsilon_at(0) == 1.0
    assert agent.epsilon_at(125_000) == pytest.approx(0.55)
    assert agent.epsilon_at(250_000) == agent.epsilon_at(300_000) == 0.1
    assert agent.epsilon_at(125_000, AlgoSpec(action_repeat=2)) == pytest.approx(0.775)
    assert agent.epsilon_at(0, AlgoSpec(eps_decay_frames=0)) == 0.1


def test_greedy_and_logging_do_not_consume_exploration_rng():
    agent, _, _ = make_agent()
    reference, _, _ = make_agent()
    for _ in range(32):
        assert agent.act_greedy(OBS) == agent.act(OBS, 0.0) == 1
        np.testing.assert_array_equal(agent.q_values(OBS), [0.0, 1.0, -1.0])
        assert agent.act(OBS, 1.0) == reference.act(OBS, 1.0)


def test_default_optimizer_and_undersized_buffer():
    agent, online, buffer = make_agent(batch_size=2)
    buffer.push(OBS, 1, 0.5, NEXT_OBS, False)
    before = online.values.detach().clone()
    assert agent.update() is None
    torch.testing.assert_close(online.values, before)
    optimizer = agent.state_dict()["optimizer"]
    params = optimizer["param_groups"][0]
    assert optimizer["state"] == {}
    assert (params["lr"], params["alpha"], params["momentum"], params["eps"]) == (
        2.5e-4, 0.95, 0.95, 0.01,
    )


@pytest.mark.parametrize(
    "done, reward, td_clip, grad_clip, expected_loss, expected_grad",
    [
        (True, 0.5, 1.0, None, 0.125, 0.5),
        (False, 0.5, 1.0, None, 2.96, -1.0),
        (True, -10.0, 1.0, None, 10.5, 1.0),
        (False, 0.5, 0.25, None, 0.25 * (3.46 - 0.125), -0.25),
        (False, 0.5, 0.0, None, 0.5 * 3.46**2, -3.46),
        (False, 0.5, 1.0, 0.25, 2.96, -0.25),
    ],
)
def test_update_applies_terminal_mask_and_clips_td_gradient(
    done, reward, td_clip, grad_clip, expected_loss, expected_grad
):
    optim = OptimSpec(td_error_clip=td_clip, grad_clip_norm=grad_clip)
    agent, online, buffer = make_agent(optim=optim)
    buffer.push(OBS, 1, reward, NEXT_OBS, done)
    target_before = online.values.detach().clone()
    with torch.no_grad():
        online.values[0, 0] = 3.0
        online.values[1, 0] = -4.0
    before = online.values.detach().clone()

    assert agent.update() == pytest.approx(expected_loss)
    torch.testing.assert_close(
        online.values.grad,
        torch.tensor([[0.0, expected_grad, 0.0], [0.0, 0.0, 0.0]]),
    )
    assert (online.values[0, 1].item() - before[0, 1].item()) * expected_grad < 0
    torch.testing.assert_close(agent.state_dict()["target"]["values"], target_before)
    agent.sync_target()
    torch.testing.assert_close(agent.state_dict()["target"]["values"], online.values)


@pytest.mark.parametrize("kind", ["rmsprop", "adam"])
def test_zero_optimizer_eps_keeps_unvisited_actions_finite(kind):
    agent, online, buffer = make_agent(optim=OptimSpec(kind=kind, eps=0))
    buffer.push(OBS, 1, 0.5, NEXT_OBS, False)
    assert np.isfinite(agent.update())
    assert torch.isfinite(online.values).all()


def test_checkpoint_preserves_both_networks_optimizer_and_exploration():
    agent, online, buffer = make_agent()
    buffer.push(OBS, 1, 0.5, NEXT_OBS, False)
    agent.update()
    snapshot = agent.state_dict()
    expected_actions = [agent.act(OBS, 1.0) for _ in range(16)]
    expected_loss = agent.update()
    expected_online = online.values.detach().clone()
    agent.sync_target()

    stream = io.BytesIO()
    torch.save(snapshot, stream)
    stream.seek(0)
    restored, restored_online, restored_buffer = make_agent(seed=99)
    restored_buffer.push(OBS, 1, 0.5, NEXT_OBS, False)
    restored.load_state_dict(torch.load(stream, weights_only=True))

    assert [restored.act(OBS, 1.0) for _ in range(16)] == expected_actions
    assert restored.update() == pytest.approx(expected_loss)
    torch.testing.assert_close(restored_online.values, expected_online)
    torch.testing.assert_close(
        restored.state_dict()["target"]["values"], snapshot["target"]["values"]
    )
