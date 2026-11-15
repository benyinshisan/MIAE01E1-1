"""Numerical regression tests; independent teammate sign-off is still required."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from torch import nn
from miae01_dqn.agent import Agent, NatureRMSprop
from miae01_dqn.buffer import ReplayBuffer
from miae01_dqn.config import AlgoSpec, OptimSpec


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_head = nn.Sequential(nn.Linear(1, 2, bias=False))
        nn.init.zeros_(self.q_head[0].weight)

    def forward(self, x):
        return self.q_head(x.float().reshape(-1, 1))


def agent(batch=1):
    return Agent(TinyNetwork(), ReplayBuffer(8, (1, 1, 1), seed=1),
                 AlgoSpec(batch_size=batch), OptimSpec(), 'cpu', seed=1)


@pytest.mark.parametrize('step,expected', [(0,1), (125000,.55), (250000,.1), (500000,.1)])
def test_frame_based_epsilon(step, expected):
    assert Agent.epsilon_at(step, AlgoSpec()) == pytest.approx(expected)


def test_centered_rmsprop_numeric_equation():
    p = nn.Parameter(torch.tensor([1.0]))
    optimizer = NatureRMSprop([p], lr=.00025, momentum=.95, alpha=.95, eps=.01)
    m = v = 0.0
    expected = 1.0
    for g in (2.0, -3.0, .5):
        m, v = .95*m + .05*g, .95*v + .05*g*g
        expected -= .00025 * g / np.sqrt(v-m*m+.01)
        p.grad = torch.tensor([g])
        optimizer.step()
        assert p.item() == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize('kind', ['rmsprop', 'adam'])
def test_zero_epsilon_and_zero_gradient_keep_parameters_finite(kind):
    a = Agent(TinyNetwork(), ReplayBuffer(8, (1, 1, 1), seed=1),
              AlgoSpec(batch_size=1), OptimSpec(kind=kind, eps=0), 'cpu', seed=1)
    x = np.ones((1, 1, 1), np.uint8)
    a.buffer.push(x, 0, 0, x, True)
    assert a.update() == 0
    for p in a.policy.parameters():
        assert torch.isfinite(p).all()
        assert torch.count_nonzero(p) == 0


def test_td_clipping_gradient_target_mask_and_sync():
    a = agent()
    x = np.ones((1,1,1), np.uint8)
    with torch.no_grad():
        a.target.q_head[0].weight.fill_(100)
    a.buffer.push(x, 0, 10, x, True)
    assert a.update() == pytest.approx(9.5)  # Terminal mask removes target Q=100.
    assert a.policy.q_head[0].weight.grad[0,0].item() == -1
    assert a.policy.q_head[0].weight[0,0].item() > 0
    assert a.target.q_head[0].weight[0,0].item() == 100
    assert all(p.grad is None for p in a.target.parameters())
    a.sync_target()
    for p,t in zip(a.policy.parameters(), a.target.parameters()):
        torch.testing.assert_close(p,t,rtol=0,atol=0)


def test_nonterminal_target_bootstraps():
    a=agent()
    with torch.no_grad():
        a.target.q_head[0].weight.fill_(.5)
    x=np.ones((1,1,1),np.uint8)
    a.buffer.push(x,0,0,x,False)
    assert a.update() == pytest.approx(.5*(.99*.5)**2)


def test_insufficient_batch_does_not_update():
    a=agent(2)
    before=deepcopy(a.policy.state_dict())
    assert a.update() is None
    assert not a.optimizer.state
    for k,v in a.policy.state_dict().items():
        torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_actions_and_state_roundtrip():
    a=agent()
    with torch.no_grad():
        a.policy.q_head[0].weight[1]=2
    x=np.ones((1,1,1),np.uint8)
    assert all(a.act(x,0)==1 for _ in range(20))
    choices=[a.act(x,1) for _ in range(2000)]
    assert 850 < sum(choices) < 1150
    b=agent()
    b.load_state_dict(a.state_dict())
    assert [a.act(x,1) for _ in range(20)] == [b.act(x,1) for _ in range(20)]
