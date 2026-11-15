"""DQN agent: action selection, gradient update, target sync.

Implementation: issue I-05.
"""

from __future__ import annotations

from typing import Any, Mapping
from copy import deepcopy

import numpy as np
import torch

from miae01_dqn.buffer import ReplayBuffer
from miae01_dqn.config import AlgoSpec, OptimSpec
from miae01_dqn.network import QNetwork


class NatureRMSprop(torch.optim.Optimizer):
    """Centered RMSProp, epsilon inside sqrt; momentum is gradient-mean decay."""
    def __init__(self, params, *, lr, momentum, alpha, eps):
        super().__init__(params, dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad, state = p.grad, self.state[p]
                if not state:
                    state['mean'] = torch.zeros_like(p)
                    state['square'] = torch.zeros_like(p)
                mean, square = state['mean'], state['square']
                mean.mul_(group['momentum']).add_(grad, alpha=1 - group['momentum'])
                square.mul_(group['alpha']).addcmul_(grad, grad, value=1 - group['alpha'])
                denominator = (square - mean.square()).clamp_min_(0).add_(group['eps']).sqrt_()
                p.addcdiv_(grad, denominator, value=-group['lr'])
        return loss


class Agent:
    def __init__(
        self,
        q_network: QNetwork,
        buffer: ReplayBuffer,
        algo: AlgoSpec,
        optim: OptimSpec,
        device: str,
        *,
        seed: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.policy = q_network.to(self.device)
        self.target = deepcopy(self.policy).eval().requires_grad_(False)
        self.buffer, self.algo, self.optim_spec = buffer, algo, optim
        self.rng = np.random.default_rng(seed)
        self.n_actions = self.policy.q_head[-1].out_features
        if optim.kind == 'rmsprop':
            self.optimizer = NatureRMSprop(self.policy.parameters(), lr=optim.lr,
                momentum=optim.momentum, alpha=optim.alpha, eps=optim.eps)
        else:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=optim.lr, eps=optim.eps)

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        if not 0 <= epsilon <= 1:
            raise ValueError('epsilon must be in [0, 1]')
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.act_greedy(obs)

    def act_greedy(self, obs: np.ndarray) -> int:
        return int(self.q_values(obs).argmax())

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs), device=self.device).unsqueeze(0)
        return self.policy(x)[0].cpu().numpy().copy()

    @staticmethod
    def epsilon_at(agent_step: int, spec: AlgoSpec) -> float:
        if agent_step < 0:
            raise ValueError('agent_step must be nonnegative')
        if spec.eps_decay_frames == 0:
            return float(spec.eps_end)
        fraction = min(agent_step * spec.action_repeat / spec.eps_decay_frames, 1.0)
        return float(spec.eps_start + fraction * (spec.eps_end - spec.eps_start))

    def update(self) -> float | None:
        if len(self.buffer) < self.algo.batch_size:
            return None
        batch = self.buffer.sample(self.algo.batch_size)
        obs = torch.as_tensor(batch.obs, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        dones = torch.as_tensor(batch.dones, device=self.device)
        with torch.no_grad():
            next_q = self.target(torch.as_tensor(batch.next_obs, device=self.device)).max(1).values
            expected = rewards + self.algo.gamma * (~dones) * next_q
        predicted = self.policy(obs).gather(1, actions[:, None]).squeeze(1)
        delta = predicted - expected
        clip = self.optim_spec.td_error_clip
        if clip > 0:
            absolute = delta.abs()
            losses = torch.where(absolute <= clip, 0.5 * delta.square(),
                                 clip * (absolute - 0.5 * clip))
        else:
            losses = 0.5 * delta.square()
        self.optimizer.zero_grad(set_to_none=True)
        # Original Torch7 backward sums per-sample clipped TD contributions.
        losses.sum().backward()
        if self.optim_spec.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.optim_spec.grad_clip_norm)
        self.optimizer.step()
        return float(losses.detach().mean().item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.policy.state_dict())

    def state_dict(self) -> Mapping[str, Any]:
        return deepcopy({'policy': self.policy.state_dict(), 'target': self.target.state_dict(),
                         'optimizer': self.optimizer.state_dict(), 'rng': self.rng.bit_generator.state})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.policy.load_state_dict(state['policy'])
        self.target.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.rng.bit_generator.state = deepcopy(state['rng'])
