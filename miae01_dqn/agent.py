"""DQN agent: action selection, gradient update, target sync.

Implementation: issue I-05.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from miae01_dqn.buffer import ReplayBuffer
from miae01_dqn.config import AlgoSpec, OptimSpec
from miae01_dqn.network import QNetwork


class Agent:
    """Owns the online network, target network and training optimizer."""

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
        self._device = torch.device(device)
        self._online = q_network.to(self._device)
        self._target = deepcopy(self._online).requires_grad_(False).eval()
        self._buffer = buffer
        self._algo = algo
        self._optim = optim
        self._rng = np.random.default_rng(seed)

        # With eps=0, an untouched weight would otherwise divide zero by zero.
        eps = optim.eps or torch.finfo(torch.float32).tiny
        if optim.kind == "rmsprop":
            self._optimizer = torch.optim.RMSprop(
                self._online.parameters(),
                lr=optim.lr,
                alpha=optim.alpha,
                eps=eps,
                momentum=optim.momentum,
            )
        else:
            self._optimizer = torch.optim.Adam(
                self._online.parameters(), lr=optim.lr, eps=eps
            )

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        if epsilon > 0 and self._rng.random() < epsilon:
            return int(self._rng.integers(self._online.n_actions))
        return self.act_greedy(obs)

    def act_greedy(self, obs: np.ndarray) -> int:
        return int(self._predict(obs).argmax().item())

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return self._predict(obs).cpu().numpy()

    @torch.no_grad()
    def _predict(self, obs: np.ndarray) -> torch.Tensor:
        inputs = torch.as_tensor(obs, device=self._device).unsqueeze(0)
        return self._online(inputs).squeeze(0)

    def epsilon_at(self, agent_step: int, spec: AlgoSpec | None = None) -> float:
        spec = self._algo if spec is None else spec
        frames = max(0, agent_step) * spec.action_repeat
        if frames >= spec.eps_decay_frames:
            return spec.eps_end
        fraction = frames / spec.eps_decay_frames
        return spec.eps_start + (spec.eps_end - spec.eps_start) * fraction

    def update(self) -> float | None:
        """Perform one gradient update and return its mean TD loss."""
        if len(self._buffer) < self._algo.batch_size:
            return None

        batch = self._buffer.sample(self._algo.batch_size)
        obs = torch.as_tensor(batch.obs, device=self._device)
        actions = torch.as_tensor(batch.actions, device=self._device)
        rewards = torch.as_tensor(batch.rewards, device=self._device)
        next_obs = torch.as_tensor(batch.next_obs, device=self._device)
        dones = torch.as_tensor(batch.dones, device=self._device)

        with torch.no_grad():
            next_q = self._target(next_obs).amax(dim=1)
            targets = rewards + self._algo.gamma * next_q.masked_fill(dones, 0.0)

        chosen_q = self._online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        # The derivative of Huber loss caps the TD error used in the update.
        if self._optim.td_error_clip > 0:
            loss = F.huber_loss(chosen_q, targets, delta=self._optim.td_error_clip)
        else:
            loss = 0.5 * F.mse_loss(chosen_q, targets)

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self._optim.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._online.parameters(), self._optim.grad_clip_norm
            )
        self._optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        """Copy online weights to the target at the caller's chosen cadence."""
        self._target.load_state_dict(self._online.state_dict())

    def state_dict(self) -> Mapping[str, Any]:
        """Snapshot both networks, the optimizer and the exploration RNG."""
        return deepcopy({
            "online": self._online.state_dict(),
            "target": self._target.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "rng": self._rng.bit_generator.state,
        })

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a snapshot for the same network architecture and optimizer."""
        self._online.load_state_dict(state["online"])
        self._target.load_state_dict(state["target"])
        self._optimizer.load_state_dict(deepcopy(state["optimizer"]))
        self._rng.bit_generator.state = state["rng"]
