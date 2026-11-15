"""Replay buffer. numpy only - this module must not import torch.

Implementation: issue I-04.

Layout: one preallocated uint8 ring of observations. Normally ``next_obs``
is read one slot ahead, so a transition occupies one slot::

    obs[i]             -> the frame observed at step i
    obs[(i + 1) % C]   -> its successor, i.e. the ``next_obs`` of that step
    actions[i], rewards[i], dones[i] -> the rest of the transition

A time-limit truncation also saves its final observation separately because
the next push overwrites the shared slot with the reset observation.

Invariants
----------
* one preallocated observation array: ``(capacity, 4, 84, 84)`` uint8, ~5.6 GB at 200k
  and ~28 GB at the paper's 1M. Storing float32 instead would be four times
  larger, which is why the division by 255 lives in the network, not here;
* the slot under the write pointer is never sampled: its successor is the slot
  the next ``push`` will overwrite, so that pair would be misaligned;
* ``len(buf)`` is the number of pushes, saturating at ``capacity``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class TransitionBatch(NamedTuple):
    obs: np.ndarray  # (B, 4, 84, 84) uint8
    actions: np.ndarray  # (B,) int64
    rewards: np.ndarray  # (B,) float32
    next_obs: np.ndarray  # (B, 4, 84, 84) uint8
    dones: np.ndarray  # (B,) bool


class ReplayBuffer:
    """Single-array ring buffer of uint8 frames, sampled uniformly.

    A transition is stored where its observation lands. The buffer reads
    ``next_obs`` from the next slot except for nonterminal truncations, which
    keep their final observation until the transition is evicted.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, int, int],
        *,
        seed: int | None = None,
    ) -> None:
        capacity = int(capacity)
        if capacity < 2:
            # With one slot the successor of the newest frame is the frame
            # itself, so no aligned (obs, next_obs) pair can exist.
            raise ValueError(f"capacity must be >= 2, got {capacity}")
        shape = tuple(int(size) for size in obs_shape)
        if len(shape) != 3 or any(size <= 0 for size in shape):
            raise ValueError(f"obs_shape must be 3 positive ints, got {obs_shape!r}")

        self._capacity = capacity
        self._obs_shape = shape
        self._obs = np.zeros((capacity, *shape), dtype=np.uint8)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=bool)
        self._truncated_next_obs: dict[int, np.ndarray] = {}
        self._pos = 0  # slot holding the newest observation
        self._full = False
        self._rng = np.random.default_rng(seed)

    # -- read-only views ---------------------------------------------------
    @property
    def capacity(self) -> int:
        """Number of slots; ``len(buf)`` reaches this value and then stops."""
        return self._capacity

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        """Shape of a single stored frame, e.g. ``(4, 84, 84)``."""
        return self._obs_shape

    @property
    def is_full(self) -> bool:
        """Whether the ring has wrapped at least once."""
        return self._full

    def __len__(self) -> int:
        """Transitions available for sampling: ``pos`` before wrapping, then ``capacity``."""
        return self._capacity if self._full else self._pos

    # -- writing -----------------------------------------------------------
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
        """Append a transition. Set ``done`` only for a terminal transition."""
        pos = self._pos
        next_pos = (pos + 1) % self._capacity
        current_frame = self._as_frame(obs, "obs")
        next_frame = self._as_frame(next_obs, "next_obs")
        self._obs[pos] = current_frame
        self._obs[next_pos] = next_frame
        self._actions[pos] = action
        self._rewards[pos] = reward
        self._dones[pos] = done

        # The next write slot's old transition is no longer sampleable.
        self._truncated_next_obs.pop(next_pos, None)
        if truncated and not done:
            self._truncated_next_obs[pos] = next_frame.copy()

        self._pos = next_pos
        if self._pos == 0:
            self._full = True

    # -- sampling ----------------------------------------------------------
    def sample(self, batch_size: int) -> TransitionBatch:
        """Draw ``batch_size`` transitions uniformly, with replacement.

        Raises ``ValueError`` instead of silently shrinking the batch, because a
        short batch would shift every downstream loss scale and hide the bug.
        """
        batch_size = int(batch_size)
        size = len(self)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if size < batch_size:
            raise ValueError(
                f"cannot sample {batch_size} transitions, buffer holds {size}"
            )

        if self._full:
            # 1..capacity-1 shifted by pos: every slot except the write pointer,
            # whose successor is about to be overwritten.
            idx = (
                self._rng.integers(1, self._capacity, size=batch_size) + self._pos
            ) % self._capacity
        else:
            idx = self._rng.integers(0, self._pos, size=batch_size)
        # Fancy indexing copies, so later pushes cannot mutate a handed-out batch.
        next_obs = self._obs[(idx + 1) % self._capacity]
        if self._truncated_next_obs:
            for batch_index, slot in enumerate(idx):
                final_obs = self._truncated_next_obs.get(int(slot))
                if final_obs is not None:
                    next_obs[batch_index] = final_obs
        return TransitionBatch(
            obs=self._obs[idx],
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            next_obs=next_obs,
            dones=self._dones[idx],
        )

    # -- helpers -----------------------------------------------------------
    def _as_frame(self, frame: np.ndarray, name: str) -> np.ndarray:
        """Validate one frame's shape and view it as uint8."""
        array = np.asarray(frame)
        if array.shape != self._obs_shape:
            raise ValueError(
                f"{name} must have shape {self._obs_shape}, got {array.shape}"
            )
        return array.astype(np.uint8, copy=False)
