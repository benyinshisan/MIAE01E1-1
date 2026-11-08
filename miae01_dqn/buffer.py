"""Replay buffer. numpy only - this module must not import torch.

Implementation: issue I-04.

Layout: one preallocated uint8 ring of observations. ``next_obs`` is never
stored; it is read one slot ahead, so a transition occupies exactly one slot::

    obs[i]             -> the frame observed at step i
    obs[(i + 1) % C]   -> its successor, i.e. the ``next_obs`` of that step
    actions[i], rewards[i], dones[i] -> the rest of the transition

Invariants
----------
* one copy of every frame: ``(capacity, 4, 84, 84)`` uint8, ~5.6 GB at 200k
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

    A transition is stored where its *observation* lands; ``next_obs`` is read
    from the next slot, so the buffer holds one copy of every frame instead of
    two. ``push`` only accepts true terminations - time-limit truncation must be
    filtered by the caller (see ``docs/architecture.md`` section 4).
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
    ) -> None:
        """Append one transition, overwriting the oldest slot once full."""
        pos = self._pos
        # Writing ``obs`` is idempotent from the second push on (it equals the
        # previous ``next_obs``) and bootstraps slot 0, which would otherwise
        # stay all-zero and be sampled as a fake frame while the ring fills up.
        self._obs[pos] = self._as_frame(obs, "obs")
        self._obs[(pos + 1) % self._capacity] = self._as_frame(next_obs, "next_obs")
        self._actions[pos] = action
        self._rewards[pos] = reward
        self._dones[pos] = done

        self._pos = (pos + 1) % self._capacity
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
        nxt = (idx + 1) % self._capacity

        # Fancy indexing copies, so later pushes cannot mutate a handed-out batch.
        return TransitionBatch(
            obs=self._obs[idx],
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            next_obs=self._obs[nxt],
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
