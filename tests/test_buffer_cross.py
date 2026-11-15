"""Issue #10: independent, public-API tests of the merged Issue #4 buffer."""

import numpy as np
import pytest

from miae01_dqn.buffer import ReplayBuffer


SHAPE = (4, 2, 2)


def frame(value):
    return np.full(SHAPE, value, dtype=np.uint8)


def push_sequence(buffer, start, stop):
    for t in range(start, stop):
        buffer.push(frame(t), t, float(t) / 2, frame(t + 1), False)


def assert_batch_aligned(batch):
    for obs, action, reward, next_obs, done in zip(*batch):
        np.testing.assert_array_equal(obs, frame(action))
        np.testing.assert_array_equal(next_obs, frame(action + 1))
        assert reward == float(action) / 2
        assert not done
    assert batch.obs.dtype == batch.next_obs.dtype == np.uint8
    assert batch.actions.dtype == np.int64
    assert batch.rewards.dtype == np.float32
    assert batch.dones.dtype == np.bool_


@pytest.mark.parametrize("writes", [1, 4, 5, 6, 10, 12])
def test_pairing_before_and_after_ring_wrap(writes):
    buffer = ReplayBuffer(5, SHAPE, seed=7)
    push_sequence(buffer, 0, writes)
    assert len(buffer) == min(writes, 5)
    assert buffer.is_full == (writes >= 5)
    seen = set()
    for _ in range(64):
        batch = buffer.sample(min(writes, 5))
        assert_batch_aligned(batch)
        seen.update(batch.actions.tolist())
    # One slot contains only the latest successor after wrapping; the other
    # four hold the most recent complete transitions. No private pointers read.
    expected = set(range(max(0, writes - 4), writes)) if writes >= 5 else set(range(writes))
    assert seen == expected


def test_overwrite_removes_oldest_and_never_samples_successor_only_slot():
    buffer = ReplayBuffer(4, SHAPE, seed=42)
    push_sequence(buffer, 0, 4)
    for _ in range(50):
        batch = buffer.sample(4)
        assert set(batch.actions) <= {1, 2, 3}
        assert_batch_aligned(batch)
    push_sequence(buffer, 4, 5)
    for _ in range(50):
        batch = buffer.sample(4)
        assert set(batch.actions) <= {2, 3, 4}
        assert_batch_aligned(batch)


def test_too_few_samples_raise_instead_of_shrinking():
    buffer = ReplayBuffer(5, SHAPE, seed=1)
    with pytest.raises(ValueError):
        buffer.sample(1)
    push_sequence(buffer, 0, 2)
    with pytest.raises(ValueError):
        buffer.sample(3)
    assert buffer.sample(2).obs.shape == (2, *SHAPE)


def test_returned_batches_are_not_mutated_by_later_pushes():
    buffer = ReplayBuffer(5, SHAPE, seed=1)
    push_sequence(buffer, 0, 4)
    batch = buffer.sample(4)
    saved = tuple(array.copy() for array in batch)
    push_sequence(buffer, 4, 12)
    for actual, expected in zip(batch, saved):
        np.testing.assert_array_equal(actual, expected)


def test_true_termination_flag_is_preserved():
    buffer = ReplayBuffer(3, SHAPE, seed=1)
    buffer.push(frame(0), 2, -1.0, frame(1), True)
    batch = buffer.sample(1)
    assert batch.dones.tolist() == [True]
    assert batch.actions.tolist() == [2]
    assert batch.rewards.tolist() == [-1.0]
    np.testing.assert_array_equal(batch.obs[0], frame(0))
    np.testing.assert_array_equal(batch.next_obs[0], frame(1))


def test_equal_seeds_reproduce_sampling_without_global_rng():
    first = ReplayBuffer(8, SHAPE, seed=3)
    second = ReplayBuffer(8, SHAPE, seed=3)
    push_sequence(first, 0, 12)
    push_sequence(second, 0, 12)
    for _ in range(5):
        for a, b in zip(first.sample(4), second.sample(4)):
            np.testing.assert_array_equal(a, b)
