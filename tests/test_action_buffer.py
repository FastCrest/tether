"""Tests for the async replan-while-execute action chunk buffer.

Goal: action-chunk-buffering. The ActionChunkBuffer data structure
implements the Physical Intelligence sliding_window pattern — the robot
pops one action per execute tick, a replan coroutine refreshes the buffer
head at the slower replan_hz.
"""
from __future__ import annotations

import numpy as np
import pytest

from tether.runtime.buffer import (
    ActionChunkBuffer,
    BufferStats,
    compute_replan_window,
)


class TestBasicBehavior:
    def test_empty_pop_returns_none(self):
        buf = ActionChunkBuffer(capacity=10)
        assert buf.pop_next() is None
        assert buf.size == 0

    def test_push_and_pop_in_order(self):
        buf = ActionChunkBuffer(capacity=10)
        chunk = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pushed = buf.push_chunk(chunk)
        assert pushed == 3
        assert buf.size == 3

        np.testing.assert_array_equal(buf.pop_next(), [1.0, 2.0])
        np.testing.assert_array_equal(buf.pop_next(), [3.0, 4.0])
        np.testing.assert_array_equal(buf.pop_next(), [5.0, 6.0])
        assert buf.pop_next() is None

    def test_push_truncates_to_capacity(self):
        buf = ActionChunkBuffer(capacity=3)
        chunk = np.arange(10 * 2).reshape(10, 2).astype(float)
        pushed = buf.push_chunk(chunk)
        assert pushed == 3
        assert buf.size == 3

    def test_peek_does_not_remove(self):
        buf = ActionChunkBuffer(capacity=5)
        buf.push_chunk(np.array([[7.0, 8.0]]))
        np.testing.assert_array_equal(buf.peek_next(), [7.0, 8.0])
        assert buf.size == 1  # peek didn't pop

    def test_rejects_non_2d(self):
        buf = ActionChunkBuffer(capacity=5)
        with pytest.raises(ValueError, match="must be 2D"):
            buf.push_chunk(np.array([1.0, 2.0, 3.0]))


class TestStaleOverwrite:
    def test_default_overwrite_discards_old(self):
        buf = ActionChunkBuffer(capacity=10)
        buf.push_chunk(np.array([[1.0], [2.0], [3.0]]))
        assert buf.size == 3

        buf.push_chunk(np.array([[10.0], [20.0]]))
        assert buf.size == 2
        np.testing.assert_array_equal(buf.pop_next(), [10.0])

    def test_stats_track_stale_overwrites(self):
        buf = ActionChunkBuffer(capacity=10)
        buf.push_chunk(np.array([[1.0], [2.0], [3.0]]))
        buf.push_chunk(np.array([[9.0]]))
        stats = buf.stats()
        assert stats.stale_overwrites == 3
        assert stats.replans == 2


class TestShouldReplan:
    def test_empty_triggers_replan(self):
        buf = ActionChunkBuffer(capacity=10)
        assert buf.should_replan(0.5) is True

    def test_full_does_not_trigger(self):
        buf = ActionChunkBuffer(capacity=4)
        buf.push_chunk(np.ones((4, 1)))
        assert buf.should_replan(0.5) is False

    def test_at_threshold_triggers(self):
        buf = ActionChunkBuffer(capacity=4)
        buf.push_chunk(np.ones((4, 1)))
        buf.pop_next()
        buf.pop_next()  # 2 left, threshold 0.5 * 4 = 2
        assert buf.should_replan(0.5) is True


class TestReplanWindow:
    def test_100hz_20hz(self):
        w = compute_replan_window(execute_hz=100, replan_hz=20)
        # per_cycle = ceil(100/20) = 5; capacity = 2*5 = 10
        assert w["capacity"] == 10
        assert w["actions_per_replan_cycle"] == 5
        assert 0.49 < w["threshold_ratio"] < 0.51

    def test_chunk_size_caps_capacity(self):
        # If replan is slow, capacity would explode — chunk_size caps it.
        w = compute_replan_window(execute_hz=200, replan_hz=2, chunk_size=20)
        # per_cycle = ceil(200/2) = 100, 2*100 = 200 → capped at 20
        assert w["capacity"] == 20

    def test_backwards_rejected(self):
        # Replanning faster than executing is pathological.
        with pytest.raises(ValueError, match="backwards"):
            compute_replan_window(execute_hz=10, replan_hz=20)

    def test_non_positive_rejected(self):
        with pytest.raises(ValueError):
            compute_replan_window(execute_hz=0, replan_hz=10)


class TestClear:
    def test_clear_empties_and_counts(self):
        buf = ActionChunkBuffer(capacity=10)
        buf.push_chunk(np.ones((5, 1)))
        buf.clear()
        assert buf.size == 0
        assert buf.stats().stale_overwrites == 5
