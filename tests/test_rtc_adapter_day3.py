"""Day 3 tests for RTC adapter (B.3) — prev_chunk_left_over carry-forward.

Day 3 scope: ActionChunkBuffer.peek_all() snapshot, merge_and_update
captures the snapshot BEFORE push_chunk wipes it, predict_chunk_with_rtc
reads it on the next call. reset() already clears it (Day 1 ✓).
"""
from __future__ import annotations

import numpy as np
import pytest

from tether.runtime.buffer import ActionChunkBuffer
from tether.runtime.rtc_adapter import RtcAdapter, RtcAdapterConfig


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _RecordingPolicy:
    def __init__(self, action_shape=(50, 7)):
        self.action_shape = action_shape
        self.calls: list[dict] = []

    def predict_action_chunk(self, **kwargs):
        self.calls.append(dict(kwargs))
        # Return 2D so the adapter doesn't need to unwrap a batch dim
        return np.zeros(self.action_shape, dtype=np.float32)


def _adapter(policy, capacity: int = 10):
    cfg = RtcAdapterConfig(enabled=False, execute_hz=100.0)
    return RtcAdapter(
        policy=policy,
        action_buffer=ActionChunkBuffer(capacity=capacity),
        config=cfg,
    )


# ---------------------------------------------------------------------------
# ActionChunkBuffer.peek_all
# ---------------------------------------------------------------------------


class TestPeekAll:
    def test_empty_returns_none(self):
        buf = ActionChunkBuffer(capacity=10)
        assert buf.peek_all() is None

    def test_returns_stacked_snapshot(self):
        buf = ActionChunkBuffer(capacity=10)
        chunk = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        buf.push_chunk(chunk)
        snap = buf.peek_all()
        assert snap is not None
        assert snap.shape == (3, 2)
        np.testing.assert_array_equal(snap, chunk)

    def test_does_not_modify_buffer(self):
        buf = ActionChunkBuffer(capacity=10)
        buf.push_chunk(np.zeros((5, 7), dtype=np.float32))
        size_before = buf.size
        _ = buf.peek_all()
        assert buf.size == size_before

    def test_returns_copy_not_view(self):
        buf = ActionChunkBuffer(capacity=10)
        buf.push_chunk(np.zeros((3, 2), dtype=np.float32))
        snap = buf.peek_all()
        snap[0, 0] = 99.0
        # Buffer's internal data unchanged
        snap2 = buf.peek_all()
        assert snap2[0, 0] == 0.0

    def test_after_pop_excludes_popped(self):
        buf = ActionChunkBuffer(capacity=10)
        buf.push_chunk(np.array([[1.0], [2.0], [3.0]], dtype=np.float32))
        buf.pop_next()
        snap = buf.peek_all()
        assert snap.shape == (2, 1)
        np.testing.assert_array_equal(snap, np.array([[2.0], [3.0]]))


# ---------------------------------------------------------------------------
# merge_and_update carry-forward
# ---------------------------------------------------------------------------


class TestMergeCarryForward:
    def test_first_merge_no_carry_state(self):
        """Buffer empty → snapshot is None → _prev_chunk_left_over stays None."""
        adapter = _adapter(_RecordingPolicy())
        actions = np.zeros((50, 7), dtype=np.float32)
        adapter.merge_and_update(actions, elapsed_time=0.05)
        assert adapter._prev_chunk_left_over is None

    def test_second_merge_captures_first_chunk(self):
        """After first merge, buffer has 50 actions. Second merge should
        snapshot those 50 (or capacity-limited count) BEFORE push wipes."""
        adapter = _adapter(_RecordingPolicy(), capacity=10)
        adapter.merge_and_update(
            np.arange(50 * 7, dtype=np.float32).reshape(50, 7),
            elapsed_time=0.05,
        )
        # Buffer now holds the first 10 actions of chunk 1 (capacity=10)
        # No pops happened — second merge snapshots all 10 before wipe
        adapter.merge_and_update(
            np.full((50, 7), 99.0, dtype=np.float32),  # second chunk
            elapsed_time=0.05,
        )
        assert adapter._prev_chunk_left_over is not None
        assert adapter._prev_chunk_left_over.shape == (10, 7)
        # Snapshot should be from chunk 1 (zeros 0..69), NOT chunk 2 (99s)
        assert adapter._prev_chunk_left_over[0, 0] == 0.0  # first action of chunk 1

    def test_merge_after_pops_captures_remaining(self):
        """If the robot has popped 4 actions from a 10-cap buffer, the
        snapshot should hold the remaining 6."""
        adapter = _adapter(_RecordingPolicy(), capacity=10)
        adapter.merge_and_update(
            np.arange(10 * 7, dtype=np.float32).reshape(10, 7),
            elapsed_time=0.05,
        )
        # Pop 4
        for _ in range(4):
            adapter.buffer.pop_next()
        assert adapter.buffer.size == 6

        adapter.merge_and_update(
            np.zeros((10, 7), dtype=np.float32),
            elapsed_time=0.05,
        )
        assert adapter._prev_chunk_left_over.shape == (6, 7)

    def test_carry_forward_is_passed_to_next_predict(self):
        """The snapshot from merge N is consumed by predict N+1."""
        policy = _RecordingPolicy()
        adapter = _adapter(policy, capacity=10)

        # Cycle 1: predict (no carry yet) + merge (now has carry)
        adapter.predict_chunk_with_rtc({"image": "x"})
        assert policy.calls[0]["prev_chunk_left_over"] is None  # first call
        adapter.merge_and_update(
            np.arange(10 * 7, dtype=np.float32).reshape(10, 7),
            elapsed_time=0.05,
        )
        assert adapter._prev_chunk_left_over is None  # buffer was empty pre-push

        # Cycle 2: predict (now has carry from cycle 1) + merge
        adapter.predict_chunk_with_rtc({"image": "y"})
        # Cycle 2's predict still sees None because buffer was empty when
        # merge 1 ran; the carry shows up on cycle 3.
        assert policy.calls[1]["prev_chunk_left_over"] is None

        adapter.merge_and_update(
            np.full((10, 7), 5.0, dtype=np.float32),
            elapsed_time=0.05,
        )
        # NOW snapshot captures chunk-1 (which is in buffer pre-push)
        assert adapter._prev_chunk_left_over is not None
        assert adapter._prev_chunk_left_over[0, 0] == 0.0  # chunk 1 first action

        # Cycle 3: predict sees the carry from cycle 2's snapshot
        adapter.predict_chunk_with_rtc({"image": "z"})
        assert policy.calls[2]["prev_chunk_left_over"] is not None
        assert policy.calls[2]["prev_chunk_left_over"].shape == (10, 7)

    def test_chunk_count_increments_on_merge(self):
        adapter = _adapter(_RecordingPolicy())
        assert adapter._chunk_count == 0
        adapter.merge_and_update(
            np.zeros((10, 7), dtype=np.float32), elapsed_time=0.05
        )
        assert adapter._chunk_count == 1
        adapter.merge_and_update(
            np.zeros((10, 7), dtype=np.float32), elapsed_time=0.05
        )
        assert adapter._chunk_count == 2

    def test_merge_records_latency(self):
        adapter = _adapter(_RecordingPolicy())
        adapter.latency.discard_first = 0
        adapter.merge_and_update(
            np.zeros((10, 7), dtype=np.float32), elapsed_time=0.123
        )
        assert adapter.latency._samples[-1] == pytest.approx(0.123)

    def test_merge_unwraps_3d_actions(self):
        """If the policy returns shape (1, T, A), merge unwraps the batch dim."""
        adapter = _adapter(_RecordingPolicy(), capacity=10)
        adapter.merge_and_update(
            np.zeros((1, 10, 7), dtype=np.float32), elapsed_time=0.05
        )
        # Buffer should have 10 actions of dim 7 — confirms unwrap worked
        assert adapter.buffer.size == 10
        assert adapter.buffer.peek_all().shape == (10, 7)


# ---------------------------------------------------------------------------
# Reset semantics (Day 1 ✓ — re-tested with carry state populated)
# ---------------------------------------------------------------------------


class TestResetClearsCarry:
    def test_reset_clears_prev_chunk_left_over(self):
        adapter = _adapter(_RecordingPolicy(), capacity=10)
        # Populate _prev_chunk_left_over via two merges
        adapter.merge_and_update(
            np.zeros((10, 7), dtype=np.float32), elapsed_time=0.05
        )
        adapter.merge_and_update(
            np.zeros((10, 7), dtype=np.float32), elapsed_time=0.05
        )
        assert adapter._prev_chunk_left_over is not None  # carry exists

        adapter.reset(episode_id="ep-2")

        assert adapter._prev_chunk_left_over is None
        assert adapter._chunk_count == 0

    def test_reset_isolates_episodes(self):
        """After reset, the next predict gets None even if old chunks
        were in the buffer."""
        policy = _RecordingPolicy()
        adapter = _adapter(policy, capacity=10)
        # Populate carry
        adapter.merge_and_update(
            np.zeros((10, 7), dtype=np.float32), elapsed_time=0.05
        )
        adapter.merge_and_update(
            np.zeros((10, 7), dtype=np.float32), elapsed_time=0.05
        )
        assert adapter._prev_chunk_left_over is not None

        # Reset, then predict — first call of new episode should see None
        adapter.reset(episode_id="ep-2")
        adapter.predict_chunk_with_rtc({"image": "fresh"})
        assert policy.calls[0]["prev_chunk_left_over"] is None
