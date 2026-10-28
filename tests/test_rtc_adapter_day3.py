"""Day 3 tests for RTC adapter (B.3) — prev_chunk_left_over carry-forward.

Scope: the leftover is the unexecuted tail of the chunk last returned to the
client, sliced at the client's ``actions_executed`` report (or a labelled
wall-clock estimate), validated against ``chunk_id``, and converted into the
policy's normalized action space before guidance. ``peek_all`` stays covered
for in-process buffer integrations.
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

    def normalize_actions(self, actions):
        return actions

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
# Leftover = returned chunk sliced at the client's report
# ---------------------------------------------------------------------------


def _chunk(n=10, d=7, base=0.0):
    return (base + np.arange(n * d, dtype=np.float32)).reshape(n, d)


class TestLeftoverFromReport:
    def test_no_previous_chunk_means_no_leftover(self):
        adapter = _adapter(_RecordingPolicy())
        plan = adapter.plan_guidance()
        assert plan.guidance.prev_chunk_left_over is None
        assert plan.leftover_source == "none"
        assert plan.telemetry()["applied"] is False

    def test_client_report_slices_returned_chunk(self):
        adapter = _adapter(_RecordingPolicy(), capacity=10)
        cid = adapter.merge_and_update(_chunk(), elapsed_time=0.05)
        plan = adapter.plan_guidance(actions_executed=4, chunk_id=cid)
        np.testing.assert_array_equal(plan.guidance.prev_chunk_left_over, _chunk()[4:])
        assert plan.leftover_source == "client"
        assert plan.actions_executed == 4
        assert plan.telemetry()["leftover_len"] == 6

    def test_unpopped_buffer_is_not_the_leftover(self):
        """Regression: HTTP /act never pops the buffer, so peek_all() returned
        the whole previous chunk as leftover. The report now decides."""
        adapter = _adapter(_RecordingPolicy(), capacity=10)
        cid = adapter.merge_and_update(_chunk(), elapsed_time=0.05)
        assert adapter.buffer.peek_all().shape == (10, 7)  # nothing popped
        plan = adapter.plan_guidance(actions_executed=7, chunk_id=cid)
        assert plan.guidance.prev_chunk_left_over.shape == (3, 7)

    def test_fully_executed_chunk_has_no_leftover(self):
        adapter = _adapter(_RecordingPolicy())
        cid = adapter.merge_and_update(_chunk(), elapsed_time=0.05)
        plan = adapter.plan_guidance(actions_executed=10, chunk_id=cid)
        assert plan.guidance.prev_chunk_left_over is None
        assert plan.leftover_source == "client"

    def test_estimate_without_report_is_labelled(self):
        adapter = _adapter(_RecordingPolicy())  # execute_hz=100
        adapter.merge_and_update(_chunk(), elapsed_time=0.05, now=100.0)
        plan = adapter.plan_guidance(now=100.03)  # 30 ms → 3 actions
        assert plan.leftover_source == "estimated"
        assert plan.actions_executed == 3
        np.testing.assert_array_equal(plan.guidance.prev_chunk_left_over, _chunk()[3:])

    def test_estimate_is_capped_at_chunk_length(self):
        adapter = _adapter(_RecordingPolicy())
        adapter.merge_and_update(_chunk(), elapsed_time=0.05, now=0.0)
        plan = adapter.plan_guidance(now=5.0)
        assert plan.actions_executed == 10
        assert plan.guidance.prev_chunk_left_over is None

    def test_chunk_ids_are_unique_per_chunk(self):
        adapter = _adapter(_RecordingPolicy())
        a = adapter.merge_and_update(_chunk(), elapsed_time=0.05)
        b = adapter.merge_and_update(_chunk(), elapsed_time=0.05)
        assert a != b and len(a) == 16


class TestReportValidation:
    def _adapter_with_chunk(self):
        adapter = _adapter(_RecordingPolicy())
        return adapter, adapter.merge_and_update(_chunk(), elapsed_time=0.05)

    @pytest.mark.parametrize("kwargs", [{"actions_executed": 2}, {"chunk_id": "abc"}])
    def test_fields_must_come_together(self, kwargs):
        from tether.runtime.rtc_adapter import RtcRequestError
        adapter, _ = self._adapter_with_chunk()
        with pytest.raises(RtcRequestError) as exc:
            adapter.plan_guidance(**kwargs)
        assert exc.value.status == 422 and exc.value.code == "rtc_report_incomplete"

    def test_stale_chunk_id_is_a_conflict(self):
        from tether.runtime.rtc_adapter import RtcRequestError
        adapter, cid = self._adapter_with_chunk()
        adapter.merge_and_update(_chunk(base=1.0), elapsed_time=0.05)
        with pytest.raises(RtcRequestError) as exc:
            adapter.plan_guidance(actions_executed=1, chunk_id=cid)
        assert exc.value.status == 409 and exc.value.code == "rtc_chunk_mismatch"

    def test_report_before_any_chunk_is_a_conflict(self):
        from tether.runtime.rtc_adapter import RtcRequestError
        adapter = _adapter(_RecordingPolicy())
        with pytest.raises(RtcRequestError) as exc:
            adapter.plan_guidance(actions_executed=0, chunk_id="abc")
        assert exc.value.status == 409

    @pytest.mark.parametrize("n", [-1, 11])
    def test_out_of_range_count_is_refused(self, n):
        from tether.runtime.rtc_adapter import RtcRequestError
        adapter, cid = self._adapter_with_chunk()
        with pytest.raises(RtcRequestError) as exc:
            adapter.plan_guidance(actions_executed=n, chunk_id=cid)
        assert exc.value.status == 422 and exc.value.code == "rtc_actions_executed_invalid"

    def test_reset_invalidates_old_chunk_id(self):
        from tether.runtime.rtc_adapter import RtcRequestError
        adapter, cid = self._adapter_with_chunk()
        adapter.reset(episode_id="ep-2")
        with pytest.raises(RtcRequestError):
            adapter.plan_guidance(actions_executed=1, chunk_id=cid)


class TestLeftoverUnits:
    class _AffinePolicy(_RecordingPolicy):
        """Processed = normalized * 2 + 1; normalize is its exact inverse."""

        def normalize_actions(self, actions):
            return (np.asarray(actions) - 1.0) / 2.0

    def test_leftover_is_converted_to_normalized_space(self):
        """Regression: the leftover was handed to guidance in processed
        (denormalized) units while guidance runs in normalized space."""
        policy = self._AffinePolicy()
        adapter = _adapter(policy)
        processed = _chunk() * 2.0 + 1.0
        cid = adapter.merge_and_update(processed, elapsed_time=0.05)
        adapter.predict_chunk_with_rtc({"image": "x"}, actions_executed=3, chunk_id=cid)
        left = policy.calls[0]["prev_chunk_left_over"]
        np.testing.assert_allclose(left, _chunk()[3:], rtol=0, atol=1e-6)
        assert left.dtype == np.float32


# ---------------------------------------------------------------------------
# merge_and_update bookkeeping
# ---------------------------------------------------------------------------


class TestMergeBookkeeping:
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
        assert adapter.buffer.size == 10
        assert adapter.buffer.peek_all().shape == (10, 7)
        assert adapter._last_chunk.shape == (10, 7)


# ---------------------------------------------------------------------------
# Reset semantics
# ---------------------------------------------------------------------------


class TestResetClearsCarry:
    def test_reset_clears_carry_state(self):
        adapter = _adapter(_RecordingPolicy(), capacity=10)
        cid = adapter.merge_and_update(_chunk(), elapsed_time=0.05)
        adapter.plan_guidance(actions_executed=1, chunk_id=cid)
        assert adapter._prev_chunk_left_over is not None

        adapter.reset(episode_id="ep-2")

        assert adapter._prev_chunk_left_over is None
        assert adapter._last_chunk is None
        assert adapter._chunk_count == 0

    def test_reset_isolates_episodes(self):
        policy = _RecordingPolicy()
        adapter = _adapter(policy, capacity=10)
        adapter.merge_and_update(_chunk(), elapsed_time=0.05)
        adapter.reset(episode_id="ep-2")
        adapter.predict_chunk_with_rtc({"image": "fresh"})
        assert policy.calls[0]["prev_chunk_left_over"] is None
