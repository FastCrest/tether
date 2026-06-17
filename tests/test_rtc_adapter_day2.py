"""Day 2 tests for RTC adapter (B.3) — predict_chunk_with_rtc body.

Day 2 scope: latency → actions_consumed dispatch, kwargs forwarded,
elapsed time recorded, fallback for policies that reject RTC kwargs.
prev_chunk_left_over is always None this day (carry-forward = Day 3).
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from tether.runtime.buffer import ActionChunkBuffer
from tether.runtime.rtc_adapter import RtcAdapter, RtcAdapterConfig


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _RtcAwarePolicy:
    """Mock policy that accepts the RTC kwargs (decomposed-style)."""

    def __init__(self, sleep_s: float = 0.0, action_shape=(1, 50, 7)):
        self.sleep_s = sleep_s
        self.action_shape = action_shape
        self.calls: list[dict] = []  # records every call's kwargs

    def predict_action_chunk(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
        return np.zeros(self.action_shape, dtype=np.float32)


class _RtcRejectingPolicy:
    """Mock policy that rejects unknown kwargs (monolithic-ONNX-style)."""

    def __init__(self, action_shape=(1, 50, 7)):
        self.action_shape = action_shape
        self.calls: list[dict] = []

    def predict_action_chunk(self, image=None, instruction=None, state=None):
        # Strict signature — TypeError on unknown kwargs
        self.calls.append({"image": image, "instruction": instruction, "state": state})
        return np.zeros(self.action_shape, dtype=np.float32)


def _adapter(
    policy,
    *,
    enabled: bool = False,
    execute_hz: float = 100.0,
    rtc_execution_horizon: int = 10,
) -> RtcAdapter:
    cfg = RtcAdapterConfig(
        enabled=enabled,
        execute_hz=execute_hz,
        rtc_execution_horizon=rtc_execution_horizon,
    )
    return RtcAdapter(
        policy=policy,
        action_buffer=ActionChunkBuffer(capacity=10),
        config=cfg,
    )


# ---------------------------------------------------------------------------
# predict_chunk_with_rtc body
# ---------------------------------------------------------------------------


class TestPredictChunkWithRtc:
    def test_returns_policy_actions(self):
        policy = _RtcAwarePolicy()
        adapter = _adapter(policy)
        actions = adapter.predict_chunk_with_rtc({"image": "fake", "state": [0.1]})
        assert isinstance(actions, np.ndarray)
        assert actions.shape == (1, 50, 7)

    def test_first_chunk_passes_prev_chunk_left_over_none(self):
        policy = _RtcAwarePolicy()
        adapter = _adapter(policy)
        adapter.predict_chunk_with_rtc({"image": "fake"})
        assert policy.calls[0]["prev_chunk_left_over"] is None

    def test_kwargs_forwarded_to_policy(self):
        policy = _RtcAwarePolicy()
        adapter = _adapter(policy)
        adapter.predict_chunk_with_rtc(
            {"image": "fake", "instruction": "pick", "state": [0.1, 0.2]}
        )
        call = policy.calls[0]
        assert call["image"] == "fake"
        assert call["instruction"] == "pick"
        assert call["state"] == [0.1, 0.2]
        assert "inference_delay" in call
        assert "prev_chunk_left_over" in call

    def test_inference_delay_is_int(self):
        """Lerobot's RTCProcessor expects an int — not a float."""
        policy = _RtcAwarePolicy()
        adapter = _adapter(policy)
        adapter.predict_chunk_with_rtc({"image": "fake"})
        assert isinstance(policy.calls[0]["inference_delay"], int)

    def test_records_elapsed_time(self):
        """The latency tracker is fed after every call."""
        policy = _RtcAwarePolicy(sleep_s=0.02)
        adapter = _adapter(policy)
        # Need to clear discard_first cold-start window for the sample to register
        adapter.latency.discard_first = 0
        adapter.predict_chunk_with_rtc({"image": "fake"})
        # First sample should be ~20ms (≥ 0.015 to avoid timer flake)
        samples = adapter.latency._samples
        assert len(samples) == 1
        assert samples[0] >= 0.015

    def test_latency_to_actions_consumed_warm(self):
        """80ms latency × 100Hz = 8 actions consumed (per plan example)."""
        policy = _RtcAwarePolicy()
        adapter = _adapter(policy, execute_hz=100.0)
        # Prime the tracker so estimate() returns a real number
        adapter.latency.discard_first = 0
        for _ in range(10):
            adapter.latency.record(0.080)
        # Now call — actions_consumed should be int(0.080 * 100) = 8
        adapter.predict_chunk_with_rtc({"image": "fake"})
        # First call uses the primed estimate (0.080) since record() happens AFTER
        # the call. So inference_delay reflects the primed value.
        assert policy.calls[0]["inference_delay"] == 8

    def test_cold_start_uses_fallback_estimate(self):
        """With no warm samples, latency.estimate() returns 0.1s fallback;
        actions_consumed = int(0.1 * 100) = 10."""
        policy = _RtcAwarePolicy()
        adapter = _adapter(policy, execute_hz=100.0)
        adapter.predict_chunk_with_rtc({"image": "fake"})
        assert policy.calls[0]["inference_delay"] == 10

    def test_execution_horizon_passed_when_enabled(self):
        """When config.enabled is True, execution_horizon is in the kwargs."""
        # Skip if lerobot not installed (enabled=True triggers RTCProcessor construction)
        from tether.runtime.rtc_adapter import _RTC_AVAILABLE
        if not _RTC_AVAILABLE:
            pytest.skip("lerobot not installed")
        policy = _RtcAwarePolicy()
        adapter = _adapter(policy, enabled=True, rtc_execution_horizon=12)
        adapter.predict_chunk_with_rtc({"image": "fake"})
        assert policy.calls[0]["execution_horizon"] == 12

    def test_execution_horizon_omitted_when_disabled(self):
        policy = _RtcAwarePolicy()
        adapter = _adapter(policy, enabled=False)
        adapter.predict_chunk_with_rtc({"image": "fake"})
        assert "execution_horizon" not in policy.calls[0]

    def test_fallback_when_policy_rejects_rtc_kwargs(self):
        """Monolithic ONNX policies have a strict signature — RTC kwargs
        cause TypeError. Adapter falls back to a plain call."""
        policy = _RtcRejectingPolicy()
        adapter = _adapter(policy)
        actions = adapter.predict_chunk_with_rtc({"image": "fake", "state": [0.1]})
        assert actions.shape == (1, 50, 7)
        # Policy was called twice: once with RTC kwargs (TypeError), once without
        # We track only the successful call
        assert len(policy.calls) == 1
        assert policy.calls[0]["image"] == "fake"

    def test_records_latency_even_on_fallback(self):
        """Fallback path still feeds the latency tracker. The timer wraps
        the entire try/except so the recorded value includes both the
        failed RTC-kwargs attempt AND the successful retry — the right
        measure for the adapter's effective per-call cost."""
        policy = _RtcRejectingPolicy()
        adapter = _adapter(policy)
        adapter.latency.discard_first = 0
        adapter.predict_chunk_with_rtc({"image": "fake"})
        assert len(adapter.latency._samples) == 1
        # Sample should be tiny but nonzero (no real work done by the mock)
        assert adapter.latency._samples[0] >= 0.0

    def test_chunk_count_not_incremented_by_predict(self):
        """chunk_count is incremented in merge_and_update, not predict.
        This separates 'we ran inference' from 'we accepted the chunk'."""
        policy = _RtcAwarePolicy()
        adapter = _adapter(policy)
        adapter.predict_chunk_with_rtc({"image": "fake"})
        assert adapter._chunk_count == 0  # only merge_and_update bumps this


# ---------------------------------------------------------------------------
# Interplay with LatencyTracker
# ---------------------------------------------------------------------------


class TestLatencyFeedback:
    def test_repeat_calls_warm_the_tracker(self):
        policy = _RtcAwarePolicy(sleep_s=0.01)
        adapter = _adapter(policy)
        adapter.latency.discard_first = 0
        for _ in range(5):
            adapter.predict_chunk_with_rtc({"image": "fake"})
        # 5 samples in the window
        assert adapter.latency.summary()["n"] == 5

    def test_actions_consumed_grows_with_observed_latency(self):
        """If real latency is high, actions_consumed grows accordingly."""
        slow_policy = _RtcAwarePolicy(sleep_s=0.05)
        adapter = _adapter(slow_policy, execute_hz=100.0)
        adapter.latency.discard_first = 0
        # Warm the tracker
        for _ in range(5):
            adapter.predict_chunk_with_rtc({"image": "fake"})
        # Make one more call — its inference_delay should reflect ~50ms × 100 = ~5
        adapter.predict_chunk_with_rtc({"image": "fake"})
        last = slow_policy.calls[-1]
        # Allow some slack for timer noise (3-15 actions)
        assert 3 <= last["inference_delay"] <= 15
