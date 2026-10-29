"""Day 2 tests for RTC adapter (B.3) — predict_chunk_with_rtc body.

Scope: latency → inference_delay dispatch, kwargs forwarded, and refusal
(no plain-inference fallback) for policies that reject the RTC kwargs.
Latency is recorded by merge_and_update, once per returned chunk.
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

    def normalize_actions(self, actions):
        return actions

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

    def test_predict_does_not_record_latency(self):
        """merge_and_update records once per returned chunk; predict used to
        record too, double-counting every chunk in the p95 window."""
        policy = _RtcAwarePolicy(sleep_s=0.02)
        adapter = _adapter(policy)
        adapter.latency.discard_first = 0
        adapter.predict_chunk_with_rtc({"image": "fake"})
        assert adapter.latency._samples == []

    def test_latency_to_actions_consumed_warm(self):
        """80ms latency × 100Hz = 8 actions consumed (per plan example)."""
        policy = _RtcAwarePolicy()
        adapter = _adapter(policy, execute_hz=100.0)
        # Prime the tracker so estimate() returns a real number
        adapter.latency.discard_first = 0
        for _ in range(10):
            adapter.latency.record(0.080)
        # Now call — inference_delay = ceil(0.080 * 100) = 8 (not 9: fp noise)
        adapter.predict_chunk_with_rtc({"image": "fake"})
        # First call uses the primed estimate (0.080) since record() happens AFTER
        # the call. So inference_delay reflects the primed value.
        assert policy.calls[0]["inference_delay"] == 8

    def test_cold_start_uses_fallback_estimate(self):
        """With no warm samples, latency.estimate() returns 0.1s fallback;
        inference_delay = ceil(0.1 * 100) = 10."""
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

    def test_inference_delay_rounds_up(self):
        """45 ms at 100 Hz executes 4.5 actions: LeRobot rounds up to 5 so the
        frozen prefix covers every action run during inference."""
        policy = _RtcAwarePolicy()
        adapter = _adapter(policy, execute_hz=100.0)
        adapter.latency.discard_first = 0
        for _ in range(10):
            adapter.latency.record(0.045)
        adapter.predict_chunk_with_rtc({"image": "fake"})
        assert policy.calls[0]["inference_delay"] == 5

    def test_policy_rejecting_rtc_kwargs_raises(self):
        """Monolithic-style signatures reject the kwargs. Plain inference
        would not apply RTC, so the adapter refuses instead of retrying."""
        policy = _RtcRejectingPolicy()
        adapter = _adapter(policy)
        with pytest.raises(RuntimeError, match="plain inference would not apply RTC"):
            adapter.predict_chunk_with_rtc({"image": "fake"})
        assert policy.calls == []

    def test_enabled_adapter_requires_normalize_actions(self):
        from tether.runtime.rtc_adapter import _RTC_AVAILABLE
        if not _RTC_AVAILABLE:
            pytest.skip("lerobot not installed")
        with pytest.raises(ValueError, match="normalize_actions"):
            _adapter(_RtcRejectingPolicy(), enabled=True)

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
    def test_repeat_cycles_warm_the_tracker(self):
        policy = _RtcAwarePolicy()
        adapter = _adapter(policy)
        adapter.latency.discard_first = 0
        for _ in range(5):
            chunk = adapter.predict_chunk_with_rtc({"image": "fake"})
            adapter.merge_and_update(chunk, elapsed_time=0.01)
        assert adapter.latency.summary()["n"] == 5

    def test_inference_delay_grows_with_recorded_latency(self):
        policy = _RtcAwarePolicy()
        adapter = _adapter(policy, execute_hz=100.0)
        adapter.latency.discard_first = 0
        for _ in range(5):
            chunk = adapter.predict_chunk_with_rtc({"image": "fake"})
            adapter.merge_and_update(chunk, elapsed_time=0.05)
        adapter.predict_chunk_with_rtc({"image": "fake"})
        assert policy.calls[-1]["inference_delay"] == 5
