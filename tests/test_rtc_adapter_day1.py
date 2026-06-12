"""Day 1 tests for RTC adapter (B.3).

Day 1 scope: RtcAdapterConfig validation, RTCProcessor construction
when enabled, no construction when disabled, lerobot config mapping
(_build_lerobot_rtc_config). Body methods (predict_chunk_with_rtc,
merge_and_update beyond pass-through) land Day 2-3 with their own tests.

Skips lerobot-dependent tests gracefully when lerobot isn't installed.
"""
from __future__ import annotations

import pytest

from tether.runtime.rtc_adapter import (
    LatencyTracker,
    RtcAdapter,
    RtcAdapterConfig,
    _RTC_AVAILABLE,
    _VALID_SCHEDULES,
    _build_lerobot_rtc_config,
    require_rtc,
)
from tether.runtime.buffer import ActionChunkBuffer


# ---------------------------------------------------------------------------
# RtcAdapterConfig validation (no lerobot needed)
# ---------------------------------------------------------------------------


class TestRtcAdapterConfigValidation:
    def test_default_disabled(self):
        cfg = RtcAdapterConfig()
        assert cfg.enabled is False
        assert cfg.prefix_attention_schedule == "LINEAR"
        assert cfg.max_guidance_weight == 10.0
        assert cfg.rtc_execution_horizon == 10

    @pytest.mark.parametrize("schedule", _VALID_SCHEDULES)
    def test_all_valid_schedules_accepted(self, schedule):
        cfg = RtcAdapterConfig(prefix_attention_schedule=schedule)
        assert cfg.prefix_attention_schedule == schedule

    def test_invalid_schedule_rejected(self):
        with pytest.raises(ValueError, match="prefix_attention_schedule"):
            RtcAdapterConfig(prefix_attention_schedule="LOGARITHMIC")

    def test_negative_max_guidance_weight_rejected(self):
        with pytest.raises(ValueError, match="max_guidance_weight"):
            RtcAdapterConfig(max_guidance_weight=-1.0)

    def test_zero_max_guidance_weight_rejected(self):
        with pytest.raises(ValueError, match="max_guidance_weight"):
            RtcAdapterConfig(max_guidance_weight=0.0)

    def test_zero_execution_horizon_rejected(self):
        with pytest.raises(ValueError, match="rtc_execution_horizon"):
            RtcAdapterConfig(rtc_execution_horizon=0)

    def test_latency_percentile_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="latency_percentile"):
            RtcAdapterConfig(latency_percentile=100)
        with pytest.raises(ValueError, match="latency_percentile"):
            RtcAdapterConfig(latency_percentile=0)

    def test_latency_percentile_p95_default(self):
        assert RtcAdapterConfig().latency_percentile == 95

    def test_latency_percentile_p99_accepted(self):
        cfg = RtcAdapterConfig(latency_percentile=99)
        assert cfg.latency_percentile == 99

    def test_adaptive_chunking_config_defaults_off(self):
        cfg = RtcAdapterConfig()
        assert cfg.adaptive_chunking_enabled is False
        assert cfg.adaptive_min_horizon == 1

    def test_invalid_adaptive_min_horizon_rejected(self):
        with pytest.raises(ValueError, match="adaptive_min_horizon"):
            RtcAdapterConfig(adaptive_min_horizon=0)


# ---------------------------------------------------------------------------
# LatencyTracker (already shipped in skeleton)
# ---------------------------------------------------------------------------


class TestLatencyTracker:
    def test_empty_returns_fallback(self):
        t = LatencyTracker()
        assert t.estimate() == pytest.approx(0.1)
        assert t.summary()["n"] == 0

    def test_cold_start_discard_applied(self):
        t = LatencyTracker(discard_first=3)
        for _ in range(3):
            t.record(1.0)  # all discarded
        assert t.summary()["n"] == 0
        t.record(0.05)
        assert t.summary()["n"] == 1
        # The 1.0s samples should NOT be in the window
        assert t.estimate() == pytest.approx(0.05)

    def test_window_size_caps_samples(self):
        t = LatencyTracker(window_size=10, discard_first=0)
        for v in range(20):
            t.record(float(v))
        assert t.summary()["n"] == 10

    def test_p95_estimate(self):
        t = LatencyTracker(discard_first=0, percentile=95)
        for v in [0.04, 0.05, 0.06, 0.07, 0.08]:
            t.record(v)
        # p95 of these samples
        est = t.estimate()
        assert 0.06 < est < 0.09


# ---------------------------------------------------------------------------
# require_rtc gating
# ---------------------------------------------------------------------------


class TestRequireRtc:
    def test_require_rtc_silent_when_available(self):
        if not _RTC_AVAILABLE:
            pytest.skip("lerobot not installed in this env")
        require_rtc()  # should not raise

    def test_require_rtc_raises_when_unavailable(self, monkeypatch):
        # Force the unavailable path even if lerobot is installed
        monkeypatch.setattr("tether.runtime.rtc_adapter._RTC_AVAILABLE", False)
        with pytest.raises(ImportError, match="lerobot"):
            require_rtc()


# ---------------------------------------------------------------------------
# _build_lerobot_rtc_config — mapping fidelity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RTC_AVAILABLE, reason="lerobot not installed")
class TestBuildLerobotRtcConfig:
    def test_default_mapping(self):
        cfg = RtcAdapterConfig(enabled=True)
        lerobot_cfg = _build_lerobot_rtc_config(cfg)
        assert lerobot_cfg.enabled is True
        assert lerobot_cfg.execution_horizon == 10
        assert lerobot_cfg.max_guidance_weight == 10.0
        assert str(lerobot_cfg.prefix_attention_schedule) == "RTCAttentionSchedule.LINEAR"

    def test_custom_schedule_propagates(self):
        cfg = RtcAdapterConfig(enabled=True, prefix_attention_schedule="EXP")
        lerobot_cfg = _build_lerobot_rtc_config(cfg)
        assert str(lerobot_cfg.prefix_attention_schedule) == "RTCAttentionSchedule.EXP"

    def test_debug_propagates(self):
        cfg = RtcAdapterConfig(enabled=True, debug=True, debug_maxlen=42)
        lerobot_cfg = _build_lerobot_rtc_config(cfg)
        assert lerobot_cfg.debug is True
        assert lerobot_cfg.debug_maxlen == 42


# ---------------------------------------------------------------------------
# RtcAdapter — construction-only (Day 1 scope)
# ---------------------------------------------------------------------------


class _FakePolicy:
    """Minimal RtcCompatiblePolicy for construction-only tests."""

    def predict_action_chunk(self, **kwargs):
        import numpy as np
        return np.zeros((1, 50, 7), dtype=np.float32)


class TestRtcAdapterConstruction:
    def test_disabled_does_not_construct_processor(self):
        cfg = RtcAdapterConfig(enabled=False)
        adapter = RtcAdapter(
            policy=_FakePolicy(),
            action_buffer=ActionChunkBuffer(capacity=10),
            config=cfg,
        )
        assert adapter._processor is None
        assert adapter.config.enabled is False

    def test_disabled_does_not_require_lerobot(self, monkeypatch):
        """Even with lerobot unavailable, disabled config must construct."""
        monkeypatch.setattr("tether.runtime.rtc_adapter._RTC_AVAILABLE", False)
        cfg = RtcAdapterConfig(enabled=False)
        adapter = RtcAdapter(
            policy=_FakePolicy(),
            action_buffer=ActionChunkBuffer(capacity=10),
            config=cfg,
        )
        assert adapter._processor is None

    def test_enabled_requires_lerobot(self, monkeypatch):
        monkeypatch.setattr("tether.runtime.rtc_adapter._RTC_AVAILABLE", False)
        cfg = RtcAdapterConfig(enabled=True)
        with pytest.raises(ImportError, match="lerobot"):
            RtcAdapter(
                policy=_FakePolicy(),
                action_buffer=ActionChunkBuffer(capacity=10),
                config=cfg,
            )

    @pytest.mark.skipif(not _RTC_AVAILABLE, reason="lerobot not installed")
    def test_enabled_constructs_processor(self):
        cfg = RtcAdapterConfig(enabled=True)
        adapter = RtcAdapter(
            policy=_FakePolicy(),
            action_buffer=ActionChunkBuffer(capacity=10),
            config=cfg,
        )
        assert adapter._processor is not None

    def test_get_stats_reports_state(self):
        cfg = RtcAdapterConfig(enabled=False)
        adapter = RtcAdapter(
            policy=_FakePolicy(),
            action_buffer=ActionChunkBuffer(capacity=10),
            config=cfg,
        )
        stats = adapter.get_stats()
        assert stats["enabled"] is False
        assert stats["chunk_count"] == 0
        assert stats["latency"]["n"] == 0
        assert "rtc_available" in stats


# ---------------------------------------------------------------------------
# Reset semantics (the parts that work without body logic)
# ---------------------------------------------------------------------------


class TestRtcAdapterReset:
    def test_reset_clears_episode_state(self):
        cfg = RtcAdapterConfig(enabled=False)
        adapter = RtcAdapter(
            policy=_FakePolicy(),
            action_buffer=ActionChunkBuffer(capacity=10),
            config=cfg,
        )
        adapter._chunk_count = 5
        adapter._prev_chunk_left_over = "fake"
        adapter.latency.record(0.1)
        adapter.latency.record(0.2)
        adapter.reset(episode_id="ep-2")

        assert adapter._chunk_count == 0
        assert adapter._prev_chunk_left_over is None
        assert adapter._active_episode_id == "ep-2"
        # New LatencyTracker → window cleared
        assert adapter.latency.summary()["n"] == 0
        assert adapter.latency._seen == 0
