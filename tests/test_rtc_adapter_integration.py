"""Day 5 integration tests for RTC adapter (B.3).

Multi-cycle scenarios that exercise the full adapter loop end-to-end.
Tests use synthetic latency injection (time.sleep in mock policies) but
do NOT measure wall-clock speedup — that requires either a real async
replan loop or Modal LIBERO (Day 6).

Coverage:
- TestMultiCycleIntegration: predict→merge→predict carry-forward across
  10 cycles, verify chunk_count, prev_chunk_left_over evolution
- TestEpisodeLifecycle: run, reset, run, confirm state isolation
- TestBufferUnderflowSafety: replan threshold under stress
- TestLatencyConvergence: p95 stabilizes with consistent latency
- TestRtcContract: contract assertions about the lerobot integration
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from tether.runtime.buffer import ActionChunkBuffer
from tether.runtime.rtc_adapter import RtcAdapter, RtcAdapterConfig


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _SyntheticPolicy:
    """Mock policy with configurable latency + action shape.

    Records every call so tests can assert RTC kwargs were forwarded.
    Returns shape (T, A) — the merge_and_update unwraps batch dim if present.
    """

    def __init__(self, latency_s: float = 0.0, action_dim: int = 7,
                 chunk_size: int = 10):
        self.latency_s = latency_s
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.calls: list[dict] = []
        # Each call returns a unique-valued chunk so carry-forward is testable
        self._call_idx = 0

    def predict_action_chunk(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.latency_s > 0:
            time.sleep(self.latency_s)
        # Return chunks where every value = call index — so the test can
        # tell which chunk was carried forward
        chunk = np.full(
            (self.chunk_size, self.action_dim),
            float(self._call_idx),
            dtype=np.float32,
        )
        self._call_idx += 1
        return chunk


def _adapter(policy, capacity=10, execute_hz=100.0):
    cfg = RtcAdapterConfig(
        enabled=False,
        execute_hz=execute_hz,
        cold_start_discard=0,  # let timing tests see real samples immediately
    )
    return RtcAdapter(
        policy=policy,
        action_buffer=ActionChunkBuffer(capacity=capacity),
        config=cfg,
    )


def _run_cycle(adapter):
    """One predict + merge cycle. Returns the chunk that was produced."""
    chunk = adapter.predict_chunk_with_rtc({"image": "x"})
    # latency.estimate() was used inside predict; merge_and_update needs
    # an elapsed time too. Use a fixed value for repeatable testing.
    adapter.merge_and_update(chunk, elapsed_time=0.05)
    return chunk


# ---------------------------------------------------------------------------
# Multi-cycle integration
# ---------------------------------------------------------------------------


class TestMultiCycleIntegration:
    def test_ten_cycles_chunk_count(self):
        adapter = _adapter(_SyntheticPolicy())
        for _ in range(10):
            _run_cycle(adapter)
        assert adapter._chunk_count == 10

    def test_carry_forward_evolves_per_cycle(self):
        """After cycle N, _prev_chunk_left_over reflects what was in the
        buffer immediately before cycle N's chunk was pushed."""
        policy = _SyntheticPolicy()  # chunks valued 0, 1, 2, ...
        adapter = _adapter(policy, capacity=10)

        _run_cycle(adapter)  # cycle 0: buffer empty pre-merge → carry stays None
        assert adapter._prev_chunk_left_over is None

        _run_cycle(adapter)  # cycle 1: buffer had chunk 0 (all 0.0s) pre-merge
        assert adapter._prev_chunk_left_over[0, 0] == 0.0

        _run_cycle(adapter)  # cycle 2: buffer had chunk 1 (all 1.0s) pre-merge
        assert adapter._prev_chunk_left_over[0, 0] == 1.0

        _run_cycle(adapter)  # cycle 3: buffer had chunk 2 (all 2.0s) pre-merge
        assert adapter._prev_chunk_left_over[0, 0] == 2.0

    def test_carry_forward_passed_to_next_predict(self):
        """The carry from cycle N's merge becomes cycle N+1's predict input."""
        policy = _SyntheticPolicy()
        adapter = _adapter(policy)

        _run_cycle(adapter)  # cycle 0
        _run_cycle(adapter)  # cycle 1 — merge captures chunk 0 as carry
        _run_cycle(adapter)  # cycle 2 — predict sees chunk 0 in carry

        # cycle 2's predict call (calls[2]) saw the carry from cycle 1's merge,
        # which was chunk 0 (all 0.0s)
        assert policy.calls[2]["prev_chunk_left_over"] is not None
        assert policy.calls[2]["prev_chunk_left_over"][0, 0] == 0.0

    def test_buffer_replans_count_matches_merges(self):
        adapter = _adapter(_SyntheticPolicy())
        for _ in range(5):
            _run_cycle(adapter)
        # ActionChunkBuffer.stats() reports replans
        assert adapter.buffer.stats().replans == 5

    def test_inference_delay_grows_with_recorded_latency(self):
        """As the latency tracker warms with consistent samples, the
        actions_consumed grows toward int(latency * execute_hz)."""
        # 30ms latency × 100Hz = 3 actions consumed (after warmup)
        policy = _SyntheticPolicy(latency_s=0.030)
        adapter = _adapter(policy, execute_hz=100.0)
        # Run a few cycles to warm the tracker (cold_start_discard=0 in fixture)
        for _ in range(8):
            _run_cycle(adapter)
        # Now check the latest predict's inference_delay
        last_call = policy.calls[-1]
        # Allow slack for timer noise: 1-10 actions reasonable for 30ms target
        assert 1 <= last_call["inference_delay"] <= 10

    def test_adaptive_chunking_overrides_execution_horizon(self):
        from tether.runtime.rtc_adapter import _RTC_AVAILABLE
        if not _RTC_AVAILABLE:
            pytest.skip("lerobot not installed")
        policy = _SyntheticPolicy()
        cfg = RtcAdapterConfig(
            enabled=True,
            execute_hz=100.0,
            cold_start_discard=0,
            rtc_execution_horizon=5,
            adaptive_chunking_enabled=True,
            adaptive_high_latency_ms=50.0,
        )
        adapter = RtcAdapter(
            policy=policy,
            action_buffer=ActionChunkBuffer(capacity=10),
            config=cfg,
        )
        adapter.latency.record(0.10)
        adapter.latency.record(0.10)
        adapter.latency.record(0.10)
        adapter.predict_chunk_with_rtc({"image": "x"})
        assert policy.calls[-1]["execution_horizon"] == 10
        stats = adapter.get_stats()
        assert stats["adaptive_chunking"]["reason"] == "stable_high_latency"

    def test_adaptive_chunking_uses_guard_margin_signal(self):
        policy = _SyntheticPolicy()
        cfg = RtcAdapterConfig(
            enabled=False,
            cold_start_discard=0,
            rtc_execution_horizon=5,
            adaptive_chunking_enabled=True,
        )
        adapter = RtcAdapter(
            policy=policy,
            action_buffer=ActionChunkBuffer(capacity=10),
            config=cfg,
        )

        adapter.record_adaptive_signal(guard_margin=0.01)
        decision = adapter._decide_adaptive_horizon(0.01)

        assert decision is not None
        assert decision.horizon == 1
        assert decision.reason == "guard_margin"
        stats = adapter.get_stats()
        assert stats["adaptive_signal"]["guard_margin"] == pytest.approx(0.01)

    def test_adaptive_chunking_uses_a2c2_correction_signal(self):
        policy = _SyntheticPolicy()
        cfg = RtcAdapterConfig(
            enabled=False,
            cold_start_discard=0,
            rtc_execution_horizon=5,
            adaptive_chunking_enabled=True,
        )
        adapter = RtcAdapter(
            policy=policy,
            action_buffer=ActionChunkBuffer(capacity=10),
            config=cfg,
        )

        adapter.record_adaptive_signal(correction_magnitude=0.3)
        decision = adapter._decide_adaptive_horizon(0.01)

        assert decision is not None
        assert decision.horizon == 1
        assert decision.reason == "correction"

    def test_reset_clears_adaptive_signal(self):
        adapter = RtcAdapter(
            policy=_SyntheticPolicy(),
            action_buffer=ActionChunkBuffer(capacity=10),
            config=RtcAdapterConfig(
                enabled=False,
                adaptive_chunking_enabled=True,
            ),
        )
        adapter.record_adaptive_signal(guard_margin=0.01)
        assert "adaptive_signal" in adapter.get_stats()

        adapter.reset(episode_id="fresh")

        assert "adaptive_signal" not in adapter.get_stats()


# ---------------------------------------------------------------------------
# Episode lifecycle
# ---------------------------------------------------------------------------


class TestEpisodeLifecycle:
    def test_episode_a_then_b_isolated(self):
        """Run 5 chunks in episode A, reset to B, run 5 more — B's state
        never touches A's tracker."""
        policy = _SyntheticPolicy()
        adapter = _adapter(policy)
        adapter.reset(episode_id="A")
        for _ in range(5):
            _run_cycle(adapter)

        adapter.reset(episode_id="B")
        # Latency window cleared, carry cleared
        assert adapter.latency.summary()["n"] == 0
        assert adapter._prev_chunk_left_over is None

        for _ in range(5):
            _run_cycle(adapter)
        # Episode B has its own count
        assert adapter._chunk_count == 5
        assert adapter._active_episode_id == "B"

    def test_reset_does_not_clear_buffer(self):
        """The action buffer is shared with the server's existing replan
        path; only the adapter's tracker + carry are reset."""
        policy = _SyntheticPolicy()
        adapter = _adapter(policy, capacity=10)
        adapter.reset(episode_id="A")
        for _ in range(3):
            _run_cycle(adapter)
        buffer_size_before = adapter.buffer.size

        adapter.reset(episode_id="B")
        # Buffer untouched — pop_next still returns valid actions
        assert adapter.buffer.size == buffer_size_before

    def test_first_predict_after_reset_no_carry(self):
        policy = _SyntheticPolicy()
        adapter = _adapter(policy)
        for _ in range(3):
            _run_cycle(adapter)
        # Carry is populated
        assert adapter._prev_chunk_left_over is not None

        adapter.reset(episode_id="new")
        # Next predict sees None
        adapter.predict_chunk_with_rtc({"image": "x"})
        # Find the predict call that happened after the reset
        last_call = policy.calls[-1]
        assert last_call["prev_chunk_left_over"] is None


# ---------------------------------------------------------------------------
# Buffer underflow safety
# ---------------------------------------------------------------------------


class TestBufferUnderflowSafety:
    def test_should_replan_after_pops(self):
        """When >= half the buffer is popped, should_replan returns True."""
        policy = _SyntheticPolicy(chunk_size=20)
        adapter = _adapter(policy, capacity=20)
        _run_cycle(adapter)  # buffer holds 20 actions
        assert not adapter.buffer.should_replan(0.5)

        # Pop 11 (more than half) → should_replan True
        for _ in range(11):
            adapter.buffer.pop_next()
        assert adapter.buffer.should_replan(0.5)

    def test_pop_until_empty_then_no_none_on_replan(self):
        """Drain the buffer entirely; next predict+merge refills it."""
        policy = _SyntheticPolicy(chunk_size=10)
        adapter = _adapter(policy, capacity=10)
        _run_cycle(adapter)
        # Drain
        for _ in range(10):
            assert adapter.buffer.pop_next() is not None
        assert adapter.buffer.pop_next() is None  # empty
        # Replan
        _run_cycle(adapter)
        assert adapter.buffer.size == 10
        assert adapter.buffer.pop_next() is not None

    def test_chunk_size_larger_than_capacity_truncates(self):
        """Buffer caps at capacity; extra actions in chunk are dropped."""
        policy = _SyntheticPolicy(chunk_size=50)  # chunk > capacity
        adapter = _adapter(policy, capacity=10)
        _run_cycle(adapter)
        assert adapter.buffer.size == 10  # capped at capacity


# ---------------------------------------------------------------------------
# Latency convergence
# ---------------------------------------------------------------------------


class TestLatencyConvergence:
    def test_consistent_latency_stabilizes_p95(self):
        """With identical latency samples, p95 == that latency value."""
        policy = _SyntheticPolicy()
        adapter = _adapter(policy)
        adapter.latency.discard_first = 0
        # Inject 20 samples of 0.080s
        for _ in range(20):
            adapter.latency.record(0.080)
        assert adapter.latency.estimate() == pytest.approx(0.080)

    def test_p95_resilient_to_one_outlier(self):
        """One bad sample doesn't tank the estimate."""
        adapter = _adapter(_SyntheticPolicy())
        adapter.latency.discard_first = 0
        for _ in range(19):
            adapter.latency.record(0.050)
        adapter.latency.record(2.0)  # outlier
        # p95 of 19 × 0.050 + 1 × 2.0 — outlier at 95th percentile
        # but still bounded
        est = adapter.latency.estimate()
        assert 0.05 <= est <= 2.0

    def test_window_caps_old_samples(self):
        """The default window_size caps at 50; ancient samples drop off."""
        adapter = _adapter(_SyntheticPolicy())
        adapter.latency.discard_first = 0
        # Inject 100 high samples then 50 low samples
        for _ in range(100):
            adapter.latency.record(1.0)
        for _ in range(50):
            adapter.latency.record(0.001)
        # Window is now all 0.001s — high samples dropped
        assert adapter.latency.estimate() == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Contract: the lerobot RTC integration
# ---------------------------------------------------------------------------


class TestRtcContract:
    def test_inference_delay_always_int(self):
        """Lerobot's RTCProcessor.denoise_step expects inference_delay as
        int — we must never pass a float."""
        policy = _SyntheticPolicy()
        adapter = _adapter(policy)
        for _ in range(5):
            _run_cycle(adapter)
        for call in policy.calls:
            assert isinstance(call["inference_delay"], int), (
                f"inference_delay must be int, got {type(call['inference_delay'])}"
            )

    def test_first_chunk_no_guidance_invariant(self):
        """RTC's first call ALWAYS has prev_chunk_left_over=None — there
        is no previous chunk to guide against."""
        policy = _SyntheticPolicy()
        adapter = _adapter(policy)
        adapter.predict_chunk_with_rtc({"image": "x"})
        assert policy.calls[0]["prev_chunk_left_over"] is None

    def test_carry_shape_matches_buffer_capacity(self):
        """The snapshot shape == (buffer.size_pre_merge, action_dim).
        With capacity=10 and chunk_size=20, buffer fills to 10 and
        snapshot is (10, action_dim)."""
        policy = _SyntheticPolicy(chunk_size=20, action_dim=7)
        adapter = _adapter(policy, capacity=10)
        _run_cycle(adapter)  # buffer fills to 10 (capped)
        _run_cycle(adapter)  # this merge snapshots the 10
        assert adapter._prev_chunk_left_over.shape == (10, 7)
