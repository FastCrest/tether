"""Tests for replay diff functions (B.2 Day 3).

Covers diff_actions, diff_latency, diff_cache, and their helpers
cosine_similarity + max_abs_diff. Pure functions, no I/O, fast.
"""
from __future__ import annotations

import math

import pytest

from tether.replay.cli import (
    cosine_similarity,
    diff_actions,
    diff_cache,
    diff_latency,
    max_abs_diff,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_length_mismatch_returns_zero(self):
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_empty_returns_zero(self):
        assert cosine_similarity([], []) == 0.0

    def test_zero_norm_returns_zero(self):
        """Degenerate input: one vector is all zeros."""
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestMaxAbsDiff:
    def test_identical(self):
        assert max_abs_diff([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_positive_delta(self):
        assert max_abs_diff([0.0, 0.0], [0.1, 0.5]) == pytest.approx(0.5)

    def test_negative_delta(self):
        assert max_abs_diff([0.0, 0.0], [-0.3, 0.1]) == pytest.approx(0.3)

    def test_length_mismatch_uses_shorter(self):
        """Only compare over the shorter prefix."""
        assert max_abs_diff([1.0, 2.0], [1.0, 2.0, 999.0]) == 0.0

    def test_empty_returns_zero(self):
        assert max_abs_diff([], []) == 0.0
        assert max_abs_diff([], [1.0]) == 0.0


# ---------------------------------------------------------------------------
# diff_actions
# ---------------------------------------------------------------------------


class TestDiffActions:
    def test_identical_passes(self):
        a = [[0.1, 0.2], [0.3, 0.4]]
        d = diff_actions(a, a)
        assert d["passed"] is True
        assert d["cosine"] == pytest.approx(1.0)
        assert d["max_abs_diff"] == 0.0

    def test_large_diff_fails(self):
        a = [[0.1, 0.2]]
        b = [[100.0, 200.0]]
        d = diff_actions(a, b)
        assert d["passed"] is False

    def test_small_diff_under_threshold_passes(self):
        a = [[0.1, 0.2, 0.3]]
        b = [[0.1001, 0.2001, 0.3001]]  # delta 1e-4
        d = diff_actions(a, b)
        assert d["passed"] is True
        assert d["max_abs_diff"] < 1e-3

    def test_custom_thresholds(self):
        """Tighter thresholds can flip pass to fail. Use non-zero recorded
        actions so cosine is well-defined (zero-norm recorded → cosine=0)."""
        a = [[0.1, 0.2]]
        b = [[0.105, 0.205]]  # delta 5e-3
        # Default: max_abs 1e-3 → fails
        assert diff_actions(a, b)["passed"] is False
        # Loose max_abs 1e-2 → passes (cosine is already ≥0.999 on close vectors)
        assert diff_actions(a, b, threshold_max_abs=1e-2)["passed"] is True

    def test_empty_actions(self):
        """Empty action chunks → cosine 0 → fails on default threshold."""
        d = diff_actions([], [])
        # With thresholds cos≥0.999, passed should be False since cos is 0
        assert d["cosine"] == 0.0
        assert d["passed"] is False


# ---------------------------------------------------------------------------
# diff_latency
# ---------------------------------------------------------------------------


class TestDiffLatency:
    def test_same_total_passes(self):
        d = diff_latency({"total_ms": 100.0}, {"total_ms": 100.0})
        assert d["passed"] is True
        assert d["delta_ms"] == 0.0
        assert d["delta_pct"] == 0.0

    def test_under_threshold_passes(self):
        # 3% delta, default threshold is 5%
        d = diff_latency({"total_ms": 100.0}, {"total_ms": 103.0})
        assert d["passed"] is True
        assert d["delta_pct"] == pytest.approx(0.03)

    def test_over_threshold_fails(self):
        # 10% delta
        d = diff_latency({"total_ms": 100.0}, {"total_ms": 110.0})
        assert d["passed"] is False

    def test_faster_replay_symmetric(self):
        # -5.1% delta (replay faster than recorded)
        d = diff_latency({"total_ms": 100.0}, {"total_ms": 94.9})
        assert d["passed"] is False  # outside ±5%
        assert d["delta_pct"] == pytest.approx(-0.051)

    def test_custom_threshold(self):
        # 8% delta, with 10% threshold
        d = diff_latency(
            {"total_ms": 100.0}, {"total_ms": 108.0}, threshold_pct=0.10
        )
        assert d["passed"] is True

    def test_zero_recorded_skips_gating(self):
        """Can't compute relative delta when recorded is 0."""
        d = diff_latency({"total_ms": 0.0}, {"total_ms": 50.0})
        assert d["passed"] is True  # no gating
        assert d["delta_pct"] is None
        assert "note" in d

    def test_per_stage_reported(self):
        rec = {"total_ms": 100.0, "stages": {"vlm_prefix_ms": 80.0, "expert_denoise_ms": 18.0}}
        rep = {"total_ms": 102.0, "stages": {"vlm_prefix_ms": 82.0, "expert_denoise_ms": 18.0}}
        d = diff_latency(rec, rep)
        assert "stages" in d
        assert d["stages"]["vlm_prefix_ms"]["delta_ms"] == pytest.approx(2.0)
        assert d["stages"]["expert_denoise_ms"]["delta_ms"] == pytest.approx(0.0)

    def test_missing_stages_skipped(self):
        rec = {"total_ms": 100.0, "stages": {"vlm_prefix_ms": 80.0}}
        rep = {"total_ms": 102.0}  # no stages
        d = diff_latency(rec, rep)
        assert d["stages"] == {}


# ---------------------------------------------------------------------------
# diff_cache
# ---------------------------------------------------------------------------


class TestDiffCache:
    def test_hit_match(self):
        d = diff_cache({"status": "hit"}, {"status": "hit"})
        assert d["passed"] is True

    def test_miss_match(self):
        d = diff_cache({"status": "miss"}, {"status": "miss"})
        assert d["passed"] is True

    def test_hit_vs_miss_fails(self):
        d = diff_cache({"status": "hit"}, {"status": "miss"})
        assert d["passed"] is False

    def test_none_treated_as_na(self):
        d = diff_cache(None, None)
        assert d["passed"] is True
        assert d["recorded_status"] == "n/a"

    def test_none_vs_hit_fails(self):
        d = diff_cache(None, {"status": "hit"})
        assert d["passed"] is False

    def test_missing_status_key_treated_as_na(self):
        d = diff_cache({}, {})
        assert d["passed"] is True
        assert d["recorded_status"] == "n/a"
