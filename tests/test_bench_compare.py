"""Tests for src/tether/bench/compare.py — the 6-gate A/B ship-rule helper."""
from __future__ import annotations

import pytest

from tether.bench.compare import compare_reports, render_compare_markdown
from tether.bench.methodology import LatencyStats
from tether.bench.report import BenchEnvironment, BenchReport


def _stats(
    mean: float = 10.0,
    p99: float = 12.0,
    p99_9: float = 15.0,
    jitter: float = 0.05,
    ci_low: float | None = None,
    ci_high: float | None = None,
    n: int = 100,
) -> LatencyStats:
    if ci_low is None:
        ci_low = mean * 0.95
    if ci_high is None:
        ci_high = mean * 1.05
    std = mean * jitter
    return LatencyStats(
        n=n,
        warmup_discarded=0,
        min_ms=mean * 0.8,
        mean_ms=mean,
        p50_ms=mean,
        p95_ms=mean * 1.15,
        p99_ms=p99,
        p99_9_ms=p99_9,
        max_ms=p99_9 * 1.05,
        std_ms=std,
        jitter=jitter,
        ci95_low_ms=ci_low,
        ci95_high_ms=ci_high,
        hz_mean=1000.0 / mean,
    )


def _env() -> BenchEnvironment:
    return BenchEnvironment(
        timestamp_utc="2026-04-24T00:00:00Z",
        tether_version="0.1.0",
        git_sha="abcd1234",
        git_dirty=False,
        python_version="3.12",
        platform="Linux-x86_64",
        gpu_name="A10G",
        cuda_version="12.4",
        ort_version="1.20.1",
        onnx_files=[],
        seed=0,
        export_dir="/tmp",
        inference_mode="onnx_cuda",
        device="cuda",
    )


def _report(stats: LatencyStats, parity: dict | None = None) -> BenchReport:
    return BenchReport(stats=stats, environment=_env(), parity=parity)


# ---------------------------------------------------------------------------
# Verdict matrix
# ---------------------------------------------------------------------------

def test_identical_reports_yield_no_signal():
    s = _stats(mean=10.0)
    r = _report(s)
    result = compare_reports(r, r)
    assert result.verdict == "NO_SIGNAL"
    assert result.ci_overlap is True


def test_clean_improvement_yields_ship():
    off = _report(_stats(mean=10.0, ci_low=9.95, ci_high=10.05, p99=12.0, p99_9=15.0, jitter=0.05))
    on = _report(
        _stats(mean=8.0, ci_low=7.95, ci_high=8.05, p99=10.0, p99_9=12.5, jitter=0.05),
        parity={"cos": 0.99999},
    )
    result = compare_reports(off, on)
    assert result.verdict == "SHIP", f"got {result.verdict}: {[g.detail for g in result.gates]}"
    assert result.mean_delta_pct < -10  # 20% improvement


def test_insufficient_gain_below_floor():
    """Mean improves 4%, below the 5% floor; CIs disjoint so signal exists."""
    off = _report(_stats(mean=10.0, ci_low=9.95, ci_high=10.05, p99=12.0, p99_9=15.0))
    on = _report(_stats(mean=9.6, ci_low=9.55, ci_high=9.65, p99=11.6, p99_9=14.6))
    result = compare_reports(off, on)
    assert result.verdict == "INSUFFICIENT_GAIN"


def test_tail_regression_blocks_ship():
    """Mean improves 15% but p99 regresses 25%."""
    off = _report(_stats(mean=10.0, ci_low=9.5, ci_high=10.5, p99=12.0, p99_9=15.0))
    on = _report(_stats(mean=8.5, ci_low=8.0, ci_high=9.0, p99=15.0, p99_9=18.0))
    result = compare_reports(off, on)
    assert result.verdict == "TAIL_REGRESSION"


def test_p99_9_regression_blocks_ship():
    """Mean + p99 fine; p99.9 regresses 15%."""
    off = _report(_stats(mean=10.0, ci_low=9.5, ci_high=10.5, p99=12.0, p99_9=15.0))
    on = _report(_stats(mean=8.0, ci_low=7.7, ci_high=8.3, p99=12.0, p99_9=17.5))
    result = compare_reports(off, on)
    assert result.verdict == "TAIL_REGRESSION"


def test_jitter_regression_blocks_ship():
    """Mean improves but jitter rises 30% (>20% cap)."""
    off = _report(_stats(mean=10.0, ci_low=9.5, ci_high=10.5, jitter=0.05))
    on = _report(_stats(mean=8.0, ci_low=7.7, ci_high=8.3, jitter=0.07))
    result = compare_reports(off, on)
    assert result.verdict == "JITTER_REGRESSION"


def test_parity_failure_blocks_ship():
    """Mean improves cleanly but parity cos below floor."""
    off = _report(_stats(mean=10.0, ci_low=9.5, ci_high=10.5))
    on = _report(_stats(mean=8.0, ci_low=7.7, ci_high=8.3), parity={"cos": 0.999})
    result = compare_reports(off, on)
    assert result.verdict == "PARITY_FAILURE"


def test_no_parity_field_does_not_fail_parity_gate():
    """If report_on has no parity, the parity gate is simply not evaluated."""
    off = _report(_stats(mean=10.0, ci_low=9.5, ci_high=10.5))
    on = _report(_stats(mean=8.0, ci_low=7.7, ci_high=8.3))  # parity=None
    result = compare_reports(off, on)
    assert result.verdict == "SHIP"
    assert "parity_cos_floor" not in [g.name for g in result.gates]


# ---------------------------------------------------------------------------
# Threshold customization
# ---------------------------------------------------------------------------

def test_relaxed_mean_floor_passes_smaller_improvement():
    off = _report(_stats(mean=10.0, ci_low=9.95, ci_high=10.05, p99=12.0, p99_9=15.0))
    on = _report(_stats(mean=9.8, ci_low=9.75, ci_high=9.85, p99=11.8, p99_9=14.8))
    # 2% improvement; default 5% floor would fail
    default_result = compare_reports(off, on)
    assert default_result.verdict == "INSUFFICIENT_GAIN"
    # Relaxed 1% floor should pass
    relaxed_result = compare_reports(off, on, mean_floor_pct=1.0)
    assert relaxed_result.verdict == "SHIP"


def test_strict_p99_cap_catches_smaller_regression():
    # Use tight disjoint CIs so the ci_no_overlap gate passes
    off = _report(_stats(mean=10.0, ci_low=9.95, ci_high=10.05, p99=12.0, p99_9=15.0))
    # 7% mean improvement, 3% p99 regression; default 5% cap allows it
    on = _report(_stats(mean=9.3, ci_low=9.25, ci_high=9.35, p99=12.36, p99_9=15.0))
    default_result = compare_reports(off, on)
    assert default_result.verdict == "SHIP"
    # Strict 2% cap rejects it
    strict_result = compare_reports(off, on, p99_cap_pct=2.0)
    assert strict_result.verdict == "TAIL_REGRESSION"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def test_render_markdown_includes_verdict_and_gates():
    off = _report(_stats(mean=10.0, ci_low=9.5, ci_high=10.5))
    on = _report(_stats(mean=8.0, ci_low=7.7, ci_high=8.3))
    result = compare_reports(off, on)
    md = render_compare_markdown(result, label_off="eager", label_on="cuda-graphs")
    assert result.verdict in md
    assert "eager" in md
    assert "cuda-graphs" in md
    assert "ci_no_overlap" in md
    assert "Latency deltas" in md
    assert "Ship-worthy gates" in md


def test_to_dict_round_trips_for_json_serialization():
    off = _report(_stats(mean=10.0, ci_low=9.5, ci_high=10.5))
    on = _report(_stats(mean=8.0, ci_low=7.7, ci_high=8.3), parity={"cos": 0.99999})
    result = compare_reports(off, on)
    d = result.to_dict()
    assert d["verdict"] == "SHIP"
    assert d["schema_version"] == 1
    assert isinstance(d["gates"], list)
    assert d["gates"][0]["name"] == "ci_no_overlap"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_zero_baseline_jitter_does_not_crash():
    off = _report(_stats(mean=10.0, ci_low=9.5, ci_high=10.5, jitter=0.0))
    on = _report(_stats(mean=8.0, ci_low=7.7, ci_high=8.3, jitter=0.05))
    result = compare_reports(off, on)
    # jitter delta is NaN when baseline is 0; verdict should still be deterministic
    assert result.verdict in {"SHIP", "JITTER_REGRESSION", "INSUFFICIENT_GAIN", "NO_SIGNAL", "TAIL_REGRESSION"}


def test_full_failed_gates_list():
    off = _report(_stats(mean=10.0, ci_low=9.5, ci_high=10.5, p99=12.0, p99_9=15.0, jitter=0.05))
    on = _report(_stats(mean=12.0, ci_low=11.5, ci_high=12.5, p99=18.0, p99_9=22.0, jitter=0.10))
    result = compare_reports(off, on)
    # Mean got worse + p99 got much worse + jitter doubled
    assert result.passes_ship_gates is False
    failed = result.failed_gates
    assert "mean_improvement_floor" in failed  # negative improvement
    assert "p99_regression_cap" in failed
