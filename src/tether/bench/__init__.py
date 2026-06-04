"""Benchmark methodology + report layer for `tether inspect bench`.

Lifted from EasyInference / ISB-1 (sibling project) — same core methodology
(warmup discard, p50/p95/p99 + tail, jitter, reproducibility envelope) with
VLA-aware semantics (per-chunk latency, diffusion-loop denoise steps,
flow-matching seed pinning).

Usage from CLI:
    from tether.bench import compute_stats, BenchReport, capture_environment
    stats = compute_stats(latencies_ms, warmup_n=20)
    env = capture_environment(export_dir=..., device=...)
    report = BenchReport(stats=stats, environment=env, ...)
    report.write_markdown("bench.md")
    report.write_json("bench.json")
"""

from tether.bench.methodology import (
    LatencyStats,
    compute_stats,
    confidence_interval_95,
)
from tether.bench.report import (
    BenchReport,
    BenchEnvironment,
    capture_environment,
)

__all__ = [
    "LatencyStats",
    "compute_stats",
    "confidence_interval_95",
    "BenchReport",
    "BenchEnvironment",
    "capture_environment",
]
