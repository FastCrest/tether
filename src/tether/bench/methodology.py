"""ISB-1 measurement methodology applied to per-chunk VLA latencies.

LatencyStats captures the distribution. compute_stats(latencies, warmup_n)
discards the first warmup_n samples, computes percentiles + jitter + CI.

Why a separate module: the existing benchmark_cmd inlines the math sloppily
(no p99.9, no jitter, no CI, no warmup discipline beyond a fixed loop).
Centralizing here means the same numbers apply across `tether inspect bench`,
the future `tether inspect bench --report-json` for CI, and any Modal
batch-bench script that ingests the same primitives.

Pure stdlib + math; no numpy required (the math is small).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class LatencyStats:
    """Distribution summary for a list of per-chunk latencies (in ms).

    `n` is the number of samples used (post-warmup discard). `warmup_discarded`
    is the number of samples dropped from the head of the input.
    """

    n: int
    warmup_discarded: int
    min_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    p99_9_ms: float
    max_ms: float
    std_ms: float
    jitter: float  # std / mean — unitless coefficient of variation
    ci95_low_ms: float  # 95% CI on the mean
    ci95_high_ms: float
    hz_mean: float  # 1000 / mean_ms (sample-rate equivalent)

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(sorted_xs: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 100])."""
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    rank = (q / 100.0) * (len(sorted_xs) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_xs[lo]
    frac = rank - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def confidence_interval_95(samples: list[float]) -> tuple[float, float]:
    """Return (low, high) of the 95% CI on the sample mean.

    Uses the normal approximation (CI = mean ± 1.96 * std / sqrt(n)).
    For n < 30 this slightly overstates confidence; tether bench typically
    runs n >= 100 so the approximation is fine. Returns (mean, mean) on
    n <= 1 (no spread to estimate).
    """
    n = len(samples)
    if n <= 1:
        m = samples[0] if samples else float("nan")
        return m, m
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)
    std = math.sqrt(var)
    half = 1.96 * std / math.sqrt(n)
    return mean - half, mean + half


def compute_stats(latencies_ms: list[float], warmup_n: int = 0) -> LatencyStats:
    """Compute the canonical bench statistics.

    Args:
        latencies_ms: per-chunk latencies in milliseconds, in measurement order.
                      WARMUP samples should be the FIRST `warmup_n` entries.
        warmup_n: number of warmup samples to discard from the head.

    Returns:
        LatencyStats with all percentiles + CI + jitter computed on the
        post-warmup tail.

    Raises:
        ValueError: if warmup_n >= len(latencies_ms) (nothing to measure).
    """
    if warmup_n < 0:
        raise ValueError(f"warmup_n must be >= 0, got {warmup_n}")
    if warmup_n >= len(latencies_ms):
        raise ValueError(
            f"warmup_n ({warmup_n}) >= len(latencies_ms) ({len(latencies_ms)}); "
            "nothing left to measure after warmup discard"
        )
    samples = list(latencies_ms[warmup_n:])
    n = len(samples)
    sorted_samples = sorted(samples)
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / max(n - 1, 1)
    std = math.sqrt(var)
    jitter = (std / mean) if mean > 0 else float("nan")
    ci_low, ci_high = confidence_interval_95(samples)
    return LatencyStats(
        n=n,
        warmup_discarded=warmup_n,
        min_ms=sorted_samples[0],
        mean_ms=mean,
        p50_ms=_percentile(sorted_samples, 50),
        p95_ms=_percentile(sorted_samples, 95),
        p99_ms=_percentile(sorted_samples, 99),
        p99_9_ms=_percentile(sorted_samples, 99.9),
        max_ms=sorted_samples[-1],
        std_ms=std,
        jitter=jitter,
        ci95_low_ms=ci_low,
        ci95_high_ms=ci_high,
        hz_mean=(1000.0 / mean) if mean > 0 else float("nan"),
    )
