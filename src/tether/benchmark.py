"""Benchmarking utilities for exported VLA models."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    stage: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    iterations: int
    hz: float

    def to_dict(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(self).items()}


def measure_latency(fn, n_warmup: int = 10, n_iterations: int = 100) -> BenchmarkResult:
    """Measure function execution latency."""
    import torch

    # Warmup
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    mean = sum(latencies) / len(latencies)

    return BenchmarkResult(
        stage="unknown",
        mean_ms=mean,
        p50_ms=latencies[len(latencies) // 2],
        p95_ms=latencies[int(len(latencies) * 0.95)],
        p99_ms=latencies[int(len(latencies) * 0.99)],
        min_ms=latencies[0],
        max_ms=latencies[-1],
        iterations=n_iterations,
        hz=1000.0 / mean if mean > 0 else 0,
    )


def save_benchmark(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save benchmark results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "results": [r.to_dict() for r in results],
        "total_mean_ms": sum(r.mean_ms for r in results),
        "total_hz": 1000.0 / sum(r.mean_ms for r in results) if results else 0,
    }
    output_path.write_text(json.dumps(data, indent=2))
    logger.info("Benchmark saved: %s", output_path)
