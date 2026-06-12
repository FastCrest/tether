from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tether.runtime.adaptive_calibration import (
    iter_adaptive_records,
    recommend_adaptive_chunk_thresholds,
    summarize_adaptive_records,
)


def _write_jsonl(path: Path, records: list[dict], *, gzip_file: bool = False) -> None:
    opener = gzip.open if gzip_file else open
    with opener(path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _record(
    *,
    latency_ms: float,
    guard_margin: float,
    correction_magnitude: float,
    uncertainty: float,
    action_delta: float,
    horizon: int,
    reason: str,
) -> dict:
    return {
        "kind": "request",
        "latency": {"total_ms": latency_ms},
        "rtc": {
            "adaptive_signal": {
                "guard_margin": guard_margin,
                "correction_magnitude": correction_magnitude,
                "uncertainty": uncertainty,
            },
            "adaptive_chunking": {
                "horizon": horizon,
                "reason": reason,
                "risk_score": 0.5,
                "replan_threshold_ratio": 0.4,
            },
            "last_action_delta": action_delta,
        },
    }


def test_iter_adaptive_records_reads_plain_and_gzip_jsonl(tmp_path):
    records = [
        {"kind": "header"},
        _record(
            latency_ms=50,
            guard_margin=0.02,
            correction_magnitude=0.1,
            uncertainty=0.2,
            action_delta=0.03,
            horizon=8,
            reason="stable",
        ),
        {"kind": "request", "latency": {"total_ms": 60}},
    ]
    plain = tmp_path / "trace.jsonl"
    gz = tmp_path / "trace.jsonl.gz"
    _write_jsonl(plain, records)
    _write_jsonl(gz, records, gzip_file=True)

    plain_records = list(iter_adaptive_records([plain]))
    gz_records = list(iter_adaptive_records([gz]))

    assert len(plain_records) == 1
    assert plain_records == gz_records
    assert plain_records[0]["rtc"]["adaptive_chunking"]["reason"] == "stable"


def test_summarize_adaptive_records_counts_reasons_and_percentiles():
    records = [
        _record(
            latency_ms=50,
            guard_margin=0.02,
            correction_magnitude=0.1,
            uncertainty=0.2,
            action_delta=0.03,
            horizon=8,
            reason="stable",
        ),
        _record(
            latency_ms=100,
            guard_margin=0.04,
            correction_magnitude=0.2,
            uncertainty=0.5,
            action_delta=0.08,
            horizon=5,
            reason="correction",
        ),
        _record(
            latency_ms=200,
            guard_margin=0.08,
            correction_magnitude=0.4,
            uncertainty=0.9,
            action_delta=0.16,
            horizon=2,
            reason="correction",
        ),
    ]

    summary = summarize_adaptive_records(records)

    assert summary["sample_count"] == 3
    assert summary["decision_count"] == 3
    assert summary["reasons"] == {"correction": 2, "stable": 1}
    assert summary["observed"]["latency_ms"]["p50"] == pytest.approx(100)
    assert summary["observed"]["guard_margin"]["p10"] == pytest.approx(0.024)


def test_recommend_adaptive_chunk_thresholds_uses_recorded_distribution():
    records = [
        _record(
            latency_ms=50,
            guard_margin=0.02,
            correction_magnitude=0.1,
            uncertainty=0.2,
            action_delta=0.03,
            horizon=8,
            reason="stable",
        ),
        _record(
            latency_ms=100,
            guard_margin=0.04,
            correction_magnitude=0.2,
            uncertainty=0.5,
            action_delta=0.08,
            horizon=5,
            reason="correction",
        ),
        _record(
            latency_ms=200,
            guard_margin=0.08,
            correction_magnitude=0.4,
            uncertainty=0.9,
            action_delta=0.16,
            horizon=2,
            reason="correction",
        ),
    ]

    recommendation = recommend_adaptive_chunk_thresholds(records)

    cfg = recommendation["recommended_config"]
    assert cfg["low_guard_margin"] == pytest.approx(0.024)
    assert cfg["high_correction_magnitude"] == pytest.approx(0.36)
    assert cfg["high_uncertainty"] == pytest.approx(0.82)
    assert cfg["high_action_delta"] == pytest.approx(0.144)
    assert cfg["high_latency_ms"] == pytest.approx(150)
    assert recommendation["defaults_used"] == []


def test_recommend_adaptive_chunk_thresholds_falls_back_to_defaults():
    recommendation = recommend_adaptive_chunk_thresholds([])

    cfg = recommendation["recommended_config"]
    assert cfg["low_guard_margin"] == pytest.approx(0.05)
    assert cfg["high_correction_magnitude"] == pytest.approx(0.2)
    assert cfg["high_uncertainty"] == pytest.approx(0.65)
    assert cfg["high_action_delta"] == pytest.approx(0.25)
    assert cfg["high_latency_ms"] == pytest.approx(120)
    assert set(recommendation["defaults_used"]) == {
        "action_delta",
        "correction_magnitude",
        "guard_margin",
        "latency_ms",
        "uncertainty",
    }
