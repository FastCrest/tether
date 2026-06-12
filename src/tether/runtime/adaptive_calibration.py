"""Offline calibration helpers for adaptive RTC action chunking.

Reads Tether JSONL trace records and recommends AdaptiveChunkConfig thresholds
from observed guard, A2C2, uncertainty, latency, and action-delta signals.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tether.runtime.adaptive_chunking import AdaptiveChunkConfig


_METRICS = (
    "guard_margin",
    "correction_magnitude",
    "uncertainty",
    "action_delta",
    "latency_ms",
    "horizon",
    "risk_score",
)


@dataclass(frozen=True)
class AdaptiveCalibrationRecommendation:
    """Serializable AAC threshold recommendation."""

    sample_count: int
    decision_count: int
    reasons: dict[str, int]
    observed: dict[str, dict[str, float]]
    recommended_config: dict[str, float]
    defaults_used: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_adaptive_records(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    """Yield request records that contain an ``rtc`` telemetry block."""
    for path_like in paths:
        path = Path(path_like)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_no}: invalid JSONL record: {exc.msg}"
                    ) from exc
                if record.get("kind") != "request":
                    continue
                if isinstance(record.get("rtc"), dict):
                    yield record


def summarize_adaptive_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize observed AAC telemetry from request records."""
    values: dict[str, list[float]] = {metric: [] for metric in _METRICS}
    reasons: Counter[str] = Counter()
    sample_count = 0
    decision_count = 0

    for record in records:
        rtc = record.get("rtc")
        if not isinstance(rtc, Mapping):
            continue
        sample_count += 1

        signal = rtc.get("adaptive_signal")
        if isinstance(signal, Mapping):
            _append_float(values["guard_margin"], signal.get("guard_margin"))
            _append_float(
                values["correction_magnitude"],
                signal.get("correction_magnitude"),
            )
            _append_float(values["uncertainty"], signal.get("uncertainty"))
            _append_float(values["action_delta"], signal.get("action_delta"))

        _append_float(values["action_delta"], rtc.get("last_action_delta"))

        decision = rtc.get("adaptive_chunking")
        if isinstance(decision, Mapping):
            decision_count += 1
            reason = str(decision.get("reason") or "unknown")
            reasons[reason] += 1
            _append_float(values["horizon"], decision.get("horizon"))
            _append_float(values["risk_score"], decision.get("risk_score"))

        latency = record.get("latency")
        if isinstance(latency, Mapping):
            _append_float(values["latency_ms"], latency.get("total_ms"))

    observed = {
        name: _describe(values_for_metric)
        for name, values_for_metric in values.items()
        if values_for_metric
    }
    return {
        "sample_count": sample_count,
        "decision_count": decision_count,
        "reasons": dict(sorted(reasons.items())),
        "observed": observed,
    }


def recommend_adaptive_chunk_config(
    records: Iterable[Mapping[str, Any]],
    *,
    defaults: AdaptiveChunkConfig | None = None,
) -> AdaptiveCalibrationRecommendation:
    """Recommend AAC thresholds from recorded telemetry.

    The recommendation is deliberately conservative:
    - low_guard_margin uses p10 of observed margins, so low-margin pilots shorten
      horizons earlier.
    - high correction, uncertainty, and action-delta thresholds use p90.
    - high_latency_ms uses p75 latency, so stable high-latency scenes can stretch
      horizons before the p95 tail.
    """
    defaults = defaults or AdaptiveChunkConfig()
    summary = summarize_adaptive_records(records)
    observed: dict[str, dict[str, float]] = summary["observed"]
    defaults_used: list[str] = []

    recommended = {
        "low_guard_margin": _recommended_percentile(
            observed,
            "guard_margin",
            "p10",
            defaults.low_guard_margin,
            defaults_used,
        ),
        "high_correction_magnitude": _recommended_percentile(
            observed,
            "correction_magnitude",
            "p90",
            defaults.high_correction_magnitude,
            defaults_used,
        ),
        "high_uncertainty": _recommended_percentile(
            observed,
            "uncertainty",
            "p90",
            defaults.high_uncertainty,
            defaults_used,
        ),
        "high_action_delta": _recommended_percentile(
            observed,
            "action_delta",
            "p90",
            defaults.high_action_delta,
            defaults_used,
        ),
        "high_latency_ms": _recommended_percentile(
            observed,
            "latency_ms",
            "p75",
            defaults.high_latency_ms,
            defaults_used,
        ),
    }
    recommended["high_uncertainty"] = max(
        recommended["high_uncertainty"],
        defaults.low_uncertainty + 1e-6,
    )

    return AdaptiveCalibrationRecommendation(
        sample_count=int(summary["sample_count"]),
        decision_count=int(summary["decision_count"]),
        reasons=dict(summary["reasons"]),
        observed=observed,
        recommended_config={
            key: round(float(value), 6)
            for key, value in recommended.items()
        },
        defaults_used=sorted(defaults_used),
    )


def recommend_adaptive_chunk_thresholds(
    records: Iterable[Mapping[str, Any]],
    *,
    defaults: AdaptiveChunkConfig | None = None,
) -> dict[str, Any]:
    """Return a plain dict recommendation for CLIs and JSON reports."""
    return recommend_adaptive_chunk_config(records, defaults=defaults).as_dict()


def _append_float(out: list[float], value: Any) -> None:
    cleaned = _clean_float(value)
    if cleaned is not None:
        out.append(cleaned)


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _describe(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "count": float(len(ordered)),
        "min": ordered[0],
        "p10": _percentile(ordered, 10),
        "p50": _percentile(ordered, 50),
        "p75": _percentile(ordered, 75),
        "p90": _percentile(ordered, 90),
        "p95": _percentile(ordered, 95),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _recommended_percentile(
    observed: Mapping[str, Mapping[str, float]],
    metric: str,
    percentile_name: str,
    default: float,
    defaults_used: list[str],
) -> float:
    stats = observed.get(metric)
    if not stats or percentile_name not in stats:
        defaults_used.append(metric)
        return float(default)
    value = _clean_float(stats[percentile_name])
    if value is None:
        defaults_used.append(metric)
        return float(default)
    return max(value, 0.0)


__all__ = [
    "AdaptiveCalibrationRecommendation",
    "iter_adaptive_records",
    "recommend_adaptive_chunk_config",
    "recommend_adaptive_chunk_thresholds",
    "summarize_adaptive_records",
]
