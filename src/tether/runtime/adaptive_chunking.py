"""Adaptive action chunk horizon selection.

The controller converts runtime signals into an execution horizon: how many
actions from the current chunk the robot can safely trust before replanning.
High uncertainty or low safety margin shortens the horizon; stable scenes under
latency pressure lengthen it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AdaptiveChunkConfig:
    enabled: bool = False
    min_horizon: int = 1
    base_horizon: int = 10
    max_horizon: int | None = None
    low_uncertainty: float = 0.20
    high_uncertainty: float = 0.65
    low_guard_margin: float = 0.05
    high_correction_magnitude: float = 0.20
    high_action_delta: float = 0.25
    high_latency_ms: float = 120.0

    def __post_init__(self) -> None:
        if self.min_horizon < 1:
            raise ValueError("min_horizon must be >= 1")
        if self.base_horizon < self.min_horizon:
            raise ValueError("base_horizon must be >= min_horizon")
        if self.max_horizon is not None and self.max_horizon < self.base_horizon:
            raise ValueError("max_horizon must be >= base_horizon when set")
        if self.low_uncertainty < 0 or self.high_uncertainty <= self.low_uncertainty:
            raise ValueError("uncertainty thresholds must satisfy 0 <= low < high")


@dataclass(frozen=True)
class AdaptiveChunkSignal:
    uncertainty: float | None = None
    guard_margin: float | None = None
    correction_magnitude: float | None = None
    action_delta: float | None = None
    latency_ms: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "uncertainty": self.uncertainty,
            "guard_margin": self.guard_margin,
            "correction_magnitude": self.correction_magnitude,
            "action_delta": self.action_delta,
            "latency_ms": self.latency_ms,
        }

    def has_values(self) -> bool:
        return any(value is not None for value in self.as_dict().values())


@dataclass(frozen=True)
class AdaptiveChunkDecision:
    horizon: int
    replan_threshold_ratio: float
    reason: str
    risk_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "replan_threshold_ratio": self.replan_threshold_ratio,
            "reason": self.reason,
            "risk_score": self.risk_score,
        }


def overlapping_action_delta(
    previous: np.ndarray | None,
    current: np.ndarray | None,
    *,
    window: int = 5,
) -> float | None:
    """L2 delta between the previous chunk tail and current chunk head."""
    if previous is None or current is None:
        return None
    prev = np.asarray(previous, dtype=np.float32)
    cur = np.asarray(current, dtype=np.float32)
    if prev.ndim != 2 or cur.ndim != 2:
        return None
    if prev.shape[1] != cur.shape[1]:
        return None
    k = min(window, prev.shape[0], cur.shape[0])
    if k <= 0:
        return None
    return float(np.linalg.norm(prev[-k:] - cur[:k]) / k)


class AdaptiveChunkController:
    """Selects an execution horizon from current runtime signals."""

    __slots__ = ("config",)

    def __init__(self, config: AdaptiveChunkConfig | None = None):
        self.config = config or AdaptiveChunkConfig()

    def decide(
        self,
        signal: AdaptiveChunkSignal | None,
        *,
        capacity: int,
    ) -> AdaptiveChunkDecision:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")

        cfg = self.config
        max_horizon = min(capacity, cfg.max_horizon or capacity)
        base_horizon = min(max(cfg.base_horizon, cfg.min_horizon), max_horizon)
        min_horizon = min(cfg.min_horizon, max_horizon)
        if not cfg.enabled:
            return self._decision(base_horizon, capacity, "disabled", 0.0)

        signal = signal or AdaptiveChunkSignal()
        risk, reason = self._risk(signal)

        if risk >= 0.66:
            return self._decision(min_horizon, capacity, reason, risk)

        latency_high = (
            signal.latency_ms is not None and signal.latency_ms >= cfg.high_latency_ms
        )
        if risk <= 0.25 and latency_high:
            return self._decision(max_horizon, capacity, "stable_high_latency", risk)
        if risk <= 0.25:
            return self._decision(base_horizon, capacity, "stable", risk)

        span = max_horizon - min_horizon
        horizon = max_horizon - int(round(risk * span))
        horizon = min(max(horizon, min_horizon), max_horizon)
        return self._decision(horizon, capacity, reason, risk)

    def _risk(self, signal: AdaptiveChunkSignal) -> tuple[float, str]:
        cfg = self.config
        risks: list[tuple[float, str]] = []

        if signal.uncertainty is not None:
            risks.append((
                _scale(signal.uncertainty, cfg.low_uncertainty, cfg.high_uncertainty),
                "uncertainty",
            ))
        if signal.guard_margin is not None:
            guard_risk = 1.0 - _scale(signal.guard_margin, 0.0, cfg.low_guard_margin)
            risks.append((guard_risk, "guard_margin"))
        if signal.correction_magnitude is not None:
            risks.append((
                _scale(signal.correction_magnitude, 0.0, cfg.high_correction_magnitude),
                "correction",
            ))
        if signal.action_delta is not None:
            risks.append((
                _scale(signal.action_delta, 0.0, cfg.high_action_delta),
                "action_delta",
            ))

        if not risks:
            return 0.0, "no_signal"
        risk, reason = max(risks, key=lambda item: item[0])
        return float(min(max(risk, 0.0), 1.0)), reason

    @staticmethod
    def _decision(
        horizon: int,
        capacity: int,
        reason: str,
        risk: float,
    ) -> AdaptiveChunkDecision:
        threshold = 1.0 - (horizon / capacity)
        threshold = min(max(threshold, 0.0), 1.0)
        return AdaptiveChunkDecision(
            horizon=int(horizon),
            replan_threshold_ratio=float(threshold),
            reason=reason,
            risk_score=float(risk),
        )


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0 if value >= high else 0.0
    return min(max((float(value) - low) / (high - low), 0.0), 1.0)


__all__ = [
    "AdaptiveChunkConfig",
    "AdaptiveChunkController",
    "AdaptiveChunkDecision",
    "AdaptiveChunkSignal",
    "overlapping_action_delta",
]
