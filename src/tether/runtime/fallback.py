"""Fallback-policy-chain primitive: ordered tiers with per-tier timeouts.

Robot never freezes. When a higher-tier policy (e.g. the main VLA) times out,
OOMs, or raises, the chain advances to the next tier (e.g. a 1-NFE distilled
student, then classical gravity-comp). Each fallback emits a Prometheus
metric so operators see degradation in real-time.

Phase 1 ships this primitive; wiring into `tether serve` (loading multiple
ONNX sessions + classical action) lands in Phase 1.5 once self-distilling-
serve matures the student tier.

Feature spec: features/01_serve/subfeatures/_dx_gaps/fallback-policy-chain/
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from tether.observability.prometheus import inc_fallback_invocation

logger = logging.getLogger(__name__)


# Predict function signature. Returns whatever the caller passes through
# (typically a dict of {actions, ...} or a numpy array).
PredictFn = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class FallbackTier:
    """One tier of a fallback chain.

    name: human label for metrics/logs (e.g. "main" / "student-1nfe" / "classical").
    predict: async callable that returns predictions or raises on failure.
    timeout_ms: per-tier timeout in milliseconds. None = no timeout (let it run).
    """

    name: str
    predict: PredictFn
    timeout_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FallbackTier.name must be non-empty")
        if self.timeout_ms is not None and self.timeout_ms <= 0:
            raise ValueError(
                f"FallbackTier.timeout_ms must be > 0 or None, got {self.timeout_ms}"
            )


@dataclass
class FallbackResult:
    """What happened when `FallbackChain.predict()` ran.

    value: whatever the succeeding tier returned (or None if all tiers failed).
    tier_used: name of the tier that succeeded (or None on total failure).
    attempts: list of (tier_name, outcome, elapsed_ms) — outcome is
        "success" | "timeout" | "exception".
    """

    value: Any
    tier_used: Optional[str]
    attempts: list[tuple[str, str, float]] = field(default_factory=list)


class AllTiersFailed(RuntimeError):
    """All fallback tiers timed out or raised."""

    def __init__(self, attempts: list[tuple[str, str, float]]):
        self.attempts = attempts
        summary = ", ".join(f"{n}({o})" for n, o, _ in attempts)
        super().__init__(f"all fallback tiers failed: {summary}")


class FallbackChain:
    """Ordered list of fallback tiers. `predict()` tries each in order; on
    timeout or exception, advances to the next. Emits
    `tether_fallback_invocations_total{target=<tier_name>}` on each
    fallback (NOT on the successful tier — that's the normal path).

    Raises `AllTiersFailed` when every tier fails.

    Usage:

        chain = FallbackChain(
            embodiment="franka",
            tiers=[
                FallbackTier(name="main", predict=main_predict, timeout_ms=200),
                FallbackTier(name="student", predict=student_predict, timeout_ms=30),
                FallbackTier(name="classical", predict=classical_predict, timeout_ms=5),
            ],
        )
        result = await chain.predict(image_b64=..., instruction=..., state=...)
        # result.value = the actions dict; result.tier_used = "main" (or "student" etc.)
    """

    __slots__ = ("_embodiment", "_tiers")

    def __init__(self, embodiment: str, tiers: list[FallbackTier]):
        if not tiers:
            raise ValueError("FallbackChain requires at least one tier")
        if not isinstance(embodiment, str) or not embodiment:
            raise ValueError("FallbackChain.embodiment must be non-empty string")
        self._embodiment = embodiment
        self._tiers = tuple(tiers)

    @property
    def embodiment(self) -> str:
        return self._embodiment

    @property
    def tiers(self) -> tuple[FallbackTier, ...]:
        return self._tiers

    async def predict(self, *args, **kwargs) -> FallbackResult:
        """Try each tier in order. Return the first successful result,
        recording all attempts.
        """
        attempts: list[tuple[str, str, float]] = []
        last_exc: Optional[BaseException] = None

        for idx, tier in enumerate(self._tiers):
            t0 = time.perf_counter()
            try:
                if tier.timeout_ms is None:
                    value = await tier.predict(*args, **kwargs)
                else:
                    value = await asyncio.wait_for(
                        tier.predict(*args, **kwargs),
                        timeout=tier.timeout_ms / 1000.0,
                    )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                attempts.append((tier.name, "success", elapsed_ms))
                # If we fell back from earlier tiers, each one emitted its
                # metric below. This tier succeeded — no metric for it.
                return FallbackResult(
                    value=value,
                    tier_used=tier.name,
                    attempts=attempts,
                )
            except asyncio.TimeoutError as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                attempts.append((tier.name, "timeout", elapsed_ms))
                inc_fallback_invocation(
                    embodiment=self._embodiment,
                    target=tier.name,
                )
                last_exc = exc
                logger.warning(
                    "fallback.tier_timeout embodiment=%s tier=%s elapsed_ms=%.1f",
                    self._embodiment, tier.name, elapsed_ms,
                )
            except Exception as exc:  # noqa: BLE001 — chain must catch everything
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                attempts.append((tier.name, "exception", elapsed_ms))
                inc_fallback_invocation(
                    embodiment=self._embodiment,
                    target=tier.name,
                )
                last_exc = exc
                logger.warning(
                    "fallback.tier_exception embodiment=%s tier=%s exc=%s: %s",
                    self._embodiment, tier.name, type(exc).__name__, exc,
                )

        # Every tier failed
        raise AllTiersFailed(attempts)
