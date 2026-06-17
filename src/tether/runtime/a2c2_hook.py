"""A2C2 serve runtime hook — Phase B.5 Day 3.

Wraps the inference-time A2C2Head (kernels/a2c2_correction.py) with:
- Latency tracker — rolling-window p95 of recent /act latencies
- Success-rate tracker — rolling-window fraction of "good" recent acts
- Auto-skip logic — apply correction only when latency p95 >= threshold
  AND success rate <= threshold (correction has positive marginal value)
- Prometheus metrics — `tether_a2c2_applied_total` + `tether_a2c2_skipped_total`

Per a2c2-correction execution plan B.5 Day 3:
- Auto-disable: if `latency_p95 < 40 ms` OR `success_rate > 90%`, skip
  the A2C2 forward pass — A2C2 only adds value at high-latency low-success
  regimes; otherwise it's pure overhead.
- Per-act logging via structured fields (a2c2_applied, latency_p95_ms,
  correction_magnitude) — picked up by the existing JSONL recorder.

The hook holds NO model state — it composes A2C2Head + the trackers. The
serve runtime owns the hook (one per loaded policy in Phase 2; one per
process in Phase 1 single-policy).
"""
from __future__ import annotations

import collections
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tether.kernels.a2c2_correction import A2C2Head

logger = logging.getLogger(__name__)


# Default thresholds per a2c2-correction execution plan B.5 acceptance
# criteria. Customers tune these via tether.yaml or `--a2c2-*` CLI flags.
_DEFAULT_LATENCY_THRESHOLD_MS = 40.0
_DEFAULT_SUCCESS_THRESHOLD = 0.90

# Rolling-window sizes — match webhooks + slo conventions (~10s of traffic
# at 10 QPS). Longer = more stable thresholds but slower adaptation.
_DEFAULT_LATENCY_WINDOW = 100
_DEFAULT_SUCCESS_WINDOW = 50

# Cold-start: until we have N samples, skip A2C2 (insufficient signal to
# decide whether it's needed). 5 is enough for a stable p95 in practice.
_MIN_SAMPLES_FOR_DECISION = 5


@dataclass(frozen=True)
class A2C2HookConfig:
    """Frozen config for the A2C2 hook. Parsed from CLI / tether.yaml at
    create_app time; immutable thereafter."""

    latency_threshold_ms: float = _DEFAULT_LATENCY_THRESHOLD_MS
    success_threshold: float = _DEFAULT_SUCCESS_THRESHOLD
    latency_window: int = _DEFAULT_LATENCY_WINDOW
    success_window: int = _DEFAULT_SUCCESS_WINDOW
    min_samples_for_decision: int = _MIN_SAMPLES_FOR_DECISION

    def __post_init__(self) -> None:
        if self.latency_threshold_ms <= 0:
            raise ValueError(
                f"latency_threshold_ms must be positive, got "
                f"{self.latency_threshold_ms}"
            )
        # Allow values > 1.0 as a documented "never skip due to success" sentinel
        # (e.g., 1.01 disables success-based skip — useful for measurement runs
        # where the success metric is /act error rate, not task success).
        # Hard cap at 2.0 to catch typos like 100 instead of 1.0.
        if not (0.0 <= self.success_threshold <= 2.0):
            raise ValueError(
                f"success_threshold must be in [0, 2.0] (1.01+ disables skip), got "
                f"{self.success_threshold}"
            )
        if self.latency_window < 1:
            raise ValueError(
                f"latency_window must be >= 1, got {self.latency_window}"
            )
        if self.success_window < 1:
            raise ValueError(
                f"success_window must be >= 1, got {self.success_window}"
            )
        if self.min_samples_for_decision < 1:
            raise ValueError(
                f"min_samples_for_decision must be >= 1, got "
                f"{self.min_samples_for_decision}"
            )


@dataclass(frozen=True)
class A2C2Decision:
    """The output of `A2C2Hook.should_apply()` — used for metric labels +
    logging, exposed for tests."""

    apply: bool
    reason: str  # bounded enum: "applied" | "cold_start" | "low_latency" | "high_success"
    latency_p95_ms: float
    success_rate: float
    samples: int


class A2C2Hook:
    """Stateful wrapper around A2C2Head with latency + success tracking.

    Thread-safe: trackers protected by a single lock; rare contention since
    the only writer is the /act handler (one event loop) and reads are
    O(N=100) via deque + numpy.
    """

    __slots__ = (
        "_head", "_config",
        "_latency_window", "_success_window", "_lock",
        "_applied_total", "_skipped_total",
    )

    def __init__(
        self,
        head: A2C2Head,
        config: A2C2HookConfig | None = None,
    ):
        self._head = head
        self._config = config or A2C2HookConfig()
        self._latency_window: collections.deque[float] = collections.deque(
            maxlen=self._config.latency_window
        )
        self._success_window: collections.deque[bool] = collections.deque(
            maxlen=self._config.success_window
        )
        self._lock = threading.Lock()
        # In-process counters for tests + diagnostics; the Prometheus
        # counters live in observability.prometheus and are emitted from
        # the apply/skip code paths separately.
        self._applied_total = 0
        self._skipped_total = 0

    @property
    def head(self) -> A2C2Head:
        return self._head

    @property
    def config(self) -> A2C2HookConfig:
        return self._config

    @property
    def applied_total(self) -> int:
        return self._applied_total

    @property
    def skipped_total(self) -> int:
        return self._skipped_total

    def record_outcome(self, latency_ms: float, success: bool) -> None:
        """Record one /act outcome. Called by the serve runtime after every
        act completes. Updates both rolling windows."""
        if latency_ms <= 0 or latency_ms != latency_ms:  # rejects NaN + non-positive
            return
        with self._lock:
            self._latency_window.append(float(latency_ms))
            self._success_window.append(bool(success))

    def latency_p95_ms(self) -> float:
        """Current p95 latency from the rolling window. Returns 0.0 when
        no samples yet (cold-start guard handled in should_apply)."""
        with self._lock:
            samples = list(self._latency_window)
        if not samples:
            return 0.0
        return float(np.percentile(samples, 95))

    def success_rate(self) -> float:
        """Current success rate from the rolling window. Returns 1.0 when
        no samples yet (we assume things are working until proven otherwise
        — cold-start path is gated by min_samples in should_apply)."""
        with self._lock:
            samples = list(self._success_window)
        if not samples:
            return 1.0
        return float(np.mean(samples))

    def sample_count(self) -> tuple[int, int]:
        """Returns (n_latency, n_success). Used by cold-start gate."""
        with self._lock:
            return len(self._latency_window), len(self._success_window)

    def should_apply(self) -> A2C2Decision:
        """Decide whether to invoke the head this /act.

        Cold-start: skip until we have min_samples in BOTH windows.
        Steady-state: apply when latency_p95 >= threshold AND
        success_rate <= threshold (i.e., latency is high AND success is
        not great — A2C2 has positive marginal value).
        """
        n_lat, n_succ = self.sample_count()
        latency_p95 = self.latency_p95_ms()
        success_rate = self.success_rate()
        cfg = self._config

        if n_lat < cfg.min_samples_for_decision or n_succ < cfg.min_samples_for_decision:
            return A2C2Decision(
                apply=False, reason="cold_start",
                latency_p95_ms=latency_p95, success_rate=success_rate,
                samples=min(n_lat, n_succ),
            )

        # Skip when latency is low (correction not needed)
        if latency_p95 < cfg.latency_threshold_ms:
            return A2C2Decision(
                apply=False, reason="low_latency",
                latency_p95_ms=latency_p95, success_rate=success_rate,
                samples=min(n_lat, n_succ),
            )

        # Skip when success is high (correction not needed)
        if success_rate > cfg.success_threshold:
            return A2C2Decision(
                apply=False, reason="high_success",
                latency_p95_ms=latency_p95, success_rate=success_rate,
                samples=min(n_lat, n_succ),
            )

        return A2C2Decision(
            apply=True, reason="applied",
            latency_p95_ms=latency_p95, success_rate=success_rate,
            samples=min(n_lat, n_succ),
        )

    def maybe_apply_to_chunk(
        self,
        *,
        actions: np.ndarray,
        observation: np.ndarray | None = None,
    ) -> tuple[np.ndarray, A2C2Decision, float]:
        """Apply the correction to each action in the chunk if should_apply()
        says yes. Returns (actions_out, decision, correction_magnitude).

        Args:
            actions: shape (chunk_size, action_dim) float32
            observation: shape (obs_dim,) float32 — caller's encoding of the
                current observation. When None, a zero vector is used (Phase 1
                degraded mode; Phase 2 wires through VLM-prefix output).

        Returns:
            actions_out: corrected actions (same shape as input)
            decision: A2C2Decision for logging + metric emission
            correction_magnitude: L2 norm of the correction across the chunk
                (0.0 when skipped). Used for /act response telemetry.
        """
        decision = self.should_apply()
        if not decision.apply:
            with self._lock:
                self._skipped_total += 1
            _emit_a2c2_metric(applied=False, reason=decision.reason)
            return actions, decision, 0.0

        if actions.ndim != 2 or actions.shape[1] != self._head.config.action_dim:
            raise ValueError(
                f"actions shape mismatch: expected (chunk_size, "
                f"{self._head.config.action_dim}), got {actions.shape}"
            )

        chunk_size = actions.shape[0]
        if chunk_size > self._head.config.chunk_size:
            # Truncate position to the head's chunk_size — the head's
            # positional encoding is bounded by its config's chunk_size,
            # but the policy may emit longer chunks in some configs.
            # Fall through with clamped position; alternatively raise.
            logger.warning(
                "a2c2.chunk_size_exceeds_head: actions=%d head=%d — clamping positions",
                chunk_size, self._head.config.chunk_size,
            )

        if observation is None:
            observation = np.zeros(self._head.config.obs_dim, dtype=np.float32)

        latency_ms = decision.latency_p95_ms
        corrected = np.empty_like(actions, dtype=np.float32)
        total_sq = 0.0
        for i in range(chunk_size):
            pos = min(i, self._head.config.chunk_size - 1)
            try:
                correction = self._head.forward(
                    base_action=actions[i].astype(np.float32, copy=False),
                    observation=observation,
                    chunk_position=pos,
                    latency_estimate_ms=latency_ms,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "a2c2.forward_failed at position=%d: %s — falling back to base action",
                    i, exc,
                )
                corrected[i] = actions[i]
                continue
            corrected[i] = actions[i] + correction
            total_sq += float(np.sum(correction * correction))

        magnitude = float(np.sqrt(total_sq))

        # Safety net per 2026-04-29-a2c2-correction_research_revisit:
        # bounded-output head should keep magnitude under sqrt(chunk_size)
        # * scale (theoretical max from per-step ±scale saturation). Scale
        # is per-head (Phase 3); default 3.0 matches Phase 1 ship.
        head_scale = self._head.config.output_saturation_scale
        chunk_safety_limit = (chunk_size ** 0.5) * head_scale
        if magnitude > chunk_safety_limit:
            logger.warning(
                "a2c2.magnitude_safety_skip: chunk_magnitude=%.2f exceeds "
                "limit=%.2f (chunk_size=%d); falling back to base actions",
                magnitude, chunk_safety_limit, chunk_size,
            )
            with self._lock:
                self._skipped_total += 1
            _emit_a2c2_metric(applied=False, reason="magnitude_safety_skip")
            return actions, decision, 0.0

        with self._lock:
            self._applied_total += 1
        _emit_a2c2_metric(applied=True, reason="applied")
        return corrected, decision, magnitude

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: A2C2HookConfig | None = None,
    ) -> "A2C2Hook":
        """Load the head from a .npz checkpoint and wrap with default config."""
        head = A2C2Head.from_checkpoint(checkpoint_path)
        return cls(head=head, config=config)


# Optional Prometheus emission — defaults to no-op when prometheus_client
# isn't installed (matches PolicyRuntime + WebhookDispatcher conventions).
try:
    from tether.observability.prometheus import (
        inc_a2c2_applied,
        inc_a2c2_skipped,
    )
    _METRICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _METRICS_AVAILABLE = False
    def inc_a2c2_applied(reason: str) -> None: pass
    def inc_a2c2_skipped(reason: str) -> None: pass


def _emit_a2c2_metric(*, applied: bool, reason: str) -> None:
    """Internal helper — emit applied or skipped counter with a reason label."""
    try:
        if applied:
            inc_a2c2_applied(reason=reason)
        else:
            inc_a2c2_skipped(reason=reason)
    except Exception as exc:  # noqa: BLE001 — metrics never break the hot path
        logger.warning("a2c2.metric_emit_failed: %s", exc)


__all__ = [
    "A2C2Decision",
    "A2C2Hook",
    "A2C2HookConfig",
]
