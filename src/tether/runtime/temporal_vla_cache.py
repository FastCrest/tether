"""Temporal reuse policy for VLA runtime caches.

The Pi0.5 decomposed server already has prefix/action/episode cache layers.
This module centralizes the safety checks that decide whether a cached entry is
still valid for the next control tick: age, language identity, visual stability,
state drift, and optional guard/action-change margins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TemporalCacheDecision:
    """Decision returned by TemporalVLAReusePolicy.assess."""

    reuse: bool
    reason: str
    steps_since: int | None = None
    image_hamming: int | None = None
    state_delta: float | None = None
    guard_margin: float | None = None
    action_delta: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reuse": self.reuse,
            "reason": self.reason,
            "steps_since": self.steps_since,
            "image_hamming": self.image_hamming,
            "state_delta": self.state_delta,
            "guard_margin": self.guard_margin,
            "action_delta": self.action_delta,
        }


class TemporalVLAReusePolicy:
    """Conservative policy for temporal cache reuse.

    The policy is intentionally model-agnostic. Callers pass the cache key
    material they already have and the policy returns a bounded reason for
    telemetry and tests.
    """

    __slots__ = (
        "enabled",
        "phash_hamming_threshold",
        "state_delta_threshold",
        "min_guard_margin",
        "action_delta_threshold",
    )

    def __init__(
        self,
        *,
        enabled: bool = True,
        phash_hamming_threshold: int = 6,
        state_delta_threshold: float = 0.05,
        min_guard_margin: float | None = None,
        action_delta_threshold: float | None = None,
    ):
        if phash_hamming_threshold < 0:
            raise ValueError("phash_hamming_threshold must be >= 0")
        if state_delta_threshold < 0:
            raise ValueError("state_delta_threshold must be >= 0")
        if min_guard_margin is not None and min_guard_margin < 0:
            raise ValueError("min_guard_margin must be >= 0 when set")
        if action_delta_threshold is not None and action_delta_threshold < 0:
            raise ValueError("action_delta_threshold must be >= 0 when set")

        self.enabled = bool(enabled)
        self.phash_hamming_threshold = int(phash_hamming_threshold)
        self.state_delta_threshold = float(state_delta_threshold)
        self.min_guard_margin = min_guard_margin
        self.action_delta_threshold = action_delta_threshold

    @staticmethod
    def state_signature(state: np.ndarray | list[float] | None) -> np.ndarray | None:
        """Normalize state for cache comparisons."""
        if state is None:
            return None
        arr = np.asarray(state, dtype=np.float32)
        if arr.size == 0:
            return None
        return arr.reshape(-1).copy()

    @staticmethod
    def max_hamming(
        cached: tuple[bytes, ...],
        current: tuple[bytes, ...],
    ) -> int | None:
        if len(cached) != len(current):
            return None
        max_dist = 0
        for left, right in zip(cached, current):
            dist = sum((a ^ b).bit_count() for a, b in zip(left, right))
            max_dist = max(max_dist, dist)
        return max_dist

    @staticmethod
    def state_delta(
        cached: np.ndarray | None,
        current: np.ndarray | None,
    ) -> float | None:
        if cached is None and current is None:
            return 0.0
        if cached is None or current is None:
            return None
        if cached.shape != current.shape:
            return None
        return float(np.linalg.norm(current - cached))

    def assess(
        self,
        *,
        cached_image_phashes: tuple[bytes, ...] | None,
        current_image_phashes: tuple[bytes, ...] | None,
        cached_lang_hash: bytes | None,
        current_lang_hash: bytes | None,
        cached_state: np.ndarray | None = None,
        current_state: np.ndarray | None = None,
        cached_step_index: int | None = None,
        current_step_index: int | None = None,
        max_age_steps: int | None = None,
        allow_lang_mismatch: bool = False,
        guard_margin: float | None = None,
        action_delta: float | None = None,
    ) -> TemporalCacheDecision:
        if not self.enabled:
            return TemporalCacheDecision(False, "disabled")
        if (
            cached_image_phashes is None
            or current_image_phashes is None
            or cached_lang_hash is None
            or current_lang_hash is None
        ):
            return TemporalCacheDecision(False, "missing_key")

        steps_since = None
        if cached_step_index is not None and current_step_index is not None:
            steps_since = current_step_index - cached_step_index
            if steps_since < 0:
                return TemporalCacheDecision(False, "negative_age", steps_since=steps_since)
            if max_age_steps is not None and steps_since > max_age_steps:
                return TemporalCacheDecision(False, "stale", steps_since=steps_since)

        if not allow_lang_mismatch and cached_lang_hash != current_lang_hash:
            return TemporalCacheDecision(False, "language_changed", steps_since=steps_since)

        image_hamming = self.max_hamming(cached_image_phashes, current_image_phashes)
        if image_hamming is None:
            return TemporalCacheDecision(False, "image_shape_changed", steps_since=steps_since)
        if image_hamming > self.phash_hamming_threshold:
            return TemporalCacheDecision(
                False,
                "image_changed",
                steps_since=steps_since,
                image_hamming=image_hamming,
            )

        state_delta = self.state_delta(cached_state, current_state)
        if state_delta is None:
            return TemporalCacheDecision(
                False,
                "state_shape_changed",
                steps_since=steps_since,
                image_hamming=image_hamming,
            )
        if state_delta > self.state_delta_threshold:
            return TemporalCacheDecision(
                False,
                "state_changed",
                steps_since=steps_since,
                image_hamming=image_hamming,
                state_delta=state_delta,
            )

        if (
            self.min_guard_margin is not None
            and guard_margin is not None
            and guard_margin < self.min_guard_margin
        ):
            return TemporalCacheDecision(
                False,
                "guard_margin_low",
                steps_since=steps_since,
                image_hamming=image_hamming,
                state_delta=state_delta,
                guard_margin=guard_margin,
            )

        if (
            self.action_delta_threshold is not None
            and action_delta is not None
            and action_delta > self.action_delta_threshold
        ):
            return TemporalCacheDecision(
                False,
                "action_delta_high",
                steps_since=steps_since,
                image_hamming=image_hamming,
                state_delta=state_delta,
                guard_margin=guard_margin,
                action_delta=action_delta,
            )

        return TemporalCacheDecision(
            True,
            "stable",
            steps_since=steps_since,
            image_hamming=image_hamming,
            state_delta=state_delta,
            guard_margin=guard_margin,
            action_delta=action_delta,
        )


__all__ = [
    "TemporalCacheDecision",
    "TemporalVLAReusePolicy",
]
