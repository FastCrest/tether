"""Action-similarity fast path (training-free FlashVLA).

Per spec features/01_serve/subfeatures/_perf_compound/action-similarity-fast-path/
action-similarity-fast-path.md (research_status: complete; locked decisions).

When an expert-produced action chunk is L2-similar to the previously emitted
chunk (default L2 < 0.05), the next predict_action_chunk() call can short-
circuit the expert and reuse the cached result. Caps consecutive skips
(default max_skips=3) so the policy doesn't drift on slow-changing scenes.

Cache stores PRE-A2C2 raw expert output per locked decision in the spec —
A2C2 corrections are recomputed on each call so reusing post-A2C2 actions
would compound corrections.

Composes with:
  - Pi05DecomposedInference.predict_action_chunk (insertion at line ~464,
    after _run_expert returns + before _action_cache populate)
  - reflex.observability.prometheus (per-skip metric for operator visibility)
  - cuda-graphs (independent — fast path skips the captured graph entirely
    when triggered, so capture state is unaffected)

Phase 1.5 ships flat thresholds (single L2 + single max_skips). Phase 2
deferred: per-task YAML overrides, adaptive thresholds, region-of-interest
similarity instead of full-vector L2.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Paper defaults (FlashVLA, arxiv 2505.21200, Table 2 + Section 4.2).
# CLI-overridable via `--action-similarity-threshold` + `--max-similar-skips`.
DEFAULT_THRESHOLD = 0.05
DEFAULT_MAX_SKIPS = 3


@dataclass
class FastPathStats:
    """Per-instance stats for the fast path. Snapshotted by the metric emitter."""

    expert_calls: int = 0       # times the expert actually ran
    skip_count: int = 0         # times we returned cached actions instead
    forced_calls: int = 0       # times we ran expert even though similar (max_skips hit)
    last_distance: float = float("nan")  # most-recent L2 distance (for debug)

    @property
    def total_calls(self) -> int:
        return self.expert_calls + self.skip_count

    @property
    def skip_rate(self) -> float:
        total = self.total_calls
        return self.skip_count / total if total > 0 else 0.0


class ActionFastPath:
    """Per-policy action-similarity fast path.

    Lifecycle (on Pi05DecomposedInference instance):
        fast_path = ActionFastPath(threshold=0.05, max_skips=3)
        # Inside predict_action_chunk:
        if fast_path.should_skip():
            return fast_path.cached_actions()  # short-circuit
        actions = self._run_expert(...)
        fast_path.observe(actions)  # update L2 + skip-counter for next call

    Thread-safety: the inference object is single-threaded per request. The
    fast path holds no locks; if the inference path becomes multi-threaded
    later (it isn't today), wrap observe() + should_skip() under the
    PolicyRuntime's request lock.
    """

    __slots__ = (
        "_threshold",
        "_max_skips",
        "_enabled",
        "_last_actions",
        "_consecutive_skips",
        "_stats",
    )

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        max_skips: int = DEFAULT_MAX_SKIPS,
        enabled: bool = True,
    ):
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        if max_skips < 0:
            raise ValueError(f"max_skips must be >= 0, got {max_skips}")
        self._threshold = float(threshold)
        self._max_skips = int(max_skips)
        self._enabled = bool(enabled)
        self._last_actions: np.ndarray | None = None
        self._consecutive_skips = 0
        self._stats = FastPathStats()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def max_skips(self) -> int:
        return self._max_skips

    @property
    def stats(self) -> FastPathStats:
        return self._stats

    def cached_actions(self) -> np.ndarray | None:
        """The currently-cached action array (or None if no cache yet).
        Caller should COPY before returning to /act so a downstream mutation
        doesn't poison the cache."""
        if self._last_actions is None:
            return None
        return self._last_actions.copy()

    def should_skip(self) -> bool:
        """True iff the next predict_action_chunk should reuse the cache.

        Conditions:
          - Fast path enabled
          - We have a cached action from a prior call
          - The previous observe() found similarity below threshold
          - We haven't hit max_skips consecutive skips already

        The "similarity" check happens in observe() AFTER the expert runs,
        not here — should_skip just checks the resulting flag (encoded as
        consecutive_skips < max_skips with cache present + similarity-flag).
        We pack this into a single counter for atomicity: if observe()
        decided this chunk was similar, it incremented `_consecutive_skips`
        (capped at max_skips); should_skip returns True iff > 0 (signalling
        "the most recent observe found similarity AND we have budget").
        """
        if not self._enabled or self._last_actions is None:
            return False
        return 0 < self._consecutive_skips <= self._max_skips

    def observe(self, actions: np.ndarray) -> None:
        """Record a freshly-produced action chunk + update similarity flag.

        Computes L2 distance between `actions` and the cached previous
        chunk (if any). If similarity below threshold AND we haven't
        exhausted the max-skips budget, increments the skip counter so the
        NEXT call's should_skip() returns True. Otherwise resets the
        counter.

        Always overwrites the cache with the new actions — even when not
        similar, the next call may find the call-after-next similar to
        THIS one.
        """
        # Increment expert_calls regardless of enabled state — caller has
        # always run the expert to produce these actions, and downstream
        # observability needs a true per-call counter (not "calls when
        # fast-path-enabled"). Caught 2026-05-07 by production smoke
        # (Run A reported expert_calls=0 even though the expert ran 20x).
        self._stats.expert_calls += 1
        if not self._enabled:
            return
        if self._last_actions is None or actions.shape != self._last_actions.shape:
            # No prior cache OR shape change → can't compare; reset.
            # expert_calls already incremented above per-call.
            self._consecutive_skips = 0
            self._stats.last_distance = float("nan")
            self._last_actions = actions.copy()
            return

        distance = float(np.linalg.norm(actions - self._last_actions))
        self._stats.last_distance = distance
        # expert_calls already incremented above per-call.

        if distance < self._threshold and self._consecutive_skips < self._max_skips:
            # Similar enough → next call can skip. Don't update cache —
            # we want subsequent skips to reuse THIS chunk's actions, not
            # drift on the noise of repeat calls.
            self._consecutive_skips += 1
        else:
            # Not similar OR max skips reached → reset.
            if self._consecutive_skips >= self._max_skips:
                self._stats.forced_calls += 1
            self._consecutive_skips = 0
            self._last_actions = actions.copy()

    def consume_skip(self) -> None:
        """Called by the inference path when should_skip() returned True
        and we're about to return the cached actions. Decrements the skip
        budget (so should_skip() flips False after `_consecutive_skips`
        cached returns)."""
        if self._consecutive_skips > 0:
            self._consecutive_skips -= 1
            self._stats.skip_count += 1

    def reset(self) -> None:
        """Clear the cache + stats. Used at episode boundaries (RTC reset)
        + by tests."""
        self._last_actions = None
        self._consecutive_skips = 0
        self._stats = FastPathStats()


__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_MAX_SKIPS",
    "ActionFastPath",
    "FastPathStats",
]
