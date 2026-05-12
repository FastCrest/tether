"""Action-similarity fast path (FlashVLA, arxiv 2505.21200).

Covers the locked-decision behavior in
features/01_serve/subfeatures/_perf_compound/action-similarity-fast-path/:
  - Disabled by default (threshold == 0); always defers to expert
  - Threshold gating: similar chunks → next call skips; dissimilar resets
  - max_skips cap: at most N consecutive skips, then forced expert
  - Cache returns a copy (caller mutations cannot poison the cache)
  - reset() clears state for episode boundaries
  - Stats: skip_count + expert_calls + skip_rate
  - Shape change resets cleanly (no crash)
"""
from __future__ import annotations

import numpy as np
import pytest

from reflex.runtime.action_fast_path import (
    DEFAULT_MAX_SKIPS,
    DEFAULT_THRESHOLD,
    ActionFastPath,
)


def _chunk(value: float, shape=(50, 7)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def test_disabled_never_skips():
    fp = ActionFastPath(threshold=0.0, max_skips=3, enabled=False)
    fp.observe(_chunk(0.0))
    fp.observe(_chunk(0.0))  # bit-identical
    assert fp.should_skip() is False


def test_threshold_zero_disables_via_constructor():
    fp = ActionFastPath(threshold=0.05, max_skips=3, enabled=False)
    fp.observe(_chunk(0.0))
    fp.observe(_chunk(0.0))
    assert fp.should_skip() is False


def test_no_skip_until_two_observations():
    fp = ActionFastPath(threshold=0.05, max_skips=3)
    assert fp.should_skip() is False
    fp.observe(_chunk(0.0))
    # First observe populates cache only — no prior chunk to compare to.
    assert fp.should_skip() is False


def test_similar_chunks_trigger_next_call_skip():
    fp = ActionFastPath(threshold=0.05, max_skips=3)
    fp.observe(_chunk(0.0))
    fp.observe(_chunk(0.0))  # L2 distance 0 < 0.05
    assert fp.should_skip() is True


def test_dissimilar_chunks_reset_skip_flag():
    fp = ActionFastPath(threshold=0.05, max_skips=3)
    fp.observe(_chunk(0.0))
    fp.observe(_chunk(10.0))  # huge L2 distance
    assert fp.should_skip() is False


def test_max_skips_caps_consecutive_skips():
    fp = ActionFastPath(threshold=0.05, max_skips=2)
    fp.observe(_chunk(0.0))
    fp.observe(_chunk(0.0))  # similar → 1 skip queued
    fp.observe(_chunk(0.0))  # similar → 2 skips queued (cap)
    fp.observe(_chunk(0.0))  # similar but at cap → resets, forces expert
    # Queue should be 0 now since cap was hit.
    assert fp._consecutive_skips == 0
    assert fp.stats.forced_calls == 1


def test_consume_skip_decrements_counter():
    fp = ActionFastPath(threshold=0.05, max_skips=3)
    fp.observe(_chunk(0.0))
    fp.observe(_chunk(0.0))  # 1 skip queued
    assert fp.should_skip() is True
    fp.consume_skip()
    assert fp.stats.skip_count == 1
    # No more queued skips → next should_skip is False.
    assert fp.should_skip() is False


def test_cached_actions_returns_copy_not_reference():
    fp = ActionFastPath(threshold=0.05, max_skips=3)
    original = _chunk(0.0)
    fp.observe(original)
    fp.observe(_chunk(0.0))  # similar → queues skip
    cached = fp.cached_actions()
    assert cached is not None
    cached[0, 0] = 999.0
    # Mutating the returned array must not poison the internal cache.
    cached_again = fp.cached_actions()
    assert cached_again is not None
    assert cached_again[0, 0] == 0.0


def test_cached_actions_none_before_first_observe():
    fp = ActionFastPath(threshold=0.05, max_skips=3)
    assert fp.cached_actions() is None


def test_reset_clears_state():
    fp = ActionFastPath(threshold=0.05, max_skips=3)
    fp.observe(_chunk(0.0))
    fp.observe(_chunk(0.0))
    fp.consume_skip()
    fp.reset()
    assert fp.cached_actions() is None
    assert fp.should_skip() is False
    assert fp.stats.expert_calls == 0
    assert fp.stats.skip_count == 0


def test_shape_change_resets_cleanly():
    fp = ActionFastPath(threshold=0.05, max_skips=3)
    fp.observe(_chunk(0.0, shape=(50, 7)))
    # Shape mismatch — must not raise, must reset cache to new shape.
    fp.observe(_chunk(0.0, shape=(50, 8)))
    assert fp.should_skip() is False


def test_stats_track_expert_calls_and_skip_rate():
    fp = ActionFastPath(threshold=0.05, max_skips=3)
    fp.observe(_chunk(0.0))  # expert call 1
    fp.observe(_chunk(0.0))  # expert call 2 (similar → queue skip)
    fp.consume_skip()        # skip 1
    assert fp.stats.expert_calls == 2
    assert fp.stats.skip_count == 1
    assert fp.stats.total_calls == 3
    assert fp.stats.skip_rate == pytest.approx(1 / 3)


def test_stats_skip_rate_zero_when_no_calls():
    fp = ActionFastPath(threshold=0.05, max_skips=3)
    assert fp.stats.skip_rate == 0.0


def test_last_distance_recorded():
    fp = ActionFastPath(threshold=0.05, max_skips=3)
    fp.observe(_chunk(0.0))
    fp.observe(_chunk(1.0))
    # L2 of (50, 7) array of 1s = sqrt(350) ~= 18.71
    assert fp.stats.last_distance == pytest.approx(np.sqrt(350.0))


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        ActionFastPath(threshold=-0.1)


def test_invalid_max_skips_raises():
    with pytest.raises(ValueError):
        ActionFastPath(max_skips=-1)


def test_paper_defaults_match_spec():
    assert DEFAULT_THRESHOLD == 0.05
    assert DEFAULT_MAX_SKIPS == 3


def test_observe_after_max_skips_resumes_caching():
    """After hitting the max_skips cap and forcing an expert call, the
    fast path must continue tracking similarity for future calls (not
    permanently disabled)."""
    fp = ActionFastPath(threshold=0.05, max_skips=2)
    fp.observe(_chunk(0.0))
    fp.observe(_chunk(0.0))
    fp.observe(_chunk(0.0))
    fp.observe(_chunk(0.0))  # cap hit → reset
    # Now subsequent similar chunks should re-queue skips.
    fp.observe(_chunk(0.0))
    assert fp.should_skip() is True


def test_expert_calls_increments_when_disabled():
    """expert_calls counter increments per observe regardless of
    enabled state. Caught 2026-05-07 production smoke (Run A reported
    expert_calls=0 because observe() early-returned when enabled=False
    even though the expert had run on each call). Fixed via increment-
    before-enabled-check; this test guards against regression."""
    fp = ActionFastPath(threshold=0.05, max_skips=3, enabled=False)
    fp.observe(_chunk(0.0))
    fp.observe(_chunk(0.0))
    fp.observe(_chunk(0.0))
    assert fp.stats.expert_calls == 3
    # Disabled mode never increments skip_count
    assert fp.stats.skip_count == 0


def test_skip_then_dissimilar_resets_cleanly():
    fp = ActionFastPath(threshold=0.05, max_skips=3)
    fp.observe(_chunk(0.0))
    fp.observe(_chunk(0.0))  # queues 1 skip
    assert fp.should_skip() is True
    fp.consume_skip()
    fp.observe(_chunk(50.0))  # huge L2 — must reset
    assert fp.should_skip() is False
