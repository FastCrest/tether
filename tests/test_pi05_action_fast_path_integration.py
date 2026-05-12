"""Integration test for ActionFastPath wired through Pi05DecomposedInference.

Closes the unit-test → wire-up gap. test_action_fast_path.py already
covers the ActionFastPath class in isolation; this test verifies the
short-circuit at predict_action_chunk:480-495 fires correctly when fed
repeated similar chunks via the actual inference path.

Approach: bypass Pi05DecomposedInference.__init__ (heavy — loads ONNX
files + reflex_config.json) and manually assemble a minimal instance
with mocked sessions + a patched _run_expert that returns
deterministic actions. Verify:
  1. Repeated similar chunks alternate expert/cache (max_skips cap is
     a theoretical bound; consume_skip after each skip means the
     counter stays at ≤1 in steady state — every other call is a skip)
  2. Dissimilar chunk resets fast-path → forces expert
  3. reset_cache() flushes fast-path between episodes
  4. threshold=0.0 (disabled) → expert always runs, never skips
  5. Cached return is a copy (caller mutation doesn't poison cache)
"""
from __future__ import annotations

import numpy as np
import pytest

from reflex.runtime.action_fast_path import ActionFastPath
from reflex.runtime.pi05_decomposed_server import Pi05DecomposedInference


@pytest.fixture
def fixture_inference():
    """Manually-assembled Pi05DecomposedInference with mocked dependencies.
    Bypasses __init__ entirely; sets only the attributes predict_action_chunk
    actually reads."""
    inst = Pi05DecomposedInference.__new__(Pi05DecomposedInference)
    # Disable all caching paths except the fast path under test
    inst.cache_level = "none"
    inst._call_index = 0
    inst._action_cache = None
    inst._cache = None  # prefix cache
    inst._stats = type(
        "FakeStats", (), {
            "action_hits": 0,
            "action_misses": 0,
        }
    )()
    inst._episode_cache = None
    # Used by the prefix cache lookup (which we skip by mocking _get_or_run_prefix)
    inst.cache_ignore_lang = False
    inst.action_cache_max_age_steps = 1000
    inst.cache_max_age_steps = 1000
    inst.cache_ttl_sec = 60.0
    inst.phash_hamming_threshold = 5
    # Config — minimal stub
    inst.config = {"decomposed": {"expert_takes_state": False}}
    inst._past_kv_names = []
    # Inject the fast path being tested
    inst._fast_path = ActionFastPath(
        threshold=0.05, max_skips=3, enabled=True,
    )

    # Mock _get_or_run_prefix → returns dummy past_kv + pad
    def fake_prefix(**kwargs):
        return ([np.zeros((1, 1), dtype=np.float32)], np.ones((1, 1), dtype=bool))
    inst._get_or_run_prefix = fake_prefix  # type: ignore[method-assign]

    # Mock _phash + _lang_hash — return constants
    inst._phash = lambda img: "ph_constant"  # type: ignore[method-assign]
    inst._lang_hash = lambda lang: "lh_constant"  # type: ignore[method-assign]
    inst._phashes_match = lambda a, b: True  # type: ignore[method-assign]

    return inst


def _call_count_run_expert(inst, action_value: float, chunk_shape=(1, 50, 7)):
    """Patch _run_expert to track call count + return predictable chunk."""
    inst._run_expert_calls = 0

    def fake_run_expert(expert_feed_base, noise):
        inst._run_expert_calls += 1
        return np.full(chunk_shape, action_value, dtype=np.float32)

    inst._run_expert = fake_run_expert  # type: ignore[method-assign]


def _predict_args(B: int = 1):
    """Standard call args for predict_action_chunk; values are placeholders
    since _phash + _get_or_run_prefix are mocked."""
    return dict(
        img_base=np.zeros((B, 3, 224, 224), dtype=np.float32),
        img_wrist_l=np.zeros((B, 3, 224, 224), dtype=np.float32),
        img_wrist_r=np.zeros((B, 3, 224, 224), dtype=np.float32),
        mask_base=np.ones(B, dtype=bool),
        mask_wrist_l=np.ones(B, dtype=bool),
        mask_wrist_r=np.ones(B, dtype=bool),
        lang_tokens=np.zeros((B, 16), dtype=np.int64),
        lang_masks=np.ones((B, 16), dtype=bool),
        noise=np.zeros((B, 50, 7), dtype=np.float32),
    )


def test_first_call_runs_expert_no_skip(fixture_inference):
    """First call: no cached fast-path state, expert must run."""
    inst = fixture_inference
    _call_count_run_expert(inst, action_value=0.0)
    actions = inst.predict_action_chunk(**_predict_args())
    assert inst._run_expert_calls == 1
    assert actions.shape == (1, 50, 7)


def test_two_similar_chunks_third_call_skips(fixture_inference):
    """After 2 similar expert calls, the 3rd call's should_skip is True
    and returns cached actions without calling _run_expert."""
    inst = fixture_inference
    _call_count_run_expert(inst, action_value=0.0)

    inst.predict_action_chunk(**_predict_args())  # call 1: expert
    inst.predict_action_chunk(**_predict_args())  # call 2: expert + queue skip
    inst.predict_action_chunk(**_predict_args())  # call 3: skip cached

    assert inst._run_expert_calls == 2  # not 3
    assert inst._fast_path.stats.skip_count == 1


def test_steady_state_alternates_expert_skip(fixture_inference):
    """With max_skips=3 + similar inputs, consume_skip decrements after
    each skip → counter stays at ≤1 → alternating pattern from call 3 on:
    expert / expert / skip / expert / skip / expert / skip ..."""
    inst = fixture_inference
    _call_count_run_expert(inst, action_value=0.0)

    for _ in range(7):
        inst.predict_action_chunk(**_predict_args())

    # Expected: calls 1+2 expert, 3 skip, 4 expert, 5 skip, 6 expert, 7 skip
    # → 4 expert calls, 3 skips
    assert inst._run_expert_calls == 4
    assert inst._fast_path.stats.skip_count == 3


def test_dissimilar_chunk_forces_expert(fixture_inference):
    """If the new expert output is far from the cached one, the fast path
    resets and the next call still runs expert (no skip queued)."""
    inst = fixture_inference

    # First two similar calls queue a skip
    _call_count_run_expert(inst, action_value=0.0)
    inst.predict_action_chunk(**_predict_args())  # expert
    inst.predict_action_chunk(**_predict_args())  # expert + queue skip

    # Now switch the expert to produce wildly different output
    _call_count_run_expert(inst, action_value=100.0)
    # NOTE: this resets _run_expert_calls counter via the new fixture
    inst.predict_action_chunk(**_predict_args())  # SHOULD skip first (uses cache from prev)
    # That returned the cached zero-chunk (skip path)
    inst.predict_action_chunk(**_predict_args())  # expert runs new value, observe → big distance, reset
    inst.predict_action_chunk(**_predict_args())  # expert runs (no skip queued after reset)

    # New _run_expert was called only on the 2 expert paths
    assert inst._run_expert_calls == 2


def test_reset_cache_flushes_fast_path(fixture_inference):
    """reset_cache() between episodes must clear the fast-path so episode
    N's actions don't bleed into episode N+1."""
    inst = fixture_inference
    _call_count_run_expert(inst, action_value=0.0)

    # Two similar calls queue a skip
    inst.predict_action_chunk(**_predict_args())
    inst.predict_action_chunk(**_predict_args())
    assert inst._fast_path.should_skip() is True

    inst.reset_cache()

    assert inst._fast_path.should_skip() is False
    assert inst._fast_path.cached_actions() is None
    # After reset, the next call must run expert (no cached state)
    _call_count_run_expert(inst, action_value=0.0)
    inst.predict_action_chunk(**_predict_args())
    assert inst._run_expert_calls == 1


def test_disabled_fast_path_always_runs_expert(fixture_inference):
    """threshold=0.0 → ActionFastPath constructed with enabled=False; no
    skip ever fires; expert runs every call."""
    inst = fixture_inference
    inst._fast_path = ActionFastPath(threshold=0.0, max_skips=3, enabled=False)
    _call_count_run_expert(inst, action_value=0.0)

    for _ in range(5):
        inst.predict_action_chunk(**_predict_args())

    assert inst._run_expert_calls == 5
    assert inst._fast_path.stats.skip_count == 0


def test_cached_actions_returned_are_copies(fixture_inference):
    """Per ActionFastPath.cached_actions docstring: returned array must be
    a copy. Mutating the returned chunk must not poison the next skip."""
    inst = fixture_inference
    _call_count_run_expert(inst, action_value=0.0)

    inst.predict_action_chunk(**_predict_args())
    inst.predict_action_chunk(**_predict_args())
    cached_first = inst.predict_action_chunk(**_predict_args())  # skip path
    # Caller mutates the returned chunk
    cached_first[0, 0, 0] = 999.0

    # Trigger another expert call so the next observe sees clean cache
    inst.predict_action_chunk(**_predict_args())  # expert + queue skip
    cached_second = inst.predict_action_chunk(**_predict_args())  # skip again
    # Mutation on cached_first must NOT have poisoned the cache
    assert cached_second[0, 0, 0] == 0.0


def test_metric_emitted_on_skip(fixture_inference, monkeypatch):
    """Verify the prometheus inc_action_skip is called on the skip path
    (defensive import — the production code wraps in try/except so missing
    prometheus dep doesn't crash unit tests)."""
    inst = fixture_inference
    _call_count_run_expert(inst, action_value=0.0)

    skip_count = {"n": 0}
    def fake_inc():
        skip_count["n"] += 1

    monkeypatch.setattr(
        "reflex.observability.prometheus.inc_action_skip",
        fake_inc,
        raising=False,
    )

    inst.predict_action_chunk(**_predict_args())  # expert
    inst.predict_action_chunk(**_predict_args())  # expert + queue
    inst.predict_action_chunk(**_predict_args())  # skip → emits metric

    assert skip_count["n"] == 1
