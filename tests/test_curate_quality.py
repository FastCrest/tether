"""Tests for src/reflex/curate/quality/ — per-episode quality scoring.

Covers each signal independently + composite + per-embodiment weights +
ranking property (high-quality episodes should outscore low-quality ones).
"""
from __future__ import annotations

import numpy as np
import pytest

from reflex.curate.quality import (
    DEFAULT_WEIGHTS,
    EMBODIMENT_WEIGHTS,
    QualityResult,
    QualityWeights,
    coverage_signal,
    efficiency_signal,
    policy_smoothness_signal,
    quality_score,
    success_signal,
    weights_for,
)
from reflex.curate.quality.baseline import baseline_median


# ── baseline_median ──────────────────────────────────────────────────────────


def test_baseline_median_exact_match() -> None:
    assert baseline_median("franka", "pick") == 120


def test_baseline_median_embodiment_wildcard_fallback() -> None:
    # franka has a "*" entry → 200
    assert baseline_median("franka", "unseen_task") == 200


def test_baseline_median_default_when_unknown_embodiment() -> None:
    assert baseline_median("ufo_arm", "pick") == 200  # default


# ── success_signal ───────────────────────────────────────────────────────────


def test_success_signal_explicit_success_high() -> None:
    actions = np.zeros((50, 3))
    actions[:, 0] = np.linspace(0, 1, 50)
    s = success_signal(success_flag=True, actions=actions)
    assert s > 0.5


def test_success_signal_explicit_failure_zero() -> None:
    actions = np.random.uniform(-1, 1, (50, 3))
    s = success_signal(success_flag=False, actions=actions)
    assert s == 0.0


def test_success_signal_no_flag_redistributes() -> None:
    """When success_flag=None, the score should be in [0, 1] from the
    other two sub-signals."""
    actions = np.zeros((50, 3))
    actions[:, 0] = np.linspace(0, 1, 50)
    s = success_signal(success_flag=None, actions=actions)
    assert 0.0 <= s <= 1.0


def test_success_signal_empty_actions() -> None:
    actions = np.zeros((0, 3))
    s = success_signal(success_flag=True, actions=actions)
    assert s == 0.0


# ── policy_smoothness_signal ─────────────────────────────────────────────────


def test_smoothness_smooth_trajectory_high() -> None:
    """A clean linear ramp should score near 1.0."""
    T = 100
    actions = np.linspace(0, 1, T).reshape(T, 1) + np.random.normal(0, 0.001, (T, 1))
    s = policy_smoothness_signal(actions=actions, execute_hz=50.0)
    assert s > 0.7


def test_smoothness_jittery_trajectory_low() -> None:
    """Pure noise should score near 0."""
    np.random.seed(42)
    actions = np.random.uniform(-1, 1, (200, 6))
    s = policy_smoothness_signal(actions=actions, execute_hz=50.0)
    assert s < 0.3


def test_smoothness_skips_gripper_dims() -> None:
    """A perfect trajectory plus a binary-toggling gripper dim should still
    score high if the gripper dim is excluded."""
    T = 100
    actions = np.zeros((T, 7))
    actions[:, :6] = np.linspace(0, 1, T).reshape(T, 1)
    # Gripper toggles wildly
    actions[:, 6] = (-1.0) ** np.arange(T)
    no_skip = policy_smoothness_signal(actions=actions, execute_hz=50.0)
    skip = policy_smoothness_signal(actions=actions, execute_hz=50.0, gripper_dims=[6])
    assert skip > no_skip
    assert skip > 0.5


def test_smoothness_within_chunk_only() -> None:
    """When chunk_ids is provided, the chunk-boundary discontinuity should
    NOT punish smoothness (within-chunk is what matters)."""
    chunk_size = 10
    chunks = 5
    T = chunk_size * chunks
    actions = np.zeros((T, 1))
    chunk_ids = np.zeros(T, dtype=int)
    for c in range(chunks):
        # Each chunk is a clean linear ramp from 0 to 1 (boundary jump
        # between chunks would normally tank smoothness)
        actions[c * chunk_size:(c + 1) * chunk_size, 0] = np.linspace(0, 1, chunk_size)
        chunk_ids[c * chunk_size:(c + 1) * chunk_size] = c

    no_chunks = policy_smoothness_signal(actions=actions, execute_hz=50.0)
    with_chunks = policy_smoothness_signal(
        actions=actions, chunk_ids=chunk_ids, execute_hz=50.0,
    )
    assert with_chunks > no_chunks


def test_smoothness_static_dim_does_not_break() -> None:
    """A constant-value dim shouldn't divide-by-zero; F3.6 range floor."""
    T = 50
    actions = np.zeros((T, 3))
    actions[:, 0] = np.linspace(0, 1, T)
    # Channel 1 + 2 are constant
    s = policy_smoothness_signal(actions=actions, execute_hz=50.0)
    assert 0.0 <= s <= 1.0
    assert not np.isnan(s)


def test_smoothness_too_short_returns_zero() -> None:
    actions = np.zeros((1, 3))
    s = policy_smoothness_signal(actions=actions, execute_hz=50.0)
    assert s == 0.0


# ── efficiency_signal ────────────────────────────────────────────────────────


def test_efficiency_at_baseline_full_score() -> None:
    s = efficiency_signal(episode_steps=120, embodiment="franka", task="pick")
    assert s == 1.0


def test_efficiency_short_episode_full_score() -> None:
    s = efficiency_signal(episode_steps=50, embodiment="franka", task="pick")
    assert s == 1.0


def test_efficiency_long_episode_decays() -> None:
    """At 4× baseline, score should be near 0."""
    median = 120  # franka pick
    s = efficiency_signal(episode_steps=int(median * 4), embodiment="franka", task="pick")
    assert s == 0.0


def test_efficiency_2x_baseline_partial() -> None:
    median = 120
    s = efficiency_signal(episode_steps=int(median * 2), embodiment="franka", task="pick")
    # 2x median → ratio=2 → score = 1 - (2-1)/3 = 2/3 ≈ 0.67
    assert 0.6 < s < 0.7


def test_efficiency_zero_steps() -> None:
    s = efficiency_signal(episode_steps=0, embodiment="franka", task="pick")
    assert s == 0.0


# ── coverage_signal ──────────────────────────────────────────────────────────


def test_coverage_first_episode_full_score() -> None:
    s = coverage_signal(episode_phash="0000000000000000", recent_phashes=())
    assert s == 1.0


def test_coverage_unique_episode_high_score() -> None:
    # All zeros vs all f's → 64-bit hamming distance is 64
    s = coverage_signal(
        episode_phash="0000000000000000",
        recent_phashes=("ffffffffffffffff",),
    )
    assert s == 1.0


def test_coverage_near_duplicate_low_score() -> None:
    # Same hash → distance 0 → score 0
    s = coverage_signal(
        episode_phash="0000000000000000",
        recent_phashes=("0000000000000000",),
    )
    assert s == 0.0


def test_coverage_no_phash_returns_neutral() -> None:
    s = coverage_signal(episode_phash=None, recent_phashes=("abc",))
    assert s == 0.5


# ── per_embodiment weights ───────────────────────────────────────────────────


def test_default_weights_sum_to_one() -> None:
    assert abs(DEFAULT_WEIGHTS.total - 1.0) < 1e-6


@pytest.mark.parametrize("emb_key", list(EMBODIMENT_WEIGHTS.keys()))
def test_all_embodiment_weights_sum_to_one(emb_key: str) -> None:
    assert abs(EMBODIMENT_WEIGHTS[emb_key].total - 1.0) < 1e-6


def test_weights_for_known_embodiment() -> None:
    w = weights_for("franka")
    assert w == DEFAULT_WEIGHTS


def test_weights_for_so100_lower_smoothness() -> None:
    """SO-100 should weight smoothness less (research finding F3.4 spec'd)."""
    w = weights_for("so100")
    assert w.smoothness < DEFAULT_WEIGHTS.smoothness


def test_weights_for_unknown_falls_back() -> None:
    assert weights_for("ufo_arm") == DEFAULT_WEIGHTS


def test_weights_for_none_falls_back() -> None:
    assert weights_for(None) == DEFAULT_WEIGHTS


def test_weights_for_loose_match_prefix() -> None:
    """franka2 / franka_panda should match franka."""
    assert weights_for("franka_panda") == DEFAULT_WEIGHTS
    assert weights_for("franka2") == DEFAULT_WEIGHTS


def test_quality_weights_negative_rejects() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        QualityWeights(success=-0.1, smoothness=0.3, efficiency=0.2, coverage=0.1)


# ── composite quality_score ──────────────────────────────────────────────────


def _make_high_quality_episode(T: int = 100) -> dict:
    actions = np.zeros((T, 7))
    actions[:, :6] = np.linspace(0, 1, T).reshape(T, 1)
    actions[:, 6] = -1.0  # gripper closed (consistent)
    chunk_ids = np.repeat(np.arange(T // 50 + 1), 50)[:T]
    return dict(
        actions=actions,
        chunk_ids=chunk_ids,
        success_flag=True,
        embodiment="franka",
        task="pick",
        execute_hz=50.0,
        gripper_dims=[6],
        episode_phash="0000000000000000",
        recent_phashes=["ffffffffffffffff"],
    )


def _make_low_quality_episode(T: int = 600) -> dict:
    np.random.seed(0)
    actions = np.random.uniform(-1, 1, (T, 6))
    chunk_ids = np.repeat(np.arange(T // 50 + 1), 50)[:T]
    return dict(
        actions=actions,
        chunk_ids=chunk_ids,
        success_flag=None,
        embodiment="so100",
        task="pick",
        execute_hz=30.0,
        gripper_dims=[5],
    )


def test_quality_score_returns_QualityResult() -> None:
    result = quality_score(**_make_high_quality_episode())
    assert isinstance(result, QualityResult)
    assert 0.0 <= result.quality_score <= 1.0
    assert "success" in result.components
    assert result.quality_version == "rule-v1"


def test_quality_score_high_quality_above_low_quality() -> None:
    """The defining property: a clean successful episode should outscore
    a noisy unsuccessful one."""
    high = quality_score(**_make_high_quality_episode())
    low = quality_score(**_make_low_quality_episode())
    assert high.quality_score > low.quality_score
    assert high.quality_score - low.quality_score > 0.5  # large discrimination


def test_quality_score_components_in_unit_range() -> None:
    result = quality_score(**_make_high_quality_episode())
    for k, v in result.components.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0, 1]"


def test_quality_score_weights_normalized() -> None:
    """Even if config supplies non-summing weights, output should be in [0, 1]."""
    custom = QualityWeights(success=2.0, smoothness=2.0, efficiency=2.0, coverage=2.0)
    result = quality_score(**_make_high_quality_episode(), weights=custom)
    weights_sum = sum(result.weights.values())
    assert abs(weights_sum - 1.0) < 1e-6
    assert 0.0 <= result.quality_score <= 1.0


def test_quality_score_to_dict_has_expected_keys() -> None:
    result = quality_score(**_make_high_quality_episode())
    d = result.to_dict()
    assert "quality_score" in d
    assert "quality_components" in d
    assert "quality_weights" in d
    assert "quality_version" in d
    assert "computed_at" in d


def test_quality_score_empty_actions_safe() -> None:
    actions = np.zeros((0, 7))
    result = quality_score(
        actions=actions,
        chunk_ids=None,
        success_flag=False,
        embodiment="franka",
        task="pick",
    )
    assert 0.0 <= result.quality_score <= 1.0


# ── ranking property ────────────────────────────────────────────────────────


def test_ranking_smoother_wins_when_other_signals_equal() -> None:
    """Hold success / efficiency / coverage equal; smoother trajectory wins."""
    T = 100
    chunk_ids = np.repeat(np.arange(T // 50 + 1), 50)[:T]
    smooth_actions = np.linspace(0, 1, T).reshape(T, 1)
    np.random.seed(1)
    noisy_actions = smooth_actions + np.random.uniform(-0.1, 0.1, (T, 1))

    common = dict(
        chunk_ids=chunk_ids, success_flag=True, embodiment="franka",
        task="pick", execute_hz=50.0, gripper_dims=[],
        episode_phash="0000000000000000", recent_phashes=["ffffffffffffffff"],
    )
    smooth_score = quality_score(actions=smooth_actions, **common).quality_score
    noisy_score = quality_score(actions=noisy_actions, **common).quality_score
    assert smooth_score > noisy_score


def test_ranking_short_episode_wins_when_other_signals_equal() -> None:
    """Hold success / smoothness / coverage equal; shorter episode wins."""
    short_ep = _make_high_quality_episode(T=80)   # below franka/pick median 120
    long_ep = _make_high_quality_episode(T=400)   # well above median
    short_score = quality_score(**short_ep).quality_score
    long_score = quality_score(**long_ep).quality_score
    assert short_score > long_score
