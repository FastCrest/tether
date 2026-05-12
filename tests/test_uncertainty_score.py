"""Per-episode uncertainty scoring (data-labeling-pipeline subsystem 3).

Covers:
  - Bit-identical samples → uncertainty 0
  - Maximally-divergent samples → uncertainty near 1
  - Constant-action dims contribute 0 (no range)
  - Components dict surfaces the right per-step / per-dim breakdown
  - argmax_step + argmax_dim point at the actual most-uncertain location
  - Input validation: shape, n_samples >= 2
  - 4-quadrant classifier maps (uncertainty, success) → training value
"""
from __future__ import annotations

import numpy as np
import pytest

from reflex.curate.quality.uncertainty import (
    UNCERTAINTY_VERSION,
    UncertaintyResult,
    classify_episode_value,
    uncertainty_score,
)


def test_identical_samples_zero_uncertainty():
    """N copies of the same chunk → variance is 0 → uncertainty is 0."""
    chunk = np.random.RandomState(0).randn(50, 7).astype(np.float32)
    samples = np.stack([chunk] * 5, axis=0)  # (5, 50, 7)
    result = uncertainty_score(samples=samples)
    assert result.uncertainty_score == 0.0
    assert result.n_samples == 5
    assert result.n_steps == 50
    assert result.n_action_dims == 7


def test_maximally_divergent_samples_high_uncertainty():
    """Samples spanning [-1, +1] uniformly → variance near max → score near 1."""
    rng = np.random.RandomState(42)
    # Uniform samples in [-1, +1] across N=50 samples for stable variance
    samples = rng.uniform(-1.0, 1.0, size=(50, 10, 4)).astype(np.float32)
    result = uncertainty_score(samples=samples)
    # Uniform in [-1, 1] has var = 4/12 = 1/3, range = 2, normalized var = (1/3) / 1 = 1/3
    # Per-dim normalized variance for uniform is 1/3, so the score lands ~0.33
    assert 0.25 < result.uncertainty_score < 0.40
    # Score is in [0, 1]
    assert 0.0 <= result.uncertainty_score <= 1.0


def test_constant_action_dim_contributes_zero():
    """A dim that's constant across N samples adds zero uncertainty."""
    rng = np.random.RandomState(0)
    # 5 samples, 10 steps, 3 dims:
    # dim 0: random (variable)
    # dim 1: constant (zero variance)
    # dim 2: random (variable)
    samples = rng.randn(5, 10, 3).astype(np.float32)
    samples[:, :, 1] = 0.5  # dim 1 constant
    result = uncertainty_score(samples=samples)
    # The constant dim doesn't push the score up; it should be roughly
    # 2/3 of what a fully-variable score would be
    assert result.uncertainty_score > 0  # the variable dims still contribute
    # Components surface argmax_dim — should NOT be the constant dim
    assert result.components["argmax_dim"] != 1


def test_score_increases_with_variance():
    """Higher cross-sample spread → higher uncertainty score."""
    rng = np.random.RandomState(0)
    base_chunk = rng.randn(50, 7).astype(np.float32)
    # Tight cluster: small per-sample noise
    tight_samples = np.stack([
        base_chunk + 0.01 * rng.randn(50, 7).astype(np.float32)
        for _ in range(5)
    ], axis=0)
    # Wider cluster
    wide_samples = np.stack([
        base_chunk + 0.5 * rng.randn(50, 7).astype(np.float32)
        for _ in range(5)
    ], axis=0)
    tight = uncertainty_score(samples=tight_samples)
    wide = uncertainty_score(samples=wide_samples)
    assert wide.uncertainty_score > tight.uncertainty_score


def test_argmax_step_points_at_most_uncertain_step():
    """One specific step has high cross-sample variance; argmax_step finds it."""
    rng = np.random.RandomState(0)
    n_samples, n_steps, n_dims = 5, 50, 7
    samples = np.zeros((n_samples, n_steps, n_dims), dtype=np.float32)
    # Step 23 has high variance, others are zero
    samples[:, 23, :] = rng.randn(n_samples, n_dims) * 2.0
    result = uncertainty_score(samples=samples)
    assert result.components["argmax_step"] == 23


def test_argmax_dim_points_at_most_variable_dim():
    """One specific action dim has high cross-sample variance; argmax_dim finds it."""
    rng = np.random.RandomState(0)
    n_samples, n_steps, n_dims = 5, 50, 7
    samples = np.zeros((n_samples, n_steps, n_dims), dtype=np.float32)
    # Dim 4 has high variance across all steps
    samples[:, :, 4] = rng.randn(n_samples, n_steps) * 2.0
    result = uncertainty_score(samples=samples)
    assert result.components["argmax_dim"] == 4


def test_components_dict_complete():
    """All expected component keys present."""
    rng = np.random.RandomState(0)
    samples = rng.randn(5, 50, 7).astype(np.float32)
    result = uncertainty_score(samples=samples)
    expected = {
        "mean_per_step_uncertainty",
        "max_per_step_uncertainty",
        "mean_per_dim_uncertainty",
        "max_per_dim_uncertainty",
        "argmax_step",
        "argmax_dim",
    }
    assert set(result.components.keys()) == expected


def test_to_dict_serializes_for_parquet():
    """to_dict produces a flat-ish dict suitable for parquet metadata storage."""
    samples = np.random.RandomState(0).randn(5, 50, 7).astype(np.float32)
    result = uncertainty_score(samples=samples)
    d = result.to_dict()
    assert "uncertainty_score" in d
    assert "uncertainty_components" in d
    assert d["uncertainty_n_samples"] == 5
    assert d["uncertainty_n_steps"] == 50
    assert d["uncertainty_n_action_dims"] == 7
    assert d["uncertainty_version"] == UNCERTAINTY_VERSION
    # Ensure components dict is a dict (not the original reference)
    assert isinstance(d["uncertainty_components"], dict)


def test_invalid_shape_raises():
    """2D samples (T, A) should raise — needs explicit (N, T, A)."""
    with pytest.raises(ValueError, match="must be 3D"):
        uncertainty_score(samples=np.zeros((50, 7), dtype=np.float32))


def test_too_few_samples_raises():
    """N=1 has no variance defined."""
    with pytest.raises(ValueError, match="≥2 samples"):
        uncertainty_score(samples=np.zeros((1, 50, 7), dtype=np.float32))


def test_score_bounded_in_unit_interval():
    """Even with extreme inputs, score stays in [0, 1]."""
    rng = np.random.RandomState(0)
    # Extreme: huge values, fully random
    samples = rng.uniform(-1000, 1000, size=(20, 30, 5)).astype(np.float32)
    result = uncertainty_score(samples=samples)
    assert 0.0 <= result.uncertainty_score <= 1.0


def test_classify_episode_value_high_success():
    """high uncertainty + success → informative_edge_case."""
    samples = np.random.RandomState(0).uniform(-1, 1, size=(10, 30, 5)).astype(np.float32)
    result = uncertainty_score(samples=samples)
    bucket = classify_episode_value(
        uncertainty=result,
        success_flag=True,
        high_uncertainty_threshold=0.0,  # force high
    )
    assert bucket == "informative_edge_case"


def test_classify_episode_value_high_failure():
    """high uncertainty + failure → edge_case_to_correct."""
    samples = np.random.RandomState(0).uniform(-1, 1, size=(10, 30, 5)).astype(np.float32)
    result = uncertainty_score(samples=samples)
    bucket = classify_episode_value(
        uncertainty=result,
        success_flag=False,
        high_uncertainty_threshold=0.0,
    )
    assert bucket == "edge_case_to_correct"


def test_classify_episode_value_low_success():
    """low uncertainty + success → redundant_known_good."""
    chunk = np.random.RandomState(0).randn(50, 7).astype(np.float32)
    samples = np.stack([chunk] * 5, axis=0)
    result = uncertainty_score(samples=samples)
    bucket = classify_episode_value(
        uncertainty=result,
        success_flag=True,
        high_uncertainty_threshold=0.5,
    )
    assert bucket == "redundant_known_good"


def test_classify_episode_value_low_failure():
    """low uncertainty + failure → model_blind_spot (the spec's worst bucket)."""
    chunk = np.random.RandomState(0).randn(50, 7).astype(np.float32)
    samples = np.stack([chunk] * 5, axis=0)
    result = uncertainty_score(samples=samples)
    bucket = classify_episode_value(
        uncertainty=result,
        success_flag=False,
        high_uncertainty_threshold=0.5,
    )
    assert bucket == "model_blind_spot"


def test_classify_episode_value_unlabeled():
    """success_flag=None → bucket suffixed with _unlabeled."""
    chunk = np.random.RandomState(0).randn(50, 7).astype(np.float32)
    samples = np.stack([chunk] * 5, axis=0)
    result = uncertainty_score(samples=samples)
    bucket = classify_episode_value(
        uncertainty=result,
        success_flag=None,
        high_uncertainty_threshold=0.5,
    )
    assert bucket == "low_uncertainty_unlabeled"


def test_version_stamped_for_back_compat():
    """uncertainty_version stamped so future format changes can be detected."""
    samples = np.random.RandomState(0).randn(5, 50, 7).astype(np.float32)
    result = uncertainty_score(samples=samples)
    assert result.uncertainty_version == UNCERTAINTY_VERSION
    assert result.uncertainty_version  # non-empty string
