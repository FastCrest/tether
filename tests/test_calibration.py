"""Unit tests for tether.eval.calibration.

Goal: calibration-metrics (weight 8). Zollo 2025 showed ECE/Brier/NLL
are the non-simulation metrics most monotonically linked to real-task
success. These tests establish the math contract — the Modal dataset
loop (to be written) will populate live numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from tether.eval.calibration import (
    compute_brier,
    compute_ece,
    compute_nll,
    discretize_action_samples,
)


class TestECE:
    def test_perfect_calibration_zero_ece(self):
        # Model always predicts 100% confident, always correct.
        probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        labels = np.array([0, 1, 0])
        r = compute_ece(probs, labels)
        assert r["ece"] == pytest.approx(0.0, abs=1e-9)

    def test_maximum_miscalibration(self):
        # Model predicts 90% confident but is always WRONG.
        probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        labels = np.array([1, 1, 1, 1])
        r = compute_ece(probs, labels)
        # Top-class confidence = 0.9; accuracy in that bin = 0.
        # ECE = |0.9 - 0| = 0.9.
        assert r["ece"] == pytest.approx(0.9, abs=1e-6)

    def test_bins_count_is_correct(self):
        probs = np.array([[0.1, 0.9]] * 10 + [[0.6, 0.4]] * 10)
        labels = np.array([1] * 10 + [0] * 10)
        r = compute_ece(probs, labels, num_bins=15)
        total = sum(r["per_bin_count"])
        assert total == 20

    def test_rejects_probs_not_summing_to_one(self):
        probs = np.array([[0.3, 0.3]])
        labels = np.array([0])
        with pytest.raises(ValueError, match="sum to 1"):
            compute_ece(probs, labels)

    def test_rejects_out_of_range_labels(self):
        probs = np.array([[0.5, 0.5]])
        labels = np.array([5])
        with pytest.raises(ValueError, match="out of range"):
            compute_ece(probs, labels)


class TestBrier:
    def test_perfect_predictions_zero_brier(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        r = compute_brier(probs, labels)
        assert r["brier"] == pytest.approx(0.0, abs=1e-9)

    def test_uniform_k_class(self):
        # Uniform over K=2: each class has p=0.5 → brier per sample =
        # (0.5-1)^2 + (0.5-0)^2 = 0.25 + 0.25 = 0.5
        probs = np.array([[0.5, 0.5]] * 10)
        labels = np.array([0] * 10)
        r = compute_brier(probs, labels)
        assert r["brier"] == pytest.approx(0.5, abs=1e-9)

    def test_upper_bound_binary(self):
        # Confident wrong → p=1 on wrong class → (1-0)^2 + (0-1)^2 = 2
        probs = np.array([[1.0, 0.0]])
        labels = np.array([1])
        r = compute_brier(probs, labels)
        assert r["brier"] == pytest.approx(2.0, abs=1e-9)

    def test_n_and_k_reported(self):
        probs = np.random.dirichlet([1, 1, 1], size=100)
        labels = np.random.randint(0, 3, size=100)
        r = compute_brier(probs, labels)
        assert r["n"] == 100
        assert r["k"] == 3


class TestNLL:
    def test_perfect_predictions_zero_nll(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        r = compute_nll(probs, labels)
        # Approximate because we clip to 1-eps to avoid log(0).
        assert r["nll"] == pytest.approx(0.0, abs=1e-9)

    def test_confident_wrong_clips_at_eps(self):
        """Without clipping, log(0) = -inf. With eps=1e-12, nll ≈ 27.6."""
        probs = np.array([[1.0, 0.0]])
        labels = np.array([1])
        r = compute_nll(probs, labels)
        assert 25.0 < r["nll"] < 30.0

    def test_uniform_yields_log_k(self):
        # Uniform over K=4 classes → p(true) = 0.25 → nll = -log(0.25) ≈ 1.386
        probs = np.array([[0.25, 0.25, 0.25, 0.25]] * 10)
        labels = np.zeros(10, dtype=int)
        r = compute_nll(probs, labels)
        assert r["nll"] == pytest.approx(np.log(4), abs=1e-6)


class TestDiscretize:
    def test_even_distribution(self):
        # 100 samples uniform over [-1, 1] with 10 bins → each bin ~0.1
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1.0, 1.0, size=10000)
        dist = discretize_action_samples(samples, num_bins=10)
        # Every bin should be close to 0.1 (uniform 10% mass)
        assert np.allclose(dist, 0.1, atol=0.02)

    def test_all_in_one_bin(self):
        samples = np.full(100, 0.5)  # 0.5 ∈ [0.4, 0.6) bin 7 of 10 over [-1,1]
        dist = discretize_action_samples(samples, num_bins=10)
        # All mass in one bin.
        assert dist.sum() == pytest.approx(1.0)
        assert (dist > 0.99).sum() == 1

    def test_clips_out_of_range(self):
        # Samples outside [-1, 1] get clipped — shouldn't raise.
        samples = np.array([-2.0, 5.0, 0.0])
        dist = discretize_action_samples(samples, num_bins=10)
        assert dist.sum() == pytest.approx(1.0)

    def test_rejects_non_1d(self):
        with pytest.raises(ValueError, match="1D"):
            discretize_action_samples(np.zeros((10, 2)), num_bins=5)


class TestIntegration:
    def test_all_three_agree_on_perfect_model(self):
        """A perfectly calibrated + correct model → ECE=0, Brier=0, NLL=0."""
        probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        labels = np.array([0, 1, 2])
        assert compute_ece(probs, labels)["ece"] == pytest.approx(0.0, abs=1e-9)
        assert compute_brier(probs, labels)["brier"] == pytest.approx(0.0, abs=1e-9)
        assert compute_nll(probs, labels)["nll"] == pytest.approx(0.0, abs=1e-9)

    def test_random_model_degrades_all_three(self):
        """A purely uniform K=10 model should have non-trivial metrics on
        all three."""
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 10, size=100)
        probs = np.full((100, 10), 0.1)

        ece = compute_ece(probs, labels)["ece"]
        brier = compute_brier(probs, labels)["brier"]
        nll = compute_nll(probs, labels)["nll"]

        # Uniform K=10 should hit all three away from zero.
        assert 0.0 <= ece < 0.15
        assert 0.8 < brier < 1.0  # 1 - 1/K = 0.9 expected
        assert nll == pytest.approx(np.log(10), abs=1e-6)
