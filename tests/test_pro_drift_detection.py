"""Tests for src/tether/pro/drift_detection.py — Phase 1 Day 10."""
from __future__ import annotations

import random

import pytest

from tether.pro.drift_detection import (
    DEFAULT_ACTION_WASSERSTEIN_MAX,
    DEFAULT_KL_DIVERGENCE_MAX,
    MIN_SAMPLES_FOR_DRIFT,
    DriftDetector,
    DriftReport,
    JointDriftScore,
    symmetric_kl_divergence,
    wasserstein_1d_simple,
)


# ---------------------------------------------------------------------------
# DriftDetector construction validation
# ---------------------------------------------------------------------------


def test_detector_rejects_zero_kl_max():
    with pytest.raises(ValueError, match="kl_divergence_max"):
        DriftDetector(kl_divergence_max=0)


def test_detector_rejects_zero_action_wasserstein_max():
    with pytest.raises(ValueError, match="action_wasserstein_max"):
        DriftDetector(action_wasserstein_max=0)


def test_detector_rejects_one_bin():
    with pytest.raises(ValueError, match="histogram_bins"):
        DriftDetector(histogram_bins=1)


def test_detector_rejects_min_samples_below_ten():
    with pytest.raises(ValueError, match="min_samples"):
        DriftDetector(min_samples=5)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def test_kl_zero_for_identical_samples():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert symmetric_kl_divergence(a, a) < 1e-9


def test_kl_zero_for_empty():
    assert symmetric_kl_divergence([], [1.0, 2.0]) == 0.0
    assert symmetric_kl_divergence([1.0], []) == 0.0


def test_kl_positive_for_disjoint_distributions():
    """Two non-overlapping distributions → positive KL. Laplace smoothing
    softens the magnitude; threshold is conservative."""
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [10.0, 11.0, 12.0, 13.0, 14.0]
    assert symmetric_kl_divergence(a, b) > 0.05


def test_kl_zero_when_all_samples_collapsed():
    """All same value in both → zero divergence."""
    a = [5.0] * 100
    b = [5.0] * 100
    assert symmetric_kl_divergence(a, b) == 0.0


def test_wasserstein_zero_for_identical():
    a = [1.0, 2.0, 3.0]
    assert wasserstein_1d_simple(a, a) == 0.0


def test_wasserstein_zero_for_empty():
    assert wasserstein_1d_simple([], [1.0]) == 0.0


def test_wasserstein_positive_for_disjoint():
    w = wasserstein_1d_simple([0.0, 1.0], [10.0, 11.0])
    assert w > 5.0


# ---------------------------------------------------------------------------
# DriftDetector.evaluate — happy + sad paths
# ---------------------------------------------------------------------------


def _matched_samples(n: int, n_joints: int = 7) -> tuple[list[list[float]], list[list[float]]]:
    """Generate identical customer + base sample lists."""
    rng = random.Random(42)
    states = [[rng.random() for _ in range(n_joints)] for _ in range(n)]
    actions = [[rng.random() for _ in range(n_joints)] for _ in range(n)]
    # Same data for both
    return states, actions


def test_evaluate_returns_insufficient_when_under_min_samples():
    detector = DriftDetector(min_samples=100)
    cust_states = [[0.1, 0.2]] * 50
    base_states = [[0.1, 0.2]] * 50
    cust_actions = [[0.1, 0.2]] * 50
    base_actions = [[0.1, 0.2]] * 50
    report = detector.evaluate(
        customer_states=cust_states, base_states=base_states,
        customer_actions=cust_actions, base_actions=base_actions,
    )
    assert not report.drift_detected
    assert report.reason == "insufficient-samples"


def test_evaluate_no_drift_for_identical_distributions():
    detector = DriftDetector()
    states, actions = _matched_samples(n=200, n_joints=7)
    report = detector.evaluate(
        customer_states=states, base_states=states,
        customer_actions=actions, base_actions=actions,
    )
    assert not report.drift_detected
    assert report.reason == "ok"
    assert report.worst_joint_index == -1


def test_evaluate_detects_kl_drift_on_state_distribution():
    """Customer state distribution shifted by +10 — large KL."""
    detector = DriftDetector(kl_divergence_max=0.1, action_wasserstein_max=10.0)
    rng = random.Random(0)
    base_states = [[rng.random() for _ in range(7)] for _ in range(200)]
    # Customer states shifted to a totally different range
    cust_states = [[10 + rng.random() for _ in range(7)] for _ in range(200)]
    actions = [[rng.random() for _ in range(7)] for _ in range(200)]
    report = detector.evaluate(
        customer_states=cust_states, base_states=base_states,
        customer_actions=actions, base_actions=actions,
    )
    assert report.drift_detected
    assert report.reason == "kl-exceeded"
    assert report.worst_joint_index >= 0


def test_evaluate_detects_action_drift_via_wasserstein():
    """Customer actions diverged → Wasserstein fires while KL stays low."""
    detector = DriftDetector(kl_divergence_max=10.0, action_wasserstein_max=0.1)
    rng = random.Random(0)
    states = [[rng.random() for _ in range(7)] for _ in range(200)]
    base_actions = [[rng.random() for _ in range(7)] for _ in range(200)]
    cust_actions = [[10 + rng.random() for _ in range(7)] for _ in range(200)]
    report = detector.evaluate(
        customer_states=states, base_states=states,
        customer_actions=cust_actions, base_actions=base_actions,
    )
    assert report.drift_detected
    assert report.reason == "action-exceeded"


def test_evaluate_per_joint_scores_one_per_joint():
    detector = DriftDetector()
    states, actions = _matched_samples(n=200, n_joints=7)
    report = detector.evaluate(
        customer_states=states, base_states=states,
        customer_actions=actions, base_actions=actions,
    )
    assert len(report.per_joint_scores) == 7
    for i, score in enumerate(report.per_joint_scores):
        assert score.joint_index == i


def test_evaluate_max_kl_property_returns_largest_per_joint():
    detector = DriftDetector()
    states, actions = _matched_samples(n=200, n_joints=7)
    report = detector.evaluate(
        customer_states=states, base_states=states,
        customer_actions=actions, base_actions=actions,
    )
    expected_max = max(s.kl_divergence for s in report.per_joint_scores)
    assert report.max_kl == expected_max


def test_evaluate_handles_empty_inputs():
    detector = DriftDetector()
    report = detector.evaluate(
        customer_states=[], base_states=[],
        customer_actions=[], base_actions=[],
    )
    assert not report.drift_detected
    assert report.reason == "insufficient-samples"


# ---------------------------------------------------------------------------
# JointDriftScore + DriftReport shapes
# ---------------------------------------------------------------------------


def test_joint_drift_score_drift_score_is_max():
    s = JointDriftScore(joint_index=0, kl_divergence=0.3, action_wasserstein=0.7)
    assert s.drift_score == 0.7
    s = JointDriftScore(joint_index=0, kl_divergence=0.9, action_wasserstein=0.4)
    assert s.drift_score == 0.9


def test_drift_report_is_frozen():
    report = DriftReport(
        drift_detected=False, reason="ok",
        n_customer_samples=100, n_base_samples=100,
        per_joint_scores=(),
        worst_joint_index=-1, worst_joint_score=0.0,
        threshold=0.5,
    )
    with pytest.raises(AttributeError):
        report.drift_detected = True  # type: ignore[misc]


def test_joint_drift_score_is_frozen():
    s = JointDriftScore(joint_index=0, kl_divergence=0.1, action_wasserstein=0.2)
    with pytest.raises(AttributeError):
        s.kl_divergence = 0.5  # type: ignore[misc]
