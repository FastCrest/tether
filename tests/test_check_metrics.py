"""Unit tests for the action-parity metric engine (reflex.check.metrics).

No GPU / Modal — validates the distributional two-sample tests + embodied metrics
on synthetic data, including the known-good vs known-bad discrimination the Modal
spike was meant to confirm (here as a fast, deterministic unit test).
"""
from __future__ import annotations

import numpy as np

from reflex.check.metrics import (
    ParityVerdict,
    binned_kl,
    compute_parity,
    embodied_metrics,
    energy_distance,
    jerk_rms,
    mmd2_rbf,
    motion_energy,
    path_length,
    two_sample_test,
)

DIM = 7
N = 150


def _samples(seed: int, shift: float = 0.0, std: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=shift, scale=std, size=(N, DIM))


# --------------------------------------------------------------------------- #
# distributional statistics                                                   #
# --------------------------------------------------------------------------- #


def test_statistics_order_same_below_shifted():
    """Same-distribution sets score far lower than mean-shifted ones, for all 3 metrics."""
    X = _samples(0)
    Y_same = _samples(1)        # same N(0,1), independent draw
    Y_shift = _samples(2, shift=0.7)
    for stat in (mmd2_rbf, energy_distance, binned_kl):
        s_same = stat(X, Y_same)
        s_shift = stat(X, Y_shift)
        # The UNBIASED MMD^2 estimator can dip slightly below 0 when X and Y share
        # a distribution (it's an unbiased estimator of a non-negative quantity) —
        # that's correct and is exactly what makes the permutation null valid.
        # energy_distance and binned_kl are non-negative by construction.
        if stat is not mmd2_rbf:
            assert s_same >= 0.0
        assert s_shift > s_same, f"{stat.__name__}: same={s_same:.4f} not < shift={s_shift:.4f}"


def test_two_sample_detects_a_real_shift():
    """A 0.7 mean shift is rejected at alpha=0.05 (power)."""
    res = two_sample_test(_samples(0), _samples(2, shift=0.7), metric="mmd", n_perm=200, seed=0)
    assert res.p_value < 0.05
    assert res.metric == "mmd" and res.statistic > 0.0
    assert res.to_dict()["p_value"] == res.p_value


def test_two_sample_fpr_is_controlled():
    """On truly same-distribution data the test must NOT systematically reject (FPR ~ alpha)."""
    rejections = 0
    trials = 20
    for s in range(trials):
        X = _samples(100 + s)
        Y = _samples(500 + s)  # same dist, different draw
        if two_sample_test(X, Y, metric="mmd", n_perm=120, seed=s).p_value < 0.05:
            rejections += 1
    # expected ~1 (5% of 20); generous bound catches an always-reject bug.
    assert rejections <= 5, f"false-positive rate too high: {rejections}/{trials}"


# --------------------------------------------------------------------------- #
# embodied-quality metrics                                                    #
# --------------------------------------------------------------------------- #


def _smooth_traj() -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, 60)
    return np.stack([np.sin(t + k) for k in range(DIM)], axis=1)


def test_embodied_jitter_raises_jerk():
    smooth = _smooth_traj()
    rng = np.random.default_rng(0)
    jittery = smooth + rng.normal(0, 0.05, smooth.shape)
    assert jerk_rms(jittery) > jerk_rms(smooth) > 0.0
    assert motion_energy(jittery) > 0.0
    assert path_length(jittery) > 0.0
    assert jerk_rms(smooth[:3]) == 0.0  # too short -> defined as 0


def test_embodied_metrics_keys():
    m = embodied_metrics(_smooth_traj())
    assert set(m) == {"jerk_rms", "motion_energy", "path_length"}


# --------------------------------------------------------------------------- #
# the verdict: known-good passes, known-bad fails                             #
# --------------------------------------------------------------------------- #


def test_compute_parity_known_good_passes():
    """A near-identical optimized distribution (orig + tiny noise) passes."""
    X = _samples(0)
    rng = np.random.default_rng(7)
    opt_good = X + rng.normal(0, 0.005, X.shape)  # a faithful optimization
    v = compute_parity(X, opt_good, metric="mmd", alpha=0.05, n_perm=200, seed=0)
    assert isinstance(v, ParityVerdict)
    assert v.passed is True
    assert v.p_value >= 0.05
    assert "holds" in v.reason


def test_compute_parity_known_bad_fails():
    """A shifted optimized distribution (a broken quantization) fails on the distribution gate."""
    X = _samples(0)
    opt_bad = _samples(3, shift=0.7)
    v = compute_parity(X, opt_bad, metric="mmd", alpha=0.05, n_perm=200, seed=0)
    assert v.passed is False
    assert v.p_value < 0.05
    assert "differs" in v.reason


def test_compute_parity_flags_embodied_regression():
    """Even with a matching action distribution, a materially jerkier trajectory fails."""
    X = _samples(0)
    rng = np.random.default_rng(7)
    opt_good = X + rng.normal(0, 0.005, X.shape)
    smooth = _smooth_traj()
    jerky = smooth + rng.normal(0, 0.2, smooth.shape)  # much jerkier
    v = compute_parity(
        X, opt_good, metric="mmd", n_perm=200, seed=0, orig_traj=smooth, opt_traj=jerky
    )
    assert v.embodied_flag is True
    assert v.passed is False
    assert v.embodied_delta["jerk_rms"] > 0.0


def test_compute_parity_verdict_serializable():
    v = compute_parity(_samples(0), _samples(1), metric="energy", n_perm=50, seed=0)
    d = v.to_dict()
    assert d["metric"] == "energy"
    assert isinstance(d["n_samples"], list) and len(d["n_samples"]) == 2
