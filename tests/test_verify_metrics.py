"""Unit tests for the distributional + embodied parity metrics (no GPU).

Deterministic: data is drawn from a seeded RNG and ``two_sample_test`` uses a
fixed internal permutation seed, so the permutation p-values are reproducible.
"""
from __future__ import annotations

import numpy as np

from reflex.verify_metrics import (
    EmbodiedParity,
    aggregate_embodied,
    energy_distance,
    jerk_rms,
    motion_energy,
    two_sample_test,
)


# --- distributional two-sample test ---------------------------------------


def test_mmd_false_positive_rate_is_controlled():
    # The correct, non-flaky way to validate a two-sample test: over many
    # same-distribution trials it must RARELY reject (false-positive rate near
    # alpha=0.05). Asserting one draw's p-value would flake ~5% of the time.
    trials, rejections = 16, 0
    for s in range(trials):
        rng = np.random.default_rng(100 + s)
        X = rng.normal(0, 1, size=(60, 5))
        Y = rng.normal(0, 1, size=(60, 5))  # same distribution
        if two_sample_test(X, Y, n_permutations=150, seed=s).distributions_differ:
            rejections += 1
    # Expected ~0.8 under the null; <=4 is robust (and deterministic via seeds).
    assert rejections <= 4, f"{rejections}/{trials} false positives — calibration off"


def test_mmd_shifted_distribution_is_rejected():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(80, 6))
    Y = rng.normal(2.0, 1, size=(80, 6))  # mean-shifted => different
    r = two_sample_test(X, Y, n_permutations=200)
    assert r.mmd_p_value < 0.05
    assert r.distributions_differ is True
    assert r.mmd2 > 0.1


def test_shift_has_larger_mmd_and_energy_than_same():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, size=(60, 4))
    same = rng.normal(0, 1, size=(60, 4))
    shifted = rng.normal(1.5, 1, size=(60, 4))
    r_same = two_sample_test(base, same, n_permutations=200)
    r_shift = two_sample_test(base, shifted, n_permutations=200)
    assert r_shift.mmd2 > r_same.mmd2
    assert r_shift.energy_distance > r_same.energy_distance
    assert r_shift.mmd_p_value < r_same.mmd_p_value


def test_two_sample_test_handles_tiny_samples():
    r = two_sample_test([[0.0, 0.0]], [[1.0, 1.0]], n_permutations=50)
    assert r.mmd_p_value == 1.0  # not enough data to reject
    assert r.distributions_differ is False
    assert r.n_permutations == 0


# --- episode-block permutation (autocorrelated trajectory data) ------------
# Per-step robot actions are autocorrelated within an episode. The i.i.d.
# step-level permutation treats correlated steps as independent and over-rejects
# (~100% false-positive on identical policies — caught on GPU 2026-06-01,
# ap-v8SDPPrHUju26ozqqdmiSR). Passing episode ids makes the test permute whole
# episodes, restoring calibration with full power. These lock that fix.


def _ar1_episode(rng, *, steps, dim=4, mu=0.0, phi=0.9, eps=0.05):
    """One autocorrelated AR(1) trajectory (smooth robot-motion analogue)."""
    x = np.zeros((steps, dim))
    x[0] = mu + rng.normal(0, eps, dim)
    for t in range(1, steps):
        x[t] = mu + phi * (x[t - 1] - mu) + rng.normal(0, eps, dim)
    return x


def _episodic_arm(rng, *, n_eps=4, steps=110, **kw):
    """(pooled_rows, episode_id_per_row) for one arm."""
    rows, groups = [], []
    for e in range(n_eps):
        rows.append(_ar1_episode(rng, steps=steps, **kw))
        groups += [e] * steps
    return np.vstack(rows), np.asarray(groups)


def test_episode_block_permutation_calibrated_but_step_level_over_rejects():
    trials, block_rej, step_rej = 12, 0, 0
    for s in range(trials):
        rng = np.random.default_rng(500 + s)
        Xb, Xg = _episodic_arm(rng)
        Yb, Yg = _episodic_arm(rng)  # SAME generating process => null is true
        if two_sample_test(
            Xb, Yb, n_permutations=120, seed=s,
            baseline_groups=Xg, candidate_groups=Yg,
        ).distributions_differ:
            block_rej += 1
        if two_sample_test(Xb, Yb, n_permutations=120, seed=s).distributions_differ:
            step_rej += 1
    # Episode-block stays calibrated on the null...
    assert block_rej <= 3, f"episode-block FPR too high: {block_rej}/{trials}"
    # ...while step-level over-rejects badly (this is why groups are mandatory).
    assert step_rej >= 9, (
        f"expected step-level to over-reject autocorrelated null, got {step_rej}/{trials}"
    )


def test_episode_block_permutation_detects_real_shift():
    rng = np.random.default_rng(7)
    Xb, Xg = _episodic_arm(rng, mu=0.0)
    Yb, Yg = _episodic_arm(rng, mu=0.4)  # real mean shift >> within-arm drift
    r = two_sample_test(
        Xb, Yb, n_permutations=200, seed=1,
        baseline_groups=Xg, candidate_groups=Yg,
    )
    assert r.distributions_differ is True  # fix keeps power, isn't blind


def test_two_sample_degenerate_grouping_does_not_fabricate_significance():
    # One episode per arm => no valid episode-level permutation; must NOT reject.
    rng = np.random.default_rng(0)
    Xb = _ar1_episode(rng, steps=120, mu=0.0)
    Yb = _ar1_episode(rng, steps=120, mu=5.0)  # wildly different, but 1 ep each
    r = two_sample_test(
        Xb, Yb, n_permutations=120,
        baseline_groups=np.zeros(120), candidate_groups=np.zeros(120),
    )
    assert r.distributions_differ is False
    assert r.mmd_p_value == 1.0


def test_block_aggregation_matches_full_kernel_mmd():
    # The episode-block path computes the observed MMD^2 from (U x U) block sums;
    # it must EXACTLY equal the full (N x N) kernel MMD^2 for the same split.
    import reflex.verify_metrics as vm

    rng = np.random.default_rng(3)
    Xb, Xg = _episodic_arm(rng, n_eps=4, steps=40)
    Yb, Yg = _episodic_arm(rng, n_eps=4, steps=40)
    r = vm.two_sample_test(
        Xb, Yb, n_permutations=10, baseline_groups=Xg, candidate_groups=Yg,
    )
    pooled = np.vstack([vm._as_2d(Xb), vm._as_2d(Yb)])
    med = vm._median_bandwidth(pooled)
    g = 1.0 / (2.0 * med)
    gammas = [g * s for s in (0.5, 1.0, 2.0)]
    K = vm._pooled_kernel(pooled, gammas)
    ref = vm._mmd2_from_kernel(
        K, np.arange(Xb.shape[0]), np.arange(Xb.shape[0], pooled.shape[0]), 3
    )
    assert abs(r.mmd2 - ref) < 1e-9, f"block {r.mmd2} != full-kernel {ref}"


def test_block_path_never_builds_full_kernel(monkeypatch):
    # The scaling guarantee: the groups path must NEVER materialize the full
    # (N, N) kernel (that is the ~2.6 GB OOM risk at the N>=30 floor). Booby-trap
    # _pooled_kernel; if the block path completes, it never touched it.
    import reflex.verify_metrics as vm

    def _boom(*_a, **_k):
        raise AssertionError("groups path must not build the full (N,N) kernel")

    monkeypatch.setattr(vm, "_pooled_kernel", _boom)
    rng = np.random.default_rng(0)
    Xb, Xg = _episodic_arm(rng, n_eps=10, steps=200)
    Yb, Yg = _episodic_arm(rng, n_eps=10, steps=200)  # ~4k rows: full kernel would be big
    r = vm.two_sample_test(
        Xb, Yb, n_permutations=50, baseline_groups=Xg, candidate_groups=Yg,
    )
    assert r.n_permutations == 50  # completed via block aggregation


def test_energy_distance_zero_for_identical_set():
    X = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    assert energy_distance(X, X) == 0.0


# --- embodied / kinematic metrics -----------------------------------------


def test_jerk_zero_for_linear_motion():
    # Constant velocity => position is linear in t => 3rd derivative is 0.
    t = np.arange(20).reshape(-1, 1)
    positions = np.hstack([t * 0.1, t * -0.2, t * 0.05])
    assert jerk_rms(positions, dt=1.0) < 1e-9


def test_jerk_positive_for_jittery_motion():
    rng = np.random.default_rng(2)
    positions = np.cumsum(rng.normal(0, 1, size=(30, 3)), axis=0)
    assert jerk_rms(positions, dt=1.0) > 0.0


def test_jerk_too_short_returns_zero():
    assert jerk_rms([[0.0], [1.0], [2.0]], dt=1.0) == 0.0  # need >= 4 steps


def test_motion_energy_sum_of_squares():
    v = np.ones((10, 3))
    assert motion_energy(v) == 30.0
    assert motion_energy(np.zeros((5, 4))) == 0.0


def test_embodied_regression_detected():
    smooth = [np.cumsum(np.full((30, 3), 0.1), axis=0)]  # linear => low jerk
    rng = np.random.default_rng(3)
    jittery = [np.cumsum(rng.normal(0, 1, size=(30, 3)), axis=0)]  # high jerk
    parity = aggregate_embodied(
        baseline_positions=smooth,
        candidate_positions=jittery,
        baseline_velocities=[np.full((30, 3), 0.1)],
        candidate_velocities=[rng.normal(0, 1, size=(30, 3))],
        baseline_completion_steps=[100.0],
        candidate_completion_steps=[105.0],
    )
    assert parity.candidate_jerk_rms > parity.baseline_jerk_rms
    assert parity.regressed() is True


def test_embodied_no_regression_for_identical():
    pos = [np.cumsum(np.full((30, 3), 0.1), axis=0)]
    vel = [np.full((30, 3), 0.1)]
    parity = aggregate_embodied(
        baseline_positions=pos,
        candidate_positions=pos,
        baseline_velocities=vel,
        candidate_velocities=vel,
        baseline_completion_steps=[100.0],
        candidate_completion_steps=[100.0],
    )
    assert parity.regressed() is False
    assert isinstance(parity, EmbodiedParity)
