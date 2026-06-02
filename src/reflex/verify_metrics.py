"""Distributional + embodied parity metrics for `reflex verify`.

`reflex verify` v0 scores *success-rate* parity (does the optimized export pass
the same tasks as the original). That is table stakes: an export can match
success rate while shifting the action distribution or moving in a way that
wrecks real hardware. This module adds the two deeper, non-bypassable signals
flagged in ``verify.py``:

* **Distributional parity** — a two-sample test on the paired per-step *applied*
  actions (original vs optimized). We ship two estimators:
  - **MMD** (maximum mean discrepancy) with a multi-bandwidth RBF kernel and a
    permutation-test p-value (Model Equality Testing, arXiv 2410.20247).
  - **Energy distance** as a second, kernel-free estimator.
  A *low* p-value means the optimized policy's action distribution differs from
  the original beyond sampling noise — a regression success rate hides.
  The permutation is **episode-aware** (shuffles whole episodes, not steps):
  per-step actions are autocorrelated within an episode, so an i.i.d. step
  permutation over-rejects badly (~100% false-positive on identical policies).

* **Embodied / kinematic parity** — scored per paired episode and aggregated:
  - **jerk** (RMS of the 3rd derivative of joint position): smoothness
    regressions are invisible to success rate but destroy real actuators.
  - **motion energy** (sum of squared joint velocities): energy-per-task parity.
  - **completion time**: an export that succeeds but is slower.

Everything here is pure NumPy (a core dep) and fully unit-testable on synthetic
arrays — no GPU, no rollout. ``verify.py`` feeds it the per-step trajectories
once the rollout primitive is widened to capture them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

ArrayLike = Sequence[Sequence[float]] | np.ndarray


# ---------------------------------------------------------------------------
# Distributional two-sample tests
# ---------------------------------------------------------------------------


def _as_2d(samples: ArrayLike) -> np.ndarray:
    """Coerce a collection of vectors into a float ``(n, d)`` matrix."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D (n_samples, n_features), got shape {arr.shape}")
    return arr


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of ``a`` and rows of ``b``."""
    a_sq = np.sum(a * a, axis=1)[:, None]
    b_sq = np.sum(b * b, axis=1)[None, :]
    sq = a_sq + b_sq - 2.0 * (a @ b.T)
    return np.maximum(sq, 0.0)  # clamp tiny negatives from float error


def _median_bandwidth(pooled: np.ndarray, *, max_n: int = 2000, seed: int = 0) -> float:
    """Median-heuristic length scale: median of pairwise distances on the pool.

    For large pools the full (N, N) distance matrix is prohibitive (~2.6 GB at
    18k rows), so the median is estimated on a deterministic subsample of
    ``max_n`` rows. Standard practice for the median heuristic — the bandwidth is
    insensitive to the exact subset — and pools of ``<= max_n`` rows use every
    row, so the result is unchanged there (small-N tests stay bit-identical).
    """
    p = pooled
    if p.shape[0] > max_n:
        idx = np.random.default_rng(seed).choice(p.shape[0], size=max_n, replace=False)
        p = p[idx]
    sq = _pairwise_sq_dists(p, p)
    iu = np.triu_indices_from(sq, k=1)
    med_sq = float(np.median(sq[iu])) if iu[0].size else 1.0
    return med_sq if med_sq > 1e-12 else 1.0


def _pooled_kernel(pooled: np.ndarray, gammas: Sequence[float]) -> np.ndarray:
    """Full pooled multi-bandwidth RBF kernel, built ONCE per test.

    Diagonal entries equal ``len(gammas)`` (each RBF is 1 on the diagonal). The
    permutation loop then computes every MMD^2 by indexing submatrices of this
    matrix instead of recomputing exp() per permutation — same math, far faster,
    and it makes the episode-block permutation cheap.
    """
    d2 = _pairwise_sq_dists(pooled, pooled)
    K = np.zeros_like(d2)
    for g in gammas:
        K += np.exp(-g * d2)
    return K


def _mmd2_from_kernel(
    K: np.ndarray, ix: np.ndarray, iy: np.ndarray, n_gammas: int
) -> float:
    """Unbiased MMD^2 between the two index sets, read off the pooled kernel.

    Equivalent to building the RBF kernels for ``X = pooled[ix]`` / ``Y =
    pooled[iy]`` and excluding the diagonal — the within-set diagonal sums to
    ``size * n_gammas`` since each RBF is 1 on the diagonal.
    """
    m, n = ix.size, iy.size
    if m < 2 or n < 2:
        return 0.0
    kxx = K[np.ix_(ix, ix)]
    kyy = K[np.ix_(iy, iy)]
    kxy = K[np.ix_(ix, iy)]
    term_xx = (kxx.sum() - m * n_gammas) / (m * (m - 1))
    term_yy = (kyy.sum() - n * n_gammas) / (n * (n - 1))
    term_xy = kxy.sum() * (2.0 / (m * n))
    return float(term_xx + term_yy - term_xy)


def _block_stats(
    pooled: np.ndarray, gammas: Sequence[float], unit_rows: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate the pooled kernel + sqrt-distance sums into per-episode blocks.

    Returns ``(K_block, D_block, counts)`` where ``K_block[p, q]`` sums the
    multi-bandwidth RBF kernel over all (step in episode p, step in episode q)
    pairs, ``D_block[p, q]`` sums Euclidean distances (for energy distance), and
    ``counts[p]`` is episode p's step count. Built block-by-block, so peak memory
    is the largest single ``(n_p, n_q)`` block — never the full ``(N, N)`` matrix.
    This is what lets the gate scale to the N>=30 episode floor (~18k steps)
    without a multi-GB kernel; permutations then read off ``K_block`` in O(U^2).
    """
    blocks = [pooled[r] for r in unit_rows]
    u = len(blocks)
    counts = np.array([b.shape[0] for b in blocks], dtype=np.int64)
    K = np.zeros((u, u))
    D = np.zeros((u, u))
    for p in range(u):
        for q in range(p, u):
            d2 = _pairwise_sq_dists(blocks[p], blocks[q])
            k = np.zeros_like(d2)
            for g in gammas:
                k += np.exp(-g * d2)
            ks, ds = float(k.sum()), float(np.sqrt(d2).sum())
            K[p, q] = K[q, p] = ks
            D[p, q] = D[q, p] = ds
    return K, D, counts


def _mmd2_from_blocks(
    K: np.ndarray, counts: np.ndarray, sel: np.ndarray, n_gammas: int,
) -> float:
    """Unbiased MMD^2 for arm A = units where ``sel`` is True (B = the rest), read
    off the episode-block kernel sums. Exactly equals the full-kernel unbiased
    MMD^2 for the same split (block sums just regroup the same pairwise terms)."""
    a = np.flatnonzero(sel)
    b = np.flatnonzero(~sel)
    m = int(counts[a].sum())
    n = int(counts[b].sum())
    if m < 2 or n < 2:
        return 0.0
    saa = K[np.ix_(a, a)].sum()
    sbb = K[np.ix_(b, b)].sum()
    sab = K[np.ix_(a, b)].sum()
    # Drop the m (resp. n) self-pairs: each step's diagonal kernel value is
    # n_gammas (every RBF is 1 on the diagonal), matching the unbiased estimator.
    term_xx = (saa - m * n_gammas) / (m * (m - 1))
    term_yy = (sbb - n * n_gammas) / (n * (n - 1))
    term_xy = sab * (2.0 / (m * n))
    return float(term_xx + term_yy - term_xy)


def _energy_from_blocks(D: np.ndarray, counts: np.ndarray, sel: np.ndarray) -> float:
    """Energy distance for the A/B split from block sqrt-distance sums (matches
    ``energy_distance``'s biased self-term convention: the i=j zeros stay in)."""
    a = np.flatnonzero(sel)
    b = np.flatnonzero(~sel)
    m = int(counts[a].sum())
    n = int(counts[b].sum())
    if m == 0 or n == 0:
        return 0.0
    d_ab = D[np.ix_(a, b)].sum() / (m * n)
    d_aa = D[np.ix_(a, a)].sum() / (m * m)
    d_bb = D[np.ix_(b, b)].sum() / (n * n)
    return float(max(2.0 * d_ab - d_aa - d_bb, 0.0))


@dataclass(frozen=True)
class TwoSampleResult:
    mmd2: float
    mmd_p_value: float  # P(MMD^2 >= observed | same distribution); low => differ
    energy_distance: float
    n_baseline: int
    n_candidate: int
    n_permutations: int

    @property
    def distributions_differ(self) -> bool:
        """True when the permutation test rejects the null at p < 0.05."""
        return self.mmd_p_value < 0.05

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "mmd2": self.mmd2,
            "mmd_p_value": self.mmd_p_value,
            "energy_distance": self.energy_distance,
            "n_baseline": self.n_baseline,
            "n_candidate": self.n_candidate,
            "n_permutations": self.n_permutations,
            "distributions_differ": self.distributions_differ,
        }


def energy_distance(X: ArrayLike, Y: ArrayLike) -> float:
    """Two-sample energy distance: ``2 E|x-y| - E|x-x'| - E|y-y'|`` (>= 0)."""
    a, b = _as_2d(X), _as_2d(Y)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    d_ab = np.sqrt(_pairwise_sq_dists(a, b)).mean()
    d_aa = np.sqrt(_pairwise_sq_dists(a, a)).mean()
    d_bb = np.sqrt(_pairwise_sq_dists(b, b)).mean()
    return float(max(2.0 * d_ab - d_aa - d_bb, 0.0))


def two_sample_test(
    baseline: ArrayLike,
    candidate: ArrayLike,
    *,
    n_permutations: int = 200,
    seed: int = 7,
    baseline_groups: ArrayLike | None = None,
    candidate_groups: ArrayLike | None = None,
) -> TwoSampleResult:
    """MMD + energy-distance two-sample test with a permutation p-value.

    ``baseline`` / ``candidate`` are ``(n_samples, n_features)`` matrices (e.g.
    per-step applied actions from the original vs the optimized policy). A low
    ``mmd_p_value`` means the action distributions differ beyond sampling noise.

    **Episode-block permutation (use it for trajectory data).** Pass
    ``baseline_groups`` / ``candidate_groups`` — one group id (episode index) per
    sample row — and the permutation shuffles whole *episodes* between the two
    arms instead of individual steps. This is mandatory for rollout actions:
    consecutive steps within an episode are autocorrelated, so the i.i.d.
    step-level permutation treats correlated samples as independent and
    over-rejects (empirically ~100% false-positive on *identical* policies; the
    episode-block test restores ~5% with full power — see
    ``scripts/_spike_mmd_autocorrelation.py``). The MMD statistic still uses every
    step; only the null distribution is generated at episode granularity. Without
    groups the test falls back to step-level permutation, valid only for genuinely
    independent samples.
    """
    X, Y = _as_2d(baseline), _as_2d(candidate)
    m, n = X.shape[0], Y.shape[0]
    if m < 2 or n < 2:
        # Not enough samples to test; report a non-significant result rather
        # than fabricate a verdict (the success-rate gate still applies).
        return TwoSampleResult(0.0, 1.0, energy_distance(X, Y), m, n, 0)

    pooled = np.vstack([X, Y])
    med_sq = _median_bandwidth(pooled)
    base_gamma = 1.0 / (2.0 * med_sq)
    gammas = [base_gamma * s for s in (0.5, 1.0, 2.0)]
    n_g = len(gammas)
    rng = np.random.default_rng(seed)

    use_blocks = baseline_groups is not None and candidate_groups is not None
    if use_blocks:
        bg = np.asarray(baseline_groups)
        cg = np.asarray(candidate_groups)
        if bg.shape[0] != m or cg.shape[0] != n:
            raise ValueError(
                "group ids must align 1:1 with samples "
                f"(got {bg.shape[0]}/{cg.shape[0]} for {m}/{n} rows)"
            )
        # Globally-unique episode ids over the pool (offset candidate so it never
        # collides with baseline), then permute whole episodes between arms.
        offset = (int(bg.max()) + 1) if bg.size else 0
        pooled_units = np.concatenate([bg, cg + offset])
        units = np.unique(pooled_units)
        n_base_units = int(np.unique(bg).size)
        if not (units.size >= 2 and 1 <= n_base_units < units.size):
            # Can't form two episode arms — don't fabricate significance.
            return TwoSampleResult(0.0, 1.0, energy_distance(X, Y), m, n, 0)

        unit_rows = [np.flatnonzero(pooled_units == u) for u in units]
        # Episode-block aggregation: build the (U, U) block sums ONCE
        # (block-by-block, bounded memory), then every permutation is O(U^2) —
        # no (N, N) kernel, so this scales to the N>=30 floor (~18k steps).
        Kb, Db, counts = _block_stats(pooled, gammas, unit_rows)
        sel0 = units < offset  # the observed split: baseline episodes
        observed = _mmd2_from_blocks(Kb, counts, sel0, n_g)
        energy = _energy_from_blocks(Db, counts, sel0)
        ge = 0
        for _ in range(n_permutations):
            order = rng.permutation(units.size)
            sel = np.zeros(units.size, dtype=bool)
            sel[order[:n_base_units]] = True
            if _mmd2_from_blocks(Kb, counts, sel, n_g) >= observed:
                ge += 1
    else:
        # No groups: i.i.d. step-level permutation (valid only for genuinely
        # independent samples). Full kernel is fine here — this path is not the
        # production trajectory path, which always supplies episode groups.
        K = _pooled_kernel(pooled, gammas)
        observed = _mmd2_from_kernel(K, np.arange(m), np.arange(m, m + n), n_g)
        energy = energy_distance(X, Y)
        ge = 0
        idx = np.arange(m + n)
        for _ in range(n_permutations):
            perm = rng.permutation(idx)
            if _mmd2_from_kernel(K, perm[:m], perm[m:], n_g) >= observed:
                ge += 1

    p_value = (1.0 + ge) / (1.0 + n_permutations)

    return TwoSampleResult(
        mmd2=observed,
        mmd_p_value=p_value,
        energy_distance=energy,
        n_baseline=m,
        n_candidate=n,
        n_permutations=n_permutations,
    )


# ---------------------------------------------------------------------------
# Embodied / kinematic metrics
# ---------------------------------------------------------------------------


def jerk_rms(positions: ArrayLike, *, dt: float = 1.0) -> float:
    """RMS jerk (3rd time-derivative of joint position) over a trajectory.

    ``positions`` is ``(T, n_joints)``. Returns 0 for trajectories too short to
    take three derivatives, or for constant-velocity (linear) motion.
    """
    p = _as_2d(positions)
    if p.shape[0] < 4 or dt <= 0:
        return 0.0
    jerk = np.diff(p, n=3, axis=0) / (dt ** 3)
    return float(np.sqrt(np.mean(jerk * jerk)))


def motion_energy(velocities: ArrayLike) -> float:
    """Sum of squared joint velocities over a trajectory (``(T, n_joints)``)."""
    v = _as_2d(velocities)
    if v.size == 0:
        return 0.0
    return float(np.sum(v * v))


@dataclass(frozen=True)
class EmbodiedParity:
    baseline_jerk_rms: float
    candidate_jerk_rms: float
    baseline_motion_energy: float
    candidate_motion_energy: float
    baseline_completion_steps: float
    candidate_completion_steps: float

    def regressed(self, *, jerk_tol: float = 1.5, energy_tol: float = 1.5, time_tol: float = 1.5) -> bool:
        """True if the candidate is materially worse on any embodied axis.

        ``*_tol`` are allowed ratios (candidate / baseline). Default 1.5 => a
        50% increase in jerk, motion energy, or completion time fails.
        """
        def worse(cand: float, base: float, tol: float) -> bool:
            if base <= 1e-9:
                return cand > 1e-6  # baseline ~0, any candidate motion is a regression
            return (cand / base) > tol
        return (
            worse(self.candidate_jerk_rms, self.baseline_jerk_rms, jerk_tol)
            or worse(self.candidate_motion_energy, self.baseline_motion_energy, energy_tol)
            or worse(self.candidate_completion_steps, self.baseline_completion_steps, time_tol)
        )

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "baseline_jerk_rms": self.baseline_jerk_rms,
            "candidate_jerk_rms": self.candidate_jerk_rms,
            "baseline_motion_energy": self.baseline_motion_energy,
            "candidate_motion_energy": self.candidate_motion_energy,
            "baseline_completion_steps": self.baseline_completion_steps,
            "candidate_completion_steps": self.candidate_completion_steps,
            "embodied_regressed": self.regressed(),
        }


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else 0.0


def aggregate_embodied(
    *,
    baseline_positions: Sequence[ArrayLike],
    candidate_positions: Sequence[ArrayLike],
    baseline_velocities: Sequence[ArrayLike],
    candidate_velocities: Sequence[ArrayLike],
    baseline_completion_steps: Sequence[float],
    candidate_completion_steps: Sequence[float],
    dt: float = 1.0,
) -> EmbodiedParity:
    """Aggregate per-episode embodied metrics into a baseline-vs-candidate parity."""
    return EmbodiedParity(
        baseline_jerk_rms=_mean([jerk_rms(p, dt=dt) for p in baseline_positions]),
        candidate_jerk_rms=_mean([jerk_rms(p, dt=dt) for p in candidate_positions]),
        baseline_motion_energy=_mean([motion_energy(v) for v in baseline_velocities]),
        candidate_motion_energy=_mean([motion_energy(v) for v in candidate_velocities]),
        baseline_completion_steps=_mean(baseline_completion_steps),
        candidate_completion_steps=_mean(candidate_completion_steps),
    )


__all__ = [
    "EmbodiedParity",
    "TwoSampleResult",
    "aggregate_embodied",
    "energy_distance",
    "jerk_rms",
    "motion_energy",
    "two_sample_test",
]
