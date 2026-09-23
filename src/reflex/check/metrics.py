"""Action-parity metric engine — the core of ``reflex verify`` (the check wedge).

Two layers, both pure NumPy (no GPU / Modal — runs anywhere):

1. **Distributional two-sample tests** on action-chunk samples drawn from the
   ORIGINAL vs OPTIMIZED policy, *conditioned on the same observations*. The
   locked statistic is **MMD (RBF kernel)**, validated by a permutation test that
   gives a false-positive-rate-controlled p-value (per ADR
   ``2026-05-31-parity-metric-mmd-provisional``; backbone: Model Equality Testing,
   arXiv 2410.20247). ``energy`` and ``binned_kl`` are provided for the empirical
   bake-off follow-up.

2. **Embodied-quality metrics** on an action trajectory (jerk, motion-energy,
   path-length) — these catch regressions that aggregate task-success hides
   (arXiv 2603.19131).

Why per-sample ``atol`` is wrong (and why this exists): a VLA policy *samples*
actions, so two correct runs differ; MSE / atol between teacher and student
anti-correlate with real-robot success. The right question is distributional:
"are the optimized policy's actions drawn from the same distribution as the
original's, on the same inputs?" — which is exactly a two-sample test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# pairwise distances                                                          #
# --------------------------------------------------------------------------- #


def _as2d(a: Any) -> np.ndarray:
    """Coerce to a float ``(n_samples, n_features)`` array."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D (n_samples, dim) array, got shape {arr.shape}")
    return arr


def _pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """``||x_i - y_j||^2`` for every pair. Shape ``(len(X), len(Y))``."""
    xx = np.einsum("ij,ij->i", X, X)[:, None]
    yy = np.einsum("ij,ij->i", Y, Y)[None, :]
    sq = xx + yy - 2.0 * (X @ Y.T)
    return np.maximum(sq, 0.0)  # clamp tiny negatives from float error


def _median_heuristic_gamma(X: np.ndarray, Y: np.ndarray) -> float:
    """RBF bandwidth ``gamma = 1 / median(pairwise squared distance)``."""
    Z = np.vstack([X, Y])
    sq = _pairwise_sq_dists(Z, Z)
    iu = np.triu_indices(len(Z), k=1)
    med = float(np.median(sq[iu])) if iu[0].size else 1.0
    return 1.0 / (med if med > 0 else 1.0)


# --------------------------------------------------------------------------- #
# distributional statistics                                                   #
# --------------------------------------------------------------------------- #


def mmd2_rbf(X: Any, Y: Any, gammas: list[float] | None = None) -> float:
    """Unbiased squared Maximum Mean Discrepancy with a multi-bandwidth RBF kernel.

    ~0 when X and Y are the same distribution; larger = more different. Multi-
    bandwidth (median heuristic × {0.5, 1, 2}) makes it robust to action scale.
    """
    X, Y = _as2d(X), _as2d(Y)
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        raise ValueError("MMD needs >= 2 samples per set")
    if gammas is None:
        g = _median_heuristic_gamma(X, Y)
        gammas = [g * s for s in (0.5, 1.0, 2.0)]
    sqxx, sqyy, sqxy = _pairwise_sq_dists(X, X), _pairwise_sq_dists(Y, Y), _pairwise_sq_dists(X, Y)
    acc = 0.0
    for g in gammas:
        kxx, kyy, kxy = np.exp(-g * sqxx), np.exp(-g * sqyy), np.exp(-g * sqxy)
        sxx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))  # exclude diagonal -> unbiased
        syy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
        sxy = kxy.mean()
        acc += sxx + syy - 2.0 * sxy
    return float(acc / len(gammas))


def energy_distance(X: Any, Y: Any) -> float:
    """Székely energy distance. ~0 when X, Y share a distribution; larger = more different."""
    X, Y = _as2d(X), _as2d(Y)
    dxy = np.sqrt(_pairwise_sq_dists(X, Y)).mean()
    dxx = np.sqrt(_pairwise_sq_dists(X, X)).mean()
    dyy = np.sqrt(_pairwise_sq_dists(Y, Y)).mean()
    return float(max(2.0 * dxy - dxx - dyy, 0.0))


def binned_kl(X: Any, Y: Any, bins: int = 24, eps: float = 1e-9) -> float:
    """Symmetrized KL averaged over action dims, on shared per-dim histograms."""
    X, Y = _as2d(X), _as2d(Y)
    d = X.shape[1]
    acc, used = 0.0, 0
    for j in range(d):
        lo = min(X[:, j].min(), Y[:, j].min())
        hi = max(X[:, j].max(), Y[:, j].max())
        if hi <= lo:
            continue
        edges = np.linspace(lo, hi, bins + 1)
        px = np.histogram(X[:, j], bins=edges)[0].astype(np.float64) + eps
        py = np.histogram(Y[:, j], bins=edges)[0].astype(np.float64) + eps
        px /= px.sum()
        py /= py.sum()
        acc += 0.5 * (np.sum(px * np.log(px / py)) + np.sum(py * np.log(py / px)))
        used += 1
    return float(acc / max(used, 1))


STATISTICS = {"mmd": mmd2_rbf, "energy": energy_distance, "binned_kl": binned_kl}


# --------------------------------------------------------------------------- #
# two-sample permutation test (FPR-controlled)                                #
# --------------------------------------------------------------------------- #


@dataclass
class TwoSampleResult:
    metric: str
    statistic: float
    p_value: float
    n_perm: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def two_sample_test(
    X: Any, Y: Any, metric: str = "mmd", n_perm: int = 200, seed: int = 0
) -> TwoSampleResult:
    """Permutation two-sample test. **H0: X and Y are the same distribution.**

    A small ``p_value`` rejects H0 → the optimized policy's action distribution
    *differs* from the original's → parity is broken. The permutation null gives
    a false-positive-rate-controlled p (this is the Model Equality Testing recipe,
    adapted from string kernels to action-chunk space).
    """
    if metric not in STATISTICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(STATISTICS)}")
    X, Y = _as2d(X), _as2d(Y)
    stat_fn = STATISTICS[metric]
    observed = stat_fn(X, Y)
    pooled = np.vstack([X, Y])
    n = len(X)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        if stat_fn(pooled[idx[:n]], pooled[idx[n:]]) >= observed:
            ge += 1
    p = (ge + 1) / (n_perm + 1)  # +1 smoothing -> valid p in (0, 1]
    return TwoSampleResult(metric=metric, statistic=float(observed), p_value=float(p), n_perm=n_perm)


# --------------------------------------------------------------------------- #
# embodied-quality metrics (per action trajectory, shape (T, action_dim))     #
# --------------------------------------------------------------------------- #


def jerk_rms(traj: Any, dt: float = 1.0) -> float:
    """RMS magnitude of the 3rd time-derivative of the action trajectory.

    Lower = smoother. Catches "passes task-success but moves jerkily" regressions
    (arXiv 2603.19131) that aggregate success-rate cannot see.
    """
    a = _as2d(traj)
    if len(a) < 4:
        return 0.0
    j = np.diff(a, n=3, axis=0) / (dt ** 3)
    return float(np.sqrt(np.mean(np.sum(j * j, axis=1))))


def motion_energy(traj: Any) -> float:
    """Sum of squared step-to-step action deltas (path energy)."""
    a = _as2d(traj)
    if len(a) < 2:
        return 0.0
    dv = np.diff(a, axis=0)
    return float(np.sum(dv * dv))


def path_length(traj: Any) -> float:
    """Total L2 path length of the action trajectory."""
    a = _as2d(traj)
    if len(a) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1)))


def embodied_metrics(traj: Any) -> dict[str, float]:
    return {
        "jerk_rms": jerk_rms(traj),
        "motion_energy": motion_energy(traj),
        "path_length": path_length(traj),
    }


# --------------------------------------------------------------------------- #
# the parity verdict                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class ParityVerdict:
    passed: bool
    metric: str
    statistic: float
    p_value: float
    alpha: float
    n_samples: tuple[int, int]
    embodied_delta: dict[str, float] = field(default_factory=dict)
    embodied_flag: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_samples"] = list(self.n_samples)
        return d


def compute_parity(
    orig_samples: Any,
    opt_samples: Any,
    *,
    metric: str = "mmd",
    alpha: float = 0.05,
    n_perm: int = 200,
    seed: int = 0,
    orig_traj: Any | None = None,
    opt_traj: Any | None = None,
    jerk_tol: float = 0.25,
) -> ParityVerdict:
    """v1 action-parity verdict for an optimized policy vs the original.

    Distributional gate (the headline): a two-sample test on action-chunk samples
    conditioned on the same observations. ``p_value < alpha`` ⇒ the optimized
    action distribution *differs* ⇒ **FAIL**.

    NOTE (honest): "no detectable difference at ``alpha``" is *not* a proof of
    equivalence — the rollout-success non-inferiority tier (``TODO(reflex-verify)``)
    closes that, and is what ties to real-robot success. Embodied gate: ``FAIL`` if
    the optimized trajectory is materially jerkier than the original (``jerk_tol``).
    """
    res = two_sample_test(orig_samples, opt_samples, metric=metric, n_perm=n_perm, seed=seed)
    dist_pass = res.p_value >= alpha

    embodied_delta: dict[str, float] = {}
    embodied_flag = False
    if orig_traj is not None and opt_traj is not None:
        o, p = embodied_metrics(orig_traj), embodied_metrics(opt_traj)
        embodied_delta = {k: float(p[k] - o[k]) for k in o}
        base = o["jerk_rms"] if o["jerk_rms"] > 0 else 1e-9
        embodied_flag = (p["jerk_rms"] - o["jerk_rms"]) / base > jerk_tol

    passed = dist_pass and not embodied_flag
    if dist_pass:
        reason = f"distribution parity holds (p={res.p_value:.3g} >= alpha={alpha})"
    else:
        reason = f"action distribution differs (p={res.p_value:.3g} < alpha={alpha})"
    if embodied_flag:
        reason += f"; embodied regression: optimized trajectory >{jerk_tol:.0%} jerkier"

    return ParityVerdict(
        passed=passed,
        metric=metric,
        statistic=res.statistic,
        p_value=res.p_value,
        alpha=alpha,
        n_samples=(len(_as2d(orig_samples)), len(_as2d(opt_samples))),
        embodied_delta=embodied_delta,
        embodied_flag=embodied_flag,
        reason=reason,
    )
