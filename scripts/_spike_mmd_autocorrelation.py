"""Diagnostic: does the verify MMD gate over-reject on autocorrelated trajectories?

The GPU re-validation (ap-v8SDPPrHUju26ozqqdmiSR) reported distributions_differ=True
(p=0.0299) comparing native vs Triton pi05 on LIBERO — a case we expected to pass
parity. The per-step applied actions fed to the gate are NOT i.i.d.: within an
episode they're highly autocorrelated (smooth robot motion). The production
permutation test shuffles individual steps as if exchangeable, so ~2500 correlated
steps are treated as ~2500 independent samples; the effective sample size is really
~#episodes. That deflates the p-value => over-rejection.

NULL-IS-TRUE experiment: both arms drawn from the SAME generating process (identical
"policy"), as episodes of autocorrelated AR(1) trajectories. A calibrated test must
reject at ~alpha (5%). We compare:
  - STEP-level permutation  (what the gate does today: shuffle steps)
  - EPISODE-block permutation (proposed fix: shuffle whole episodes)

Fast self-contained MMD: same multi-bandwidth RBF + median-heuristic gammas as
reflex.verify_metrics, but the pooled kernel is built ONCE per trial and each
permutation is a quadratic form h^T K h (biased MMD; a permutation test is
self-consistent for biased-vs-unbiased, so calibration is identical). A sanity
block checks this fast statistic tracks the production one. No GPU, no cost.

    python scripts/_spike_mmd_autocorrelation.py
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Fast cached-kernel MMD (multi-bandwidth RBF, median heuristic) — same math
# as reflex.verify_metrics.two_sample_test, but K built once per trial.
# --------------------------------------------------------------------------
def _pooled_sqdists(P):
    sq = np.sum(P * P, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (P @ P.T)
    return np.maximum(d2, 0.0)


def _kernel(d2):
    iu = np.triu_indices_from(d2, k=1)
    med_sq = float(np.median(d2[iu])) or 1.0
    base = 1.0 / (2.0 * med_sq)
    K = np.zeros_like(d2)
    for s in (0.5, 1.0, 2.0):
        K += np.exp(-(base * s) * d2)
    return K


def _mmd_biased(K, labels):
    """h^T K h with h_i = +1/m on arm A, -1/n on arm B."""
    m = int(np.sum(labels == 0))
    n = int(labels.size - m)
    if m < 1 or n < 1:
        return 0.0
    h = np.where(labels == 0, 1.0 / m, -1.0 / n)
    return float(h @ (K @ h))


def _perm_pvalue(P, unit_ids, base_unit_ids, *, n_perm=120, seed=7):
    """Permutation MMD. `unit_ids[i]` = the exchangeable unit (step idx for the
    step-level test, episode idx for the block test) of pooled row i.
    Permuting at the unit level = step-level when units are unique per row,
    block-level when units repeat within an episode."""
    K = _kernel(_pooled_sqdists(P))
    units = np.unique(unit_ids)
    base_set = set(base_unit_ids.tolist())
    labels = np.where(np.isin(unit_ids, list(base_set)), 0, 1)
    observed = _mmd_biased(K, labels)
    n_base_units = len(base_set)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(units)
        sel = set(perm[:n_base_units].tolist())
        lab = np.where(np.isin(unit_ids, list(sel)), 0, 1)
        if _mmd_biased(K, lab) >= observed:
            ge += 1
    return (1.0 + ge) / (1.0 + n_perm)


# --------------------------------------------------------------------------
# Data: autocorrelated AR(1) episodes (smooth robot-motion analogue)
# --------------------------------------------------------------------------
def _ar1(rng, *, steps, dim=7, mu=None, phi=0.92, eps=0.05):
    if mu is None:
        mu = np.zeros(dim)
    x = np.zeros((steps, dim))
    x[0] = mu + rng.normal(0, eps, dim)
    for t in range(1, steps):
        x[t] = mu + phi * (x[t - 1] - mu) + rng.normal(0, eps, dim)
    return x


def _arm_pool(rng, *, n_eps, mean_steps, ep_offset, **kw):
    """Returns (pooled_rows, episode_id_per_row). ep ids are globally unique."""
    rows, ep_ids = [], []
    for e in range(n_eps):
        steps = int(rng.integers(mean_steps - 30, mean_steps + 30))
        rows.append(_ar1(rng, steps=steps, **kw))
        ep_ids += [ep_offset + e] * steps
    return np.vstack(rows), np.asarray(ep_ids)


def _fpr(label, trial, *, trials=60, alpha=0.05):
    rej = sum(1 for s in range(2000, 2000 + trials) if trial(np.random.default_rng(s)) < alpha)
    print(f"{label:54s} reject {rej:3d}/{trials} = {rej/trials:6.1%}  (target ~{alpha:.0%})", flush=True)
    return rej / trials


def _build(rng, n_eps, mean_steps, *, shift=0.0, dim=7):
    b, b_ep = _arm_pool(rng, n_eps=n_eps, mean_steps=mean_steps, ep_offset=0, dim=dim)
    c, c_ep = _arm_pool(rng, n_eps=n_eps, mean_steps=mean_steps, ep_offset=n_eps, dim=dim,
                        mu=np.full(dim, shift) if shift else None)
    P = np.vstack([b, c])
    ep_ids = np.concatenate([b_ep, c_ep])
    step_ids = np.arange(P.shape[0])          # each row its own unit -> step-level
    base_step_ids = np.arange(b.shape[0])
    base_ep_ids = np.unique(b_ep)
    return P, ep_ids, step_ids, base_step_ids, base_ep_ids


def main():
    DIM, N_EPS, MEAN_STEPS = 7, 4, 140
    print("=== NULL IS TRUE (same generating process for both arms) ===")
    print("Calibrated test rejects ~5%. Over-rejection => mis-calibrated.\n", flush=True)

    def iid_step(rng):  # sanity: i.i.d., step-level -> should be ~5%
        X = rng.normal(0, 1, (N_EPS * MEAN_STEPS, DIM))
        Y = rng.normal(0, 1, (N_EPS * MEAN_STEPS, DIM))
        P = np.vstack([X, Y]); ids = np.arange(P.shape[0])
        return _perm_pvalue(P, ids, np.arange(X.shape[0]))
    _fpr("1. i.i.d. null, STEP-level (sanity, expect ~5%)", iid_step)

    def ac_step(rng):  # the gate today
        P, ep, st, bst, bep = _build(rng, N_EPS, MEAN_STEPS)
        return _perm_pvalue(P, st, bst)
    _fpr("2. autocorrelated null, STEP-level (the gate TODAY)", ac_step)

    def ac_block(rng):  # the fix
        P, ep, st, bst, bep = _build(rng, N_EPS, MEAN_STEPS)
        return _perm_pvalue(P, ep, bep)
    _fpr("3. autocorrelated null, EPISODE-block (proposed FIX)", ac_block)

    print("\n=== POWER (a REAL shift must still be caught) ===", flush=True)
    def ac_block_shift(rng):
        P, ep, st, bst, bep = _build(rng, N_EPS, MEAN_STEPS, shift=0.15)
        return _perm_pvalue(P, ep, bep)
    _fpr("4. autocorrelated +0.15 shift, EPISODE-block (power, want HIGH)", ac_block_shift)
    # Reference: step-level "power" on the same shift (inflated, for contrast).
    def ac_step_shift(rng):
        P, ep, st, bst, bep = _build(rng, N_EPS, MEAN_STEPS, shift=0.15)
        return _perm_pvalue(P, st, bst)
    _fpr("5. autocorrelated +0.15 shift, STEP-level (contrast)", ac_step_shift)


if __name__ == "__main__":
    main()
