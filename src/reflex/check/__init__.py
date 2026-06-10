"""reflex.check — the action-parity verdict engine (the core of ``reflex verify``).

Decides whether an OPTIMIZED VLA policy still behaves like the ORIGINAL, using a
distributional two-sample test on action-chunk samples (conditioned on the same
observations) plus embodied-quality metrics. Locked metric: MMD (RBF) permutation
test, per ADR 2026-05-31-parity-metric-mmd-provisional (backbone: Model Equality
Testing, arXiv 2410.20247). Pure NumPy — the verdict math runs anywhere, no GPU.
"""
from __future__ import annotations

from reflex.check.metrics import (
    STATISTICS,
    ParityVerdict,
    TwoSampleResult,
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

__all__ = [
    "STATISTICS",
    "ParityVerdict",
    "TwoSampleResult",
    "binned_kl",
    "compute_parity",
    "embodied_metrics",
    "energy_distance",
    "jerk_rms",
    "mmd2_rbf",
    "motion_energy",
    "path_length",
    "two_sample_test",
]
