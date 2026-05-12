"""Per-episode uncertainty scoring for flow-matching VLAs.

Subsystem 3 of the data-labeling-pipeline (per spec
features/01_serve/subfeatures/_ecosystem/data-labeling-pipeline/).
Subsystems 1 (success) + 2 (quality) live in `signals.py` + `composite.py`;
this module is the orthogonal third axis the spec calls out.

Design (per data-labeling-pipeline_research.md, Lens 1 + Lens 2):

  - For flow-matching VLAs (pi0, pi0.5, SmolVLA), N inference passes with
    different initial noise produce N different action chunks. Their
    variance is the model's intrinsic uncertainty about the action — high
    variance = the model isn't confident which action is correct.
  - Architecturally natural: same model, different starting noise. No
    architectural change needed. Unlike MC Dropout (Djupskas NLDL 2026)
    which is unreliable for extrapolation OOD inputs.
  - Per the sidecar's "high uncertainty + success = informative edge case"
    framing, this score is ORTHOGONAL to quality_score(). Downstream
    consumers (self-distilling-serve, data marketplace) combine the two
    axes to surface the most-valuable training examples.

Phase 1.5 v1 ships variance-based scoring + components dict. Phase 2
adds inter-mode divergence (Diff-DAgger / He&Cao) + post-hoc temperature
calibration (arxiv 2507.17383) — both deferred per the sidecar's
"deferred" pile.

The scoring function is pure-numpy and takes pre-generated samples as
input. Sample generation is the CALLER's responsibility (offline replay,
serving instrumentation, or a future `reflex traces uncertainty` CLI).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

UNCERTAINTY_VERSION = "rule-v1"


@dataclass(frozen=True)
class UncertaintyResult:
    """Output of uncertainty_score(). Orthogonal to QualityResult — both
    get embedded into the per-episode parquet record alongside the success
    label. Downstream training data selection combines uncertainty +
    quality + success per the spec's "informative edge case" framing."""

    uncertainty_score: float           # [0, 1] — higher = more uncertain
    components: dict[str, float]       # per-step / per-dim breakdown
    n_samples: int
    n_steps: int
    n_action_dims: int
    uncertainty_version: str
    computed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "uncertainty_score": self.uncertainty_score,
            "uncertainty_components": dict(self.components),
            "uncertainty_n_samples": self.n_samples,
            "uncertainty_n_steps": self.n_steps,
            "uncertainty_n_action_dims": self.n_action_dims,
            "uncertainty_version": self.uncertainty_version,
            "computed_at": self.computed_at,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_per_dim(samples: np.ndarray) -> np.ndarray:
    """Per-dim variance normalized by per-dim observed range.

    Returns (T, A) array of normalized variance values in [0, ~1].
    Zero-range dims (constant action) get zero normalized variance,
    matching the intuition "no variance, no uncertainty."
    """
    # samples shape: (N, T, A)
    var = samples.var(axis=0, ddof=0)  # (T, A) — variance across N samples
    rng = samples.max(axis=0) - samples.min(axis=0)  # (T, A) — observed range
    # Avoid div-by-zero on constant dims
    safe_rng = np.where(rng > 1e-9, rng, 1.0)
    # Variance bounded by (range/2)^2 for any distribution → divide by
    # (range/2)^2 = range^2 / 4 to land in [0, 1].
    return np.where(rng > 1e-9, var / ((safe_rng / 2.0) ** 2), 0.0)


def uncertainty_score(
    *,
    samples: np.ndarray,
) -> UncertaintyResult:
    """Compute per-episode uncertainty from N flow-matching inference samples.

    Args:
        samples: shape (N, T, A) — N inference passes (different noise),
            T action steps per chunk, A action dimensions.

            For pi0.5 typical: N=5, T=50, A=7 → shape (5, 50, 7).

    Returns:
        UncertaintyResult with score in [0, 1] (higher = more uncertain) +
        components breakdown for downstream filtering.

    Methodology:
        1. Per-dim variance across N samples → (T, A)
        2. Normalize each dim by per-dim observed range² to land in [0, 1]
        3. Mean across action dims → per-step uncertainty (T,)
        4. Mean across time steps → episode uncertainty (scalar)

        Components dict carries per-step max + per-dim max for downstream
        debugging ("which timestep was most uncertain", "which joint was
        most variable").

    Raises:
        ValueError: if samples shape isn't 3D or N < 2 (need ≥2 samples
            for variance to be defined).
    """
    if samples.ndim != 3:
        raise ValueError(
            f"samples must be 3D (N, T, A), got shape {samples.shape}"
        )
    n_samples, n_steps, n_dims = samples.shape
    if n_samples < 2:
        raise ValueError(
            f"need ≥2 samples for variance (got N={n_samples})"
        )

    samples_f = samples.astype(np.float64, copy=False)

    norm_var = _normalize_per_dim(samples_f)  # (T, A)
    per_step = norm_var.mean(axis=1)  # (T,)
    score = float(per_step.mean())
    # Clip to [0, 1] — theoretical bound is 1.0 for uniform distributions
    # but rounding + range estimation can exceed slightly.
    score = float(min(1.0, max(0.0, score)))

    components = {
        "mean_per_step_uncertainty": float(per_step.mean()),
        "max_per_step_uncertainty": float(per_step.max()),
        "mean_per_dim_uncertainty": float(norm_var.mean(axis=0).mean()),
        "max_per_dim_uncertainty": float(norm_var.mean(axis=0).max()),
        # Argmax over time → which step was most uncertain (debugging hint)
        "argmax_step": int(per_step.argmax()),
        # Argmax over dims → which action dim was most variable
        "argmax_dim": int(norm_var.mean(axis=0).argmax()),
    }

    return UncertaintyResult(
        uncertainty_score=score,
        components=components,
        n_samples=n_samples,
        n_steps=n_steps,
        n_action_dims=n_dims,
        uncertainty_version=UNCERTAINTY_VERSION,
        computed_at=_utc_now_iso(),
    )


def classify_episode_value(
    *,
    uncertainty: UncertaintyResult,
    success_flag: bool | None,
    high_uncertainty_threshold: float = 0.3,
) -> str:
    """Surface the spec's 4 quadrants of training value:

        high uncertainty + success → "informative_edge_case"  [highest value]
        high uncertainty + failure → "edge_case_to_correct"
        low uncertainty + success  → "redundant_known_good"
        low uncertainty + failure  → "model_blind_spot"      [also valuable]

    Per the data-labeling-pipeline spec's Notes section + the sidecar's
    Lens 6 (Valle et al. arxiv 2507.17049: "selecting training data by
    quality AND uncertainty produces better policies than either alone").

    Returns the bucket label as a string. None success_flag → bucket
    suffix '_unlabeled'.
    """
    high = uncertainty.uncertainty_score >= high_uncertainty_threshold
    if success_flag is None:
        return ("high_uncertainty_unlabeled" if high
                else "low_uncertainty_unlabeled")
    if high and success_flag:
        return "informative_edge_case"
    if high and not success_flag:
        return "edge_case_to_correct"
    if not high and success_flag:
        return "redundant_known_good"
    return "model_blind_spot"


__all__ = [
    "UNCERTAINTY_VERSION",
    "UncertaintyResult",
    "uncertainty_score",
    "classify_episode_value",
]
