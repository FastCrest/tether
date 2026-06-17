"""Pro-tier distribution-shift detection — gates the distill scheduler.

Per ADR 2026-04-25-self-distilling-serve-architecture: a distill kick
on data that has drifted FAR from the base distribution can produce a
worse student than no distill at all. The drift detector gates this:
when shift exceeds threshold, the scheduler refuses to kick + emits a
warning ("customer's distribution has changed materially; investigate
before continuing to distill").

Two metrics:
- Per-joint state distribution KL-divergence between customer + base
  (catches scene/embodiment drift like new lighting or unusual objects)
- Action-space fingerprinting via Wasserstein-1 over per-joint action
  histograms (catches "wrist got bent mid-dataset" — bimodal action
  distribution that would teach the student a confused policy)

Pure module — no I/O, no asyncio. Caller (Day 5+ wiring in
DistillScheduler) feeds in customer + base sample windows; receives a
DriftReport with the per-joint scores + an overall pass/fail decision.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)


# Default thresholds per ADR (tunable via tether.yaml).
# KL-divergence > this means materially different; refuse distill kick.
DEFAULT_KL_DIVERGENCE_MAX = 0.5

# Wasserstein-1 between per-joint action histograms; same intuition as
# the post-swap-monitor T1 scale — low single-digits OK; >1.0 = drift.
DEFAULT_ACTION_WASSERSTEIN_MAX = 1.0

# Histogram bin count for KL + Wasserstein. Higher = more sensitive
# but noisier; 32 is a balanced default that matches sklearn convention.
DEFAULT_HISTOGRAM_BINS = 32

# Minimum sample count to compute drift confidently. Below this, the
# detector returns drift_score=0 + reason="insufficient-samples" rather
# than a noisy score.
MIN_SAMPLES_FOR_DRIFT = 100


@dataclass(frozen=True)
class JointDriftScore:
    """Per-joint drift score — one entry per joint dimension."""

    joint_index: int
    kl_divergence: float
    action_wasserstein: float

    @property
    def drift_score(self) -> float:
        """Combined score — max of the two metrics. Conservative:
        ANY metric exceeding threshold triggers a fail."""
        return max(self.kl_divergence, self.action_wasserstein)


@dataclass(frozen=True)
class DriftReport:
    """Frozen output of DriftDetector.evaluate()."""

    drift_detected: bool
    reason: str  # "ok" | "kl-exceeded" | "action-exceeded" | "insufficient-samples"
    n_customer_samples: int
    n_base_samples: int
    per_joint_scores: tuple[JointDriftScore, ...]
    worst_joint_index: int  # -1 when no drift
    worst_joint_score: float
    threshold: float

    @property
    def max_kl(self) -> float:
        if not self.per_joint_scores:
            return 0.0
        return max(s.kl_divergence for s in self.per_joint_scores)

    @property
    def max_action_wasserstein(self) -> float:
        if not self.per_joint_scores:
            return 0.0
        return max(s.action_wasserstein for s in self.per_joint_scores)


class DriftDetector:
    """Distribution-shift detector for the Pro distill pipeline.

    Lifecycle:
        detector = DriftDetector()
        report = detector.evaluate(
            customer_states=[...],   # list of state vectors
            base_states=[...],        # baseline state distribution
            customer_actions=[...],   # list of action chunks
            base_actions=[...],       # baseline action distribution
        )
        if report.drift_detected:
            # caller (DistillScheduler) refuses kick + logs warning
            ...
    """

    __slots__ = (
        "_kl_max", "_action_wasserstein_max",
        "_histogram_bins", "_min_samples",
    )

    def __init__(
        self,
        *,
        kl_divergence_max: float = DEFAULT_KL_DIVERGENCE_MAX,
        action_wasserstein_max: float = DEFAULT_ACTION_WASSERSTEIN_MAX,
        histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
        min_samples: int = MIN_SAMPLES_FOR_DRIFT,
    ):
        if kl_divergence_max <= 0:
            raise ValueError(f"kl_divergence_max must be > 0, got {kl_divergence_max}")
        if action_wasserstein_max <= 0:
            raise ValueError(
                f"action_wasserstein_max must be > 0, got {action_wasserstein_max}"
            )
        if histogram_bins < 2:
            raise ValueError(f"histogram_bins must be >= 2, got {histogram_bins}")
        if min_samples < 10:
            raise ValueError(f"min_samples must be >= 10, got {min_samples}")
        self._kl_max = float(kl_divergence_max)
        self._action_wasserstein_max = float(action_wasserstein_max)
        self._histogram_bins = int(histogram_bins)
        self._min_samples = int(min_samples)

    def evaluate(
        self,
        *,
        customer_states: list[list[float]],
        base_states: list[list[float]],
        customer_actions: list[list[float]],
        base_actions: list[list[float]],
    ) -> DriftReport:
        """Compare customer + base distributions per-joint. Returns a
        DriftReport with first-detector-wins precedence (KL → Wasserstein)."""
        n_cust = len(customer_states)
        n_base = len(base_states)

        if n_cust < self._min_samples or n_base < self._min_samples:
            return DriftReport(
                drift_detected=False, reason="insufficient-samples",
                n_customer_samples=n_cust, n_base_samples=n_base,
                per_joint_scores=(),
                worst_joint_index=-1, worst_joint_score=0.0,
                threshold=self._kl_max,
            )

        # Determine joint count from the first sample (assume consistent
        # across all samples — caller's responsibility).
        joint_count_state = len(customer_states[0]) if customer_states else 0
        joint_count_action = len(customer_actions[0]) if customer_actions else 0
        joint_count = max(joint_count_state, joint_count_action)
        if joint_count == 0:
            return DriftReport(
                drift_detected=False, reason="insufficient-samples",
                n_customer_samples=n_cust, n_base_samples=n_base,
                per_joint_scores=(),
                worst_joint_index=-1, worst_joint_score=0.0,
                threshold=self._kl_max,
            )

        scores: list[JointDriftScore] = []
        worst_idx = -1
        worst_score = 0.0
        worst_reason = "ok"

        for j in range(joint_count):
            # State distribution KL
            kl = 0.0
            if j < joint_count_state:
                cust_j = [s[j] for s in customer_states if j < len(s)]
                base_j = [s[j] for s in base_states if j < len(s)]
                if cust_j and base_j:
                    kl = symmetric_kl_divergence(
                        cust_j, base_j, n_bins=self._histogram_bins,
                    )

            # Action distribution Wasserstein
            wd = 0.0
            if j < joint_count_action:
                cust_a = [a[j] for a in customer_actions if j < len(a)]
                base_a = [a[j] for a in base_actions if j < len(a)]
                if cust_a and base_a:
                    wd = wasserstein_1d_simple(cust_a, base_a)

            score = JointDriftScore(
                joint_index=j, kl_divergence=kl, action_wasserstein=wd,
            )
            scores.append(score)

            if score.drift_score > worst_score:
                worst_score = score.drift_score
                worst_idx = j
                worst_reason = (
                    "kl-exceeded" if kl >= wd else "action-exceeded"
                )

        # First-failing-metric precedence: KL takes priority for the
        # reason field when it's higher (more sensitive to small
        # distribution changes); Wasserstein when it's the worse signal.
        max_kl_val = max((s.kl_divergence for s in scores), default=0.0)
        max_wd_val = max((s.action_wasserstein for s in scores), default=0.0)
        kl_failed = max_kl_val > self._kl_max
        wd_failed = max_wd_val > self._action_wasserstein_max

        drift_detected = kl_failed or wd_failed
        if not drift_detected:
            reason = "ok"
        elif kl_failed and (not wd_failed or max_kl_val / self._kl_max >= max_wd_val / self._action_wasserstein_max):
            reason = "kl-exceeded"
        else:
            reason = "action-exceeded"

        return DriftReport(
            drift_detected=drift_detected,
            reason=reason,
            n_customer_samples=n_cust, n_base_samples=n_base,
            per_joint_scores=tuple(scores),
            worst_joint_index=worst_idx if drift_detected else -1,
            worst_joint_score=worst_score if drift_detected else 0.0,
            threshold=self._kl_max if reason in ("kl-exceeded", "ok") else self._action_wasserstein_max,
        )


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def symmetric_kl_divergence(
    samples_a: list[float], samples_b: list[float], *, n_bins: int = 32,
) -> float:
    """Symmetric KL = (KL(P||Q) + KL(Q||P)) / 2 over histograms of the
    two samples. Add-one Laplace smoothing prevents div-by-zero on empty
    bins. Returns 0 when either sample is empty.

    Symmetric form (Jeffrey divergence) avoids the directionality issue
    of plain KL — we don't have a privileged "reference" distribution
    here; both customer + base are observations."""
    if not samples_a or not samples_b:
        return 0.0
    lo = min(min(samples_a), min(samples_b))
    hi = max(max(samples_a), max(samples_b))
    if lo == hi:
        return 0.0  # all samples identical → no divergence
    bin_width = (hi - lo) / n_bins
    hist_a = [0] * n_bins
    hist_b = [0] * n_bins
    for v in samples_a:
        idx = min(n_bins - 1, max(0, int((v - lo) / bin_width)))
        hist_a[idx] += 1
    for v in samples_b:
        idx = min(n_bins - 1, max(0, int((v - lo) / bin_width)))
        hist_b[idx] += 1
    # Add-one Laplace smoothing
    p = [(c + 1) / (sum(hist_a) + n_bins) for c in hist_a]
    q = [(c + 1) / (sum(hist_b) + n_bins) for c in hist_b]
    kl_pq = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
    kl_qp = sum(qi * math.log(qi / pi) for pi, qi in zip(p, q))
    return 0.5 * (kl_pq + kl_qp)


def wasserstein_1d_simple(
    samples_a: list[float], samples_b: list[float],
) -> float:
    """Simple 1D Wasserstein-1 — sorted-pair L1 with linear-interp
    resampling to equal length. Same as eval_gate.wasserstein_1d but
    re-implemented here to avoid cross-module dependency."""
    if not samples_a or not samples_b:
        return 0.0
    a_sorted = sorted(samples_a)
    b_sorted = sorted(samples_b)
    n = max(len(a_sorted), len(b_sorted))

    def _resample(values: list[float], target: int) -> list[float]:
        if len(values) == target:
            return values
        out = []
        for i in range(target):
            pos = i * (len(values) - 1) / max(1, target - 1)
            lo_idx = int(pos)
            hi_idx = min(lo_idx + 1, len(values) - 1)
            frac = pos - lo_idx
            out.append(values[lo_idx] * (1 - frac) + values[hi_idx] * frac)
        return out

    a_res = _resample(a_sorted, n)
    b_res = _resample(b_sorted, n)
    return sum(abs(a - b) for a, b in zip(a_res, b_res)) / n


__all__ = [
    "DEFAULT_ACTION_WASSERSTEIN_MAX",
    "DEFAULT_HISTOGRAM_BINS",
    "DEFAULT_KL_DIVERGENCE_MAX",
    "DriftDetector",
    "DriftReport",
    "JointDriftScore",
    "MIN_SAMPLES_FOR_DRIFT",
    "symmetric_kl_divergence",
    "wasserstein_1d_simple",
]
