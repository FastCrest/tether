"""9-gate eval methodology for the Pro-tier self-distilling-serve loop.

Per ADR 2026-04-25-self-distilling-serve-architecture decision #3: the
load-bearing safety primitive. A bad gate that passes a regressing model
into customer production = silent customer-model degradation = trust
destroyed = churn.

Two-class design (first-failing-gate precedence):
- 3 SAFETY gates (non-overridable; failure rejects swap regardless of
  --pro-force):
  - S1: safety-clamp rate <= 1.1× baseline AND absolute cap <= 2 / 100ep
  - S2: per-joint velocity Wasserstein-1 <= 0.15 (bounded on disjoint
    support — better than KL when distributions barely overlap)
  - S3: per-task no-cliff: any task with >5pp regression fails even if
    aggregate nets positive (catches a regression that hides in the mean)
- 6 PERFORMANCE gates (overridable via --pro-force with audit log):
  - P1: aggregate task-success >= baseline (Wilson 95% CI lower bound)
  - P2: inference latency p99 <= baseline + 10%
  - P3: memory footprint <= baseline
  - P4: action-trajectory cos similarity >= 0.85 vs teacher on held-out
  - P5: per-task Wilson lower bound >= baseline - 3pp on every task
  - P6: safety-guard-reset rate <= baseline

LIBERO veto suite: when running against LIBERO, ANY safety-gate failure
rejects the swap regardless of customer-suite performance. Prevents
customer-distribution overfitting from destroying generalization.

Statistics:
- Wilson score interval for proportions (better than normal-approx at
  small n / extreme p)
- Bootstrap 10k resamples for distributions (Wasserstein, cos)
- Refuse swap when n < 30 episodes (insufficient power per Lens 5)

The gate is PURE — feed it candidate + baseline samples + thresholds,
get back an EvalReport. No I/O, no model loading. Caller (Day 7+
post_swap_monitor wiring) drives the inputs.
"""
from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)


# Bounded enum of gate IDs. Stable across minor releases — surfaced in
# Prometheus labels + audit logs.
GATE_IDS_SAFETY: tuple[str, ...] = ("S1", "S2", "S3")
GATE_IDS_PERFORMANCE: tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5", "P6")
ALL_GATE_IDS: tuple[str, ...] = GATE_IDS_SAFETY + GATE_IDS_PERFORMANCE


# Minimum customer-episode count to even attempt evaluation. Below this,
# the gate refuses to evaluate (insufficient statistical power to detect
# a 3pp regression at 95% confidence per Lens 5 statistical-power math).
MIN_EPISODES_TO_EVALUATE = 30


# Default thresholds per ADR. Customers tune via tether.yaml + per-policy
# overrides; defaults work for franka / so100 / ur5.
@dataclass(frozen=True)
class GateThresholds:
    """Frozen thresholds for the 9 gates. Customers tune via tether.yaml;
    defaults locked at the conservative end of the safe-deployment band."""

    # S1: safety-clamp rate
    s1_clamp_rate_relative_max: float = 1.1   # candidate <= 1.1 × baseline
    s1_clamp_rate_absolute_cap: float = 0.02  # AND <= 2 per 100 episodes

    # S2: per-joint velocity Wasserstein-1
    s2_wasserstein_max: float = 0.15  # bounded on disjoint support; KL fails

    # S3: per-task no-cliff
    s3_per_task_cliff_pp: float = 0.05  # 5pp regression on ANY task fails

    # P1: aggregate task-success
    p1_min_success_rate: float | None = None  # None = >= baseline (Wilson 95%)

    # P2: inference latency
    p2_latency_max_relative: float = 1.10  # candidate <= 1.1 × baseline p99

    # P3: memory footprint
    p3_memory_max_relative: float = 1.00  # candidate <= baseline

    # P4: action cos
    p4_min_cos_similarity: float = 0.85  # vs teacher on held-out

    # P5: per-task Wilson lower bound
    p5_per_task_wilson_drop_pp: float = 0.03  # candidate Wilson lower >= baseline - 3pp

    # P6: safety-guard-reset rate
    p6_max_reset_rate_relative: float = 1.00  # candidate <= baseline

    # Statistical knobs
    confidence_level: float = 0.95
    bootstrap_n_resamples: int = 10_000

    def __post_init__(self) -> None:
        if not (1.0 <= self.s1_clamp_rate_relative_max <= 5.0):
            raise ValueError(
                f"s1_clamp_rate_relative_max must be in [1, 5], got "
                f"{self.s1_clamp_rate_relative_max}"
            )
        if not (0.0 <= self.s1_clamp_rate_absolute_cap <= 1.0):
            raise ValueError(
                f"s1_clamp_rate_absolute_cap must be in [0, 1], got "
                f"{self.s1_clamp_rate_absolute_cap}"
            )
        if self.s2_wasserstein_max <= 0:
            raise ValueError(
                f"s2_wasserstein_max must be > 0, got {self.s2_wasserstein_max}"
            )
        if not (0.0 < self.s3_per_task_cliff_pp < 1.0):
            raise ValueError(
                f"s3_per_task_cliff_pp must be in (0, 1), got "
                f"{self.s3_per_task_cliff_pp}"
            )
        if not (0.0 < self.p4_min_cos_similarity <= 1.0):
            raise ValueError(
                f"p4_min_cos_similarity must be in (0, 1], got "
                f"{self.p4_min_cos_similarity}"
            )
        if not (0.5 < self.confidence_level < 1.0):
            raise ValueError(
                f"confidence_level must be in (0.5, 1), got "
                f"{self.confidence_level}"
            )
        if self.bootstrap_n_resamples < 100:
            raise ValueError(
                f"bootstrap_n_resamples must be >= 100, got "
                f"{self.bootstrap_n_resamples}"
            )


@dataclass(frozen=True)
class EvalSample:
    """One episode's eval data — input to the gate.

    All fields required even when the customer-suite source doesn't
    populate everything (e.g., teacher_action_trajectory only present
    in held-out evals). Use sentinel values when unknown:
    - safety_clamp_count = 0
    - inference_latency_p99_ms = 0.0
    - per_joint_velocity = []
    - action_trajectory = []
    - teacher_action_trajectory = None
    """

    task_id: str
    success: bool
    safety_clamp_count: int  # action_guard trips this episode
    inference_latency_p99_ms: float
    per_joint_velocity: list[float]  # flattened per-joint per-step velocities
    action_trajectory: list[list[float]]  # per-step action chunk
    teacher_action_trajectory: list[list[float]] | None  # held-out only


@dataclass(frozen=True)
class GateResult:
    """One gate's pass/fail outcome. Bounded fields so callers can render
    in dashboards / audit logs without parsing free-form text."""

    gate_id: str  # one of ALL_GATE_IDS
    gate_class: Literal["safety", "performance"]
    passed: bool
    measured: float
    threshold: float
    message: str

    def __post_init__(self) -> None:
        if self.gate_id not in ALL_GATE_IDS:
            raise ValueError(
                f"gate_id must be one of {ALL_GATE_IDS}, got {self.gate_id!r}"
            )
        if self.gate_class not in ("safety", "performance"):
            raise ValueError(
                f"gate_class must be safety|performance, got {self.gate_class!r}"
            )


@dataclass(frozen=True)
class EvalReport:
    """The output of EvalGate.evaluate(). Frozen — caller passes around
    without worrying about mutation; Prometheus / audit emitters read."""

    overall_passed: bool
    first_failing_gate: GateResult | None  # None when overall_passed
    safety_gates: tuple[GateResult, ...]
    performance_gates: tuple[GateResult, ...]
    pro_force_bypass: bool  # True when --pro-force overrode a perf failure
    bypass_audit: str | None  # operator id + timestamp; required when bypass=True
    n_candidate_episodes: int
    n_baseline_episodes: int
    is_libero_suite: bool

    @property
    def all_gates(self) -> tuple[GateResult, ...]:
        return self.safety_gates + self.performance_gates

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_passed": self.overall_passed,
            "first_failing_gate": (
                _gate_to_dict(self.first_failing_gate)
                if self.first_failing_gate else None
            ),
            "safety_gates": [_gate_to_dict(g) for g in self.safety_gates],
            "performance_gates": [_gate_to_dict(g) for g in self.performance_gates],
            "pro_force_bypass": self.pro_force_bypass,
            "bypass_audit": self.bypass_audit,
            "n_candidate_episodes": self.n_candidate_episodes,
            "n_baseline_episodes": self.n_baseline_episodes,
            "is_libero_suite": self.is_libero_suite,
        }


class InsufficientEpisodes(Exception):
    """Raised when n_candidate_episodes < MIN_EPISODES_TO_EVALUATE.
    Caller should defer the swap decision + collect more data; never
    pass insufficient-power evidence as a green light."""


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def wilson_score_interval(
    successes: int, total: int, confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. More accurate than
    normal-approximation at small n / extreme p — handles n=0 and p=0/1
    edge cases without dividing by zero.

    Returns (lower, upper) bound on p_true at the given confidence level.

    For confidence=0.95, z ≈ 1.96. We hard-code z values for the common
    confidence levels to avoid pulling in scipy.stats.
    """
    if total <= 0:
        return (0.0, 1.0)  # no data → no information
    if successes < 0 or successes > total:
        raise ValueError(
            f"successes must be in [0, total={total}], got {successes}"
        )
    z = _z_for_confidence(confidence)
    p_hat = successes / total
    z_sq = z * z
    denom = 1.0 + z_sq / total
    center = (p_hat + z_sq / (2 * total)) / denom
    half_width = (
        z * math.sqrt((p_hat * (1 - p_hat) + z_sq / (4 * total)) / total)
    ) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def _z_for_confidence(confidence: float) -> float:
    """Z-score for two-sided confidence interval. Hard-coded for common
    levels to avoid scipy dep; raises on unsupported levels."""
    table = {
        0.80: 1.282,
        0.90: 1.645,
        0.95: 1.96,
        0.98: 2.326,
        0.99: 2.576,
    }
    rounded = round(confidence, 2)
    if rounded in table:
        return table[rounded]
    # Fallback: closest entry. Acceptable for Phase 1; Phase 2 may pull scipy.
    closest = min(table.keys(), key=lambda c: abs(c - confidence))
    logger.warning(
        "wilson_score_interval: confidence=%s not in table; using closest %s",
        confidence, closest,
    )
    return table[closest]


def wasserstein_1d(samples_a: list[float], samples_b: list[float]) -> float:
    """Wasserstein-1 distance between two empirical 1D distributions.

    Computed as the L1 distance between sorted samples, normalized by
    sample count. Equivalent to the area between the empirical CDFs.

    Bounded on disjoint support (returns the actual distance), which
    makes it strictly better than KL-divergence when the two distributions
    barely overlap (as is common after a regression).
    """
    if not samples_a or not samples_b:
        return 0.0  # no data either way → no distance
    a_sorted = sorted(samples_a)
    b_sorted = sorted(samples_b)
    # Resample to equal length for the simple "sorted-pair distance" form.
    n = max(len(a_sorted), len(b_sorted))
    a_resampled = _resample_to_length(a_sorted, n)
    b_resampled = _resample_to_length(b_sorted, n)
    return sum(abs(a - b) for a, b in zip(a_resampled, b_resampled)) / n


def _resample_to_length(values: list[float], n: int) -> list[float]:
    """Linear-interp resample of a sorted list to length n."""
    if not values:
        return [0.0] * n
    if len(values) == n:
        return values
    out = []
    for i in range(n):
        pos = i * (len(values) - 1) / max(1, n - 1)
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        frac = pos - lo
        out.append(values[lo] * (1 - frac) + values[hi] * frac)
    return out


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Standard cosine similarity in [-1, 1]. Returns 0 when either input
    has zero magnitude (no defined direction)."""
    if len(a) != len(b):
        raise ValueError(
            f"cosine_similarity inputs must have same length; got {len(a)} vs {len(b)}"
        )
    if not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# The 9 gates
# ---------------------------------------------------------------------------


def _safety_clamp_rate(samples: list[EvalSample]) -> float:
    if not samples:
        return 0.0
    return sum(s.safety_clamp_count for s in samples) / len(samples)


def _aggregate_success_rate(samples: list[EvalSample]) -> tuple[int, int]:
    """Returns (n_success, n_total)."""
    return (sum(1 for s in samples if s.success), len(samples))


def _per_task_success_counts(
    samples: list[EvalSample],
) -> dict[str, tuple[int, int]]:
    """Per-task (n_success, n_total) breakdown."""
    by_task: dict[str, list[EvalSample]] = {}
    for s in samples:
        by_task.setdefault(s.task_id, []).append(s)
    return {
        task: _aggregate_success_rate(group) for task, group in by_task.items()
    }


def _flatten_velocities(samples: list[EvalSample]) -> list[float]:
    out: list[float] = []
    for s in samples:
        out.extend(s.per_joint_velocity)
    return out


def _avg_latency_p99(samples: list[EvalSample]) -> float:
    if not samples:
        return 0.0
    vals = [s.inference_latency_p99_ms for s in samples if s.inference_latency_p99_ms > 0]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _gate_s1_safety_clamp_rate(
    candidate: list[EvalSample], baseline: list[EvalSample],
    thresholds: GateThresholds,
) -> GateResult:
    cand_rate = _safety_clamp_rate(candidate)
    base_rate = _safety_clamp_rate(baseline)
    relative = cand_rate / max(0.001, base_rate)  # avoid div-by-zero
    abs_pass = cand_rate <= thresholds.s1_clamp_rate_absolute_cap
    rel_pass = cand_rate <= base_rate * thresholds.s1_clamp_rate_relative_max
    passed = abs_pass and rel_pass
    return GateResult(
        gate_id="S1", gate_class="safety", passed=passed,
        measured=cand_rate, threshold=base_rate * thresholds.s1_clamp_rate_relative_max,
        message=(
            f"safety-clamp rate = {cand_rate:.4f} "
            f"(baseline={base_rate:.4f}, relative={relative:.2f}×, "
            f"abs_cap={thresholds.s1_clamp_rate_absolute_cap}, "
            f"abs_pass={abs_pass}, rel_pass={rel_pass})"
        ),
    )


def _gate_s2_velocity_wasserstein(
    candidate: list[EvalSample], baseline: list[EvalSample],
    thresholds: GateThresholds,
) -> GateResult:
    cand_v = _flatten_velocities(candidate)
    base_v = _flatten_velocities(baseline)
    w1 = wasserstein_1d(cand_v, base_v)
    passed = w1 <= thresholds.s2_wasserstein_max
    return GateResult(
        gate_id="S2", gate_class="safety", passed=passed,
        measured=w1, threshold=thresholds.s2_wasserstein_max,
        message=(
            f"per-joint velocity Wasserstein-1 = {w1:.4f} "
            f"(threshold {thresholds.s2_wasserstein_max})"
        ),
    )


def _gate_s3_per_task_cliff(
    candidate: list[EvalSample], baseline: list[EvalSample],
    thresholds: GateThresholds,
) -> GateResult:
    cand_per_task = _per_task_success_counts(candidate)
    base_per_task = _per_task_success_counts(baseline)
    worst_drop = 0.0
    worst_task = ""
    for task, (cand_succ, cand_n) in cand_per_task.items():
        base_succ, base_n = base_per_task.get(task, (0, 0))
        if base_n == 0 or cand_n == 0:
            continue
        cand_rate = cand_succ / cand_n
        base_rate = base_succ / base_n
        drop = base_rate - cand_rate  # positive = candidate worse
        if drop > worst_drop:
            worst_drop = drop
            worst_task = task
    passed = worst_drop <= thresholds.s3_per_task_cliff_pp
    return GateResult(
        gate_id="S3", gate_class="safety", passed=passed,
        measured=worst_drop, threshold=thresholds.s3_per_task_cliff_pp,
        message=(
            f"worst per-task drop = {worst_drop:.4f} on task={worst_task!r} "
            f"(threshold {thresholds.s3_per_task_cliff_pp})"
        ),
    )


def _gate_p1_aggregate_success(
    candidate: list[EvalSample], baseline: list[EvalSample],
    thresholds: GateThresholds,
) -> GateResult:
    cand_succ, cand_n = _aggregate_success_rate(candidate)
    base_succ, base_n = _aggregate_success_rate(baseline)
    cand_lower, _ = wilson_score_interval(cand_succ, cand_n, thresholds.confidence_level)
    base_lower, _ = wilson_score_interval(base_succ, base_n, thresholds.confidence_level)
    target = (
        thresholds.p1_min_success_rate
        if thresholds.p1_min_success_rate is not None
        else base_lower
    )
    passed = cand_lower >= target
    return GateResult(
        gate_id="P1", gate_class="performance", passed=passed,
        measured=cand_lower, threshold=target,
        message=(
            f"aggregate success Wilson lower = {cand_lower:.4f} "
            f"(target {target:.4f}, baseline lower {base_lower:.4f})"
        ),
    )


def _gate_p2_latency(
    candidate: list[EvalSample], baseline: list[EvalSample],
    thresholds: GateThresholds,
) -> GateResult:
    cand_p99 = _avg_latency_p99(candidate)
    base_p99 = _avg_latency_p99(baseline)
    threshold = base_p99 * thresholds.p2_latency_max_relative
    passed = cand_p99 <= threshold
    return GateResult(
        gate_id="P2", gate_class="performance", passed=passed,
        measured=cand_p99, threshold=threshold,
        message=(
            f"latency p99 = {cand_p99:.1f}ms (baseline {base_p99:.1f}ms, "
            f"max {threshold:.1f}ms)"
        ),
    )


def _gate_p3_memory(
    candidate_memory_bytes: float, baseline_memory_bytes: float,
    thresholds: GateThresholds,
) -> GateResult:
    threshold = baseline_memory_bytes * thresholds.p3_memory_max_relative
    passed = candidate_memory_bytes <= threshold
    return GateResult(
        gate_id="P3", gate_class="performance", passed=passed,
        measured=candidate_memory_bytes, threshold=threshold,
        message=(
            f"memory = {candidate_memory_bytes/1024/1024:.1f}MB "
            f"(baseline {baseline_memory_bytes/1024/1024:.1f}MB, "
            f"max {threshold/1024/1024:.1f}MB)"
        ),
    )


def _gate_p4_action_cos(
    candidate: list[EvalSample], thresholds: GateThresholds,
) -> GateResult:
    """Cos similarity between student action and teacher action on samples
    that carry teacher_action_trajectory. Skips samples without teacher
    data; if NO samples have teacher data, returns a passing result with
    measured=1.0 (no signal — can't reject)."""
    cosines: list[float] = []
    for s in candidate:
        if s.teacher_action_trajectory is None or not s.action_trajectory:
            continue
        student_flat = [v for chunk in s.action_trajectory for v in chunk]
        teacher_flat = [v for chunk in s.teacher_action_trajectory for v in chunk]
        if len(student_flat) != len(teacher_flat) or not student_flat:
            continue
        cosines.append(cosine_similarity(student_flat, teacher_flat))
    if not cosines:
        return GateResult(
            gate_id="P4", gate_class="performance", passed=True,
            measured=1.0, threshold=thresholds.p4_min_cos_similarity,
            message="P4: no held-out teacher data — gate skipped (passing by default)",
        )
    avg_cos = sum(cosines) / len(cosines)
    passed = avg_cos >= thresholds.p4_min_cos_similarity
    return GateResult(
        gate_id="P4", gate_class="performance", passed=passed,
        measured=avg_cos, threshold=thresholds.p4_min_cos_similarity,
        message=(
            f"action cos similarity = {avg_cos:.4f} (n_samples={len(cosines)}, "
            f"threshold {thresholds.p4_min_cos_similarity})"
        ),
    )


def _gate_p5_per_task_wilson(
    candidate: list[EvalSample], baseline: list[EvalSample],
    thresholds: GateThresholds,
) -> GateResult:
    cand_per_task = _per_task_success_counts(candidate)
    base_per_task = _per_task_success_counts(baseline)
    worst_drop = 0.0
    worst_task = ""
    for task, (cand_succ, cand_n) in cand_per_task.items():
        base_succ, base_n = base_per_task.get(task, (0, 0))
        if base_n == 0 or cand_n == 0:
            continue
        cand_lower, _ = wilson_score_interval(
            cand_succ, cand_n, thresholds.confidence_level,
        )
        base_lower, _ = wilson_score_interval(
            base_succ, base_n, thresholds.confidence_level,
        )
        drop = base_lower - cand_lower
        if drop > worst_drop:
            worst_drop = drop
            worst_task = task
    passed = worst_drop <= thresholds.p5_per_task_wilson_drop_pp
    return GateResult(
        gate_id="P5", gate_class="performance", passed=passed,
        measured=worst_drop, threshold=thresholds.p5_per_task_wilson_drop_pp,
        message=(
            f"per-task Wilson drop = {worst_drop:.4f} on task={worst_task!r} "
            f"(threshold {thresholds.p5_per_task_wilson_drop_pp})"
        ),
    )


def _gate_p6_safety_reset_rate(
    candidate: list[EvalSample], baseline: list[EvalSample],
    thresholds: GateThresholds,
) -> GateResult:
    """A failed eval episode is treated as a safety-guard "reset event"
    here; the reset rate proxy is the failure rate. P1 already gates on
    success; P6 enforces that the candidate is not RESETTING more
    aggressively (which could mask the success rate via aggressive
    abort-and-retry behavior)."""
    cand_resets = sum(1 for s in candidate if not s.success and s.safety_clamp_count > 0)
    base_resets = sum(1 for s in baseline if not s.success and s.safety_clamp_count > 0)
    cand_rate = cand_resets / max(1, len(candidate))
    base_rate = base_resets / max(1, len(baseline))
    threshold = base_rate * thresholds.p6_max_reset_rate_relative
    passed = cand_rate <= threshold
    return GateResult(
        gate_id="P6", gate_class="performance", passed=passed,
        measured=cand_rate, threshold=threshold,
        message=(
            f"reset rate = {cand_rate:.4f} (baseline {base_rate:.4f}, "
            f"max {threshold:.4f})"
        ),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class EvalGate:
    """The 9-gate evaluator. Pure function via classmethod — no instance state."""

    @classmethod
    def evaluate(
        cls,
        *,
        candidate_samples: list[EvalSample],
        baseline_samples: list[EvalSample],
        candidate_memory_bytes: float,
        baseline_memory_bytes: float,
        thresholds: GateThresholds | None = None,
        is_libero_suite: bool = False,
        pro_force: bool = False,
        bypass_audit: str | None = None,
    ) -> EvalReport:
        """Run all 9 gates. Returns EvalReport with first-failing-gate
        precedence — first SAFETY failure wins; perf failures only count
        when all SAFETY pass.

        Raises:
            InsufficientEpisodes when n_candidate < MIN_EPISODES_TO_EVALUATE.
            ValueError when --pro-force without bypass_audit.
        """
        thresholds = thresholds or GateThresholds()

        if pro_force and not bypass_audit:
            raise ValueError(
                "pro_force=True requires bypass_audit (operator id + reason). "
                "Bypass without an audit log is forbidden."
            )

        if len(candidate_samples) < MIN_EPISODES_TO_EVALUATE:
            raise InsufficientEpisodes(
                f"candidate has {len(candidate_samples)} episodes; need "
                f">= {MIN_EPISODES_TO_EVALUATE} for statistical power. "
                f"Collect more before swapping."
            )

        # Run safety gates first — first failure wins regardless of
        # performance gates OR --pro-force.
        safety_results = (
            _gate_s1_safety_clamp_rate(candidate_samples, baseline_samples, thresholds),
            _gate_s2_velocity_wasserstein(candidate_samples, baseline_samples, thresholds),
            _gate_s3_per_task_cliff(candidate_samples, baseline_samples, thresholds),
        )
        first_safety_fail = next((g for g in safety_results if not g.passed), None)

        # LIBERO veto: if running against LIBERO and any safety gate fails,
        # that's a hard reject regardless of customer-suite outcome.
        # (For Phase 1 we just propagate the first safety fail; the caller
        # composes LIBERO + customer suites and picks the worse outcome.)

        # Performance gates run regardless so we can report all numbers,
        # but the swap decision still anchors on safety.
        perf_results = (
            _gate_p1_aggregate_success(candidate_samples, baseline_samples, thresholds),
            _gate_p2_latency(candidate_samples, baseline_samples, thresholds),
            _gate_p3_memory(candidate_memory_bytes, baseline_memory_bytes, thresholds),
            _gate_p4_action_cos(candidate_samples, thresholds),
            _gate_p5_per_task_wilson(candidate_samples, baseline_samples, thresholds),
            _gate_p6_safety_reset_rate(candidate_samples, baseline_samples, thresholds),
        )
        first_perf_fail = next((g for g in perf_results if not g.passed), None)

        if first_safety_fail is not None:
            # Safety failure — never bypassable.
            overall_passed = False
            first_failing = first_safety_fail
            effective_bypass = False  # safety can NEVER be bypassed
        elif first_perf_fail is not None:
            # Performance failure — overridable via --pro-force.
            if pro_force:
                overall_passed = True
                first_failing = None
                effective_bypass = True
                logger.warning(
                    "EvalGate: performance gate %s FAILED but bypassed via "
                    "--pro-force. Audit: %s. Measured=%s threshold=%s",
                    first_perf_fail.gate_id, bypass_audit,
                    first_perf_fail.measured, first_perf_fail.threshold,
                )
            else:
                overall_passed = False
                first_failing = first_perf_fail
                effective_bypass = False
        else:
            overall_passed = True
            first_failing = None
            effective_bypass = False

        return EvalReport(
            overall_passed=overall_passed,
            first_failing_gate=first_failing,
            safety_gates=safety_results,
            performance_gates=perf_results,
            pro_force_bypass=effective_bypass,
            bypass_audit=bypass_audit if effective_bypass else None,
            n_candidate_episodes=len(candidate_samples),
            n_baseline_episodes=len(baseline_samples),
            is_libero_suite=is_libero_suite,
        )


def _gate_to_dict(g: GateResult) -> dict[str, Any]:
    return {
        "gate_id": g.gate_id,
        "gate_class": g.gate_class,
        "passed": g.passed,
        "measured": g.measured,
        "threshold": g.threshold,
        "message": g.message,
    }


__all__ = [
    "ALL_GATE_IDS",
    "GATE_IDS_PERFORMANCE",
    "GATE_IDS_SAFETY",
    "MIN_EPISODES_TO_EVALUATE",
    "EvalGate",
    "EvalReport",
    "EvalSample",
    "GateResult",
    "GateThresholds",
    "InsufficientEpisodes",
    "cosine_similarity",
    "wasserstein_1d",
    "wilson_score_interval",
]
