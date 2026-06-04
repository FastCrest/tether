"""Tests for src/tether/pro/eval_gate.py — Phase 1 self-distilling-serve Day 5-6.

Per ADR 2026-04-25-self-distilling-serve-architecture decision #3:
9-gate methodology, 3 SAFETY non-overridable + 6 PERFORMANCE with
--pro-force bypass. First-failing-gate precedence; safety failures are
never bypassable; insufficient episodes raises rather than passes.
"""
from __future__ import annotations

import math

import pytest

from tether.pro.eval_gate import (
    ALL_GATE_IDS,
    GATE_IDS_PERFORMANCE,
    GATE_IDS_SAFETY,
    MIN_EPISODES_TO_EVALUATE,
    EvalGate,
    EvalReport,
    EvalSample,
    GateResult,
    GateThresholds,
    InsufficientEpisodes,
    cosine_similarity,
    wasserstein_1d,
    wilson_score_interval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_sample(
    *,
    task_id: str = "pick_block",
    success: bool = True,
    safety_clamp_count: int = 0,
    inference_latency_p99_ms: float = 50.0,
    per_joint_velocity: list[float] | None = None,
    action_trajectory: list[list[float]] | None = None,
    teacher_action_trajectory: list[list[float]] | None = None,
) -> EvalSample:
    return EvalSample(
        task_id=task_id,
        success=success,
        safety_clamp_count=safety_clamp_count,
        inference_latency_p99_ms=inference_latency_p99_ms,
        per_joint_velocity=per_joint_velocity or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        action_trajectory=action_trajectory or [[0.0] * 7 for _ in range(50)],
        teacher_action_trajectory=teacher_action_trajectory,
    )


def _identity_baseline_and_candidate(n: int = 50) -> tuple[list[EvalSample], list[EvalSample]]:
    """Generate matched samples (candidate identical to baseline) — should
    pass all 9 gates."""
    baseline = [_mk_sample() for _ in range(n)]
    candidate = [_mk_sample() for _ in range(n)]
    return baseline, candidate


# ---------------------------------------------------------------------------
# Constants + bounded enums
# ---------------------------------------------------------------------------


def test_gate_id_buckets_are_disjoint_and_cover_all():
    assert set(GATE_IDS_SAFETY) & set(GATE_IDS_PERFORMANCE) == set()
    assert set(GATE_IDS_SAFETY) | set(GATE_IDS_PERFORMANCE) == set(ALL_GATE_IDS)
    assert len(GATE_IDS_SAFETY) == 3
    assert len(GATE_IDS_PERFORMANCE) == 6
    assert len(ALL_GATE_IDS) == 9


def test_min_episodes_threshold_locked():
    """Per Lens 5: 30 episodes minimum for statistical power. Locked."""
    assert MIN_EPISODES_TO_EVALUATE == 30


# ---------------------------------------------------------------------------
# GateThresholds validation
# ---------------------------------------------------------------------------


def test_thresholds_default_construction_succeeds():
    GateThresholds()


def test_thresholds_rejects_clamp_relative_below_one():
    with pytest.raises(ValueError, match="s1_clamp_rate_relative_max"):
        GateThresholds(s1_clamp_rate_relative_max=0.5)


def test_thresholds_rejects_negative_wasserstein():
    with pytest.raises(ValueError, match="s2_wasserstein_max"):
        GateThresholds(s2_wasserstein_max=-0.1)


def test_thresholds_rejects_cliff_at_or_above_one():
    with pytest.raises(ValueError, match="s3_per_task_cliff_pp"):
        GateThresholds(s3_per_task_cliff_pp=1.0)


def test_thresholds_rejects_cos_above_one():
    with pytest.raises(ValueError, match="p4_min_cos_similarity"):
        GateThresholds(p4_min_cos_similarity=1.5)


def test_thresholds_rejects_confidence_at_half():
    with pytest.raises(ValueError, match="confidence_level"):
        GateThresholds(confidence_level=0.5)


def test_thresholds_rejects_low_bootstrap_resamples():
    with pytest.raises(ValueError, match="bootstrap_n_resamples"):
        GateThresholds(bootstrap_n_resamples=50)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def test_wilson_zero_total_returns_full_range():
    lo, hi = wilson_score_interval(0, 0)
    assert lo == 0.0 and hi == 1.0


def test_wilson_all_success_lower_bound_below_one():
    """At n=10, p_hat=1.0, Wilson lower < 1 (accounts for sample-size
    uncertainty unlike normal-approximation which would give 1.0)."""
    lo, hi = wilson_score_interval(10, 10)
    assert lo < 1.0
    assert hi == 1.0


def test_wilson_all_failure_upper_bound_above_zero():
    lo, hi = wilson_score_interval(0, 10)
    assert lo == 0.0
    assert hi > 0.0


def test_wilson_rejects_invalid_successes():
    with pytest.raises(ValueError, match="successes"):
        wilson_score_interval(11, 10)
    with pytest.raises(ValueError, match="successes"):
        wilson_score_interval(-1, 10)


def test_wilson_central_estimate_close_to_phat():
    """At large n, the interval is centered near p_hat."""
    lo, hi = wilson_score_interval(80, 100)
    center = (lo + hi) / 2
    assert abs(center - 0.80) < 0.05


def test_wasserstein_zero_for_identical_samples():
    assert wasserstein_1d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_wasserstein_positive_for_disjoint_samples():
    """Distributions with zero overlap still produce a real distance — KL
    can't do this."""
    w = wasserstein_1d([0.0, 1.0], [10.0, 11.0])
    assert w == pytest.approx(10.0, abs=0.5)


def test_wasserstein_handles_empty_inputs():
    assert wasserstein_1d([], [1.0, 2.0]) == 0.0
    assert wasserstein_1d([1.0], []) == 0.0


def test_cosine_identical_vectors_returns_one():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_returns_zero():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_zero_vector_returns_zero():
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


def test_cosine_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        cosine_similarity([1, 2], [1, 2, 3])


# ---------------------------------------------------------------------------
# EvalGate.evaluate — happy path (identity)
# ---------------------------------------------------------------------------


def test_identity_baseline_candidate_passes_all_gates():
    baseline, candidate = _identity_baseline_and_candidate(n=50)
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=100_000_000,
        baseline_memory_bytes=100_000_000,
    )
    assert report.overall_passed
    assert report.first_failing_gate is None
    assert all(g.passed for g in report.safety_gates)
    assert all(g.passed for g in report.performance_gates)


# ---------------------------------------------------------------------------
# Insufficient episodes — refuses to evaluate
# ---------------------------------------------------------------------------


def test_insufficient_episodes_raises():
    baseline = [_mk_sample() for _ in range(50)]
    candidate = [_mk_sample() for _ in range(10)]  # under threshold
    with pytest.raises(InsufficientEpisodes):
        EvalGate.evaluate(
            candidate_samples=candidate,
            baseline_samples=baseline,
            candidate_memory_bytes=100_000_000,
            baseline_memory_bytes=100_000_000,
        )


# ---------------------------------------------------------------------------
# SAFETY gates
# ---------------------------------------------------------------------------


def test_s1_safety_clamp_rate_failure():
    """Candidate doubles the safety-clamp rate → S1 fails."""
    baseline = [_mk_sample(safety_clamp_count=1) for _ in range(50)]
    candidate = [_mk_sample(safety_clamp_count=10) for _ in range(50)]
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=100_000_000,
        baseline_memory_bytes=100_000_000,
    )
    assert not report.overall_passed
    assert report.first_failing_gate.gate_id == "S1"


def test_s2_velocity_wasserstein_failure():
    """Candidate has a wildly different velocity distribution → S2 fails."""
    baseline = [_mk_sample(per_joint_velocity=[0.1] * 7) for _ in range(50)]
    candidate = [_mk_sample(per_joint_velocity=[5.0] * 7) for _ in range(50)]
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=100_000_000,
        baseline_memory_bytes=100_000_000,
    )
    assert not report.overall_passed
    assert report.first_failing_gate.gate_id == "S2"


def test_s3_per_task_cliff_failure():
    """Candidate regresses on one task by 50pp; S3 catches it even though
    aggregate may net positive."""
    baseline = [
        *[_mk_sample(task_id="task_a", success=True) for _ in range(30)],
        *[_mk_sample(task_id="task_b", success=True) for _ in range(30)],
    ]
    # Candidate: still 100% on task_a, 0% on task_b (50pp drop on task_b)
    candidate = [
        *[_mk_sample(task_id="task_a", success=True) for _ in range(30)],
        *[_mk_sample(task_id="task_b", success=False) for _ in range(30)],
    ]
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=100_000_000,
        baseline_memory_bytes=100_000_000,
    )
    assert not report.overall_passed
    assert report.first_failing_gate.gate_id == "S3"


# ---------------------------------------------------------------------------
# PERFORMANCE gates
# ---------------------------------------------------------------------------


def test_p1_aggregate_success_failure():
    """Candidate's Wilson lower < baseline lower → P1 fails."""
    baseline = [_mk_sample(success=True) for _ in range(50)]
    # Candidate: 60% success (significant drop)
    candidate = [
        _mk_sample(success=(i < 30)) for i in range(50)
    ]
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=100_000_000,
        baseline_memory_bytes=100_000_000,
    )
    # Candidate may also fail S3 (per-task cliff), but the FIRST failure
    # we care about is P1 OR S3 — both possible. Verify NOT passed.
    assert not report.overall_passed


def test_p2_latency_failure():
    """Candidate latency > baseline × 1.10 → P2 fails."""
    baseline = [_mk_sample(inference_latency_p99_ms=50.0) for _ in range(50)]
    candidate = [_mk_sample(inference_latency_p99_ms=100.0) for _ in range(50)]
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=100_000_000,
        baseline_memory_bytes=100_000_000,
    )
    assert not report.overall_passed
    # Since safety gates pass (identical clamp + velocity + per-task), P2 should fire
    assert report.first_failing_gate.gate_id == "P2"


def test_p3_memory_failure():
    """Candidate uses more memory than baseline → P3 fails."""
    baseline, candidate = _identity_baseline_and_candidate(n=50)
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=200_000_000,  # 2× baseline
        baseline_memory_bytes=100_000_000,
    )
    assert not report.overall_passed
    assert report.first_failing_gate.gate_id == "P3"


def test_p4_action_cos_skipped_without_teacher_data():
    """When no samples carry teacher trajectory, P4 passes by default
    (no signal — can't reject)."""
    baseline, candidate = _identity_baseline_and_candidate(n=50)
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=100_000_000,
        baseline_memory_bytes=100_000_000,
    )
    p4 = next(g for g in report.performance_gates if g.gate_id == "P4")
    assert p4.passed
    assert "skipped" in p4.message


def test_p4_action_cos_failure_when_teacher_diverges():
    """Candidate's actions diverge from teacher → P4 fails."""
    baseline = [_mk_sample() for _ in range(50)]
    # Candidate actions opposite-sign teacher; cos = -1
    student_chunk = [[1.0] * 7 for _ in range(50)]
    teacher_chunk = [[-1.0] * 7 for _ in range(50)]
    candidate = [
        _mk_sample(
            action_trajectory=student_chunk,
            teacher_action_trajectory=teacher_chunk,
        )
        for _ in range(50)
    ]
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=100_000_000,
        baseline_memory_bytes=100_000_000,
    )
    # Velocity distribution is identical (default), success identical, so
    # safety gates pass. P4 should fire.
    p4 = next(g for g in report.performance_gates if g.gate_id == "P4")
    assert not p4.passed


def test_p5_per_task_wilson_drop_failure():
    """Candidate's per-task Wilson lower bound drops > 3pp on a task → P5 fails."""
    baseline = [_mk_sample(task_id="task_a", success=True) for _ in range(60)]
    # Candidate: only 70% on task_a (Wilson lower ~ 0.58, baseline ~ 0.94)
    candidate = [
        _mk_sample(task_id="task_a", success=(i < 42)) for i in range(60)
    ]
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=100_000_000,
        baseline_memory_bytes=100_000_000,
    )
    # Multiple gates fail; verify at least P5 is in the failure
    failed = [g for g in report.performance_gates if not g.passed]
    assert any(g.gate_id == "P5" for g in failed)


# ---------------------------------------------------------------------------
# --pro-force bypass behavior
# ---------------------------------------------------------------------------


def test_pro_force_bypass_overrides_performance_failure():
    """A perf failure with --pro-force + audit → overall_passed=True with
    pro_force_bypass=True."""
    baseline, candidate = _identity_baseline_and_candidate(n=50)
    # Candidate uses too much memory (P3 fails)
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=200_000_000,
        baseline_memory_bytes=100_000_000,
        pro_force=True,
        bypass_audit="ops_engineer_42 @ 2026-04-25 — accepting 2x memory for known model size increase",
    )
    assert report.overall_passed
    assert report.pro_force_bypass
    assert report.bypass_audit is not None


def test_pro_force_bypass_does_not_override_safety_failure():
    """Even with --pro-force, a SAFETY failure rejects the swap. Non-
    overridable per ADR."""
    baseline = [_mk_sample(safety_clamp_count=0) for _ in range(50)]
    candidate = [_mk_sample(safety_clamp_count=10) for _ in range(50)]  # S1 fails
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=100_000_000,
        baseline_memory_bytes=100_000_000,
        pro_force=True,
        bypass_audit="ops_engineer_42 — attempting to override safety",
    )
    assert not report.overall_passed
    assert report.first_failing_gate.gate_id == "S1"
    assert not report.pro_force_bypass  # safety can never be bypassed


def test_pro_force_without_audit_raises():
    """Bypass without audit log is forbidden."""
    baseline, candidate = _identity_baseline_and_candidate(n=50)
    with pytest.raises(ValueError, match="bypass_audit"):
        EvalGate.evaluate(
            candidate_samples=candidate,
            baseline_samples=baseline,
            candidate_memory_bytes=200_000_000,
            baseline_memory_bytes=100_000_000,
            pro_force=True,
            bypass_audit=None,
        )


# ---------------------------------------------------------------------------
# EvalReport shape + serialization
# ---------------------------------------------------------------------------


def test_report_has_all_expected_fields():
    baseline, candidate = _identity_baseline_and_candidate(n=50)
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=100_000_000,
        baseline_memory_bytes=100_000_000,
    )
    assert isinstance(report, EvalReport)
    assert report.n_candidate_episodes == 50
    assert report.n_baseline_episodes == 50
    assert len(report.safety_gates) == 3
    assert len(report.performance_gates) == 6
    assert len(report.all_gates) == 9


def test_report_to_dict_is_json_serializable():
    import json
    baseline, candidate = _identity_baseline_and_candidate(n=50)
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=100_000_000,
        baseline_memory_bytes=100_000_000,
    )
    d = report.to_dict()
    s = json.dumps(d)
    restored = json.loads(s)
    assert restored["overall_passed"] is True
    assert len(restored["safety_gates"]) == 3
    assert len(restored["performance_gates"]) == 6


def test_gate_result_rejects_invalid_gate_id():
    with pytest.raises(ValueError, match="gate_id"):
        GateResult(
            gate_id="X1", gate_class="safety", passed=True,
            measured=0.0, threshold=0.0, message="",
        )


def test_gate_result_rejects_invalid_gate_class():
    with pytest.raises(ValueError, match="gate_class"):
        GateResult(
            gate_id="S1", gate_class="bogus", passed=True,  # type: ignore[arg-type]
            measured=0.0, threshold=0.0, message="",
        )


# ---------------------------------------------------------------------------
# First-failing-gate precedence
# ---------------------------------------------------------------------------


def test_safety_failure_blocks_perf_failures_in_first_failing_gate():
    """When BOTH safety + perf fail, first_failing_gate must be SAFETY."""
    baseline = [_mk_sample(safety_clamp_count=0, inference_latency_p99_ms=50) for _ in range(50)]
    candidate = [_mk_sample(safety_clamp_count=10, inference_latency_p99_ms=200) for _ in range(50)]
    report = EvalGate.evaluate(
        candidate_samples=candidate,
        baseline_samples=baseline,
        candidate_memory_bytes=100_000_000,
        baseline_memory_bytes=100_000_000,
    )
    assert report.first_failing_gate.gate_id in ("S1", "S2", "S3")
    assert report.first_failing_gate.gate_class == "safety"
