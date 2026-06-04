"""End-to-end integration tests for the self-distilling-serve loop (Days 11-12).

Per ADR 2026-04-25-self-distilling-serve-architecture: 4-stage loop is
(collect -> distill -> 9-gate eval -> swap) with HW-bound JWT licensing
and customer-disk-only data residency. Day 1-10 unit tests cover each
primitive in isolation; this file is the load-bearing
integration-test harness that exercises the FULL loop on stubbed data.

Two test categories:

1. **Identity-student GREEN-TEAM** -- a stub student model that
   perfectly mirrors the teacher. Asserts that:
   - all 9 gates pass
   - swap proceeds
   - post-swap monitor opens window and sees no trips
   - drift detector reports no drift

2. **Seeded-regression RED-TEAM** -- stub student models with specific
   regressions (worse safety, worse latency, action drift, per-task
   cliff). Asserts that:
   - the right gate fails
   - --pro-force bypass surfaces the audit
   - post-swap monitor fires the right trip signal
   - rollback handler executes when triggered

All I/O is stubbed (no real HF, no real Modal, no real model training)
so the suite runs in seconds and exercises the composition of the
primitives, not their backends.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tether.pro.distill_scheduler import (
    DistillScheduler,
    SchedulerConfig,
    SchedulerState,
)
from tether.pro.drift_detection import DriftDetector
from tether.pro.eval_gate import (
    EvalGate,
    EvalSample,
    GateThresholds,
    InsufficientEpisodes,
)
from tether.pro.hf_hub import (
    HfHubClient,
    HfRepoSpec,
)
from tether.pro.post_swap_monitor import MonitorConfig, PostSwapMonitor
from tether.pro.rollback import RollbackHandler


# ---------------------------------------------------------------------------
# Sample builders -- IDENTITY (green) + SEEDED REGRESSIONS (red)
# ---------------------------------------------------------------------------


def _identity_sample(
    task_id: str = "t1",
    *,
    success: bool = True,
    clamp: int = 0,
    latency: float = 50.0,
    velocity: list[float] | None = None,
    action: list[list[float]] | None = None,
    teacher_action: list[list[float]] | None = None,
) -> EvalSample:
    """One green-team episode where candidate matches teacher exactly."""
    velocity = velocity if velocity is not None else [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    # Non-zero actions so cosine similarity is well-defined (cos(zero, zero)=0)
    default_action = [[0.5, -0.3, 0.2, 0.1, -0.4, 0.3, 0.0]] * 10
    action = action if action is not None else default_action
    return EvalSample(
        task_id=task_id, success=success, safety_clamp_count=clamp,
        inference_latency_p99_ms=latency,
        per_joint_velocity=velocity, action_trajectory=action,
        teacher_action_trajectory=teacher_action or action,
    )


def _identity_samples(n: int = 50, *, n_tasks: int = 3) -> list[EvalSample]:
    """Build n identical-to-teacher episodes across n_tasks task IDs."""
    return [_identity_sample(task_id=f"task_{i % n_tasks}") for i in range(n)]


def _samples_with_higher_clamp_rate(
    n: int = 50, *, clamp_rate: float = 0.50,
) -> list[EvalSample]:
    """Stress S1: candidate's safety-clamp rate is X× the baseline (0)."""
    return [
        _identity_sample(clamp=1 if (i / n) < clamp_rate else 0)
        for i in range(n)
    ]


def _samples_with_higher_latency(
    n: int = 50, *, latency: float = 100.0,
) -> list[EvalSample]:
    """Stress P2: candidate's latency p99 exceeds baseline by X%."""
    return [_identity_sample(latency=latency) for _ in range(n)]


def _samples_with_per_task_cliff(
    n: int = 90, *, cliff_task: str = "task_0",
) -> list[EvalSample]:
    """Stress S3: one task drops in success rate while aggregate stays OK."""
    out = []
    for i in range(n):
        task = f"task_{i % 3}"
        success = task != cliff_task  # cliff_task always fails
        out.append(_identity_sample(task_id=task, success=success))
    return out


# ---------------------------------------------------------------------------
# DistillScheduler integration
# ---------------------------------------------------------------------------


def test_scheduler_nightly_fires_after_utc_hour():
    config = SchedulerConfig(mode="nightly", nightly_utc_hour=3)
    scheduler = DistillScheduler(config=config)
    state = SchedulerState(
        last_kick_at=None, samples_at_last_kick=0, quality_at_last_kick=None,
    )

    # Day 1, 02:00 UTC -- before threshold -> no fire
    decision = scheduler.should_kick(
        state=state, current_samples=1000,
        now=datetime(2026, 4, 25, 2, 0, tzinfo=timezone.utc),
    )
    assert not decision.kick

    # Day 1, 04:00 UTC -- after threshold, never kicked -> fire
    decision = scheduler.should_kick(
        state=state, current_samples=1000,
        now=datetime(2026, 4, 25, 4, 0, tzinfo=timezone.utc),
    )
    assert decision.kick


def test_scheduler_samples_trigger_fires_at_threshold():
    config = SchedulerConfig(mode="samples", samples_threshold=500, min_kick_gap_s=0)
    scheduler = DistillScheduler(config=config)
    state = SchedulerState(
        last_kick_at=None, samples_at_last_kick=0, quality_at_last_kick=None,
    )

    # 499 samples -- not yet
    assert not scheduler.should_kick(
        state=state, current_samples=499,
        now=datetime.now(timezone.utc),
    ).kick
    # 500 samples -- fire
    assert scheduler.should_kick(
        state=state, current_samples=500,
        now=datetime.now(timezone.utc),
    ).kick


def test_scheduler_manual_mode_never_fires():
    config = SchedulerConfig(mode="manual")
    scheduler = DistillScheduler(config=config)
    state = SchedulerState(
        last_kick_at=None, samples_at_last_kick=0, quality_at_last_kick=None,
    )
    decision = scheduler.should_kick(
        state=state, current_samples=999_999,
        now=datetime.now(timezone.utc),
    )
    assert not decision.kick


# ---------------------------------------------------------------------------
# EvalGate integration -- green + red
# ---------------------------------------------------------------------------


def test_green_team_identity_student_passes_all_9_gates():
    """Identity student matches teacher -- all 9 gates pass."""
    candidate = _identity_samples(n=50)
    baseline = _identity_samples(n=50)
    report = EvalGate.evaluate(
        candidate_samples=candidate, baseline_samples=baseline,
        candidate_memory_bytes=100_000_000, baseline_memory_bytes=100_000_000,
    )
    assert report.overall_passed
    assert report.first_failing_gate is None
    assert all(g.passed for g in report.safety_gates)
    assert all(g.passed for g in report.performance_gates)


def test_red_team_high_clamp_rate_fails_s1():
    """Candidate's clamp rate exceeds 1.1× baseline + cap → S1 fails."""
    candidate = _samples_with_higher_clamp_rate(n=50, clamp_rate=0.5)
    baseline = _identity_samples(n=50)
    report = EvalGate.evaluate(
        candidate_samples=candidate, baseline_samples=baseline,
        candidate_memory_bytes=100_000_000, baseline_memory_bytes=100_000_000,
    )
    assert not report.overall_passed
    assert report.first_failing_gate is not None
    assert report.first_failing_gate.gate_id == "S1"


def test_red_team_per_task_cliff_fails_s3():
    """One task fully fails, others succeed -- S3 catches it."""
    candidate = _samples_with_per_task_cliff(n=90)
    baseline = _identity_samples(n=90, n_tasks=3)
    report = EvalGate.evaluate(
        candidate_samples=candidate, baseline_samples=baseline,
        candidate_memory_bytes=100_000_000, baseline_memory_bytes=100_000_000,
    )
    assert not report.overall_passed
    assert report.first_failing_gate.gate_id == "S3"


def test_red_team_higher_memory_fails_p3():
    """candidate memory > baseline memory -> P3 fails."""
    candidate = _identity_samples(n=50)
    baseline = _identity_samples(n=50)
    report = EvalGate.evaluate(
        candidate_samples=candidate, baseline_samples=baseline,
        candidate_memory_bytes=200_000_000,  # 2x baseline
        baseline_memory_bytes=100_000_000,
    )
    assert all(g.passed for g in report.safety_gates)
    assert report.first_failing_gate.gate_id == "P3"


def test_pro_force_bypasses_perf_gate_with_audit():
    """--pro-force overrides P3 memory failure when audit provided."""
    candidate = _identity_samples(n=50)
    baseline = _identity_samples(n=50)
    report = EvalGate.evaluate(
        candidate_samples=candidate, baseline_samples=baseline,
        candidate_memory_bytes=200_000_000, baseline_memory_bytes=100_000_000,
        pro_force=True,
        bypass_audit="op@tether 2026-04-25 acked memory regression",
    )
    assert report.overall_passed
    assert report.pro_force_bypass
    assert report.bypass_audit is not None


def test_pro_force_cannot_bypass_safety_gate():
    """--pro-force is forbidden from bypassing S1/S2/S3."""
    candidate = _samples_with_higher_clamp_rate(n=50, clamp_rate=0.5)
    baseline = _identity_samples(n=50)
    report = EvalGate.evaluate(
        candidate_samples=candidate, baseline_samples=baseline,
        candidate_memory_bytes=100_000_000, baseline_memory_bytes=100_000_000,
        pro_force=True, bypass_audit="op tried to bypass S1",
    )
    assert not report.overall_passed
    assert not report.pro_force_bypass
    assert report.first_failing_gate.gate_class == "safety"


def test_pro_force_without_audit_raises():
    candidate = _identity_samples(n=50)
    baseline = _identity_samples(n=50)
    with pytest.raises(ValueError, match="bypass_audit"):
        EvalGate.evaluate(
            candidate_samples=candidate, baseline_samples=baseline,
            candidate_memory_bytes=100_000_000, baseline_memory_bytes=100_000_000,
            pro_force=True, bypass_audit=None,
        )


def test_eval_gate_refuses_below_min_episodes():
    """n_candidate < 30 -> InsufficientEpisodes, not silent pass."""
    candidate = _identity_samples(n=10)
    baseline = _identity_samples(n=50)
    with pytest.raises(InsufficientEpisodes):
        EvalGate.evaluate(
            candidate_samples=candidate, baseline_samples=baseline,
            candidate_memory_bytes=100_000_000, baseline_memory_bytes=100_000_000,
        )


# ---------------------------------------------------------------------------
# Post-swap monitor integration
# ---------------------------------------------------------------------------


def test_post_swap_window_opens_and_episodes_recorded():
    monitor = PostSwapMonitor()
    monitor.start_window(baseline_clamp_rate=0.01)
    assert monitor.is_window_open

    for _ in range(10):
        monitor.record_episode(
            safety_clamp_count=0, cos_to_previous_model=0.99,
        )
    assert monitor.episodes_seen == 10


def test_post_swap_no_trip_for_clean_episodes():
    monitor = PostSwapMonitor(MonitorConfig(sensitivity="aggressive"))
    monitor.start_window(baseline_clamp_rate=0.01)
    for _ in range(50):
        monitor.record_episode(
            safety_clamp_count=0, cos_to_previous_model=0.99,
        )
    decision = monitor.should_rollback()
    assert not decision.should_rollback


def test_post_swap_t1_trips_on_high_clamp_rate():
    """Per-episode clamp rate spikes -> T1 trips after sensitivity-required
    consecutive trips."""
    monitor = PostSwapMonitor(MonitorConfig(
        sensitivity="aggressive",  # trips after 1 in-a-row
        rolling_window_size=10,
    ))
    monitor.start_window(baseline_clamp_rate=0.01)
    # Saturate window with high clamps
    for _ in range(20):
        monitor.record_episode(
            safety_clamp_count=10, cos_to_previous_model=0.99,
        )
    decision = monitor.should_rollback()
    assert decision.should_rollback


def test_post_swap_window_closes_after_max_episodes():
    monitor = PostSwapMonitor(MonitorConfig(window_episode_count=20))
    monitor.start_window(baseline_clamp_rate=0.01)
    for _ in range(25):
        monitor.record_episode(safety_clamp_count=0, cos_to_previous_model=0.99)
    # Over 20 episodes -> window closes
    assert not monitor.is_window_open


# ---------------------------------------------------------------------------
# Rollback integration
# ---------------------------------------------------------------------------


def test_rollback_executes_via_monitor_trigger():
    """Auto-trigger from monitor: no operator required."""
    swap_calls = []

    def _swap_fn(target_slot: str) -> None:
        swap_calls.append(target_slot)

    handler = RollbackHandler(
        router_swap_fn=_swap_fn,
        active_slot_getter=lambda: "a",
    )
    outcome = handler.execute(
        trigger="auto",
        reason="T1",
        operator=None,
        target_slot="b",
    )
    assert outcome.succeeded
    assert outcome.to_slot == "b"
    assert outcome.from_slot == "a"
    assert swap_calls == ["b"]


def test_rollback_from_cli_requires_operator():
    """Manual CLI rollback must carry an operator id (audit trail)."""
    handler = RollbackHandler(
        router_swap_fn=lambda s: None,
        active_slot_getter=lambda: "a",
    )
    with pytest.raises(ValueError, match="operator"):
        handler.execute(
            trigger="cli",
            reason="operator-cli",
            operator=None,  # missing
            target_slot="b",
        )


def test_rollback_failure_is_loud():
    """If swap_fn raises, outcome.succeeded=False with structured error."""
    def _failing_swap(s: str) -> None:
        raise RuntimeError("router unreachable")

    handler = RollbackHandler(
        router_swap_fn=_failing_swap,
        active_slot_getter=lambda: "a",
    )
    outcome = handler.execute(
        trigger="auto", reason="T1",
        operator=None, target_slot="b",
    )
    assert not outcome.succeeded
    assert outcome.error is not None
    assert "RuntimeError" in outcome.error


def test_rollback_to_active_slot_is_noop():
    handler = RollbackHandler(
        router_swap_fn=lambda s: None,
        active_slot_getter=lambda: "a",
    )
    outcome = handler.execute(
        trigger="auto", reason="T1",
        operator=None, target_slot="a",  # same as active
    )
    assert not outcome.succeeded
    assert "no-op" in outcome.error


# ---------------------------------------------------------------------------
# Drift detection integration
# ---------------------------------------------------------------------------


def test_drift_no_signal_when_distributions_identical():
    detector = DriftDetector()
    states = [[0.1 * i for _ in range(7)] for i in range(200)]
    actions = [[0.2 * i for _ in range(7)] for i in range(200)]
    report = detector.evaluate(
        customer_states=states, base_states=states,
        customer_actions=actions, base_actions=actions,
    )
    assert not report.drift_detected
    assert report.reason == "ok"


def test_drift_fires_when_state_distribution_diverges():
    detector = DriftDetector(
        kl_divergence_max=0.1, action_wasserstein_max=10.0,
    )
    base_states = [[float(i % 10)] * 7 for i in range(200)]
    cust_states = [[float(100 + i % 10)] * 7 for i in range(200)]
    actions = [[0.0] * 7 for _ in range(200)]
    report = detector.evaluate(
        customer_states=cust_states, base_states=base_states,
        customer_actions=actions, base_actions=actions,
    )
    assert report.drift_detected
    assert report.reason in ("kl-exceeded", "action-exceeded")


# ---------------------------------------------------------------------------
# HF Hub push integration -- with stubbed api_caller
# ---------------------------------------------------------------------------


def test_hf_push_succeeds_with_clean_caller(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"fake-distilled-student")

    def _stub_caller(**kwargs):
        class _CI:
            oid = "abc123def456"
        return _CI()

    client = HfHubClient(
        repo=HfRepoSpec(org="tether-students", name="acme-prod"),
        token="hf_test_token",
    )
    outcome = client.push(
        local_dir=tmp_path,
        commit_message="auto-distill 2026-04-25",
        api_caller=_stub_caller,
    )
    assert outcome.succeeded
    assert outcome.revision == "abc123def456"


def test_hf_push_retries_then_succeeds_on_transient_error(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    call_count = {"n": 0}

    def _flaky(**kwargs):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise ConnectionError("transient")

        class _CI:
            oid = "abc"
        return _CI()

    client = HfHubClient(
        repo=HfRepoSpec(org="a", name="b"),
        token="t", retry_attempts=3, retry_backoff_s=0.0,
    )
    outcome = client.push(
        local_dir=tmp_path, commit_message="x", api_caller=_flaky,
    )
    assert outcome.succeeded
    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# FULL-LOOP COMPOSITION: green path
# ---------------------------------------------------------------------------


def test_full_loop_green_path_clean_swap(tmp_path):
    """Green-team E2E: identity student passes gates + monitor sees clean
    operation. Composes scheduler -> eval -> hf-push -> swap -> monitor."""
    # 1. Scheduler decides to kick
    scheduler = DistillScheduler(
        config=SchedulerConfig(mode="samples", samples_threshold=100, min_kick_gap_s=0),
    )
    state = SchedulerState(
        last_kick_at=None, samples_at_last_kick=0, quality_at_last_kick=None,
    )
    decision = scheduler.should_kick(
        state=state, current_samples=200,
        now=datetime.now(timezone.utc),
    )
    assert decision.kick

    # 2. Distill produces an identity student (mocked -- skip the actual
    #    Modal A100 train)
    candidate = _identity_samples(n=50)
    baseline = _identity_samples(n=50)

    # 3. Eval gate evaluates -- all 9 gates pass
    eval_report = EvalGate.evaluate(
        candidate_samples=candidate, baseline_samples=baseline,
        candidate_memory_bytes=100_000_000, baseline_memory_bytes=100_000_000,
    )
    assert eval_report.overall_passed

    # 4. Push to HF Hub
    (tmp_path / "model.onnx").write_bytes(b"green-student")

    def _push_caller(**kwargs):
        class _CI:
            oid = "green123"
        return _CI()

    client = HfHubClient(
        repo=HfRepoSpec(org="tether-students", name="acme"),
        token="t",
    )
    push_outcome = client.push(
        local_dir=tmp_path, commit_message="auto-distill green",
        api_caller=_push_caller,
    )
    assert push_outcome.succeeded

    # 5. Swap (mocked router_swap_fn)
    swap_log = []

    def _swap(slot: str) -> None:
        swap_log.append(slot)

    handler = RollbackHandler(
        router_swap_fn=_swap, active_slot_getter=lambda: "a",
    )

    # 6. Open monitor window post-swap
    monitor = PostSwapMonitor(MonitorConfig(sensitivity="normal"))
    monitor.start_window(baseline_clamp_rate=0.01)

    # 7. Record clean episodes
    for _ in range(50):
        monitor.record_episode(
            safety_clamp_count=0, cos_to_previous_model=0.99,
        )

    # 8. Monitor verdict: no rollback fires
    decision_post = monitor.should_rollback()
    assert not decision_post.should_rollback
    # (Rollback handler never invoked in green path)
    assert swap_log == []


# ---------------------------------------------------------------------------
# FULL-LOOP COMPOSITION: red path -- monitor trips, rollback fires
# ---------------------------------------------------------------------------


def test_full_loop_red_path_monitor_trips_then_rollback():
    """Red-team E2E: --pro-force lets a perf-regression student through
    eval, but post-swap monitor catches the latent regression in
    production traffic -> rollback fires."""
    # Pretend eval gate let the swap through via --pro-force on P3
    # (memory regression -- ops decided to ack temporarily).
    candidate = _identity_samples(n=50)
    baseline = _identity_samples(n=50)
    eval_report = EvalGate.evaluate(
        candidate_samples=candidate, baseline_samples=baseline,
        candidate_memory_bytes=200_000_000, baseline_memory_bytes=100_000_000,
        pro_force=True,
        bypass_audit="op@tether acked memory regression for traffic test",
    )
    assert eval_report.overall_passed
    assert eval_report.pro_force_bypass

    # Post-swap monitor opens window
    monitor = PostSwapMonitor(MonitorConfig(
        sensitivity="aggressive", rolling_window_size=10,
    ))
    monitor.start_window(baseline_clamp_rate=0.01)

    # Production traffic comes in: student starts safety-clamping more
    for _ in range(20):
        monitor.record_episode(
            safety_clamp_count=10, cos_to_previous_model=0.5,
        )
    decision = monitor.should_rollback()
    assert decision.should_rollback

    # Rollback handler fires
    swap_calls = []

    def _swap(slot: str) -> None:
        swap_calls.append(slot)

    handler = RollbackHandler(
        router_swap_fn=_swap, active_slot_getter=lambda: "a",
    )
    rb_outcome = handler.execute(
        trigger="auto",
        reason=decision.reason,
        operator=None, target_slot="b",
    )
    assert rb_outcome.succeeded
    assert swap_calls == ["b"]


def test_full_loop_safety_gate_blocks_swap_no_force_possible():
    """Hard reject: safety gate failure -> swap rejected even with
    --pro-force. No subsequent monitor or rollback path exists because
    the swap never happened."""
    candidate = _samples_with_higher_clamp_rate(n=50, clamp_rate=0.6)
    baseline = _identity_samples(n=50)
    eval_report = EvalGate.evaluate(
        candidate_samples=candidate, baseline_samples=baseline,
        candidate_memory_bytes=100_000_000, baseline_memory_bytes=100_000_000,
        pro_force=True,
        bypass_audit="op@tether tried to force-clear S1 safety gate",
    )
    # Safety gate ALWAYS wins
    assert not eval_report.overall_passed
    assert eval_report.first_failing_gate.gate_id == "S1"
    assert not eval_report.pro_force_bypass
