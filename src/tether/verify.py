"""Action-parity gate orchestrator for `tether verify` (v0).

`tether verify` answers a single customer question: *does my OPTIMIZED export
(ONNX / Triton) still behave like the ORIGINAL native-PyTorch policy?* It runs
both policies through the same LIBERO loop, pairs their per-episode outcomes,
scores the pair through the shipped Pro 9-gate evaluator, and emits a PASS/FAIL
verdict plus a written ``PARITY.md`` receipt.

v0 deliberately REUSES shipped components rather than inventing new metrics:

* :func:`tether.eval.libero_rollout.run_libero_rollout` gathers paired
  (original, optimized) episode outcomes — it already supports ``use_native``
  to flip between native-PyTorch (the *original*) and ONNX/Triton inference
  (the *optimized* export) on the exact same proven loop. We call it twice on
  the same suite + seed + task set and pair the results by ``task_id``.
* :class:`tether.pro.eval_gate.EvalGate` does ALL the metric math: Wasserstein-1
  on joint velocities (S2), action cosine similarity (P4), Wilson-CI aggregate
  + per-task success (P1/P5), the per-task success-cliff veto (S3), and the
  n>=30 statistical-power floor. We map original→``baseline_samples`` and
  optimized→``candidate_samples`` and let the gate decide.

Verification Evidence v1 captures executed actions, inference latency, joint
velocity, safety-clamp counts, success, and process/device memory for both
arms. Missing required evidence is typed with a bounded reason and fails the
first affected gate; it is never replaced with a numeric or empty sentinel.

This module is import-light: ``run_libero_rollout`` (and therefore torch /
LIBERO / mujoco) is imported lazily inside :func:`gather_paired_samples`, so
importing :mod:`tether.verify` for the verdict types or the unit tests costs
nothing. The scoring + aggregation layer is pure and fully mockable via the
``gather_fn`` seam on :func:`run_verify`.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from tether.pro.eval_gate import (
    EvalGate,
    EvalReport,
    EvalSample,
    GateThresholds,
    InsufficientEpisodes,
    MIN_EPISODES_TO_EVALUATE,
)
from tether.verify_metrics import (
    EmbodiedParity,
    TwoSampleResult,
    aggregate_embodied,
    two_sample_test,
)
from tether.verification_evidence import (
    EvidenceValue,
    UnavailableReason,
    serialize_evidence,
    normalize_verification_device,
)

logger = logging.getLogger(__name__)


# Suites we accept today. Mirrors the `tether eval` Phase-1 surface (LIBERO
# only); SimplerEnv / customer suites are a separate roadmap item.
SUPPORTED_SUITES: tuple[str, ...] = ("libero",)


# A "gather" callable returns paired episode-outcome dicts, in the exact shape
# `run_libero_rollout` returns. The default implementation runs the real
# rollouts; tests inject a synthetic stub of the same signature. Keeping this a
# plain Callable (not a Protocol) keeps the test seam trivial.
GatherFn = Callable[..., tuple[dict[str, Any], dict[str, Any]]]


@dataclass(frozen=True)
class ParityVerdict:
    """Structured outcome of `tether verify` — frozen so the CLI / report
    writer pass it around without worrying about mutation.

    Wraps the Pro :class:`~tether.pro.eval_gate.EvalReport` (the real scoring)
    with the verify-specific framing: which export, which original, which
    suite, and the headline success rates that make the verdict legible
    without re-deriving them from the gate internals.
    """

    passed: bool
    eval_report: EvalReport  # the Pro 9-gate report (source of truth)
    optimized_ref: str  # path / HF id of the export under test
    original_ref: str  # path / HF id of the native-PyTorch reference
    suite: str
    target: str
    n_episodes: int  # paired episode count (== candidate == baseline)
    original_success_rate: float | EvidenceValue[float]  # in [0, 1] when available
    optimized_success_rate: float | EvidenceValue[float]  # in [0, 1] when available
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )
    two_sample: TwoSampleResult | None = None  # distributional parity (None if no per-step data)
    two_sample_episodes: int = 0  # # episodes the distributional test compared (both arms succeeded)
    embodied: EmbodiedParity | None = None  # kinematic parity (None if no per-step data)

    @property
    def candidate_not_worse(self) -> bool:
        """Rich-verdict helper: candidate is >= baseline on success AND has no
        embodied regression. A 'different but not worse' export is True here even
        when ``two_sample.distributions_differ`` (the distributional flag alone
        doesn't make the candidate worse — it surfaces a shift for review)."""
        no_embodied_regress = self.embodied is None or not self.embodied.regressed()
        delta = self.success_rate_delta
        return (
            not isinstance(delta, EvidenceValue)
            and delta >= 0.0
            and no_embodied_regress
        )

    @property
    def success_rate_delta(self) -> float | EvidenceValue[float]:
        """optimized - original. Negative => the export regressed."""
        if isinstance(self.original_success_rate, EvidenceValue):
            return EvidenceValue.unavailable(
                self.original_success_rate.reason or UnavailableReason.MISSING_FIELD
            )
        if isinstance(self.optimized_success_rate, EvidenceValue):
            return EvidenceValue.unavailable(
                self.optimized_success_rate.reason or UnavailableReason.MISSING_FIELD
            )
        return self.optimized_success_rate - self.original_success_rate

    @property
    def first_failing_gate_id(self) -> str | None:
        g = self.eval_report.first_failing_gate
        return g.gate_id if g is not None else None

    def to_dict(self) -> dict[str, Any]:
        diagnostics = {
            "two_sample": (
                {"status": "EVALUATED", "value": self.two_sample.to_dict()}
                if self.two_sample is not None
                else {
                    "status": "NOT_EVALUATED",
                    "reason": UnavailableReason.MISSING_FIELD.value,
                }
            ),
            "embodied": (
                {"status": "EVALUATED", "value": self.embodied.to_dict()}
                if self.embodied is not None
                else {
                    "status": "NOT_EVALUATED",
                    "reason": UnavailableReason.MISSING_FIELD.value,
                }
            ),
        }
        return {
            "passed": self.passed,
            "optimized_ref": self.optimized_ref,
            "original_ref": self.original_ref,
            "suite": self.suite,
            "target": self.target,
            "n_episodes": self.n_episodes,
            "original_success_rate": serialize_evidence(self.original_success_rate),
            "optimized_success_rate": serialize_evidence(self.optimized_success_rate),
            "success_rate_delta": serialize_evidence(self.success_rate_delta),
            "first_failing_gate_id": self.first_failing_gate_id,
            "generated_at": self.generated_at,
            "two_sample": self.two_sample.to_dict() if self.two_sample else None,
            "two_sample_episodes": self.two_sample_episodes,
            "candidate_not_worse": self.candidate_not_worse,
            "embodied": self.embodied.to_dict() if self.embodied else None,
            "diagnostics": diagnostics,
            "eval_report": self.eval_report.to_dict(),
        }


# ---------------------------------------------------------------------------
# Rollout-results -> EvalSample adapter
# ---------------------------------------------------------------------------


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return ordered[
        max(0, math.ceil((percentile / 100.0) * len(ordered)) - 1)
    ]


def _nearest_rank_p99(values: list[float]) -> float | None:
    return _nearest_rank(values, 99.0)


def _episode_index(results: dict[str, Any]) -> dict[tuple[Any, Any], dict[str, Any]]:
    indexed: dict[tuple[Any, Any], dict[str, Any]] = {}
    for task in results.get("per_task", []) or []:
        task_idx = task.get("task_idx")
        for episode in task.get("episodes", []) or []:
            indexed[(task_idx, episode.get("ep"))] = episode
    return indexed


def _captured_velocities(episode: dict[str, Any]) -> list[float] | None:
    measured = episode.get("joint_velocities")
    if measured:
        return [
            float(value)
            for row in measured
            for value in np.asarray(row, dtype=np.float64).reshape(-1)
        ]

    actions = episode.get("actions") or []
    timestamps = episode.get("action_timestamps_s") or []
    if len(actions) < 2 or len(actions) != len(timestamps):
        return None
    velocities: list[float] = []
    for previous, current, t0, t1 in zip(
        actions[:-1], actions[1:], timestamps[:-1], timestamps[1:]
    ):
        dt = float(t1) - float(t0)
        if dt <= 0:
            return None
        prev_array = np.asarray(previous, dtype=np.float64).reshape(-1)
        cur_array = np.asarray(current, dtype=np.float64).reshape(-1)
        if prev_array.shape != cur_array.shape:
            return None
        velocities.extend((np.abs(cur_array - prev_array) / dt).tolist())
    return velocities or None


def _rollout_results_to_samples(
    results: dict[str, Any],
    *,
    teacher_results: dict[str, Any] | None = None,
) -> list[EvalSample]:
    """Adapt captured rollout records without inventing missing measurements."""

    teacher_index = _episode_index(teacher_results or {})
    samples: list[EvalSample] = []
    for task in results.get("per_task", []) or []:
        task_id = str(task.get("task_idx", task.get("task_description", "unknown")))
        for ep in task.get("episodes", []) or []:
            raw_actions = ep.get("actions") or None
            actions = (
                [np.asarray(action, dtype=np.float64).reshape(-1).tolist() for action in raw_actions]
                if raw_actions
                else None
            )
            latency = _nearest_rank_p99(
                [float(value) for value in ep.get("inference_latency_ms", []) or []]
            )
            teacher_actions: list[list[float]] | None = None
            if teacher_results is not None:
                teacher_ep = teacher_index.get((task.get("task_idx"), ep.get("ep")))
                raw_teacher = teacher_ep.get("actions") if teacher_ep else None
                if raw_teacher and actions:
                    candidate_array = np.asarray(actions, dtype=np.float64)
                    teacher_array = np.asarray(raw_teacher, dtype=np.float64)
                    if candidate_array.shape == teacher_array.shape:
                        teacher_actions = teacher_array.tolist()
            samples.append(
                EvalSample(
                    task_id=task_id,
                    success=(bool(ep["success"]) if "success" in ep else None),
                    safety_clamp_count=(
                        int(ep["safety_clamp_count"])
                        if "safety_clamp_count" in ep
                        else None
                    ),
                    inference_latency_p99_ms=latency,
                    per_joint_velocity=_captured_velocities(ep),
                    action_trajectory=actions,
                    teacher_action_trajectory=teacher_actions,
                )
            )
    return samples


def _success_rate(samples: list[EvalSample]) -> float:
    if not samples:
        return 0.0
    if any(sample.success is None for sample in samples):
        raise ValueError("success evidence is unavailable")
    return sum(1 for s in samples if s.success) / len(samples)


def _unavailable_reason_from_result(
    results: dict[str, Any],
    key: str,
    default: UnavailableReason = UnavailableReason.MISSING_FIELD,
) -> UnavailableReason:
    payload = results.get(key)
    if isinstance(payload, dict) and payload.get("status") == "unavailable":
        try:
            return UnavailableReason(str(payload.get("reason")))
        except ValueError:
            return UnavailableReason.MEASUREMENT_FAILED
    return default


def _combine_arm_evidence(
    baseline: EvidenceValue[Any],
    candidate: EvidenceValue[Any],
) -> EvidenceValue[dict[str, Any]]:
    if not baseline.is_available:
        return EvidenceValue.unavailable(
            baseline.reason or UnavailableReason.MISSING_FIELD
        )
    if not candidate.is_available:
        return EvidenceValue.unavailable(
            candidate.reason or UnavailableReason.MISSING_FIELD
        )
    return EvidenceValue.available(
        {"baseline": baseline.value, "candidate": candidate.value}
    )


def _arm_sample_evidence(
    results: dict[str, Any], samples: list[EvalSample]
) -> dict[str, EvidenceValue[Any]]:
    n_episodes = len(samples)
    success = (
        EvidenceValue.available(
            {
                "episodes": n_episodes,
                "successes": sum(1 for sample in samples if sample.success),
            }
        )
        if samples and all(sample.success is not None for sample in samples)
        else EvidenceValue.unavailable(UnavailableReason.MISSING_FIELD)
    )

    safety = (
        EvidenceValue.available(
            {
                "episodes": n_episodes,
                "clamp_count": sum(
                    int(sample.safety_clamp_count or 0) for sample in samples
                ),
            }
        )
        if samples and all(
            sample.safety_clamp_count is not None for sample in samples
        )
        else EvidenceValue.unavailable(
            _unavailable_reason_from_result(results, "safety_evidence")
        )
    )

    actions = (
        EvidenceValue.available(
            {
                "episodes": n_episodes,
                "steps": sum(len(sample.action_trajectory or []) for sample in samples),
            }
        )
        if samples and all(sample.action_trajectory for sample in samples)
        else EvidenceValue.unavailable(UnavailableReason.MISSING_FIELD)
    )

    velocities = (
        EvidenceValue.available(
            {
                "episodes": n_episodes,
                "samples": sum(len(sample.per_joint_velocity or []) for sample in samples),
            }
        )
        if samples and all(sample.per_joint_velocity for sample in samples)
        else EvidenceValue.unavailable(UnavailableReason.MISSING_FIELD)
    )

    latency = (
        EvidenceValue.available(
            {
                "episodes": n_episodes,
                "episode_p99_ms": [
                    float(sample.inference_latency_p99_ms)
                    for sample in samples
                    if sample.inference_latency_p99_ms is not None
                ],
            }
        )
        if samples and all(
            sample.inference_latency_p99_ms is not None
            and sample.inference_latency_p99_ms > 0
            for sample in samples
        )
        else EvidenceValue.unavailable(UnavailableReason.MISSING_FIELD)
    )
    return {
        "success": success,
        "safety_clamps": safety,
        "executed_actions": actions,
        "joint_velocity": velocities,
        "latency": latency,
    }


def _memory_evidence(
    results: dict[str, Any], *, expected_device: str
) -> EvidenceValue[Any]:
    from tether.eval.evidence_capture import (
        MEMORY_CAPTURE_JITTER_S,
        MEMORY_DEADLINE_EPSILON_S,
        MEMORY_SAMPLE_INTERVAL_S,
    )

    payload = results.get("memory_evidence")
    if not isinstance(payload, dict):
        return EvidenceValue.unavailable(UnavailableReason.MISSING_FIELD)
    try:
        value = EvidenceValue.from_dict(payload)
    except (TypeError, ValueError):
        return EvidenceValue.unavailable(UnavailableReason.MEASUREMENT_FAILED)
    if not value.is_available:
        return value
    summary = value.value
    if not isinstance(summary, dict):
        return EvidenceValue.unavailable(UnavailableReason.MEASUREMENT_FAILED)
    required = (
        "sample_hz",
        "backend",
        "process_identity",
        "window",
        "samples",
        "process_rss",
        "device_allocated",
        "combined",
    )
    if any(name not in summary for name in required):
        return EvidenceValue.unavailable(UnavailableReason.MISSING_FIELD)
    if summary["sample_hz"] != 10.0 or summary["backend"] != expected_device:
        return EvidenceValue.unavailable(UnavailableReason.MEASUREMENT_FAILED)
    identity = summary["process_identity"]
    window = summary["window"]
    if (
        not isinstance(identity, dict)
        or not isinstance(identity.get("pid"), int)
        or identity["pid"] <= 0
        or not isinstance(identity.get("capture_id"), str)
        or not identity["capture_id"]
        or not isinstance(window, dict)
    ):
        return EvidenceValue.unavailable(UnavailableReason.MEASUREMENT_FAILED)
    samples = summary["samples"]
    if not isinstance(samples, list) or not samples:
        return EvidenceValue.unavailable(UnavailableReason.MEASUREMENT_FAILED)
    rss_values: list[float] = []
    device_values: list[float] = []
    for sample in samples:
        if not isinstance(sample, dict):
            return EvidenceValue.unavailable(UnavailableReason.MEASUREMENT_FAILED)
        rss = sample.get("process_rss_bytes")
        device = sample.get("device_allocated_bytes")
        scheduled = sample.get("scheduled_monotonic_s")
        captured = sample.get("captured_monotonic_s")
        if (
            not isinstance(rss, (int, float))
            or not isinstance(device, (int, float))
            or not isinstance(scheduled, (int, float))
            or not isinstance(captured, (int, float))
        ):
            return EvidenceValue.unavailable(UnavailableReason.MEASUREMENT_FAILED)
        if (
            not all(
                math.isfinite(float(value))
                for value in (rss, device, scheduled, captured)
            )
            or float(rss) < 0
            or float(device) < 0
            or float(captured) < float(scheduled)
        ):
            return EvidenceValue.unavailable(UnavailableReason.MEASUREMENT_FAILED)
        rss_values.append(float(rss))
        device_values.append(float(device))
    captured_times = [float(sample["captured_monotonic_s"]) for sample in samples]
    scheduled_times = [float(sample["scheduled_monotonic_s"]) for sample in samples]
    started = window.get("started_monotonic_s")
    ended = window.get("ended_monotonic_s")
    duration = window.get("duration_s")
    expected_count = window.get("expected_samples")
    captured_count = window.get("captured_samples")
    reported_max_gap = window.get("max_gap_s")
    if not all(
        isinstance(value, (int, float))
        for value in (
            started,
            ended,
            duration,
            expected_count,
            captured_count,
            reported_max_gap,
        )
    ):
        return EvidenceValue.unavailable(UnavailableReason.MEASUREMENT_FAILED)
    assert isinstance(started, (int, float))
    assert isinstance(ended, (int, float))
    assert isinstance(duration, (int, float))
    assert isinstance(expected_count, (int, float))
    assert isinstance(captured_count, (int, float))
    assert isinstance(reported_max_gap, (int, float))
    interval = MEMORY_SAMPLE_INTERVAL_S
    gaps = [
        current - previous
        for previous, current in zip(captured_times, captured_times[1:])
    ]
    if (
        float(duration) < interval
        or float(ended) <= float(started)
        or not all(
            math.isfinite(float(value))
            for value in (started, ended, duration, reported_max_gap)
        )
        or not math.isclose(
            float(ended) - float(started),
            float(duration),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or int(captured_count) != len(samples)
        or float(captured_count) != int(captured_count)
        or float(expected_count) != int(expected_count)
        or int(expected_count)
        != max(
            2,
            math.floor(
                (float(duration) + MEMORY_DEADLINE_EPSILON_S) / interval
            ) + 1,
        )
        or len(samples) != int(expected_count)
        or any(
            not math.isclose(
                scheduled,
                float(started) + (index * interval),
                rel_tol=0.0,
                abs_tol=MEMORY_DEADLINE_EPSILON_S,
            )
            for index, scheduled in enumerate(scheduled_times)
        )
        or float(ended) - scheduled_times[-1]
        > interval + MEMORY_CAPTURE_JITTER_S
        or any(
            not interval - MEMORY_CAPTURE_JITTER_S
            <= gap
            <= interval + MEMORY_CAPTURE_JITTER_S
            for gap in gaps
        )
        or not math.isclose(
            float(reported_max_gap),
            max(gaps) if gaps else 0.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or any(
            not 0.0 <= captured - scheduled <= MEMORY_CAPTURE_JITTER_S
            for captured, scheduled in zip(captured_times, scheduled_times)
        )
    ):
        return EvidenceValue.unavailable(UnavailableReason.MEASUREMENT_FAILED)
    if summary["backend"] == "cpu" and any(device != 0 for device in device_values):
        return EvidenceValue.unavailable(UnavailableReason.MEASUREMENT_FAILED)

    expected_series = {
        "process_rss": rss_values,
        "device_allocated": device_values,
        "combined": [
            rss + device for rss, device in zip(rss_values, device_values)
        ],
    }
    for name, values in expected_series.items():
        captured = summary[name]
        if not isinstance(captured, dict):
            return EvidenceValue.unavailable(UnavailableReason.MISSING_FIELD)
        expected_peak = max(values)
        expected_p95 = _nearest_rank(values, 95.0)
        if (
            captured.get("peak_bytes") != expected_peak
            or captured.get("p95_bytes") != expected_p95
        ):
            return EvidenceValue.unavailable(UnavailableReason.MEASUREMENT_FAILED)
    return value


def _verification_evidence(
    original_results: dict[str, Any],
    optimized_results: dict[str, Any],
    baseline_samples: list[EvalSample],
    candidate_samples: list[EvalSample],
    *,
    verification_device: str,
) -> dict[str, EvidenceValue[Any]]:
    baseline = _arm_sample_evidence(original_results, baseline_samples)
    candidate = _arm_sample_evidence(optimized_results, candidate_samples)
    combined = {
        name: _combine_arm_evidence(baseline[name], candidate[name])
        for name in (
            "success",
            "safety_clamps",
            "executed_actions",
            "joint_velocity",
            "latency",
        )
    }
    original_safety_identity = original_results.get("safety_evidence")
    optimized_safety_identity = optimized_results.get("safety_evidence")
    if (
        isinstance(original_safety_identity, dict)
        and original_safety_identity.get("status") == "available"
        and isinstance(optimized_safety_identity, dict)
        and optimized_safety_identity.get("status") == "available"
        and original_safety_identity != optimized_safety_identity
    ):
        combined["safety_clamps"] = EvidenceValue.unavailable(
            UnavailableReason.MEASUREMENT_FAILED
        )
    combined["memory"] = _combine_arm_evidence(
        _memory_evidence(
            original_results, expected_device=verification_device
        ),
        _memory_evidence(
            optimized_results, expected_device=verification_device
        ),
    )
    if len(baseline_samples) != len(candidate_samples):
        combined["memory"] = EvidenceValue.unavailable(
            UnavailableReason.MEASUREMENT_FAILED
        )
    if combined["memory"].is_available:
        memory_value = combined["memory"].value
        assert isinstance(memory_value, dict)
        baseline_identity = memory_value["baseline"]["process_identity"]
        candidate_identity = memory_value["candidate"]["process_identity"]
        if (
            baseline_identity["pid"] == candidate_identity["pid"]
            or baseline_identity["capture_id"] == candidate_identity["capture_id"]
        ):
            combined["memory"] = EvidenceValue.unavailable(
                UnavailableReason.MEASUREMENT_FAILED
            )
    if candidate_samples and all(
        sample.teacher_action_trajectory for sample in candidate_samples
    ):
        combined["teacher_trajectory"] = EvidenceValue.available(
            {
                "episodes": len(candidate_samples),
                "steps": sum(
                    len(sample.teacher_action_trajectory or [])
                    for sample in candidate_samples
                ),
            }
        )
    else:
        combined["teacher_trajectory"] = EvidenceValue.unavailable(
            UnavailableReason.MISSING_FIELD
        )
    return combined


def _validate_rollout_device(
    results: dict[str, Any], *, expected: str, arm: str
) -> None:
    actual = results.get("verification_device")
    if actual != expected:
        raise ValueError(
            f"{arm} rollout device mismatch: expected {expected!r}, got {actual!r}"
        )


def _memory_p95(evidence: EvidenceValue[Any], arm: str) -> float | None:
    if not evidence.is_available or not isinstance(evidence.value, dict):
        return None
    arm_value = evidence.value.get(arm)
    if not isinstance(arm_value, dict):
        return None
    combined = arm_value.get("combined")
    if not isinstance(combined, dict):
        return None
    value = combined.get("p95_bytes")
    return float(value) if isinstance(value, (int, float)) else None


def _collect_step_actions(results: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Stack every per-step *applied* action into ``(N, D)`` + an ``(N,)`` array of
    episode ids (globally unique across tasks) so the two-sample test can permute
    whole episodes, not steps.

    The applied action — what the policy actually commanded each control step —
    has identical layout (7-dim) for BOTH the native and the optimized arm, so
    the two-sample test compares like with like. The model-internal *predicted
    chunk* does NOT: native ``select_action`` exposes one action per call while
    the decomposed path returns a full multi-step chunk, so their flattened
    widths differ (7 vs 350) and are not comparable — comparing those silently
    no-ops the distributional gate, which is the bug this collector fixes.

    The episode ids matter just as much: per-step actions are autocorrelated
    within an episode, so the two-sample test MUST permute at episode granularity
    (see ``verify_metrics.two_sample_test``). Without them the test over-rejects.

    Returns ``((0, 0), (0,))`` when the rollout didn't capture trajectories (tap
    off or older results) — the optional two-sample diagnostic is then marked
    ``NOT_EVALUATED`` while required trajectory gates fail closed.
    """
    rows: list[np.ndarray] = []
    groups: list[int] = []
    ep_uid = 0
    for task in results.get("per_task", []) or []:
        for ep in task.get("episodes", []) or []:
            acts = ep.get("actions", []) or []
            for act in acts:
                rows.append(np.asarray(act, dtype=np.float64).reshape(-1))
                groups.append(ep_uid)
            if acts:
                ep_uid += 1
    if not rows:
        return np.empty((0, 0)), np.empty((0,))
    width = min(r.shape[0] for r in rows)
    if not width:
        return np.empty((0, 0)), np.empty((0,))
    return np.vstack([r[:width] for r in rows]), np.asarray(groups)


def _collect_paired_succeeded_step_actions(
    original_results: dict[str, Any], optimized_results: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-step applied actions for both arms, RESTRICTED to episodes BOTH arms
    succeeded (matched by ``(task_idx, ep)``).

    Conditioning on commonly-succeeded episodes isolates the *policy* shift
    (e.g. bf16 vs fp32 numerics) from the *outcome* shift. If one arm fails more
    episodes, those failures inject very different actions (a robot flailing for
    520 steps) that make the pooled action distributions differ for a reason
    unrelated to per-step policy fidelity — i.e. the distributional test would
    flag a difference that is really just "this arm succeeds less", which the
    success-rate gate already measures. Comparing actions only where both arms
    accomplished the same task answers the question the gate actually wants:
    *given the same successful outcome, does the optimized policy act the same?*

    Returns ``(base_actions, base_groups, cand_actions, cand_groups,
    n_episodes)``. Each arm carries its own per-episode group ids (the
    two-sample test offsets them so the arms' episodes stay distinct units);
    a shared id per commonly-succeeded episode keeps the bookkeeping simple.
    Returns empties + 0 when no episode succeeded in both arms (the two-sample
    test is then marked ``NOT_EVALUATED``; required trajectory gates still
    fail closed).
    """
    def _index(results: dict[str, Any]) -> dict:
        idx = {}
        for task in results.get("per_task", []) or []:
            ti = task.get("task_idx")
            for ep in task.get("episodes", []) or []:
                idx[(ti, ep.get("ep"))] = ep
        return idx

    oi, ci = _index(original_results), _index(optimized_results)
    b_rows: list[np.ndarray] = []
    c_rows: list[np.ndarray] = []
    b_g: list[int] = []
    c_g: list[int] = []
    uid = 0
    for key in sorted(oi.keys() & ci.keys()):
        oe, ce = oi[key], ci[key]
        if not (oe.get("success") and ce.get("success")):
            continue
        oa = oe.get("actions") or []
        ca = ce.get("actions") or []
        if not oa or not ca:
            continue
        for a in oa:
            b_rows.append(np.asarray(a, dtype=np.float64).reshape(-1))
            b_g.append(uid)
        for a in ca:
            c_rows.append(np.asarray(a, dtype=np.float64).reshape(-1))
            c_g.append(uid)
        uid += 1
    empty = (np.empty((0, 0)), np.empty((0,)), np.empty((0, 0)), np.empty((0,)), 0)
    if not b_rows or not c_rows:
        return empty
    width = min(min(r.shape[0] for r in b_rows), min(r.shape[0] for r in c_rows))
    if not width:
        return empty
    base = np.vstack([r[:width] for r in b_rows])
    cand = np.vstack([r[:width] for r in c_rows])
    return base, np.asarray(b_g), cand, np.asarray(c_g), uid


def _collect_eef_and_steps(results: dict[str, Any]) -> tuple[list[np.ndarray], list[float]]:
    """Per-episode end-effector position trajectories + completion-step counts."""
    positions: list[np.ndarray] = []
    steps: list[float] = []
    for task in results.get("per_task", []) or []:
        for ep in task.get("episodes", []) or []:
            eef = ep.get("eef_positions") or []
            if len(eef) > 1:
                positions.append(np.asarray(eef, dtype=np.float64))
            steps.append(float(ep.get("steps", 0) or 0))
    return positions, steps


# ---------------------------------------------------------------------------
# Paired-sample gathering (the only side-effecting / model-loading seam)
# ---------------------------------------------------------------------------


def _execute_verification_arm(
    *,
    arm: str,
    optimized_ref: str,
    original_checkpoint: str,
    task_suite_name: str,
    num_episodes: int,
    task_indices: list[int] | None,
    seed: int,
    preprocessor_ref: str | None,
    verification_device: str,
    safety_limits: Any | None,
    safety_config_sha256: str | None,
) -> dict[str, Any]:
    """Load and execute exactly one verification arm in the current process."""

    from tether.eval.evidence_capture import ProcessDeviceMemorySampler
    from tether.eval.libero_rollout import (
        load_pi05_policy_and_processors,
        load_verification_policy_context_and_processors,
        run_libero_rollout,
    )
    from tether.export_config import load_tether_config

    verification_device = normalize_verification_device(verification_device)
    export_config = load_tether_config(optimized_ref, inspect_artifacts=True)
    model_type = str(export_config["model_type"])

    if arm == "original":
        policy, preprocessor, postprocessor = load_pi05_policy_and_processors(
            student_checkpoint=original_checkpoint,
            decomposed_dir=optimized_ref,
            preprocessor_ref=preprocessor_ref,
            model_type=model_type,
            require_exact_checkpoint=True,
            device=verification_device,
        )
        inference = None
        use_native = True
        label = "ORIGINAL"
    elif arm == "optimized":
        policy, preprocessor, postprocessor = (
            load_verification_policy_context_and_processors(
                checkpoint=original_checkpoint,
                model_type=model_type,
                preprocessor_ref=preprocessor_ref,
                decomposed_dir=optimized_ref,
                device=verification_device,
            )
        )
        from tether.runtime.verify_inference import load_verification_inference

        inference = load_verification_inference(
            optimized_ref, device=verification_device
        )
        use_native = False
        label = "OPTIMIZED"
    else:
        raise ValueError(f"unknown verification arm: {arm!r}")

    return run_libero_rollout(
        inference=inference,
        use_native=use_native,
        label=label,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task_suite_name=task_suite_name,
        num_episodes=num_episodes,
        task_indices=task_indices,
        seed=seed,
        capture_trajectories=True,
        safety_limits=safety_limits,
        safety_config_sha256=safety_config_sha256,
        verification_device=verification_device,
        memory_sampler=ProcessDeviceMemorySampler(device=verification_device),
    )


def _verification_arm_worker(result_queue: Any, kwargs: dict[str, Any]) -> None:
    """Spawn target that reports either the arm result or a readable failure."""

    import traceback

    try:
        result_queue.put(("ok", _execute_verification_arm(**kwargs)))
    except BaseException:  # noqa: BLE001 - child failure must reach the parent
        result_queue.put(("error", traceback.format_exc()))


def _terminate_and_join_process(process: Any) -> None:
    """Leave no live verification child after any parent-side failure."""

    try:
        alive = process.is_alive()
    except (AssertionError, OSError, ValueError):
        alive = False
    if alive:
        process.terminate()
        try:
            process.join(timeout=10)
        except (AssertionError, OSError, ValueError):
            pass
    try:
        alive = process.is_alive()
    except (AssertionError, OSError, ValueError):
        alive = False
    if alive:
        process.kill()
    try:
        process.join(timeout=10)
    except (AssertionError, OSError, ValueError):
        pass


def _validate_paired_rollout_identity(
    original: dict[str, Any],
    optimized: dict[str, Any],
    *,
    seed: int,
    safety_limits: Any | None,
) -> None:
    """Reject paired arms that did not use one seed/safety identity."""

    from tether.eval.libero_rollout import PAIRED_SEED_PROTOCOL

    for arm, results in (("original", original), ("optimized", optimized)):
        if results.get("seed") != seed or results.get("seed_protocol") != PAIRED_SEED_PROTOCOL:
            raise ValueError(f"{arm} rollout seed identity mismatch")
        if safety_limits is None:
            continue
        evidence = results.get("safety_evidence")
        if not isinstance(evidence, dict) or evidence.get("status") != "available":
            raise ValueError(f"{arm} rollout safety identity is unavailable")
        value = evidence.get("value")
        if (
            not isinstance(value, dict)
            or value.get("sha256") != safety_limits.sha256
            or value.get("limits") != safety_limits.to_dict()
        ):
            raise ValueError(f"{arm} rollout safety identity mismatch")


def _run_isolated_verification_arm(**kwargs: Any) -> dict[str, Any]:
    """Run an arm in a fresh spawned process so memory attribution is isolated."""

    import multiprocessing
    import queue
    import time

    context = multiprocessing.get_context("spawn")
    result_queue: Any | None = None
    process: Any | None = None
    try:
        result_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=_verification_arm_worker,
            args=(result_queue, kwargs),
            name=f"tether-verify-{kwargs['arm']}",
        )
        process.start()
        deadline = time.monotonic() + (6 * 60 * 60)
        while True:
            try:
                status, payload = result_queue.get(timeout=1.0)
                break
            except queue.Empty as exc:
                if not process.is_alive():
                    process.join(timeout=10)
                    raise RuntimeError(
                        f"verification {kwargs['arm']} arm exited with code "
                        f"{process.exitcode} without returning results"
                    ) from exc
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"verification {kwargs['arm']} arm timed out"
                    ) from exc
        process.join(timeout=10)
        if process.is_alive():
            raise RuntimeError(f"verification {kwargs['arm']} arm did not exit")
        if status != "ok":
            raise RuntimeError(
                f"verification {kwargs['arm']} arm failed in isolated process:\n{payload}"
            )
        if process.exitcode != 0:
            raise RuntimeError(
                f"verification {kwargs['arm']} arm exited with code {process.exitcode}"
            )
        if not isinstance(payload, dict):
            raise RuntimeError(f"verification {kwargs['arm']} returned invalid results")
        return payload
    except BaseException:
        if process is not None:
            _terminate_and_join_process(process)
        raise
    finally:
        if result_queue is not None:
            try:
                result_queue.close()
                result_queue.join_thread()
            except (EOFError, OSError, ValueError):
                if process is not None:
                    _terminate_and_join_process(process)
                raise


def gather_paired_samples(
    *,
    optimized_ref: str,
    original_ref: str | None,
    suite: str,
    task_suite_name: str,
    num_episodes: int,
    task_indices: list[int] | None,
    seed: int,
    preprocessor_ref: str | None = None,
    verification_device: str = "cpu",
    safety_config: str | None = None,
    isolate_processes: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the ORIGINAL (native PyTorch) and OPTIMIZED (ONNX/Triton) policies
    through the SAME LIBERO loop and return both rollout result dicts.

    Returns ``(original_results, optimized_results)`` — both in the shape
    documented on :func:`tether.eval.libero_rollout.run_libero_rollout`.

    This is the only function that loads models / runs simulation, and the only
    one that imports torch + LIBERO. It is isolated behind the ``gather_fn``
    seam on :func:`run_verify` precisely so the scoring path can be unit-tested
    with synthetic samples and zero GPU.

    v0 reuses :func:`run_libero_rollout` with ``use_native`` flipped between the
    two arms — the identical primitive the shipped side-by-side eval uses
    (``scripts/modal_fast_kernels_l3_side_by_side.py``). Same ``seed`` + same
    ``task_indices`` keeps the two arms paired: episode *i* of task *t* sees the
    same LIBERO initial state in both arms.
    """
    verification_device = normalize_verification_device(verification_device)
    from tether.eval.evidence_capture import load_canonical_safety_limits
    from tether.export_config import load_tether_config

    export_config = load_tether_config(optimized_ref, inspect_artifacts=True)
    original_checkpoint = original_ref or str(export_config["model_id"])
    canonical_safety = (
        load_canonical_safety_limits(safety_config) if safety_config else None
    )
    safety_config_sha256 = canonical_safety.sha256 if canonical_safety else None
    arm_kwargs = dict(
        optimized_ref=optimized_ref,
        original_checkpoint=original_checkpoint,
        task_suite_name=task_suite_name,
        num_episodes=num_episodes,
        task_indices=task_indices,
        seed=seed,
        preprocessor_ref=preprocessor_ref,
        verification_device=verification_device,
        safety_limits=canonical_safety,
        safety_config_sha256=safety_config_sha256,
    )
    if isolate_processes:
        logger.info("verify: running ORIGINAL arm in isolated process")
        original_results = _run_isolated_verification_arm(
            arm="original", **arm_kwargs
        )
        logger.info("verify: running OPTIMIZED arm in isolated process")
        optimized_results = _run_isolated_verification_arm(
            arm="optimized", **arm_kwargs
        )
        _validate_paired_rollout_identity(
            original_results,
            optimized_results,
            seed=seed,
            safety_limits=canonical_safety,
        )
        return original_results, optimized_results

    # In-process execution is retained only as a unit-test seam. Its same-PID
    # memory evidence deliberately cannot produce a green P3 verdict.
    from tether.eval.libero_rollout import (
        load_pi05_policy_and_processors,
        run_libero_rollout,
    )

    policy, preprocessor, postprocessor = load_pi05_policy_and_processors(
        student_checkpoint=original_checkpoint,
        decomposed_dir=optimized_ref,
        preprocessor_ref=preprocessor_ref,
        model_type=str(export_config["model_type"]),
        require_exact_checkpoint=True,
        device=verification_device,
    )

    common = dict(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task_suite_name=task_suite_name,
        num_episodes=num_episodes,
        task_indices=task_indices,
        seed=seed,
        capture_trajectories=True,
        safety_limits=canonical_safety,
        safety_config_sha256=safety_config_sha256,
        verification_device=verification_device,
    )

    from tether.eval.evidence_capture import ProcessDeviceMemorySampler

    # ARM A — original: native lerobot select_action (the reference behavior).
    logger.info("verify: running ORIGINAL arm (native PyTorch)")
    original_results = run_libero_rollout(
        inference=None,
        use_native=True,
        label="ORIGINAL",
        memory_sampler=ProcessDeviceMemorySampler(device=verification_device),
        **common,
    )

    # ARM B — optimized: construct the runtime from the declared export files.
    # Failure to load or support that artifact aborts verification; there is no
    # native-weight fallback because that would make parity trivially pass.
    logger.info("verify: loading and running OPTIMIZED export artifacts")
    from tether.runtime.verify_inference import load_verification_inference

    inference = load_verification_inference(optimized_ref, device=verification_device)
    optimized_results = run_libero_rollout(
        inference=inference,
        use_native=False,
        label="OPTIMIZED",
        memory_sampler=ProcessDeviceMemorySampler(device=verification_device),
        **common,
    )

    _validate_paired_rollout_identity(
        original_results,
        optimized_results,
        seed=seed,
        safety_limits=canonical_safety,
    )

    return original_results, optimized_results


# ---------------------------------------------------------------------------
# Public orchestrator — PURE scoring given a gather seam
# ---------------------------------------------------------------------------


def run_verify(
    *,
    optimized_ref: str,
    original_ref: str | None = None,
    suite: str = "libero",
    target: str = "unknown",
    task_suite_name: str = "libero_10",
    num_episodes: int = 30,
    task_indices: list[int] | None = None,
    seed: int = 7,
    thresholds: GateThresholds | None = None,
    preprocessor_ref: str | None = None,
    safety_config: str | None = None,
    verification_device: str = "cpu",
    gather_fn: GatherFn | None = None,
) -> ParityVerdict:
    """Resolve original + optimized policies, gather paired samples, score via
    the Pro 9-gate evaluator, and return a :class:`ParityVerdict`.

    The scoring + aggregation in this function is PURE given ``gather_fn`` — it
    does no I/O and loads no models itself. ``gather_fn`` defaults to
    :func:`gather_paired_samples` (which runs the real rollouts); unit tests
    pass a stub that returns synthetic paired result dicts.

    Raises:
        ValueError: unsupported suite.
        InsufficientEpisodes: fewer than ``MIN_EPISODES_TO_EVALUATE`` paired
            episodes (propagated from :class:`EvalGate`) — verify refuses to
            return a green light on under-powered evidence, matching the gate.
    """
    if suite not in SUPPORTED_SUITES:
        raise ValueError(
            f"Unsupported suite: {suite!r}. v0 supports: "
            f"{', '.join(SUPPORTED_SUITES)}."
        )
    verification_device = normalize_verification_device(verification_device)

    resolved_original_ref = original_ref
    if resolved_original_ref is None:
        config_path = Path(optimized_ref) / "tether_config.json"
        if config_path.is_file():
            from tether.export_config import load_tether_config

            resolved_original_ref = str(
                load_tether_config(optimized_ref, inspect_artifacts=True)["model_id"]
            )
        else:
            # Synthetic gather functions used by callers may not have an export
            # directory. Real verification always resolves from tether_config.
            resolved_original_ref = optimized_ref

    gather = gather_fn or gather_paired_samples
    original_results, optimized_results = gather(
        optimized_ref=optimized_ref,
        original_ref=resolved_original_ref,
        suite=suite,
        task_suite_name=task_suite_name,
        num_episodes=num_episodes,
        task_indices=task_indices,
        seed=seed,
        preprocessor_ref=preprocessor_ref,
        safety_config=safety_config,
        verification_device=verification_device,
    )

    _validate_rollout_device(
        original_results, expected=verification_device, arm="original"
    )
    _validate_rollout_device(
        optimized_results, expected=verification_device, arm="optimized"
    )

    # ORIGINAL -> baseline, OPTIMIZED -> candidate. The gate asks "is the
    # candidate as good as the baseline?" which is exactly the parity question.
    baseline_samples = _rollout_results_to_samples(original_results)
    candidate_samples = _rollout_results_to_samples(
        optimized_results, teacher_results=original_results
    )
    evidence = _verification_evidence(
        original_results,
        optimized_results,
        baseline_samples,
        candidate_samples,
        verification_device=verification_device,
    )
    candidate_memory_bytes = _memory_p95(evidence["memory"], "candidate")
    baseline_memory_bytes = _memory_p95(evidence["memory"], "baseline")

    report: EvalReport = EvalGate.evaluate(
        candidate_samples=candidate_samples,
        baseline_samples=baseline_samples,
        candidate_memory_bytes=candidate_memory_bytes,
        baseline_memory_bytes=baseline_memory_bytes,
        thresholds=thresholds,
        is_libero_suite=(suite == "libero"),
        pro_force=False,
        bypass_audit=None,
        evidence=evidence,
    )

    # Distributional + embodied parity — the v0 TODOs, now wired. Computed from
    # the per-step trajectories the widened rollout tap captures. When the tap
    # is off / older results lack them, these stay None and only success-rate
    # parity applies (no silent degrade — the verdict records which ran).
    base_actions, base_groups, cand_actions, cand_groups, n_cmp = (
        _collect_paired_succeeded_step_actions(original_results, optimized_results)
    )
    two_sample: TwoSampleResult | None = None
    if (
        base_actions.size
        and cand_actions.size
        and base_actions.shape[1] == cand_actions.shape[1]
    ):
        # Episode-aware + outcome-conditioned: permute whole episodes (per-step
        # actions are autocorrelated → step-level permutation over-rejects ~100%)
        # over ONLY episodes both arms succeeded (isolates the per-step policy
        # shift from the outcome shift — see the collector docstring).
        two_sample = two_sample_test(
            base_actions,
            cand_actions,
            baseline_groups=base_groups,
            candidate_groups=cand_groups,
        )

    base_pos, base_steps = _collect_eef_and_steps(original_results)
    cand_pos, cand_steps = _collect_eef_and_steps(optimized_results)
    embodied: EmbodiedParity | None = None
    if base_pos and cand_pos:
        embodied = aggregate_embodied(
            baseline_positions=base_pos,
            candidate_positions=cand_pos,
            baseline_velocities=[np.diff(p, axis=0) for p in base_pos],
            candidate_velocities=[np.diff(p, axis=0) for p in cand_pos],
            baseline_completion_steps=base_steps,
            candidate_completion_steps=cand_steps,
        )

    # Non-bypassable: a shifted action distribution or an embodied regression
    # fails the verdict even when success-rate parity passed.
    passed = report.overall_passed
    if two_sample is not None and two_sample.distributions_differ:
        passed = False
    if embodied is not None and embodied.regressed():
        passed = False

    success_evidence = evidence["success"]
    if success_evidence.is_available:
        original_success_rate: float | EvidenceValue[float] = _success_rate(
            baseline_samples
        )
        optimized_success_rate: float | EvidenceValue[float] = _success_rate(
            candidate_samples
        )
    else:
        reason = success_evidence.reason or UnavailableReason.MISSING_FIELD
        original_success_rate = EvidenceValue.unavailable(reason)
        optimized_success_rate = EvidenceValue.unavailable(reason)

    return ParityVerdict(
        passed=passed,
        two_sample_episodes=n_cmp,
        eval_report=report,
        optimized_ref=optimized_ref,
        original_ref=resolved_original_ref,
        suite=suite,
        target=target,
        n_episodes=len(candidate_samples),
        original_success_rate=original_success_rate,
        optimized_success_rate=optimized_success_rate,
        two_sample=two_sample,
        embodied=embodied,
    )


__all__ = [
    "MIN_EPISODES_TO_EVALUATE",
    "SUPPORTED_SUITES",
    "InsufficientEpisodes",
    "ParityVerdict",
    "gather_paired_samples",
    "run_verify",
]
