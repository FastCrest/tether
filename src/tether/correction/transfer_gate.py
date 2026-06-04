"""B.4 transfer-validation gate logic.

Pure-Python (no torch) so the gate decision can be unit-tested without GPU. The
training + eval scripts in scripts/{train_a2c2,validate_a2c2_transfer}.py produce
the inputs to compute_gate_report; this module owns the thresholds + decision rule.

Gate definition (from features/01_serve/subfeatures/_rtc_a2c2/a2c2-correction_plan.md):

  Inputs:
    - in_dist_mse      : MSE on held-out split of the training distribution
    - held_out_mses    : list of MSE per held-out trace (e.g. 3x synthetic-latency
                         serve traces or real Jetson traces)
    - task_success_on  : task-success rate with A2C2-on, latency >= eval_latency_ms
    - task_success_off : task-success rate with A2C2-off, same conditions
    - eval_latency_ms  : the latency floor at which the success delta is measured

  Decision:
    - PROCEED:  max(mse_ratio) <= 1.2  AND  delta >= +5 pp
    - PAUSE:    1.2 < max(mse_ratio) <= 2.0  OR  0 <= delta < +5 pp
    - ABORT:    max(mse_ratio) > 2.0  OR  delta < 0 pp

The PROCEED / PAUSE / ABORT terminology matches the plan's "Decision Checkpoint"
section exactly. PAUSE is not the same as ABORT — it triggers human-in-the-loop
investigation rather than auto-abort.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class GateDecision(str, Enum):
    PROCEED = "PROCEED"
    PAUSE = "PAUSE"
    ABORT = "ABORT"


@dataclass(frozen=True)
class GateThresholds:
    """Acceptance thresholds. Defaults match the plan; override per-deployment."""

    mse_ratio_proceed_max: float = 1.2
    mse_ratio_abort_min: float = 2.0
    success_delta_proceed_min: float = 0.05  # +5 percentage points
    success_delta_abort_max: float = 0.0     # delta < 0 → A2C2 hurts → abort
    eval_latency_ms_floor: float = 40.0      # min latency for delta to count


@dataclass
class GateReport:
    """Decision + per-trace breakdown + human-readable summary lines."""

    decision: GateDecision
    in_dist_mse: float
    held_out_mses: list[float]
    mse_ratios: list[float]
    max_mse_ratio: float
    task_success_on: float
    task_success_off: float
    success_delta: float
    eval_latency_ms: float
    thresholds: GateThresholds
    failure_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d

    def to_markdown(self) -> str:
        lines = [
            "# A2C2 Transfer Validation Gate Report",
            "",
            f"**Decision: {self.decision.value}**",
            "",
            "## MSE ratio (out-of-distribution / in-distribution)",
            "",
            f"- in_dist_mse: {self.in_dist_mse:.6f}",
            f"- proceed-max threshold: {self.thresholds.mse_ratio_proceed_max:.2f}",
            f"- abort-min threshold:   {self.thresholds.mse_ratio_abort_min:.2f}",
            "",
            "| Trace | Held-out MSE | Ratio | Pass |",
            "|-------|-------------:|------:|------|",
        ]
        for i, (m, r) in enumerate(zip(self.held_out_mses, self.mse_ratios)):
            mark = "OK" if r <= self.thresholds.mse_ratio_proceed_max else (
                "PAUSE" if r <= self.thresholds.mse_ratio_abort_min else "ABORT"
            )
            lines.append(f"| {i} | {m:.6f} | {r:.3f} | {mark} |")
        lines += [
            "",
            f"**max ratio: {self.max_mse_ratio:.3f}**",
            "",
            "## Task-success delta (high-latency regime)",
            "",
            f"- success_on:        {self.task_success_on:.4f}",
            f"- success_off:       {self.task_success_off:.4f}",
            f"- delta:             {self.success_delta:+.4f}",
            f"- eval_latency_ms:   {self.eval_latency_ms:.1f}",
            f"- proceed-min threshold: +{self.thresholds.success_delta_proceed_min:.2f}",
            f"- abort-max threshold:    {self.thresholds.success_delta_abort_max:+.2f}",
            "",
        ]
        if self.failure_reasons:
            lines += ["## Failure reasons", ""]
            lines += [f"- {r}" for r in self.failure_reasons]
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""]
            lines += [f"- {n}" for n in self.notes]
            lines.append("")
        return "\n".join(lines)

    def write(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".json":
            p.write_text(json.dumps(self.to_dict(), indent=2))
        else:
            p.write_text(self.to_markdown())


def compute_gate_report(
    in_dist_mse: float,
    held_out_mses: list[float],
    task_success_on: float,
    task_success_off: float,
    eval_latency_ms: float,
    thresholds: GateThresholds | None = None,
    notes: list[str] | None = None,
) -> GateReport:
    """Apply gate thresholds to measured numbers; return decision + report.

    Strict zero-handling: in_dist_mse <= 0 is unphysical -> ABORT with reason.
    Strict latency floor: if eval_latency_ms < threshold, success delta is
    non-comparable to plan -> PAUSE with reason (not auto-ABORT, so a low-latency
    evaluator can still surface the data for human review).
    """
    th = thresholds or GateThresholds()
    failure_reasons: list[str] = []

    if in_dist_mse <= 0:
        return GateReport(
            decision=GateDecision.ABORT,
            in_dist_mse=in_dist_mse,
            held_out_mses=list(held_out_mses),
            mse_ratios=[float("inf")] * len(held_out_mses),
            max_mse_ratio=float("inf"),
            task_success_on=task_success_on,
            task_success_off=task_success_off,
            success_delta=task_success_on - task_success_off,
            eval_latency_ms=eval_latency_ms,
            thresholds=th,
            failure_reasons=[
                f"in_dist_mse <= 0 ({in_dist_mse}); training did not converge or labels degenerate"
            ],
            notes=list(notes or []),
        )

    if not held_out_mses:
        return GateReport(
            decision=GateDecision.ABORT,
            in_dist_mse=in_dist_mse,
            held_out_mses=[],
            mse_ratios=[],
            max_mse_ratio=float("inf"),
            task_success_on=task_success_on,
            task_success_off=task_success_off,
            success_delta=task_success_on - task_success_off,
            eval_latency_ms=eval_latency_ms,
            thresholds=th,
            failure_reasons=["no held-out MSE measurements provided"],
            notes=list(notes or []),
        )

    mse_ratios = [m / in_dist_mse for m in held_out_mses]
    max_ratio = max(mse_ratios)
    delta = task_success_on - task_success_off

    mse_abort = max_ratio > th.mse_ratio_abort_min
    delta_abort = delta < th.success_delta_abort_max
    mse_proceed = max_ratio <= th.mse_ratio_proceed_max
    delta_proceed = delta >= th.success_delta_proceed_min
    latency_ok = eval_latency_ms >= th.eval_latency_ms_floor

    if mse_abort:
        failure_reasons.append(
            f"max MSE ratio {max_ratio:.3f} > abort threshold {th.mse_ratio_abort_min}"
        )
    if delta_abort:
        failure_reasons.append(
            f"task-success delta {delta:+.3f} < abort threshold {th.success_delta_abort_max:+.2f} "
            "(A2C2 makes deployment WORSE)"
        )

    if mse_abort or delta_abort:
        decision = GateDecision.ABORT
    elif mse_proceed and delta_proceed and latency_ok:
        decision = GateDecision.PROCEED
    else:
        decision = GateDecision.PAUSE
        if not mse_proceed:
            failure_reasons.append(
                f"max MSE ratio {max_ratio:.3f} > proceed threshold "
                f"{th.mse_ratio_proceed_max} (transfer is weak)"
            )
        if not delta_proceed:
            failure_reasons.append(
                f"task-success delta {delta:+.3f} < proceed threshold "
                f"+{th.success_delta_proceed_min:.2f} (transfer gain too small)"
            )
        if not latency_ok:
            failure_reasons.append(
                f"eval_latency_ms {eval_latency_ms:.1f} < floor "
                f"{th.eval_latency_ms_floor} (delta not measured at intended regime)"
            )

    return GateReport(
        decision=decision,
        in_dist_mse=in_dist_mse,
        held_out_mses=list(held_out_mses),
        mse_ratios=mse_ratios,
        max_mse_ratio=max_ratio,
        task_success_on=task_success_on,
        task_success_off=task_success_off,
        success_delta=delta,
        eval_latency_ms=eval_latency_ms,
        thresholds=th,
        failure_reasons=failure_reasons,
        notes=list(notes or []),
    )
