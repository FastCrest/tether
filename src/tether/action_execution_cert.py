"""Action-execution certificate for realtime robot policy proof packets."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _int(value: Any) -> int | None:
    number = _number(value)
    if number is None:
        return None
    rounded = int(number)
    return rounded if rounded > 0 else None


def _round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    status: str,
    *,
    metric: str,
    actual: Any = None,
    expected: Any = None,
    remediation: str = "",
) -> None:
    check: dict[str, Any] = {
        "name": name,
        "status": status,
        "metric": metric,
        "actual": actual,
        "expected": expected,
    }
    if remediation:
        check["remediation"] = remediation
    checks.append(check)


def _pass_fail(condition: bool) -> str:
    return "pass" if condition else "fail"


def _max_abs_delta(left: list[float], right: list[float]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def _velocity(prev: list[float], cur: list[float]) -> list[float] | None:
    if not prev or not cur or len(prev) != len(cur):
        return None
    return [b - a for a, b in zip(prev, cur, strict=True)]


def _coerce_chunk(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or not value:
        return None
    rows: list[list[float]] = []
    width: int | None = None
    for row in value:
        if not isinstance(row, list) or not row:
            return None
        out: list[float] = []
        for item in row:
            number = _number(item)
            if number is None:
                return None
            out.append(number)
        if width is None:
            width = len(out)
        elif len(out) != width:
            return None
        rows.append(out)
    return rows


def _first_mapping(*values: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _sample_response(sample: dict[str, Any]) -> dict[str, Any]:
    response = sample.get("response")
    return response if isinstance(response, dict) else {}


def _extract_actions(sample: dict[str, Any]) -> list[list[float]] | None:
    response = _sample_response(sample)
    candidates = (
        sample.get("actions"),
        sample.get("action_chunk"),
        sample.get("guarded_actions"),
        sample.get("raw_actions"),
        response.get("actions"),
        response.get("action_chunk"),
        response.get("guarded_actions"),
        response.get("raw_actions"),
    )
    for candidate in candidates:
        chunk = _coerce_chunk(candidate)
        if chunk is not None:
            return chunk
    return None


def _extract_execution(sample: dict[str, Any]) -> dict[str, Any]:
    response = _sample_response(sample)
    return _first_mapping(
        sample.get("execution"),
        sample.get("action_execution"),
        sample.get("adaptive_action_chunking"),
        sample.get("rtc"),
        response.get("execution"),
        response.get("action_execution"),
        response.get("adaptive_action_chunking"),
        response.get("rtc"),
    )


def _execution_horizon(execution: dict[str, Any]) -> int | None:
    for key in (
        "executed_horizon",
        "execute_horizon",
        "execution_horizon",
        "action_horizon",
        "horizon_actions",
        "actions_to_execute",
        "execute_steps",
    ):
        value = _int(execution.get(key))
        if value is not None:
            return value
    return None


def _truthy_sequence(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def _phase_evidence(execution: dict[str, Any]) -> bool:
    for key in (
        "phase_transition_indices",
        "phase_transition_index",
        "phase_boundaries",
        "low_speed_transition_indices",
    ):
        if _truthy_sequence(execution.get(key)) or _number(execution.get(key)) is not None:
            return True
    reason = str(execution.get("adaptive_reason") or execution.get("horizon_reason") or "")
    lowered = reason.lower()
    return any(token in lowered for token in ("phase", "transition", "low_speed"))


def _runtime_attribution(execution: dict[str, Any]) -> bool:
    for key in (
        "adaptive_reason",
        "horizon_reason",
        "cache_status",
        "cache_hit",
        "scheduler",
        "action_source",
        "chunk_source",
        "execution_mode",
        "policy_version",
    ):
        value = execution.get(key)
        if value not in (None, "", [], {}):
            return True
    return False


def _deadline_cause(sample: dict[str, Any], execution: dict[str, Any]) -> str | None:
    for key in ("deadline_cause", "deadline_source", "deadline_reason"):
        value = execution.get(key, sample.get(key))
        if value not in (None, "", [], {}):
            return str(value)
    return None


def _deadline_exceeded(sample: dict[str, Any], execution: dict[str, Any]) -> bool:
    return bool(sample.get("deadline_exceeded") or execution.get("deadline_exceeded"))


def _execution_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    samples = receipt.get("act_samples") or []
    if not isinstance(samples, list):
        return []
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        chunk = _extract_actions(sample)
        execution = _extract_execution(sample)
        horizon = _execution_horizon(execution)
        roundtrip_ms = _number(sample.get("roundtrip_ms"))
        rows.append(
            {
                "sample": sample.get("sample") or idx + 1,
                "roundtrip_ms": roundtrip_ms,
                "actions": chunk,
                "execution": execution,
                "horizon": horizon,
                "deadline_exceeded": sample.get("deadline_exceeded"),
                "deadline_cause": sample.get("deadline_cause"),
                "deadline_source": sample.get("deadline_source"),
                "deadline_reason": sample.get("deadline_reason"),
            }
        )
    return rows


def build_action_execution_certificate(
    receipt: dict[str, Any],
    *,
    control_hz: float | None = None,
    max_stale_action_window_ms: float = 100.0,
    max_chunk_boundary_delta: float = 0.15,
    max_velocity_discontinuity: float = 0.2,
    require_phase_aware_horizon: bool = False,
    require_runtime_attribution: bool = True,
) -> dict[str, Any]:
    """Build a pass/fail certificate for action chunk execution evidence."""

    if control_hz is not None and control_hz <= 0:
        raise ValueError("control_hz must be > 0 when supplied")
    if max_stale_action_window_ms < 0:
        raise ValueError("max_stale_action_window_ms must be >= 0")
    if max_chunk_boundary_delta < 0:
        raise ValueError("max_chunk_boundary_delta must be >= 0")
    if max_velocity_discontinuity < 0:
        raise ValueError("max_velocity_discontinuity must be >= 0")

    rows = _execution_rows(receipt)
    chunk_rows = [row for row in rows if row["actions"] is not None]
    chunks: list[list[list[float]]] = [row["actions"] for row in chunk_rows]
    dims = [len(chunk[0]) for chunk in chunks if chunk]
    sizes = [len(chunk) for chunk in chunks]
    shape_consistent = bool(chunks) and len(set(dims)) == 1
    horizons = [row["horizon"] for row in chunk_rows if row["horizon"] is not None]
    missing_horizons = sum(1 for row in chunk_rows if row["horizon"] is None)
    horizon_over_chunk = sum(
        1
        for row in chunk_rows
        if row["horizon"] is not None and row["actions"] is not None
        and row["horizon"] > len(row["actions"])
    )

    period_ms = 1000.0 / control_hz if control_hz else None
    stale_windows: list[float] = []
    if period_ms is not None:
        for row in chunk_rows:
            horizon = row["horizon"]
            roundtrip_ms = row["roundtrip_ms"]
            if horizon is None or roundtrip_ms is None:
                continue
            stale_windows.append(max(0.0, roundtrip_ms - horizon * period_ms))

    boundary_deltas: list[float] = []
    velocity_jumps: list[float] = []
    if shape_consistent:
        for prev, cur in zip(chunk_rows, chunk_rows[1:], strict=False):
            prev_actions = prev["actions"]
            cur_actions = cur["actions"]
            if prev_actions is None or cur_actions is None:
                continue
            prev_horizon = prev["horizon"] or len(prev_actions)
            boundary_idx = max(0, min(prev_horizon, len(prev_actions)) - 1)
            boundary_delta = _max_abs_delta(prev_actions[boundary_idx], cur_actions[0])
            if boundary_delta is not None:
                boundary_deltas.append(boundary_delta)

            if boundary_idx > 0 and len(cur_actions) > 1:
                prev_velocity = _velocity(
                    prev_actions[boundary_idx - 1],
                    prev_actions[boundary_idx],
                )
                cur_velocity = _velocity(cur_actions[0], cur_actions[1])
                if prev_velocity is not None and cur_velocity is not None:
                    jump = _max_abs_delta(prev_velocity, cur_velocity)
                    if jump is not None:
                        velocity_jumps.append(jump)

    phase_samples = 0
    attribution_samples = 0
    deadline_missing = 0
    deadline_required = 0
    adaptive_reasons: Counter[str] = Counter()
    cache_statuses: Counter[str] = Counter()
    for row in rows:
        execution = row["execution"]
        if _phase_evidence(execution):
            phase_samples += 1
        if _runtime_attribution(execution):
            attribution_samples += 1
        reason = execution.get("adaptive_reason") or execution.get("horizon_reason")
        if reason not in (None, "", [], {}):
            adaptive_reasons[str(reason)] += 1
        cache = execution.get("cache_status")
        if cache not in (None, "", [], {}):
            cache_statuses[str(cache)] += 1
        if _deadline_exceeded(row, execution):
            deadline_required += 1
            if _deadline_cause(row, execution) is None:
                deadline_missing += 1

    max_stale = max(stale_windows) if stale_windows else None
    max_boundary = max(boundary_deltas) if boundary_deltas else None
    max_velocity_jump = max(velocity_jumps) if velocity_jumps else None

    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "action_chunks_present",
        _pass_fail(bool(chunks)),
        metric="act_samples.actions",
        actual=len(chunks),
        expected=">= 1",
        remediation="Preserve /act response actions in the proof packet before requesting an execution certificate.",
    )
    _add_check(
        checks,
        "action_shape_consistent",
        _pass_fail(shape_consistent),
        metric="act_samples.actions.shape",
        actual={"dims": sorted(set(dims)), "chunk_sizes": [min(sizes), max(sizes)] if sizes else []},
        expected="non-empty chunks with one action_dim",
        remediation="Record complete action chunks with a stable action dimension.",
    )
    _add_check(
        checks,
        "control_hz_defined",
        _pass_fail(bool(control_hz and control_hz > 0)),
        metric="control_hz",
        actual=control_hz,
        expected="> 0",
        remediation="Pass `--control-hz` or include a control rate in the proof profile.",
    )
    _add_check(
        checks,
        "execution_horizon_present",
        _pass_fail(bool(chunks) and missing_horizons == 0),
        metric="act_samples.action_execution.executed_horizon",
        actual={"present": len(horizons), "missing": missing_horizons},
        expected="one positive horizon per action chunk",
        remediation="Have the runtime emit action_execution.executed_horizon for each /act sample.",
    )
    _add_check(
        checks,
        "execution_horizon_within_chunk",
        _pass_fail(bool(chunks) and horizon_over_chunk == 0 and missing_horizons == 0),
        metric="action_execution.executed_horizon",
        actual={"over_chunk": horizon_over_chunk},
        expected="<= chunk_size",
        remediation="Clamp the execution horizon to the available action chunk length.",
    )

    if max_stale is None:
        _add_check(
            checks,
            "stale_action_window_within_budget",
            "fail",
            metric="action_execution.stale_window_ms.max",
            actual=None,
            expected=f"<= {max_stale_action_window_ms}",
            remediation="Record roundtrip_ms, control_hz, and executed_horizon to quantify stale action windows.",
        )
    else:
        _add_check(
            checks,
            "stale_action_window_within_budget",
            _pass_fail(max_stale <= max_stale_action_window_ms),
            metric="action_execution.stale_window_ms.max",
            actual=_round(max_stale, 3),
            expected=f"<= {max_stale_action_window_ms}",
            remediation="Increase execution horizon, lower serving latency, or lower the control rate.",
        )

    if len(chunks) < 2:
        _add_check(
            checks,
            "chunk_boundary_delta_within_budget",
            "skip",
            metric="action_execution.chunk_boundary_delta.max_abs",
            actual=None,
            expected=f"<= {max_chunk_boundary_delta}",
        )
    elif max_boundary is None:
        _add_check(
            checks,
            "chunk_boundary_delta_within_budget",
            "fail",
            metric="action_execution.chunk_boundary_delta.max_abs",
            actual=None,
            expected=f"<= {max_chunk_boundary_delta}",
            remediation="Record compatible consecutive chunks so boundary alignment can be checked.",
        )
    else:
        _add_check(
            checks,
            "chunk_boundary_delta_within_budget",
            _pass_fail(max_boundary <= max_chunk_boundary_delta),
            metric="action_execution.chunk_boundary_delta.max_abs",
            actual=_round(max_boundary, 6),
            expected=f"<= {max_chunk_boundary_delta}",
            remediation="Use RTC/BID smoothing or overlap-aware chunk alignment before deployment.",
        )

    if len(chunks) < 2:
        _add_check(
            checks,
            "velocity_discontinuity_within_budget",
            "skip",
            metric="action_execution.velocity_discontinuity.max_abs",
            actual=None,
            expected=f"<= {max_velocity_discontinuity}",
        )
    elif max_velocity_jump is None:
        _add_check(
            checks,
            "velocity_discontinuity_within_budget",
            "fail",
            metric="action_execution.velocity_discontinuity.max_abs",
            actual=None,
            expected=f"<= {max_velocity_discontinuity}",
            remediation="Record at least two actions per consecutive chunk to check boundary velocity.",
        )
    else:
        _add_check(
            checks,
            "velocity_discontinuity_within_budget",
            _pass_fail(max_velocity_jump <= max_velocity_discontinuity),
            metric="action_execution.velocity_discontinuity.max_abs",
            actual=_round(max_velocity_jump, 6),
            expected=f"<= {max_velocity_discontinuity}",
            remediation="Smooth/fuse chunks or shorten the executed horizon near sharp transitions.",
        )

    phase_status = _pass_fail(phase_samples > 0) if require_phase_aware_horizon else "skip"
    _add_check(
        checks,
        "phase_aware_horizon_present",
        phase_status,
        metric="action_execution.phase_transition_samples",
        actual=phase_samples,
        expected=">= 1" if require_phase_aware_horizon else "not required",
        remediation="Emit phase or low-speed transition evidence when using adaptive action chunking.",
    )
    _add_check(
        checks,
        "runtime_attribution_present",
        _pass_fail(attribution_samples > 0) if require_runtime_attribution else "skip",
        metric="action_execution.runtime_attribution_samples",
        actual=attribution_samples,
        expected=">= 1" if require_runtime_attribution else "not required",
        remediation="Emit action_execution cache, scheduler, or adaptive-horizon reasons in /act responses.",
    )
    _add_check(
        checks,
        "deadline_cause_present",
        _pass_fail(deadline_missing == 0),
        metric="action_execution.deadline_cause_missing",
        actual={"required": deadline_required, "missing": deadline_missing},
        expected=0,
        remediation="Attach deadline_cause/deadline_source when a sample exceeds its runtime deadline.",
    )

    summary = {
        "pass": sum(1 for check in checks if check["status"] == "pass"),
        "fail": sum(1 for check in checks if check["status"] == "fail"),
        "skip": sum(1 for check in checks if check["status"] == "skip"),
        "failed_checks": [
            check["name"] for check in checks if check["status"] == "fail"
        ],
    }
    decision = "PASS" if summary["fail"] == 0 else "FAIL"

    return {
        "schema_version": 1,
        "kind": "tether.action_execution_certificate",
        "generated_at": _now_iso(),
        "decision": decision,
        "passed": decision == "PASS",
        "control_hz": _round(control_hz, 3),
        "control_period_ms": _round(period_ms, 3),
        "metrics": {
            "sample_count": len(rows),
            "action_chunk_count": len(chunks),
            "action_dim": dims[0] if shape_consistent and dims else None,
            "chunk_size_min": min(sizes) if sizes else None,
            "chunk_size_max": max(sizes) if sizes else None,
            "execution_horizon_min": min(horizons) if horizons else None,
            "execution_horizon_max": max(horizons) if horizons else None,
            "missing_execution_horizon_count": missing_horizons,
            "horizon_over_chunk_count": horizon_over_chunk,
            "stale_action_window_ms": {
                "p95_ms": _round(_percentile(stale_windows, 0.95), 3),
                "max_ms": _round(max_stale, 3),
            },
            "chunk_boundary_delta": {
                "samples": len(boundary_deltas),
                "max_abs": _round(max_boundary, 6),
            },
            "velocity_discontinuity": {
                "samples": len(velocity_jumps),
                "max_abs": _round(max_velocity_jump, 6),
            },
            "phase_transition_samples": phase_samples,
            "runtime_attribution_samples": attribution_samples,
            "deadline_cause_required": deadline_required,
            "deadline_cause_missing": deadline_missing,
            "adaptive_reasons": dict(sorted(adaptive_reasons.items())),
            "cache_statuses": dict(sorted(cache_statuses.items())),
        },
        "thresholds": {
            "max_stale_action_window_ms": max_stale_action_window_ms,
            "max_chunk_boundary_delta": max_chunk_boundary_delta,
            "max_velocity_discontinuity": max_velocity_discontinuity,
            "require_phase_aware_horizon": require_phase_aware_horizon,
            "require_runtime_attribution": require_runtime_attribution,
        },
        "checks": checks,
        "summary": summary,
    }


def format_action_execution_markdown(report: dict[str, Any]) -> str:
    """Format an action-execution certificate section as Markdown."""

    metrics = report.get("metrics") or {}
    stale = metrics.get("stale_action_window_ms") or {}
    boundary = metrics.get("chunk_boundary_delta") or {}
    velocity = metrics.get("velocity_discontinuity") or {}
    summary = report.get("summary") or {}
    lines = [
        "## Action Execution",
        "",
        f"- Decision: **{report.get('decision', 'FAIL')}**",
        f"- Control period: `{report.get('control_period_ms')} ms`",
        f"- Action chunks: `{metrics.get('action_chunk_count')}`",
        f"- Execution horizon: `{metrics.get('execution_horizon_min')}` to `{metrics.get('execution_horizon_max')}` actions",
        f"- Stale action window max: `{stale.get('max_ms')} ms`",
        f"- Chunk-boundary delta max: `{boundary.get('max_abs')}`",
        f"- Velocity discontinuity max: `{velocity.get('max_abs')}`",
        f"- Runtime attribution samples: `{metrics.get('runtime_attribution_samples')}`",
        "",
        f"{summary.get('pass', 0)} pass, {summary.get('fail', 0)} fail, "
        f"{summary.get('skip', 0)} skip.",
        "",
        "| Check | Status | Actual | Expected |",
        "|---|---|---:|---:|",
    ]
    for check in report.get("checks") or []:
        lines.append(
            f"| `{check.get('name')}` | {check.get('status')} | "
            f"{check.get('actual')} | {check.get('expected')} |"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "build_action_execution_certificate",
    "format_action_execution_markdown",
]
