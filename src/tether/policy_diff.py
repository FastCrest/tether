"""Policy rollout diffing for ``tether policy diff``.

The command intentionally starts from recorded traces instead of live robot
traffic: compare two runs over the same observations, or compare production
actions against shadow actions already embedded in one trace.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from tether.replay.cli import diff_actions
from tether.replay.readers import load_reader

POLICY_DIFF_SCHEMA_VERSION = 1
FailOn = Literal["none", "actions", "latency", "guard", "shape", "any"]


class PolicyDiffError(ValueError):
    """Raised when input traces cannot be diffed."""


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] + frac * (vals[hi] - vals[lo])


def _load_trace(
    path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[Any, dict[str, Any]]]:
    reader = load_reader(path)
    header = reader.read_header()
    requests: list[dict[str, Any]] = []
    shadow_results: dict[Any, dict[str, Any]] = {}
    for kind, rec in reader.read_records():
        if kind == "request":
            requests.append(rec)
        elif kind == "shadow_result":
            shadow_results[rec.get("seq")] = rec
    return header, requests, shadow_results


def _action_shape(actions: Any) -> dict[str, int]:
    if not isinstance(actions, list):
        return {"num_actions": 0, "action_dim": 0}
    first = actions[0] if actions else []
    dim = len(first) if isinstance(first, list) else 0
    return {"num_actions": len(actions), "action_dim": dim}


def _guard_counts(record: dict[str, Any]) -> dict[str, int | bool]:
    evidence = record.get("evidence")
    safety = evidence.get("safety") if isinstance(evidence, dict) else None
    if isinstance(safety, dict):
        return {
            "clamped": bool(safety.get("clamped")),
            "clamp_count": int(safety.get("clamp_count") or 0),
            "violation_count": int(safety.get("violation_count") or 0),
        }
    guard = record.get("guard") if isinstance(record.get("guard"), dict) else {}
    violations = guard.get("violations") if isinstance(guard, dict) else None
    return {
        "clamped": bool(guard.get("clamped")) if isinstance(guard, dict) else False,
        "clamp_count": int(guard.get("clamp_count") or 0) if isinstance(guard, dict) else 0,
        "violation_count": len(violations) if isinstance(violations, list) else 0,
    }


def _latency_diff(
    baseline: dict[str, Any],
    candidate: dict[str, Any] | None,
    *,
    max_regression_pct: float,
) -> dict[str, Any]:
    base_ms = float((baseline.get("latency") or {}).get("total_ms") or 0.0)
    if candidate is None:
        return {
            "baseline_total_ms": base_ms,
            "candidate_total_ms": None,
            "delta_ms": None,
            "delta_pct": None,
            "passed": True,
            "note": "candidate latency unavailable",
        }
    cand_ms = float((candidate.get("latency") or {}).get("total_ms") or 0.0)
    if base_ms <= 0:
        return {
            "baseline_total_ms": base_ms,
            "candidate_total_ms": cand_ms,
            "delta_ms": cand_ms - base_ms,
            "delta_pct": None,
            "passed": True,
            "note": "baseline latency <= 0; gating skipped",
        }
    delta_ms = cand_ms - base_ms
    delta_pct = delta_ms / base_ms
    return {
        "baseline_total_ms": base_ms,
        "candidate_total_ms": cand_ms,
        "delta_ms": delta_ms,
        "delta_pct": delta_pct,
        "passed": delta_pct <= max_regression_pct,
        "threshold_pct": max_regression_pct,
    }


def _request_fingerprint(record: dict[str, Any]) -> dict[str, Any]:
    request = record.get("request") if isinstance(record.get("request"), dict) else {}
    return {
        "instruction": request.get("instruction") or "",
        "image_sha256": request.get("image_sha256"),
        "episode_id": request.get("episode_id"),
    }


def _metadata_warnings(
    baseline_header: dict[str, Any],
    candidate_header: dict[str, Any] | None,
) -> list[str]:
    if candidate_header is None:
        return []
    warnings: list[str] = []
    for key in ("embodiment", "model_type", "export_kind"):
        a = baseline_header.get(key)
        b = candidate_header.get(key)
        if a != b:
            warnings.append(f"{key} differs: baseline={a!r}, candidate={b!r}")
    return warnings


def _shadow_candidate_record(
    record: dict[str, Any],
    shadow_result: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    routing = record.get("routing")
    if not isinstance(routing, dict):
        routing = {}
    if shadow_result is not None:
        response = (
            shadow_result.get("response")
            if isinstance(shadow_result.get("response"), dict)
            else {}
        )
        shadow_actions = response.get("actions")
        if not isinstance(shadow_actions, list):
            return None
        result_routing = (
            shadow_result.get("routing")
            if isinstance(shadow_result.get("routing"), dict)
            else routing
        )
        return {
            "kind": "request",
            "seq": record.get("seq"),
            "request": record.get("request") if isinstance(record.get("request"), dict) else {},
            "response": {
                "actions": shadow_actions,
                **_action_shape(shadow_actions),
            },
            "latency": shadow_result.get("latency"),
            "guard": result_routing.get("shadow_guard") if isinstance(result_routing.get("shadow_guard"), dict) else None,
            "routing": result_routing,
        }
    if not isinstance(routing, dict):
        return None
    shadow_actions = routing.get("shadow_actions")
    if shadow_actions is None:
        shadow = routing.get("shadow")
        if isinstance(shadow, dict):
            shadow_actions = shadow.get("actions")
    if not isinstance(shadow_actions, list):
        return None
    return {
        "kind": "request",
        "seq": record.get("seq"),
        "request": record.get("request") if isinstance(record.get("request"), dict) else {},
        "response": {
            "actions": shadow_actions,
            **_action_shape(shadow_actions),
        },
        "latency": None,
        "guard": routing.get("shadow_guard") if isinstance(routing.get("shadow_guard"), dict) else None,
        "routing": routing,
    }


def _shadow_skip_record(
    record: dict[str, Any],
    shadow_result: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if shadow_result is not None:
        result_routing = (
            shadow_result.get("routing")
            if isinstance(shadow_result.get("routing"), dict)
            else {}
        )
        if shadow_result.get("error") or result_routing.get("shadow_error"):
            err = shadow_result.get("error")
            message = (
                err.get("message")
                if isinstance(err, dict)
                else result_routing.get("shadow_error")
            )
            return {
                "seq": record.get("seq"),
                "status": "shadow_error",
                "passed": False,
                "error": str(message or "")[:500],
            }
        return None
    routing = record.get("routing")
    if not isinstance(routing, dict):
        return None
    if routing.get("shadow_sampled") is False:
        return {
            "seq": record.get("seq"),
            "status": "shadow_skipped",
            "passed": True,
            "reason": routing.get("shadow_skip_reason") or "not_sampled",
        }
    if routing.get("shadow_error"):
        return {
            "seq": record.get("seq"),
            "status": "shadow_error",
            "passed": False,
            "error": str(routing.get("shadow_error"))[:500],
        }
    if routing.get("shadow_pending") is True:
        return {
            "seq": record.get("seq"),
            "status": "shadow_pending",
            "passed": False,
            "reason": "shadow_result not recorded yet",
        }
    return None


def _compare_record_pair(
    baseline: dict[str, Any],
    candidate: dict[str, Any] | None,
    *,
    min_action_cos: float,
    max_action_delta: float,
    max_latency_regression_pct: float,
) -> dict[str, Any]:
    base_actions = (baseline.get("response") or {}).get("actions") or []
    cand_actions = (candidate.get("response") or {}).get("actions") if candidate else None
    if not isinstance(cand_actions, list):
        return {
            "seq": baseline.get("seq"),
            "status": "missing_candidate",
            "passed": False,
        }

    base_shape = _action_shape(base_actions)
    cand_shape = _action_shape(cand_actions)
    shape_match = base_shape == cand_shape
    action = diff_actions(
        base_actions,
        cand_actions,
        threshold_cos=min_action_cos,
        threshold_max_abs=max_action_delta,
    )
    latency = _latency_diff(
        baseline,
        candidate,
        max_regression_pct=max_latency_regression_pct,
    )
    base_guard = _guard_counts(baseline)
    cand_guard = _guard_counts(candidate)
    guard_regressed = (
        int(cand_guard["violation_count"]) > int(base_guard["violation_count"])
        or int(cand_guard["clamp_count"]) > int(base_guard["clamp_count"])
    )
    request_match = _request_fingerprint(baseline) == _request_fingerprint(candidate)
    passed = bool(action["passed"] and latency["passed"] and shape_match and not guard_regressed)
    return {
        "seq": baseline.get("seq"),
        "status": "compared",
        "passed": passed,
        "request_match": request_match,
        "shape": {
            "baseline": base_shape,
            "candidate": cand_shape,
            "passed": shape_match,
        },
        "actions": action,
        "latency": latency,
        "guard": {
            "baseline": base_guard,
            "candidate": cand_guard,
            "regressed": guard_regressed,
            "passed": not guard_regressed,
        },
    }


def diff_policy_traces(
    *,
    baseline_trace: str | Path,
    candidate_trace: str | Path | None = None,
    shadow: bool = False,
    min_action_cos: float = 0.995,
    max_action_delta: float = 0.10,
    max_latency_regression_pct: float = 0.10,
) -> dict[str, Any]:
    """Return a machine-readable policy diff report.

    Modes:
    - trace pair: compare request records by ``seq`` across two traces.
    - shadow: compare ``response.actions`` against appended ``shadow_result``
      rows, with legacy inline ``routing.shadow_actions`` still supported.
    """
    if not shadow and candidate_trace is None:
        raise PolicyDiffError("candidate_trace is required unless shadow=True")

    base_header, base_requests, shadow_results = _load_trace(baseline_trace)
    candidate_header: dict[str, Any] | None = None
    candidate_by_seq: dict[Any, dict[str, Any]] = {}
    if shadow:
        mode = "shadow_trace"
    else:
        mode = "trace_pair"
        candidate_header, candidate_requests, _ = _load_trace(candidate_trace or "")
        candidate_by_seq = {rec.get("seq"): rec for rec in candidate_requests}

    per_request: list[dict[str, Any]] = []
    for base in base_requests:
        if shadow:
            shadow_result = shadow_results.get(base.get("seq"))
            shadow_skip = _shadow_skip_record(base, shadow_result)
            if shadow_skip is not None:
                per_request.append(shadow_skip)
                continue
            candidate = _shadow_candidate_record(base, shadow_result)
        else:
            candidate = candidate_by_seq.get(base.get("seq"))
        per_request.append(
            _compare_record_pair(
                base,
                candidate,
                min_action_cos=min_action_cos,
                max_action_delta=max_action_delta,
                max_latency_regression_pct=max_latency_regression_pct,
            )
        )

    compared = [row for row in per_request if row.get("status") == "compared"]
    action_deltas = [
        float(row["actions"]["max_abs_diff"])
        for row in compared
        if isinstance(row.get("actions"), dict)
    ]
    action_cosines = [
        float(row["actions"]["cosine"])
        for row in compared
        if isinstance(row.get("actions"), dict)
    ]
    latency_regressions = [
        row for row in compared
        if isinstance(row.get("latency"), dict) and row["latency"].get("passed") is False
    ]
    action_failures = [
        row for row in compared
        if isinstance(row.get("actions"), dict) and row["actions"].get("passed") is False
    ]
    shape_failures = [
        row for row in compared
        if isinstance(row.get("shape"), dict) and row["shape"].get("passed") is False
    ]
    guard_regressions = [
        row for row in compared
        if isinstance(row.get("guard"), dict) and row["guard"].get("regressed") is True
    ]
    request_mismatches = [row for row in compared if row.get("request_match") is False]
    missing = [row for row in per_request if row.get("status") == "missing_candidate"]
    shadow_skipped = [row for row in per_request if row.get("status") == "shadow_skipped"]
    shadow_errors = [row for row in per_request if row.get("status") == "shadow_error"]
    shadow_pending = [row for row in per_request if row.get("status") == "shadow_pending"]
    verdict = "pass"
    if (
        not compared
        or action_failures
        or latency_regressions
        or shape_failures
        or guard_regressions
        or missing
        or shadow_errors
        or shadow_pending
    ):
        verdict = "fail"
    elif request_mismatches or _metadata_warnings(base_header, candidate_header):
        verdict = "warn"

    return {
        "kind": "tether.policy_diff",
        "schema_version": POLICY_DIFF_SCHEMA_VERSION,
        "mode": mode,
        "thresholds": {
            "min_action_cos": min_action_cos,
            "max_action_delta": max_action_delta,
            "max_latency_regression_pct": max_latency_regression_pct,
        },
        "baseline": {
            "trace": str(baseline_trace),
            "model_hash": base_header.get("model_hash"),
            "config_hash": base_header.get("config_hash"),
            "model_type": base_header.get("model_type"),
            "export_kind": base_header.get("export_kind"),
            "embodiment": base_header.get("embodiment"),
        },
        "candidate": (
            {"source": "shadow_result or routing.shadow_actions"}
            if shadow
            else {
                "trace": str(candidate_trace),
                "model_hash": (candidate_header or {}).get("model_hash"),
                "config_hash": (candidate_header or {}).get("config_hash"),
                "model_type": (candidate_header or {}).get("model_type"),
                "export_kind": (candidate_header or {}).get("export_kind"),
                "embodiment": (candidate_header or {}).get("embodiment"),
            }
        ),
        "summary": {
            "verdict": verdict,
            "baseline_requests": len(base_requests),
            "compared": len(compared),
            "shadow_skipped": len(shadow_skipped),
            "shadow_errors": len(shadow_errors),
            "shadow_pending": len(shadow_pending),
            "missing_candidate": len(missing),
            "request_mismatches": len(request_mismatches),
            "action_failures": len(action_failures),
            "latency_regressions": len(latency_regressions),
            "shape_failures": len(shape_failures),
            "guard_regressions": len(guard_regressions),
            "max_action_delta": max(action_deltas) if action_deltas else 0.0,
            "mean_action_delta": (
                sum(action_deltas) / len(action_deltas) if action_deltas else 0.0
            ),
            "p95_action_delta": _percentile(action_deltas, 0.95),
            "min_action_cosine": min(action_cosines) if action_cosines else 0.0,
            "metadata_warnings": _metadata_warnings(base_header, candidate_header),
        },
        "per_request": per_request,
    }


def should_fail(report: dict[str, Any], fail_on: FailOn) -> bool:
    if fail_on == "none":
        return False
    summary = report.get("summary") or {}
    if fail_on == "any":
        return summary.get("verdict") == "fail"
    key_by_mode = {
        "actions": "action_failures",
        "latency": "latency_regressions",
        "guard": "guard_regressions",
        "shape": "shape_failures",
    }
    return int(summary.get(key_by_mode[fail_on]) or 0) > 0


def write_report(report: dict[str, Any], output: str | Path) -> None:
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def format_policy_diff(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"tether policy diff - {str(summary['verdict']).upper()}",
        f"mode: {report['mode']}",
        f"compared: {summary['compared']}/{summary['baseline_requests']}",
        (
            "actions: "
            f"max_abs={summary['max_action_delta']:.3g}, "
            f"p95_abs={summary['p95_action_delta']:.3g}, "
            f"min_cos={summary['min_action_cosine']:.6f}"
        ),
        (
            "failures: "
            f"actions={summary['action_failures']}, "
            f"latency={summary['latency_regressions']}, "
            f"shape={summary['shape_failures']}, "
            f"guard={summary['guard_regressions']}, "
            f"missing={summary['missing_candidate']}"
        ),
    ]
    if summary.get("request_mismatches"):
        lines.append(f"request mismatches: {summary['request_mismatches']}")
    for warning in summary.get("metadata_warnings") or []:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)


__all__ = [
    "POLICY_DIFF_SCHEMA_VERSION",
    "PolicyDiffError",
    "diff_policy_traces",
    "format_policy_diff",
    "should_fail",
    "write_report",
]
