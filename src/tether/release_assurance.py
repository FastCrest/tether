"""Release assurance packet for robot policy updates.

This is the customer-facing composition layer over deployment proof, realtime
serving certificates, action-execution checks, policy diff, and shadow rollout
gates. Lower-level commands remain available; this module answers the operator
question directly: promote, hold, or roll back this release?
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from tether.promote import PromotionError, decide_promotion
from tether.realtime_cert import (
    RealtimeCertificateError,
    build_realtime_certificate,
    format_realtime_certificate_markdown,
    load_deploy_proof,
)

RELEASE_ASSURANCE_SCHEMA_VERSION = 1
ReleaseDecision = Literal["PROMOTE", "HOLD", "ROLLBACK"]


class ReleaseAssuranceError(ValueError):
    """Raised when release assurance input artifacts cannot be loaded."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_packet_dir(packet: str | Path) -> Path:
    path = Path(packet).expanduser().resolve()
    if path.is_file():
        if path.name != "deployment-proof.json":
            raise ReleaseAssuranceError(
                "packet file must be deployment-proof.json; pass the packet directory otherwise"
            )
        return path.parent
    return path


def _failed_checks(report: dict[str, Any] | None) -> list[str]:
    if not isinstance(report, dict):
        return []
    summary = report.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("failed_checks"), list):
        return [str(item) for item in summary["failed_checks"]]
    return [
        str(check.get("name"))
        for check in report.get("checks") or []
        if isinstance(check, dict) and check.get("status") == "fail"
    ]


def _component(
    name: str,
    *,
    present: bool,
    decision: str,
    artifact: str = "",
    failed_checks: list[str] | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "present": bool(present),
        "decision": decision,
        "artifact": artifact,
        "failed_checks": failed_checks or [],
        "summary": summary or {},
    }


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _add_signal(
    signals: list[dict[str, Any]],
    name: str,
    status: str,
    *,
    value: Any = None,
    threshold: Any = None,
    source: str,
    remediation: str = "",
) -> None:
    row = {
        "name": name,
        "status": status,
        "value": value,
        "threshold": threshold,
        "source": source,
    }
    if remediation:
        row["remediation"] = remediation
    signals.append(row)


def _risk_signals(
    *,
    proof: dict[str, Any],
    promotion: dict[str, Any] | None,
    realtime: dict[str, Any] | None,
    shadow: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []

    proof_checks = proof.get("checks") or []
    proof_failures = [
        check.get("name")
        for check in proof_checks
        if isinstance(check, dict) and check.get("status") == "fail"
    ]
    _add_signal(
        signals,
        "deployment_proof_failures",
        "fail" if proof_failures else "pass",
        value=len(proof_failures),
        threshold=0,
        source="deployment-proof",
        remediation="Fix failed deployment-proof checks before release.",
    )

    latency = proof.get("latency") if isinstance(proof.get("latency"), dict) else {}
    guard_violations = int(latency.get("guard_violations") or 0) if latency else 0
    _add_signal(
        signals,
        "guard_violations",
        "fail" if guard_violations else "pass",
        value=guard_violations,
        threshold=0,
        source="deployment-proof",
        remediation="Inspect ActionGuard violations and clamp causes.",
    )

    policy_summary = (
        (promotion.get("policy_diff") or {}).get("summary")
        if isinstance(promotion, dict)
        else None
    )
    if not isinstance(policy_summary, dict):
        proof_policy = proof.get("policy_diff") if isinstance(proof.get("policy_diff"), dict) else {}
        embedded = proof_policy.get("report") if isinstance(proof_policy, dict) else None
        policy_summary = embedded.get("summary") if isinstance(embedded, dict) else None
    if isinstance(policy_summary, dict):
        action_failures = int(policy_summary.get("action_failures") or 0)
        latency_regressions = int(policy_summary.get("latency_regressions") or 0)
        guard_regressions = int(policy_summary.get("guard_regressions") or 0)
        shadow_pending = int(policy_summary.get("shadow_pending") or 0)
        shadow_errors = int(policy_summary.get("shadow_errors") or 0)
        _add_signal(
            signals,
            "policy_action_delta",
            "fail" if action_failures else "pass",
            value=action_failures,
            threshold=0,
            source="policy-diff",
            remediation="Compare candidate action chunks and collect more shadow data.",
        )
        _add_signal(
            signals,
            "policy_latency_regression",
            "warn" if latency_regressions else "pass",
            value=latency_regressions,
            threshold=0,
            source="policy-diff",
            remediation="Profile candidate latency before promotion.",
        )
        _add_signal(
            signals,
            "policy_guard_regression",
            "fail" if guard_regressions else "pass",
            value=guard_regressions,
            threshold=0,
            source="policy-diff",
            remediation="Do not promote until guard regressions are understood.",
        )
        _add_signal(
            signals,
            "shadow_completion",
            "fail" if shadow_errors else ("warn" if shadow_pending else "pass"),
            value={"pending": shadow_pending, "errors": shadow_errors},
            threshold={"pending": 0, "errors": 0},
            source="shadow-rollout",
            remediation="Wait for shadow_result rows to flush or fix shadow runtime errors.",
        )
    else:
        _add_signal(
            signals,
            "policy_diff_present",
            "warn",
            value=False,
            threshold=True,
            source="policy-diff",
            remediation="Add candidate or shadow policy diff evidence before production rollout.",
        )

    if isinstance(realtime, dict):
        for check in realtime.get("checks") or []:
            if not isinstance(check, dict):
                continue
            name = str(check.get("name") or "")
            if name in {
                "roundtrip_p95_within_budget",
                "jitter_within_budget",
                "deadline_misses_within_budget",
                "control_budget_misses_within_budget",
                "action_execution_certificate",
            }:
                status = str(check.get("status") or "unknown")
                _add_signal(
                    signals,
                    name,
                    "fail" if status == "fail" else ("warn" if status == "skip" else "pass"),
                    value=check.get("actual"),
                    threshold=check.get("expected"),
                    source="realtime-certificate",
                    remediation=str(check.get("remediation") or ""),
                )

        execution = realtime.get("execution_certificate")
        if isinstance(execution, dict):
            metrics = execution.get("metrics") or {}
            stale = metrics.get("stale_action_window_ms") or {}
            boundary = metrics.get("chunk_boundary_delta") or {}
            velocity = metrics.get("velocity_discontinuity") or {}
            thresholds = execution.get("thresholds") or {}
            for signal_name, value, limit, remediation in (
                (
                    "stale_action_window_ms",
                    _numeric(stale.get("max_ms")),
                    _numeric(thresholds.get("max_stale_action_window_ms")),
                    "Shorten serving latency or increase executed horizon.",
                ),
                (
                    "chunk_boundary_delta",
                    _numeric(boundary.get("max_abs")),
                    _numeric(thresholds.get("max_chunk_boundary_delta")),
                    "Smooth, fuse, or shorten action chunks near boundaries.",
                ),
                (
                    "velocity_discontinuity",
                    _numeric(velocity.get("max_abs")),
                    _numeric(thresholds.get("max_velocity_discontinuity")),
                    "Reduce boundary velocity jumps before release.",
                ),
            ):
                status = "warn"
                if value is not None and limit is not None:
                    status = "pass" if value <= limit else "fail"
                _add_signal(
                    signals,
                    signal_name,
                    status,
                    value=value,
                    threshold=limit,
                    source="action-execution-certificate",
                    remediation=remediation,
                )
    else:
        _add_signal(
            signals,
            "realtime_certificate_present",
            "warn",
            value=False,
            threshold=True,
            source="realtime-certificate",
            remediation="Run release assurance with --realtime or --control-hz for control-loop evidence.",
        )

    if shadow is None:
        _add_signal(
            signals,
            "shadow_rollout_present",
            "warn",
            value=False,
            threshold=True,
            source="shadow-rollout",
            remediation="Mirror a candidate with --shadow-policy and pass --shadow-trace before fleet rollout.",
        )

    return signals


def _gaps(
    *,
    proof: dict[str, Any],
    realtime: dict[str, Any] | None,
    shadow: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []

    def add(control: str, severity: str, message: str, next_step: str) -> None:
        gaps.append(
            {
                "control": control,
                "severity": severity,
                "message": message,
                "next_step": next_step,
            }
        )

    security = proof.get("security")
    security = security if isinstance(security, dict) else {}
    if not security.get("enabled"):
        add(
            "runtime_auth",
            "warn",
            "Proof packet does not show API-key auth enforcement.",
            "Run `tether prove` with --api-key for production promotion packets.",
        )

    trace = proof.get("trace")
    trace = trace if isinstance(trace, dict) else {}
    if not trace.get("files"):
        add(
            "trace_forensics",
            "warn",
            "Proof packet has no recorded /act trace files.",
            "Run `tether prove` with --record-dir so rollback/debug evidence exists.",
        )

    safety = proof.get("safety_stress")
    safety = safety if isinstance(safety, dict) else {}
    if not safety.get("enabled"):
        add(
            "runtime_safety",
            "warn",
            "ActionGuard stress evidence is missing.",
            "Pass --embodiment or --safety-config during proof collection.",
        )

    if realtime is None:
        add(
            "realtime_serving",
            "warn",
            "No realtime serving certificate is attached.",
            "Run with --realtime and --control-hz for the target robot loop.",
        )
    elif not isinstance(realtime.get("execution_certificate"), dict):
        add(
            "action_execution",
            "warn",
            "Realtime certificate lacks action-execution continuity evidence.",
            "Run with --execution-cert after /act responses include action_execution telemetry.",
        )

    if shadow is None:
        add(
            "shadow_rollout",
            "warn",
            "No shadow rollout gate is attached.",
            "Pass --shadow-trace collected from `tether serve --shadow-policy --record`.",
        )

    return gaps


def _confidence(
    *,
    components: list[dict[str, Any]],
    risk_signals: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> int:
    score = 100
    for component in components:
        if not component.get("present"):
            score -= 8
        elif component.get("decision") in {"FAIL", "BLOCK", "HOLD"}:
            score -= 25
        elif component.get("decision") == "ROLLBACK":
            score -= 40
    for signal in risk_signals:
        if signal.get("status") == "fail":
            score -= 12
        elif signal.get("status") == "warn":
            score -= 4
    for gap in gaps:
        score -= 6 if gap.get("severity") == "warn" else 12
    return max(0, min(100, score))


# Risk signals that gate the release decision regardless of the promotion
# profile: zero-tolerance runtime/behavioral safety breaches that the profile
# does not (and should not) make tunable. A failing hard-safety signal forces
# HOLD (or ROLLBACK for an active candidate) even when the promotion gate passed.
_HARD_SAFETY_SIGNALS = frozenset({"guard_violations", "policy_guard_regression"})


def _blocking_safety_signals(risk_signals: list[dict[str, Any]]) -> list[str]:
    """Names of failing hard-safety signals that must block promotion."""
    return [
        str(signal.get("name"))
        for signal in risk_signals
        if signal.get("status") == "fail" and signal.get("name") in _HARD_SAFETY_SIGNALS
    ]


def _final_decision(
    *,
    promotion: dict[str, Any],
    realtime: dict[str, Any] | None,
    shadow: dict[str, Any] | None,
    candidate_active: bool,
    safety_blocked: bool = False,
) -> ReleaseDecision:
    blocking = safety_blocked
    if promotion.get("decision") != "PROMOTE":
        blocking = True
    if realtime is not None and realtime.get("decision") != "PASS":
        blocking = True
    if shadow is not None and shadow.get("decision") != "PROMOTE":
        blocking = True
    if not blocking:
        return "PROMOTE"
    if candidate_active or promotion.get("decision") == "ROLLBACK" or (
        shadow is not None and shadow.get("decision") == "ROLLBACK"
    ):
        return "ROLLBACK"
    return "HOLD"


def build_release_assurance(
    *,
    packet: str | Path,
    profile_path: str | Path | None = None,
    candidate_active: bool = False,
    realtime: bool = False,
    target: str = "",
    control_hz: float | None = None,
    max_roundtrip_p95_ms: float | None = None,
    max_jitter_p95_minus_p50_ms: float | None = None,
    max_deadline_misses: int = 0,
    max_control_budget_misses: int = 0,
    max_act_errors: int = 0,
    execution_cert: bool = False,
    max_stale_action_window_ms: float = 100.0,
    max_chunk_boundary_delta: float = 0.15,
    max_velocity_discontinuity: float = 0.2,
    require_phase_aware_horizon: bool = False,
    require_runtime_attribution: bool = True,
    shadow_trace: str | Path | None = None,
    shadow_min_compared: int = 1,
    shadow_wait_timeout_s: float = 0.0,
    shadow_poll_s: float = 0.25,
    shadow_fail_on: str = "any",
    shadow_min_action_cos: float = 0.995,
    shadow_max_action_delta: float = 0.10,
    shadow_max_latency_regression_pct: float = 0.10,
) -> dict[str, Any]:
    """Build one release assurance report from existing release evidence."""

    packet_dir = _resolve_packet_dir(packet)
    proof_path = packet_dir / "deployment-proof.json"
    try:
        proof = load_deploy_proof(proof_path)
    except RealtimeCertificateError as exc:
        raise ReleaseAssuranceError(str(exc)) from exc

    try:
        shadow_report = None
        effective_profile_path = profile_path or ("lab-shadow" if shadow_trace else None)
        if shadow_trace:
            from tether.shadow_rollout import run_shadow_rollout_gate

            shadow_report = run_shadow_rollout_gate(
                trace=shadow_trace,
                packet_dir=packet_dir,
                profile=effective_profile_path or "lab-shadow",
                candidate_active=candidate_active,
                min_compared=shadow_min_compared,
                wait_timeout_s=shadow_wait_timeout_s,
                poll_s=shadow_poll_s,
                fail_on=shadow_fail_on,
                min_action_cos=shadow_min_action_cos,
                max_action_delta=shadow_max_action_delta,
                max_latency_regression_pct=shadow_max_latency_regression_pct,
                use_existing_packet=True,
            )
        promotion = decide_promotion(
            packet_dir,
            profile_path=effective_profile_path,
            candidate_active=candidate_active,
        )
    except (PromotionError, ValueError) as exc:
        raise ReleaseAssuranceError(str(exc)) from exc

    realtime_report = None
    realtime_requested = bool(
        realtime
        or execution_cert
        or target
        or (control_hz is not None and control_hz > 0)
        or max_roundtrip_p95_ms is not None
        or max_jitter_p95_minus_p50_ms is not None
    )
    if realtime_requested:
        try:
            realtime_report = build_realtime_certificate(
                proof,
                target=target,
                control_hz=control_hz,
                max_roundtrip_p95_ms=max_roundtrip_p95_ms,
                max_jitter_p95_minus_p50_ms=max_jitter_p95_minus_p50_ms,
                max_deadline_misses=max_deadline_misses,
                max_control_budget_misses=max_control_budget_misses,
                max_act_errors=max_act_errors,
                execution_cert=execution_cert,
                max_stale_action_window_ms=max_stale_action_window_ms,
                max_chunk_boundary_delta=max_chunk_boundary_delta,
                max_velocity_discontinuity=max_velocity_discontinuity,
                require_phase_aware_horizon=require_phase_aware_horizon,
                require_runtime_attribution=require_runtime_attribution,
            )
        except RealtimeCertificateError as exc:
            raise ReleaseAssuranceError(str(exc)) from exc

    components = [
        _component(
            "deployment_proof",
            present=True,
            decision="PASS" if proof.get("passed") else "FAIL",
            artifact=str(proof_path),
            failed_checks=_failed_checks(proof),
            summary={"passed": bool(proof.get("passed"))},
        ),
        _component(
            "promotion_gate",
            present=True,
            decision=str(promotion.get("decision") or "BLOCK"),
            artifact=str(packet_dir / "promotion-decision.json")
            if (packet_dir / "promotion-decision.json").exists()
            else "",
            failed_checks=_failed_checks(promotion),
            summary=promotion.get("summary") if isinstance(promotion.get("summary"), dict) else {},
        ),
        _component(
            "realtime_certificate",
            present=realtime_report is not None,
            decision=str((realtime_report or {}).get("decision") or "NOT_RUN"),
            artifact="",
            failed_checks=_failed_checks(realtime_report),
            summary=(realtime_report or {}).get("summary")
            if isinstance((realtime_report or {}).get("summary"), dict)
            else {},
        ),
        _component(
            "shadow_rollout",
            present=shadow_report is not None,
            decision=str((shadow_report or {}).get("decision") or "NOT_RUN"),
            artifact=str((shadow_report or {}).get("packet_dir") or ""),
            failed_checks=list(((shadow_report or {}).get("promotion") or {}).get("failed_checks") or []),
            summary=((shadow_report or {}).get("policy_diff") or {}).get("summary")
            if isinstance(((shadow_report or {}).get("policy_diff") or {}).get("summary"), dict)
            else {},
        ),
    ]

    risk_signals = _risk_signals(
        proof=proof,
        promotion=promotion,
        realtime=realtime_report,
        shadow=shadow_report,
    )
    gaps = _gaps(proof=proof, realtime=realtime_report, shadow=shadow_report)
    blocking_signals = _blocking_safety_signals(risk_signals)
    decision = _final_decision(
        promotion=promotion,
        realtime=realtime_report,
        shadow=shadow_report,
        candidate_active=candidate_active,
        safety_blocked=bool(blocking_signals),
    )
    confidence = _confidence(components=components, risk_signals=risk_signals, gaps=gaps)

    failed_components = [
        component["name"]
        for component in components
        if component["present"]
        and component["decision"] not in {"PASS", "PROMOTE"}
    ]
    # blocked_by must agree with the decision: empty when we PROMOTE (so a failing but
    # profile-permitted signal no longer contradicts a PROMOTE verdict), and the full
    # set of failing signals when the release is held/rolled back. Hard-safety signals
    # (blocking_signals) independently force a non-PROMOTE decision above.
    failed_signals = (
        [signal["name"] for signal in risk_signals if signal.get("status") == "fail"]
        if decision != "PROMOTE"
        else []
    )

    return {
        "schema_version": RELEASE_ASSURANCE_SCHEMA_VERSION,
        "kind": "tether.release_assurance",
        "generated_at": _now_iso(),
        "decision": decision,
        "passed": decision == "PROMOTE",
        "candidate_active": bool(candidate_active),
        "confidence": confidence,
        "packet_dir": str(packet_dir),
        "profile": promotion.get("profile") or {},
        "source": {
            "deployment_proof": str(proof_path),
            "shadow_trace": str(shadow_trace) if shadow_trace else "",
        },
        "components": components,
        "risk_signals": risk_signals,
        "gaps": gaps,
        "blocked_by": {
            "components": failed_components,
            "signals": failed_signals,
        },
        "promotion": promotion,
        "realtime_certificate": realtime_report,
        "shadow_rollout": shadow_report,
    }


def format_release_assurance_human(report: dict[str, Any]) -> str:
    lines = [
        f"tether release assure - {report.get('decision')}",
        f"packet:     {report.get('packet_dir')}",
        f"profile:    {(report.get('profile') or {}).get('name', 'default')}",
        f"confidence: {report.get('confidence', 0)}/100",
    ]
    lines.append("components:")
    for component in report.get("components") or []:
        lines.append(
            "  - "
            f"{component.get('name')}: {component.get('decision')}"
            f"{'' if component.get('present') else ' (not run)'}"
        )
    blocked = report.get("blocked_by") or {}
    blockers = list(blocked.get("components") or []) + list(blocked.get("signals") or [])
    if blockers:
        lines.append("blocked by:")
        for item in blockers[:12]:
            lines.append(f"  - {item}")
    gaps = report.get("gaps") or []
    if gaps:
        lines.append("open evidence gaps:")
        for gap in gaps[:8]:
            lines.append(f"  - {gap.get('control')}: {gap.get('message')}")
    return "\n".join(lines)


def format_release_assurance_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Tether Release Assurance",
        "",
        f"- Decision: **{report.get('decision')}**",
        f"- Confidence: `{report.get('confidence', 0)}/100`",
        f"- Packet: `{report.get('packet_dir')}`",
        f"- Profile: `{(report.get('profile') or {}).get('name', 'default')}`",
        "",
        "## Components",
        "",
        "| Component | Present | Decision | Failed checks |",
        "|---|---:|---:|---|",
    ]
    for component in report.get("components") or []:
        failed = ", ".join(component.get("failed_checks") or [])
        lines.append(
            f"| `{component.get('name')}` | {bool(component.get('present'))} | "
            f"`{component.get('decision')}` | {failed or '-'} |"
        )

    lines.extend([
        "",
        "## Risk Signals",
        "",
        "| Signal | Status | Value | Threshold | Source |",
        "|---|---:|---:|---:|---|",
    ])
    for signal in report.get("risk_signals") or []:
        lines.append(
            f"| `{signal.get('name')}` | `{signal.get('status')}` | "
            f"{signal.get('value')} | {signal.get('threshold')} | {signal.get('source')} |"
        )

    gaps = report.get("gaps") or []
    lines.extend(["", "## Evidence Gaps", ""])
    if not gaps:
        lines.append("No open evidence gaps recorded.")
    else:
        for gap in gaps:
            lines.extend([
                f"### {gap.get('control')}",
                "",
                f"- Severity: `{gap.get('severity')}`",
                f"- Gap: {gap.get('message')}",
                f"- Next step: `{gap.get('next_step')}`",
                "",
            ])

    realtime = report.get("realtime_certificate")
    if isinstance(realtime, dict):
        lines.extend(["", format_realtime_certificate_markdown(realtime).rstrip(), ""])

    return "\n".join(lines).rstrip() + "\n"


def write_release_assurance_packet(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir).expanduser().resolve()
    packet_dir = report.get("packet_dir")
    if packet_dir and out == Path(str(packet_dir)).expanduser().resolve():
        raise ReleaseAssuranceError(
            "--output-dir must be separate from the input proof packet directory "
            "so release-assurance artifacts do not overwrite proof MANIFEST.json"
        )
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "release-assurance.json"
    md_path = out / "release-assurance.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(format_release_assurance_markdown(report), encoding="utf-8")

    files = []
    for path in (json_path, md_path):
        files.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest = {
        "kind": "tether.release_assurance_manifest",
        "schema_version": 1,
        "generated_at": _now_iso(),
        "files": files,
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "RELEASE_ASSURANCE_SCHEMA_VERSION",
    "ReleaseAssuranceError",
    "build_release_assurance",
    "format_release_assurance_human",
    "format_release_assurance_markdown",
    "write_release_assurance_packet",
]
