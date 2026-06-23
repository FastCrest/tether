"""Realtime serving certificate for deployment proof packets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RealtimeCertificateError(ValueError):
    """Raised when a realtime certificate cannot be built."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _threshold(receipt: dict[str, Any], key: str) -> Any:
    profile = receipt.get("profile") or {}
    thresholds = profile.get("thresholds") or {}
    return thresholds.get(key)


def _latency_metric(latency: dict[str, Any], section: str, key: str) -> float | None:
    block = latency.get(section) or {}
    return _number(block.get(key))


def _resolve_proof_path(path: str | Path) -> Path:
    proof_path = Path(path).expanduser()
    if proof_path.is_dir():
        proof_path = proof_path / "deployment-proof.json"
    if not proof_path.exists():
        raise RealtimeCertificateError(
            f"deployment proof not found: {proof_path}. Run `tether prove` first."
        )
    return proof_path.resolve()


def load_deploy_proof(path: str | Path) -> dict[str, Any]:
    """Load a deployment proof packet directory or deployment-proof.json file."""

    proof_path = _resolve_proof_path(path)
    try:
        receipt = json.loads(proof_path.read_text())
    except json.JSONDecodeError as exc:
        raise RealtimeCertificateError(f"invalid deployment proof JSON: {exc}") from exc
    if not isinstance(receipt, dict):
        raise RealtimeCertificateError("deployment proof JSON must be an object")
    if receipt.get("kind") != "tether.deployment_proof":
        raise RealtimeCertificateError(
            "deployment-proof.json is not a Tether deployment proof"
        )
    receipt["_source_proof_path"] = str(proof_path)
    return receipt


def _control_hz(
    receipt: dict[str, Any],
    latency: dict[str, Any],
    override_hz: float | None,
) -> tuple[float | None, str]:
    if override_hz and override_hz > 0:
        return float(override_hz), "argument"
    budget_hz = _number((latency.get("control_budget") or {}).get("control_hz"))
    if budget_hz and budget_hz > 0:
        return budget_hz, "latency.control_budget"
    profile_hz = _number(_threshold(receipt, "control_hz"))
    if profile_hz and profile_hz > 0:
        return profile_hz, "profile.thresholds.control_hz"
    return None, "missing"


def _roundtrip_samples(receipt: dict[str, Any]) -> list[float]:
    samples = receipt.get("act_samples") or []
    values: list[float] = []
    if not isinstance(samples, list):
        return values
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        value = _number(sample.get("roundtrip_ms"))
        if value is not None:
            values.append(value)
    return values


def _control_budget_summary(
    receipt: dict[str, Any],
    latency: dict[str, Any],
    control_hz: float | None,
) -> dict[str, Any]:
    budget = latency.get("control_budget") or {}
    period_ms = _number(budget.get("period_ms"))
    if control_hz and control_hz > 0:
        period_ms = 1000.0 / control_hz

    samples = _roundtrip_samples(receipt)
    missed_samples: int | None
    if period_ms and samples:
        missed_samples = sum(1 for value in samples if value > period_ms)
        missed_source = "act_samples"
    else:
        missed = _number(budget.get("missed_samples"))
        missed_samples = int(missed) if missed is not None else None
        missed_source = "latency.control_budget" if missed is not None else "missing"

    return {
        "control_hz": round(control_hz, 3) if control_hz else None,
        "period_ms": round(period_ms, 3) if period_ms else None,
        "missed_samples": missed_samples,
        "missed_samples_source": missed_source,
        "roundtrip_sample_count": len(samples),
    }


def _profile_jitter_threshold(receipt: dict[str, Any]) -> float | None:
    return _number(_threshold(receipt, "max_jitter_p95_minus_p50_ms"))


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


def _summarize_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pass": sum(1 for check in checks if check["status"] == "pass"),
        "fail": sum(1 for check in checks if check["status"] == "fail"),
        "skip": sum(1 for check in checks if check["status"] == "skip"),
        "failed_checks": [
            check["name"] for check in checks if check["status"] == "fail"
        ],
    }


def build_realtime_certificate(
    receipt: dict[str, Any],
    *,
    control_hz: float | None = None,
    target: str = "",
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
) -> dict[str, Any]:
    """Build a pass/fail realtime-serving certificate from proof latency evidence."""

    if receipt.get("kind") != "tether.deployment_proof":
        raise RealtimeCertificateError("receipt is not a Tether deployment proof")
    if control_hz is not None and control_hz <= 0:
        raise RealtimeCertificateError("control_hz must be > 0 when supplied")

    latency = receipt.get("latency") or {}
    if not isinstance(latency, dict):
        latency = {}

    effective_hz, control_source = _control_hz(receipt, latency, control_hz)
    control_budget = _control_budget_summary(receipt, latency, effective_hz)
    roundtrip_p95 = _latency_metric(latency, "roundtrip_ms", "p95_ms")
    roundtrip_p50 = _latency_metric(latency, "roundtrip_ms", "p50_ms")
    roundtrip_p99 = _latency_metric(latency, "roundtrip_ms", "p99_ms")
    roundtrip_max = _latency_metric(latency, "roundtrip_ms", "max_ms")
    jitter = _number((latency.get("jitter") or {}).get("p95_minus_p50_ms"))
    deadline_misses = _number(latency.get("deadline_misses"))
    act_errors = _number(latency.get("act_errors"))

    p95_budget_ms = max_roundtrip_p95_ms
    if p95_budget_ms is None and control_budget["period_ms"] is not None:
        p95_budget_ms = float(control_budget["period_ms"])

    jitter_budget_ms = max_jitter_p95_minus_p50_ms
    if jitter_budget_ms is None:
        jitter_budget_ms = _profile_jitter_threshold(receipt)

    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "deployment_proof_passed",
        _pass_fail(receipt.get("passed") is True),
        metric="receipt.passed",
        actual=receipt.get("passed"),
        expected=True,
        remediation="Fix failed deployment-proof checks before certifying realtime serving.",
    )
    _add_check(
        checks,
        "latency_summary_present",
        _pass_fail(bool(latency and latency.get("roundtrip_ms"))),
        metric="latency.roundtrip_ms",
        actual=bool(latency and latency.get("roundtrip_ms")),
        expected=True,
        remediation="Run `tether prove` with /act samples before building the certificate.",
    )
    _add_check(
        checks,
        "control_hz_defined",
        _pass_fail(bool(effective_hz and effective_hz > 0)),
        metric="control_hz",
        actual=effective_hz,
        expected="> 0",
        remediation="Pass `--control-hz` or rerun `tether prove --control-hz <hz>`.",
    )

    if roundtrip_p95 is None or p95_budget_ms is None:
        _add_check(
            checks,
            "roundtrip_p95_within_budget",
            "fail",
            metric="latency.roundtrip_ms.p95_ms",
            actual=roundtrip_p95,
            expected=f"<= {p95_budget_ms}" if p95_budget_ms is not None else "control budget",
            remediation="Collect proof latency with a control rate before certifying realtime serving.",
        )
    else:
        _add_check(
            checks,
            "roundtrip_p95_within_budget",
            _pass_fail(roundtrip_p95 <= p95_budget_ms),
            metric="latency.roundtrip_ms.p95_ms",
            actual=roundtrip_p95,
            expected=f"<= {round(p95_budget_ms, 3)}",
            remediation="Reduce model/server latency or lower the control rate.",
        )

    if jitter_budget_ms is None:
        _add_check(
            checks,
            "jitter_within_budget",
            "skip",
            metric="latency.jitter.p95_minus_p50_ms",
            actual=jitter,
            expected="not configured",
        )
    elif jitter is None:
        _add_check(
            checks,
            "jitter_within_budget",
            "fail",
            metric="latency.jitter.p95_minus_p50_ms",
            actual=None,
            expected=f"<= {jitter_budget_ms}",
            remediation="Rerun `tether prove` with enough samples to compute jitter.",
        )
    else:
        _add_check(
            checks,
            "jitter_within_budget",
            _pass_fail(jitter <= jitter_budget_ms),
            metric="latency.jitter.p95_minus_p50_ms",
            actual=jitter,
            expected=f"<= {jitter_budget_ms}",
            remediation="Reduce queueing jitter, GPU contention, or batching delays.",
        )

    if deadline_misses is None:
        _add_check(
            checks,
            "deadline_misses_within_budget",
            "fail",
            metric="latency.deadline_misses",
            actual=None,
            expected=f"<= {max_deadline_misses}",
            remediation="Rerun `tether prove` with deadline-aware serve evidence.",
        )
    else:
        _add_check(
            checks,
            "deadline_misses_within_budget",
            _pass_fail(deadline_misses <= max_deadline_misses),
            metric="latency.deadline_misses",
            actual=int(deadline_misses),
            expected=f"<= {max_deadline_misses}",
            remediation="Investigate deadline misses before deployment.",
        )

    missed_samples = control_budget.get("missed_samples")
    if missed_samples is None:
        _add_check(
            checks,
            "control_budget_misses_within_budget",
            "fail",
            metric="latency.control_budget.missed_samples",
            actual=None,
            expected=f"<= {max_control_budget_misses}",
            remediation="Rerun `tether prove --control-hz <hz>` or pass a proof with act samples.",
        )
    else:
        _add_check(
            checks,
            "control_budget_misses_within_budget",
            _pass_fail(int(missed_samples) <= max_control_budget_misses),
            metric="latency.control_budget.missed_samples",
            actual=int(missed_samples),
            expected=f"<= {max_control_budget_misses}",
            remediation="The server missed the robot control-loop period.",
        )

    if act_errors is None:
        _add_check(
            checks,
            "act_errors_within_budget",
            "fail",
            metric="latency.act_errors",
            actual=None,
            expected=f"<= {max_act_errors}",
            remediation="Rerun `tether prove` with /act samples that report errors.",
        )
    else:
        _add_check(
            checks,
            "act_errors_within_budget",
            _pass_fail(act_errors <= max_act_errors),
            metric="latency.act_errors",
            actual=int(act_errors),
            expected=f"<= {max_act_errors}",
            remediation="Fix /act errors before certifying realtime serving.",
        )

    summary = _summarize_checks(checks)
    decision = "PASS" if summary["fail"] == 0 else "FAIL"

    report = {
        "schema_version": 1,
        "kind": "tether.realtime_serving_certificate",
        "generated_at": _now_iso(),
        "decision": decision,
        "passed": decision == "PASS",
        "target": target,
        "source": {
            "deployment_proof": receipt.get("_source_proof_path", ""),
            "export_dir": receipt.get("export_dir", ""),
            "profile": (receipt.get("profile") or {}).get("name", ""),
        },
        "control_budget": {
            **control_budget,
            "control_hz_source": control_source,
            "roundtrip_p95_budget_ms": (
                round(float(p95_budget_ms), 3) if p95_budget_ms is not None else None
            ),
            "max_control_budget_misses": max_control_budget_misses,
        },
        "latency": {
            "samples": latency.get("samples"),
            "roundtrip_ms": {
                "p50_ms": roundtrip_p50,
                "p95_ms": roundtrip_p95,
                "p99_ms": roundtrip_p99,
                "max_ms": roundtrip_max,
            },
            "jitter_p95_minus_p50_ms": jitter,
            "deadline_misses": int(deadline_misses) if deadline_misses is not None else None,
            "act_errors": int(act_errors) if act_errors is not None else None,
        },
        "thresholds": {
            "max_roundtrip_p95_ms": p95_budget_ms,
            "max_jitter_p95_minus_p50_ms": jitter_budget_ms,
            "max_deadline_misses": max_deadline_misses,
            "max_control_budget_misses": max_control_budget_misses,
            "max_act_errors": max_act_errors,
        },
        "checks": checks,
        "summary": summary,
    }

    if execution_cert:
        from tether.action_execution_cert import build_action_execution_certificate

        execution_report = build_action_execution_certificate(
            receipt,
            control_hz=effective_hz,
            max_stale_action_window_ms=max_stale_action_window_ms,
            max_chunk_boundary_delta=max_chunk_boundary_delta,
            max_velocity_discontinuity=max_velocity_discontinuity,
            require_phase_aware_horizon=require_phase_aware_horizon,
            require_runtime_attribution=require_runtime_attribution,
        )
        report["execution_certificate"] = execution_report
        _add_check(
            checks,
            "action_execution_certificate",
            "pass" if execution_report.get("decision") == "PASS" else "fail",
            metric="execution_certificate.decision",
            actual=execution_report.get("decision"),
            expected="PASS",
            remediation="Fix failed action-execution checks before promoting the policy.",
        )
        summary = _summarize_checks(checks)
        decision = "PASS" if summary["fail"] == 0 else "FAIL"
        report["summary"] = summary
        report["decision"] = decision
        report["passed"] = decision == "PASS"

    return report


def format_realtime_certificate_human(report: dict[str, Any]) -> str:
    """Format a realtime certificate for terminals and issue comments."""

    decision = report.get("decision", "FAIL")
    source = report.get("source") or {}
    control = report.get("control_budget") or {}
    latency = report.get("latency") or {}
    roundtrip = latency.get("roundtrip_ms") or {}
    summary = report.get("summary") or {}
    lines = [
        f"tether realtime-serving-cert - {decision}",
        f"target:  {report.get('target') or 'unspecified'}",
        f"proof:   {source.get('deployment_proof') or 'unknown'}",
        (
            "control: "
            f"{control.get('control_hz') or 'unknown'} Hz, "
            f"period={control.get('period_ms') or 'unknown'} ms"
        ),
        (
            "roundtrip: "
            f"p50={roundtrip.get('p50_ms')}ms "
            f"p95={roundtrip.get('p95_ms')}ms "
            f"p99={roundtrip.get('p99_ms')}ms "
            f"max={roundtrip.get('max_ms')}ms"
        ),
        (
            "misses: "
            f"control_budget={control.get('missed_samples')} "
            f"deadline={latency.get('deadline_misses')} "
            f"act_errors={latency.get('act_errors')}"
        ),
        (
            "checks:  "
            f"{summary.get('pass', 0)} pass, "
            f"{summary.get('fail', 0)} fail, "
            f"{summary.get('skip', 0)} skip"
        ),
    ]
    failed = summary.get("failed_checks") or []
    if failed:
        lines.append("failed:  " + ", ".join(failed))
    execution = report.get("execution_certificate")
    if isinstance(execution, dict):
        metrics = execution.get("metrics") or {}
        stale = metrics.get("stale_action_window_ms") or {}
        boundary = metrics.get("chunk_boundary_delta") or {}
        velocity = metrics.get("velocity_discontinuity") or {}
        lines.append(
            "execution: "
            f"{execution.get('decision', 'FAIL')} "
            f"stale_max={stale.get('max_ms')}ms "
            f"boundary_delta={boundary.get('max_abs')} "
            f"velocity_jump={velocity.get('max_abs')}"
        )
        failed_execution = (execution.get("summary") or {}).get("failed_checks") or []
        if failed_execution:
            lines.append("execution_failed: " + ", ".join(failed_execution))
    return "\n".join(lines)


def format_realtime_certificate_markdown(report: dict[str, Any]) -> str:
    """Format a realtime certificate as Markdown."""

    source = report.get("source") or {}
    control = report.get("control_budget") or {}
    latency = report.get("latency") or {}
    roundtrip = latency.get("roundtrip_ms") or {}
    summary = report.get("summary") or {}
    lines = [
        "# Tether Realtime Serving Certificate",
        "",
        f"- Decision: **{report.get('decision', 'FAIL')}**",
        f"- Target: `{report.get('target') or 'unspecified'}`",
        f"- Source proof: `{source.get('deployment_proof') or 'unknown'}`",
        f"- Control rate: `{control.get('control_hz') or 'unknown'} Hz`",
        f"- Control period: `{control.get('period_ms') or 'unknown'} ms`",
        f"- Roundtrip p95 budget: `{control.get('roundtrip_p95_budget_ms')} ms`",
        f"- Control-budget misses: `{control.get('missed_samples')}`",
        f"- Deadline misses: `{latency.get('deadline_misses')}`",
        f"- Act errors: `{latency.get('act_errors')}`",
        "",
        "## Latency",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| roundtrip p50 | {roundtrip.get('p50_ms')} ms |",
        f"| roundtrip p95 | {roundtrip.get('p95_ms')} ms |",
        f"| roundtrip p99 | {roundtrip.get('p99_ms')} ms |",
        f"| roundtrip max | {roundtrip.get('max_ms')} ms |",
        f"| jitter p95-p50 | {latency.get('jitter_p95_minus_p50_ms')} ms |",
        "",
        "## Checks",
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
    execution = report.get("execution_certificate")
    if isinstance(execution, dict):
        from tether.action_execution_cert import format_action_execution_markdown

        lines.extend(["", format_action_execution_markdown(execution).rstrip()])
    return "\n".join(lines) + "\n"


def _latency_cell(value: Any) -> str:
    """Format a millisecond metric for a table cell; missing -> em dash."""

    if value is None:
        return "—"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _certificate_model_label(report: dict[str, Any]) -> str:
    """Best-effort model name for a certificate row.

    The certificate has no explicit model field, so derive it from the source
    export directory (e.g. ``~/.cache/tether/exports/smolvla-base`` ->
    ``smolvla-base``). Falls back to the target, then ``unknown``.
    """

    export_dir = ((report.get("source") or {}).get("export_dir") or "").rstrip("/")
    name = Path(export_dir).name if export_dir else ""
    return name or (report.get("target") or "unknown")


def format_realtime_certificates_markdown_table(
    reports: list[dict[str, Any]],
    *,
    title: str = "Realtime serving latency",
) -> str:
    """Render multiple realtime certificates as one comparison table.

    One row per certificate (model x target) with roundtrip p50/p95/p99/max and
    the PASS/FAIL decision. Companion to :func:`format_realtime_certificate_markdown`
    (which renders a single certificate vertically); this is the cross-run table
    used to publish the README latency section and the Jetson latency results doc.
    """

    lines = [
        f"## {title}",
        "",
        "| Model | Target | Control | p50 ms | p95 ms | p99 ms | max ms | Decision |",
        "|---|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for report in reports:
        control = report.get("control_budget") or {}
        roundtrip = (report.get("latency") or {}).get("roundtrip_ms") or {}
        hz = control.get("control_hz")
        control_cell = f"{float(hz):g} Hz" if hz else "—"
        decision = report.get("decision", "FAIL")
        lines.append(
            f"| `{_certificate_model_label(report)}` "
            f"| `{report.get('target') or 'unspecified'}` "
            f"| {control_cell} "
            f"| {_latency_cell(roundtrip.get('p50_ms'))} "
            f"| {_latency_cell(roundtrip.get('p95_ms'))} "
            f"| {_latency_cell(roundtrip.get('p99_ms'))} "
            f"| {_latency_cell(roundtrip.get('max_ms'))} "
            f"| **{decision}** |"
        )
    return "\n".join(lines) + "\n"


def write_realtime_certificate(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write realtime certificate JSON, Markdown, and MANIFEST."""

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads(json.dumps(report))
    report["artifacts"] = {
        "json": str(out / "realtime-serving-cert.json"),
        "markdown": str(out / "realtime-serving-cert.md"),
        "manifest": str(out / "MANIFEST.json"),
    }
    (out / "realtime-serving-cert.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (out / "realtime-serving-cert.md").write_text(
        format_realtime_certificate_markdown(report)
    )
    files = []
    for path in sorted(p for p in out.iterdir() if p.is_file() and p.name != "MANIFEST.json"):
        files.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "files": files,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


__all__ = [
    "RealtimeCertificateError",
    "build_realtime_certificate",
    "format_realtime_certificate_human",
    "format_realtime_certificate_markdown",
    "load_deploy_proof",
    "write_realtime_certificate",
]
