"""Shadow rollout gate composition.

This module turns a recorded shadow trace into the operator decision people
actually need: PROMOTE, HOLD, or ROLLBACK.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tether.policy_diff import diff_policy_traces, should_fail, write_report
from tether.promote import decide_promotion, write_promotion_report

SHADOW_ROLLOUT_SCHEMA_VERSION = 1
VALID_FAIL_ON = {"none", "actions", "latency", "guard", "shape", "any"}


class ShadowRolloutError(ValueError):
    """Raised when the shadow rollout gate cannot run."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_shadow_trace_ready(
    trace: str | Path,
    *,
    min_compared: int,
    timeout_s: float,
    poll_s: float,
    min_action_cos: float,
    max_action_delta: float,
    max_latency_regression_pct: float,
) -> dict[str, Any]:
    deadline = time.time() + max(0.0, float(timeout_s))
    last_report: dict[str, Any] | None = None
    while True:
        last_report = diff_policy_traces(
            baseline_trace=trace,
            shadow=True,
            min_action_cos=min_action_cos,
            max_action_delta=max_action_delta,
            max_latency_regression_pct=max_latency_regression_pct,
        )
        summary = last_report.get("summary") or {}
        compared = int(summary.get("compared") or 0)
        pending = int(summary.get("shadow_pending") or 0)
        if compared >= int(min_compared) and pending == 0:
            return last_report
        if time.time() >= deadline:
            return last_report
        time.sleep(max(0.05, float(poll_s)))


def _write_synthetic_proof_packet(
    packet_dir: Path,
    *,
    trace: str | Path,
    policy_diff: dict[str, Any],
    policy_diff_fail_on: str,
) -> None:
    checks: list[dict[str, Any]] = [
        {
            "name": "shadow_trace_present",
            "status": "pass",
            "category": "shadow",
            "expected": "readable shadow trace",
            "actual": str(trace),
        },
        {
            "name": "policy_diff_gate",
            "status": (
                "fail"
                if should_fail(policy_diff, policy_diff_fail_on)  # type: ignore[arg-type]
                else "pass"
            ),
            "category": "promotion",
            "expected": f"fail_on={policy_diff_fail_on} not tripped",
            "actual": (policy_diff.get("summary") or {}).get("verdict"),
        },
    ]
    proof = {
        "kind": "tether.deployment_proof",
        "schema_version": 1,
        "generated_at": _now_iso(),
        "passed": all(check["status"] == "pass" for check in checks),
        "mode": "shadow-rollout",
        "output_dir": str(packet_dir),
        "export_dir": "",
        "checks": checks,
        "policy_diff": {
            "enabled": True,
            "shadow": True,
            "baseline_trace": str(trace),
            "candidate_trace": None,
            "fail_on": policy_diff_fail_on,
            "report": policy_diff,
            "report_artifact": "policy-diff.json",
        },
    }
    packet_dir.mkdir(parents=True, exist_ok=True)
    (packet_dir / "deployment-proof.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = []
    for name in ("deployment-proof.json", "policy-diff.json"):
        path = packet_dir / name
        files.append(
            {
                "name": name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest = {
        "kind": "tether.manifest",
        "schema_version": 1,
        "generated_at": _now_iso(),
        "files": files,
    }
    (packet_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


_DERIVED_PACKET_OUTPUTS = {
    "MANIFEST.json",
    "promotion-decision.json",
    "release-assurance.json",
    "release-assurance.md",
}


def _refresh_packet_manifest(packet_dir: Path) -> None:
    files = []
    for path in sorted(
        p
        for p in packet_dir.iterdir()
        if p.is_file() and p.name not in _DERIVED_PACKET_OUTPUTS
    ):
        files.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest = {
        "kind": "tether.manifest",
        "schema_version": 1,
        "generated_at": _now_iso(),
        "files": files,
    }
    (packet_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_shadow_rollout_gate(
    *,
    trace: str | Path,
    packet_dir: str | Path,
    profile: str | Path = "lab-shadow",
    candidate_active: bool = False,
    min_compared: int = 1,
    wait_timeout_s: float = 0.0,
    poll_s: float = 0.25,
    fail_on: str = "any",
    min_action_cos: float = 0.995,
    max_action_delta: float = 0.10,
    max_latency_regression_pct: float = 0.10,
    use_existing_packet: bool = False,
) -> dict[str, Any]:
    """Run a shadow diff and promotion decision, returning one report."""
    if fail_on not in VALID_FAIL_ON:
        raise ShadowRolloutError(
            f"fail_on must be one of {sorted(VALID_FAIL_ON)}, got {fail_on!r}"
        )
    if min_compared < 1:
        raise ShadowRolloutError("min_compared must be >= 1")

    trace_path = Path(trace).expanduser()
    if not trace_path.exists():
        raise ShadowRolloutError(f"shadow trace not found: {trace_path}")
    packet_path = Path(packet_dir).expanduser()
    packet_path.mkdir(parents=True, exist_ok=True)

    policy_diff = _ensure_shadow_trace_ready(
        trace_path,
        min_compared=min_compared,
        timeout_s=wait_timeout_s,
        poll_s=poll_s,
        min_action_cos=min_action_cos,
        max_action_delta=max_action_delta,
        max_latency_regression_pct=max_latency_regression_pct,
    )
    policy_diff_path = packet_path / "policy-diff.json"
    write_report(policy_diff, policy_diff_path)

    if use_existing_packet:
        proof_path = packet_path / "deployment-proof.json"
        if not proof_path.exists():
            raise ShadowRolloutError(
                f"--use-existing-packet set but {proof_path} does not exist"
            )
        _refresh_packet_manifest(packet_path)
    else:
        _write_synthetic_proof_packet(
            packet_path,
            trace=trace_path,
            policy_diff=policy_diff,
            policy_diff_fail_on=fail_on,
        )

    report = decide_promotion(
        packet_path,
        profile_path=profile,
        candidate_active=candidate_active,
    )
    promotion_path = packet_path / "promotion-decision.json"
    write_promotion_report(report, promotion_path)
    decision = "HOLD" if report.get("decision") == "BLOCK" else report.get("decision")
    return {
        "kind": "tether.shadow_rollout_gate",
        "schema_version": SHADOW_ROLLOUT_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "decision": decision,
        "trace": str(trace_path),
        "packet_dir": str(packet_path),
        "profile": str(profile),
        "candidate_active": bool(candidate_active),
        "wait": {
            "timeout_s": float(wait_timeout_s),
            "poll_s": float(poll_s),
            "min_compared": int(min_compared),
        },
        "policy_diff": {
            "path": str(policy_diff_path),
            "summary": policy_diff.get("summary"),
        },
        "promotion": {
            "path": str(promotion_path),
            "decision": report.get("decision"),
            "summary": report.get("summary"),
            "failed_checks": (report.get("summary") or {}).get("failed_checks") or [],
        },
        "artifacts": {
            "deployment_proof": str(packet_path / "deployment-proof.json"),
            "policy_diff": str(policy_diff_path),
            "promotion_decision": str(promotion_path),
            "manifest": str(packet_path / "MANIFEST.json"),
        },
    }


def format_shadow_rollout_human(report: dict[str, Any]) -> str:
    policy = report.get("policy_diff") or {}
    summary = policy.get("summary") or {}
    promotion = report.get("promotion") or {}
    lines = [
        f"tether policy shadow-gate - {report.get('decision')}",
        f"trace:   {report.get('trace')}",
        f"packet:  {report.get('packet_dir')}",
        f"profile: {report.get('profile')}",
        (
            "shadow: "
            f"compared={summary.get('compared', 0)}/"
            f"{summary.get('baseline_requests', 0)}, "
            f"pending={summary.get('shadow_pending', 0)}, "
            f"errors={summary.get('shadow_errors', 0)}, "
            f"verdict={summary.get('verdict')}"
        ),
        f"checks:  {(promotion.get('summary') or {}).get('pass', 0)} pass, "
        f"{(promotion.get('summary') or {}).get('fail', 0)} fail",
    ]
    failed = promotion.get("failed_checks") or []
    if failed:
        lines.append("failed gates:")
        for name in failed[:10]:
            lines.append(f"  - {name}")
    return "\n".join(lines)


__all__ = [
    "SHADOW_ROLLOUT_SCHEMA_VERSION",
    "ShadowRolloutError",
    "format_shadow_rollout_human",
    "run_shadow_rollout_gate",
]
