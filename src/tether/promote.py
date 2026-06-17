"""Promotion decision layer for deployment proof packets.

``tether prove`` collects evidence. ``tether promote`` consumes that evidence
and returns the operator decision companies actually need before rollout.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

PROMOTION_SCHEMA_VERSION = 1
Decision = Literal["PROMOTE", "BLOCK", "ROLLBACK"]

DEFAULT_PROMOTION_PROFILE: dict[str, Any] = {
    "schema_version": 1,
    "name": "default",
    "description": "Default promotion gate: verify packet integrity and fail on explicit proof or policy regressions.",
    "thresholds": {
        "require_manifest": True,
        "require_manifest_hashes": True,
        "require_deployment_passed": True,
        "require_no_proof_error": True,
        "max_failed_checks": 0,
        "require_auth": False,
        "require_metrics": False,
        "require_record_trace": False,
        "require_guard": False,
        "require_policy_diff": "auto",
        "allowed_policy_verdicts": ["pass"],
        "max_policy_action_failures": 0,
        "max_policy_latency_regressions": 0,
        "max_policy_guard_regressions": 0,
        "max_policy_shape_failures": 0,
        "max_policy_missing_candidate": 0,
        "max_policy_shadow_pending": 0,
        "max_policy_shadow_errors": 0,
        "max_roundtrip_p95_ms": None,
        "max_warm_roundtrip_p95_ms": None,
        "max_deadline_misses": None,
        "max_control_budget_misses": None,
    },
}

BUILTIN_PROMOTION_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "ci-default": {
        "name": "ci-default",
        "description": "CI gate for package/release checks. Requires packet integrity and passing proof checks, but keeps latency hardware-neutral.",
    },
    "lab-shadow": {
        "name": "lab-shadow",
        "description": "Lab or shadow rollout gate. Requires policy-diff evidence, allows warn verdicts, and keeps runtime latency permissive.",
        "thresholds": {
            "require_policy_diff": True,
            "allowed_policy_verdicts": ["pass", "warn"],
            "max_policy_action_failures": 0,
            "max_policy_latency_regressions": 3,
            "max_policy_guard_regressions": 0,
            "max_policy_shape_failures": 0,
            "max_policy_missing_candidate": 0,
        },
    },
    "warehouse-safe": {
        "name": "warehouse-safe",
        "description": "Strict warehouse production gate. Requires auth, trace evidence, guard stress, policy diff, and realtime budget checks.",
        "thresholds": {
            "require_auth": True,
            "require_metrics": True,
            "require_record_trace": True,
            "require_guard": True,
            "require_policy_diff": True,
            "allowed_policy_verdicts": ["pass"],
            "max_roundtrip_p95_ms": 80,
            "max_warm_roundtrip_p95_ms": 40,
            "max_deadline_misses": 0,
            "max_control_budget_misses": 0,
        },
    },
    "contact-strict": {
        "name": "contact-strict",
        "description": "Strictest preset for contact-rich manipulation. Same evidence as warehouse-safe with tighter latency budgets.",
        "thresholds": {
            "require_auth": True,
            "require_metrics": True,
            "require_record_trace": True,
            "require_guard": True,
            "require_policy_diff": True,
            "allowed_policy_verdicts": ["pass"],
            "max_roundtrip_p95_ms": 50,
            "max_warm_roundtrip_p95_ms": 25,
            "max_deadline_misses": 0,
            "max_control_budget_misses": 0,
        },
    },
}


class PromotionError(ValueError):
    """Raised when a proof packet or promotion profile cannot be loaded."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _copy_jsonable(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


def _normalize_profile_name(value: str | Path) -> str:
    return str(value).strip().lower().replace("_", "-")


def builtin_promotion_profiles() -> dict[str, dict[str, Any]]:
    """Return built-in promotion profiles, merged with default thresholds."""
    profiles: dict[str, dict[str, Any]] = {}
    for name, override in BUILTIN_PROMOTION_PROFILE_OVERRIDES.items():
        merged = _deep_merge(_copy_jsonable(DEFAULT_PROMOTION_PROFILE), override)
        merged["profile_source"] = "builtin"
        profiles[name] = merged
    return profiles


def list_promotion_profiles() -> list[dict[str, Any]]:
    """Return concise metadata for built-in promotion profiles."""
    rows = []
    for name, profile in builtin_promotion_profiles().items():
        thresholds = profile.get("thresholds") or {}
        rows.append(
            {
                "name": name,
                "description": profile.get("description", ""),
                "require_policy_diff": thresholds.get("require_policy_diff"),
                "require_auth": thresholds.get("require_auth"),
                "require_metrics": thresholds.get("require_metrics"),
                "require_guard": thresholds.get("require_guard"),
                "require_record_trace": thresholds.get("require_record_trace"),
                "max_roundtrip_p95_ms": thresholds.get("max_roundtrip_p95_ms"),
                "max_warm_roundtrip_p95_ms": thresholds.get("max_warm_roundtrip_p95_ms"),
            }
        )
    return rows


def get_builtin_promotion_profile(name: str | Path) -> dict[str, Any]:
    normalized = _normalize_profile_name(name)
    profiles = builtin_promotion_profiles()
    if normalized not in profiles:
        raise PromotionError(
            "unknown built-in promotion profile "
            f"{name!r}; available: {', '.join(sorted(profiles))}"
        )
    return _copy_jsonable(profiles[normalized])


def load_promotion_profile(profile_path: str | Path | None = None) -> dict[str, Any]:
    profile = _copy_jsonable(DEFAULT_PROMOTION_PROFILE)
    if profile_path is None or str(profile_path) == "":
        return profile

    path = Path(profile_path).expanduser()
    if not path.exists():
        normalized = _normalize_profile_name(profile_path)
        if normalized in BUILTIN_PROMOTION_PROFILE_OVERRIDES:
            return get_builtin_promotion_profile(normalized)
        raise PromotionError(
            f"promotion profile not found: {path}; built-ins: "
            f"{', '.join(sorted(BUILTIN_PROMOTION_PROFILE_OVERRIDES))}"
        )

    raw = (
        yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.suffix.lower() in {".yml", ".yaml"}
        else json.loads(path.read_text(encoding="utf-8"))
    )
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise PromotionError("promotion profile must be a mapping")

    loaded = _deep_merge(profile, raw)
    loaded["profile_path"] = str(path.resolve())
    loaded["profile_source"] = "file"
    if not isinstance(loaded.get("thresholds"), dict):
        raise PromotionError("promotion profile must contain a thresholds mapping")
    return loaded


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
            raise PromotionError(
                "packet file must be deployment-proof.json; pass the packet directory otherwise"
            )
        return path.parent
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PromotionError(f"missing required packet artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PromotionError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PromotionError(f"expected object JSON in {path}")
    return payload


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    category: str,
    expected: Any = None,
    actual: Any = None,
    remediation: str = "",
) -> None:
    checks.append(
        {
            "name": name,
            "category": category,
            "status": "pass" if passed else "fail",
            "expected": expected,
            "actual": actual,
            "remediation": remediation,
        }
    )


def _threshold(profile: dict[str, Any], key: str) -> Any:
    return (profile.get("thresholds") or {}).get(key)


def _summary_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "pass": sum(1 for check in checks if check.get("status") == "pass"),
        "fail": sum(1 for check in checks if check.get("status") == "fail"),
    }


def _verify_manifest(packet_dir: Path, *, require_hashes: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    manifest_path = packet_dir / "MANIFEST.json"
    try:
        manifest = _load_json(manifest_path)
    except PromotionError as exc:
        return {"present": False, "manifest": None, "checks": [
            {
                "name": "packet_manifest_present",
                "category": "reproducibility",
                "status": "fail",
                "expected": "MANIFEST.json",
                "actual": str(exc),
                "remediation": "Run `tether prove` to regenerate a hashed proof packet.",
            }
        ]}

    _add_check(
        checks,
        "packet_manifest_present",
        True,
        category="reproducibility",
        expected="MANIFEST.json",
        actual=str(manifest_path),
    )

    mismatches: list[dict[str, Any]] = []
    for item in manifest.get("files") or []:
        name = item.get("name")
        if not name:
            mismatches.append({"name": "", "reason": "missing name"})
            continue
        path = packet_dir / str(name)
        if not path.exists():
            mismatches.append({"name": name, "reason": "missing file"})
            continue
        expected_size = item.get("size_bytes")
        expected_sha = item.get("sha256")
        if expected_size is not None and int(expected_size) != path.stat().st_size:
            mismatches.append({"name": name, "reason": "size mismatch"})
            continue
        if expected_sha and _sha256_file(path) != expected_sha:
            mismatches.append({"name": name, "reason": "sha256 mismatch"})

    _add_check(
        checks,
        "packet_manifest_hashes",
        not mismatches or not require_hashes,
        category="reproducibility",
        expected="all MANIFEST.json hashes match" if require_hashes else "optional",
        actual=mismatches,
        remediation="Do not promote from a modified proof packet; rerun `tether prove`.",
    )
    return {"present": True, "manifest": manifest, "checks": checks}


def _load_policy_diff(packet_dir: Path, proof: dict[str, Any]) -> dict[str, Any] | None:
    path = packet_dir / "policy-diff.json"
    if path.exists():
        return _load_json(path)
    embedded = ((proof.get("policy_diff") or {}).get("report"))
    return embedded if isinstance(embedded, dict) else None


def _evaluate_latency_thresholds(
    *,
    checks: list[dict[str, Any]],
    profile: dict[str, Any],
    proof: dict[str, Any],
) -> None:
    latency = proof.get("latency") or {}
    threshold_map = (
        ("max_roundtrip_p95_ms", (latency.get("roundtrip_ms") or {}).get("p95_ms"), "roundtrip_p95"),
        (
            "max_warm_roundtrip_p95_ms",
            (latency.get("warm_roundtrip_ms") or {}).get("p95_ms"),
            "warm_roundtrip_p95",
        ),
        ("max_deadline_misses", latency.get("deadline_misses"), "deadline_misses"),
        (
            "max_control_budget_misses",
            (latency.get("control_budget") or {}).get("missed_samples"),
            "control_budget_misses",
        ),
    )
    for key, actual, name in threshold_map:
        limit = _threshold(profile, key)
        if limit is None:
            continue
        _add_check(
            checks,
            name,
            actual is not None and float(actual) <= float(limit),
            category="runtime",
            expected=f"<= {limit}",
            actual=actual,
            remediation="Regenerate the proof with a safer/faster deployment profile.",
        )


def _evaluate_proof_evidence(
    *,
    checks: list[dict[str, Any]],
    profile: dict[str, Any],
    proof: dict[str, Any],
) -> None:
    if _threshold(profile, "require_auth"):
        security = proof.get("security") or {}
        security_checks = security.get("checks") or []
        auth_ok = bool(security.get("enabled")) and all(
            check.get("status") == "pass" for check in security_checks
        )
        _add_check(
            checks,
            "proof_auth_required",
            auth_ok,
            category="security",
            expected="API-key proof checks passed",
            actual={
                "enabled": bool(security.get("enabled")),
                "failed_checks": [
                    check.get("name")
                    for check in security_checks
                    if check.get("status") == "fail"
                ],
            },
            remediation="Run `tether prove` with --api-key and keep production endpoints protected.",
        )

    if _threshold(profile, "require_metrics"):
        metrics = proof.get("metrics") or {}
        metrics_ok = int(metrics.get("status_code") or 0) == 200 and bool(
            metrics.get("metric_names") or []
        )
        _add_check(
            checks,
            "proof_metrics_required",
            metrics_ok,
            category="observability",
            expected="/metrics scrape with Tether metric families",
            actual={
                "status_code": metrics.get("status_code"),
                "metric_names": metrics.get("metric_names") or [],
            },
            remediation="Rerun `tether prove` against a serve runtime with /metrics enabled.",
        )

    if _threshold(profile, "require_record_trace"):
        trace = proof.get("trace") or {}
        trace_files = trace.get("files") or []
        _add_check(
            checks,
            "proof_trace_required",
            bool(trace_files),
            category="forensics",
            expected="recorded /act trace files in the proof packet",
            actual={"record_dir": trace.get("record_dir"), "files": trace_files},
            remediation="Run `tether prove` with --record-dir and replayable /act traffic.",
        )

    if _threshold(profile, "require_guard"):
        guard = proof.get("safety_stress") or {}
        guard_checks = guard.get("checks") or []
        guard_ok = bool(guard.get("enabled")) and all(
            check.get("status") == "pass" for check in guard_checks
        )
        _add_check(
            checks,
            "proof_guard_required",
            guard_ok,
            category="safety",
            expected="ActionGuard stress evidence passed",
            actual={
                "enabled": bool(guard.get("enabled")),
                "source": guard.get("source"),
                "failed_checks": [
                    check.get("name")
                    for check in guard_checks
                    if check.get("status") == "fail"
                ],
            },
            remediation="Run `tether prove` with --safety-config or --embodiment and fix guard stress failures.",
        )


def _evaluate_policy_diff(
    *,
    checks: list[dict[str, Any]],
    profile: dict[str, Any],
    proof: dict[str, Any],
    policy_diff: dict[str, Any] | None,
) -> None:
    require_policy_diff = _threshold(profile, "require_policy_diff")
    proof_policy_meta = proof.get("policy_diff") or {}
    policy_enabled = bool(proof_policy_meta.get("enabled"))
    required = require_policy_diff is True or (
        require_policy_diff == "auto" and policy_enabled
    )
    if required:
        _add_check(
            checks,
            "policy_diff_present",
            policy_diff is not None,
            category="promotion",
            expected="policy-diff.json or embedded policy_diff.report",
            actual=bool(policy_diff),
            remediation="Run `tether prove` with policy-diff flags before promotion.",
        )
    if policy_diff is None:
        return

    summary = policy_diff.get("summary") or {}
    allowed = set(_threshold(profile, "allowed_policy_verdicts") or ["pass"])
    verdict = str(summary.get("verdict") or "unknown")
    _add_check(
        checks,
        "policy_diff_verdict",
        verdict in allowed,
        category="promotion",
        expected=sorted(allowed),
        actual=verdict,
        remediation="Inspect policy-diff.json; do not promote a regressed policy.",
    )

    threshold_map = (
        ("max_policy_action_failures", "action_failures"),
        ("max_policy_latency_regressions", "latency_regressions"),
        ("max_policy_guard_regressions", "guard_regressions"),
        ("max_policy_shape_failures", "shape_failures"),
        ("max_policy_missing_candidate", "missing_candidate"),
        ("max_policy_shadow_pending", "shadow_pending"),
        ("max_policy_shadow_errors", "shadow_errors"),
    )
    for threshold_key, summary_key in threshold_map:
        limit = _threshold(profile, threshold_key)
        if limit is None:
            continue
        actual = int(summary.get(summary_key) or 0)
        _add_check(
            checks,
            f"policy_{summary_key}",
            actual <= int(limit),
            category="promotion",
            expected=f"<= {limit}",
            actual=actual,
            remediation="Fix the candidate policy or collect more shadow evidence.",
        )


def decide_promotion(
    packet: str | Path,
    *,
    profile_path: str | Path | None = None,
    candidate_active: bool = False,
) -> dict[str, Any]:
    """Evaluate a proof packet and return a promotion decision report."""
    packet_dir = _resolve_packet_dir(packet)
    profile = load_promotion_profile(profile_path)
    checks: list[dict[str, Any]] = []

    proof = _load_json(packet_dir / "deployment-proof.json")
    if proof.get("kind") != "tether.deployment_proof":
        raise PromotionError("deployment-proof.json is not a Tether deployment proof")

    if _threshold(profile, "require_manifest"):
        manifest_evidence = _verify_manifest(
            packet_dir,
            require_hashes=bool(_threshold(profile, "require_manifest_hashes")),
        )
        checks.extend(manifest_evidence["checks"])
    else:
        manifest_evidence = {"present": (packet_dir / "MANIFEST.json").exists(), "manifest": None}

    if _threshold(profile, "require_deployment_passed"):
        _add_check(
            checks,
            "deployment_proof_passed",
            bool(proof.get("passed")),
            category="deployment",
            expected=True,
            actual=proof.get("passed"),
            remediation="Fix failed deployment-proof checks before promotion.",
        )
    if _threshold(profile, "require_no_proof_error"):
        _add_check(
            checks,
            "deployment_proof_error_absent",
            not proof.get("error"),
            category="deployment",
            expected="no proof error",
            actual=proof.get("error"),
            remediation="Inspect deployment-proof.json and server.log.",
        )

    failed_proof_checks = [
        check for check in proof.get("checks") or [] if check.get("status") == "fail"
    ]
    max_failed = _threshold(profile, "max_failed_checks")
    if max_failed is not None:
        _add_check(
            checks,
            "deployment_failed_checks",
            len(failed_proof_checks) <= int(max_failed),
            category="deployment",
            expected=f"<= {max_failed}",
            actual=len(failed_proof_checks),
            remediation="Address failed checks in deployment-proof.json.",
        )

    _evaluate_proof_evidence(checks=checks, profile=profile, proof=proof)
    _evaluate_latency_thresholds(checks=checks, profile=profile, proof=proof)
    policy_diff = _load_policy_diff(packet_dir, proof)
    _evaluate_policy_diff(
        checks=checks,
        profile=profile,
        proof=proof,
        policy_diff=policy_diff,
    )

    counts = _summary_counts(checks)
    decision: Decision = "PROMOTE"
    if counts["fail"]:
        decision = "ROLLBACK" if candidate_active else "BLOCK"

    failed = [check for check in checks if check.get("status") == "fail"]
    return {
        "kind": "tether.promotion_decision",
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "decision": decision,
        "candidate_active": bool(candidate_active),
        "packet_dir": str(packet_dir),
        "profile": profile,
        "proof": {
            "path": str(packet_dir / "deployment-proof.json"),
            "passed": bool(proof.get("passed")),
            "error": proof.get("error"),
            "check_failures": len(failed_proof_checks),
            "output_dir": proof.get("output_dir"),
            "export_dir": proof.get("export_dir"),
        },
        "policy_diff": {
            "present": policy_diff is not None,
            "path": str(packet_dir / "policy-diff.json") if (packet_dir / "policy-diff.json").exists() else "",
            "summary": (policy_diff or {}).get("summary") if policy_diff else None,
        },
        "manifest": {
            "present": bool(manifest_evidence.get("present")),
        },
        "summary": {
            "pass": counts["pass"],
            "fail": counts["fail"],
            "failed_checks": [check["name"] for check in failed],
        },
        "checks": checks,
    }


def write_promotion_report(report: dict[str, Any], output: str | Path) -> None:
    Path(output).expanduser().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def format_promotion_human(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    proof = report.get("proof") or {}
    policy = report.get("policy_diff") or {}
    lines = [
        f"tether promote - {report.get('decision')}",
        f"packet:  {report.get('packet_dir')}",
        f"profile: {(report.get('profile') or {}).get('name', 'default')}",
        f"proof:   {'PASS' if proof.get('passed') else 'FAIL'} ({proof.get('check_failures', 0)} proof check failures)",
        f"policy:  {'present' if policy.get('present') else 'not-present'}",
        f"checks:  {summary.get('pass', 0)} pass, {summary.get('fail', 0)} fail",
    ]
    failed = [check for check in report.get("checks") or [] if check.get("status") == "fail"]
    if failed:
        lines.append("failed gates:")
        for check in failed[:10]:
            lines.append(
                f"  - {check.get('name')}: expected {check.get('expected')}, "
                f"actual {check.get('actual')}"
            )
    return "\n".join(lines)


__all__ = [
    "BUILTIN_PROMOTION_PROFILE_OVERRIDES",
    "DEFAULT_PROMOTION_PROFILE",
    "PROMOTION_SCHEMA_VERSION",
    "PromotionError",
    "builtin_promotion_profiles",
    "decide_promotion",
    "format_promotion_human",
    "get_builtin_promotion_profile",
    "list_promotion_profiles",
    "load_promotion_profile",
    "write_promotion_report",
]
