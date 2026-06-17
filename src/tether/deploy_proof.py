"""Deployment proof packet generation for ``tether deploy-proof``.

This is the company-facing sibling of ``tether smoke``: smoke proves a fresh
install can serve a tiny local export; deploy-proof proves a specific export can
start, answer /act, satisfy profile thresholds, and produce a hashed evidence
packet.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from tether import __version__
from tether.smoke import (
    _log_tail,
    _offline_env,
    _require_serve_runtime_deps,
    _run_deploy_doctor,
    _stop_process,
    _wait_for_health,
    find_free_port,
)


class DeployProofError(RuntimeError):
    """Raised when deploy-proof setup cannot proceed."""


DEFAULT_PROFILE: dict[str, Any] = {
    "schema_version": 1,
    "name": "default",
    "thresholds": {
        "max_doctor_failures": 0,
        "max_act_errors": 0,
        "require_active_providers": True,
        "require_metrics": True,
        "require_auth": False,
        "require_record_trace": "auto",
        "require_policy_diff": False,
        "policy_diff_fail_on": "any",
        "require_guard": False,
        "max_deadline_misses": None,
        "max_first_roundtrip_ms": None,
        "max_roundtrip_p95_ms": None,
        "max_warm_roundtrip_p95_ms": None,
        "max_jitter_p95_minus_p50_ms": None,
        "control_hz": None,
        "max_missed_control_budget": None,
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _default_output_dir() -> Path:
    home = Path(os.environ.get("TETHER_HOME", Path.home() / ".cache" / "tether")).expanduser()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return home / "deploy-proof" / stamp


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_deploy_profile(profile_path: str | Path | None = None) -> dict[str, Any]:
    """Load a deployment proof profile from JSON/YAML and merge defaults."""

    profile = json.loads(json.dumps(DEFAULT_PROFILE))
    if profile_path is None or str(profile_path) == "":
        return profile

    path = Path(profile_path).expanduser()
    if not path.exists():
        raise DeployProofError(f"profile not found: {path}")

    raw = yaml.safe_load(path.read_text()) if path.suffix.lower() in {".yml", ".yaml"} else json.loads(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise DeployProofError("deploy-proof profile must be a mapping")

    loaded = _deep_merge(profile, raw)
    loaded["profile_path"] = str(path.resolve())
    if "thresholds" not in loaded or not isinstance(loaded["thresholds"], dict):
        raise DeployProofError("deploy-proof profile must contain a thresholds mapping")
    return loaded


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_file_manifest(root: str | Path) -> dict[str, Any]:
    """Hash every file under ``root`` using stable relative paths."""

    base = Path(root).expanduser().resolve()
    files: list[dict[str, Any]] = []
    if not base.exists():
        return {"root": str(base), "files": files}

    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        files.append(
            {
                "path": path.relative_to(base).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {"root": str(base), "files": files}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 1)
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return round(float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac), 1)


def summarize_deploy_latency(
    samples: list[dict[str, Any]],
    *,
    control_hz: float | None = None,
) -> dict[str, Any]:
    """Summarize /act latency with deployment-oriented fields."""

    def _values(key: str, source: list[dict[str, Any]]) -> list[float]:
        return [
            float(sample[key])
            for sample in source
            if isinstance(sample.get(key), (int, float))
        ]

    def _summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
        return {
            "p50_ms": _percentile(values, 50.0),
            "p95_ms": _percentile(values, 95.0),
            "p99_ms": _percentile(values, 99.0),
            "max_ms": round(max(values), 1),
        }

    roundtrip = _values("roundtrip_ms", samples)
    inference = _values("latency_ms", samples)
    warm = samples[1:]
    first = samples[0] if samples else {}
    jitter_ms = 0.0
    if roundtrip:
        jitter_ms = round(_percentile(roundtrip, 95.0) - _percentile(roundtrip, 50.0), 1)

    budget_ms = None
    missed_budget = 0
    if control_hz and control_hz > 0:
        budget_ms = 1000.0 / control_hz
        missed_budget = sum(1 for value in roundtrip if value > budget_ms)

    deadline_misses = sum(1 for sample in samples if sample.get("deadline_exceeded") is True)
    act_errors = sum(1 for sample in samples if sample.get("error"))
    guard_violations = sum(
        len(sample.get("guard_violations") or [])
        for sample in samples
        if isinstance(sample.get("guard_violations"), list)
    )

    return {
        "samples": len(samples),
        "ttfa_ms": (
            round(float(first["roundtrip_ms"]), 1)
            if isinstance(first.get("roundtrip_ms"), (int, float))
            else 0.0
        ),
        "first_sample": {
            "inference_ms": (
                round(float(first["latency_ms"]), 1)
                if isinstance(first.get("latency_ms"), (int, float))
                else 0.0
            ),
            "roundtrip_ms": (
                round(float(first["roundtrip_ms"]), 1)
                if isinstance(first.get("roundtrip_ms"), (int, float))
                else 0.0
            ),
        },
        "inference_ms": _summary(inference),
        "roundtrip_ms": _summary(roundtrip),
        "warm_inference_ms": _summary(_values("latency_ms", warm)),
        "warm_roundtrip_ms": _summary(_values("roundtrip_ms", warm)),
        "jitter": {"p95_minus_p50_ms": jitter_ms},
        "control_budget": {
            "control_hz": control_hz,
            "period_ms": round(budget_ms, 3) if budget_ms else None,
            "missed_samples": missed_budget,
        },
        "deadline_misses": deadline_misses,
        "act_errors": act_errors,
        "guard_violations": guard_violations,
    }


def _request(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(  # noqa: S310
        url,
        data=payload,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8", errors="replace")
            parsed: Any
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = raw
            return {
                "status_code": int(response.status),
                "body": parsed,
                "headers": dict(response.headers),
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = raw
        return {
            "status_code": int(exc.code),
            "body": parsed,
            "headers": dict(exc.headers),
        }


def _read_text_url(url: str, *, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return {
                "status_code": int(response.status),
                "body": response.read().decode("utf-8", errors="replace"),
                "headers": dict(response.headers),
            }
    except urllib.error.HTTPError as exc:
        return {
            "status_code": int(exc.code),
            "body": exc.read().decode("utf-8", errors="replace"),
            "headers": dict(exc.headers),
        }


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {"X-Tether-Key": api_key} if api_key else {}


def _redact_command(cmd: list[str]) -> list[str]:
    redacted = list(cmd)
    for idx, value in enumerate(redacted):
        if value == "--api-key" and idx + 1 < len(redacted):
            redacted[idx + 1] = "<redacted>"
    return redacted


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


def _run_security_checks(
    base_url: str,
    *,
    api_key: str | None,
    require_auth: bool,
    timeout_s: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    if require_auth and not api_key:
        _add_check(
            checks,
            "api_key_configured",
            False,
            category="security",
            expected="--api-key supplied when profile requires auth",
            actual="no api key",
            remediation="Run deploy-proof with --api-key and serve production with --api-key.",
        )
        return {"enabled": False, "checks": checks, "probes": probes}

    if not api_key:
        _add_check(
            checks,
            "api_key_configured",
            True,
            category="security",
            expected="auth optional in current profile",
            actual="no api key",
        )
        return {"enabled": False, "checks": checks, "probes": probes}

    body = {"instruction": "deploy-proof-auth-check", "state": [0.0] * 6}
    for name, method, path, request_body in (
        ("config_requires_auth", "GET", "/config", None),
        ("act_requires_auth", "POST", "/act", body),
        ("guard_status_requires_auth", "GET", "/guard/status", None),
        ("guard_reset_requires_auth", "POST", "/guard/reset", None),
    ):
        probe = _request(
            method,
            f"{base_url}{path}",
            body=request_body,
            timeout_s=timeout_s,
        )
        probes.append({"name": name, "path": path, "status_code": probe["status_code"]})
        _add_check(
            checks,
            name,
            probe["status_code"] == 401,
            category="security",
            expected=401,
            actual=probe["status_code"],
            remediation="Protected runtime endpoints must use the serve API-key dependency.",
        )

    authed_config = _request(
        "GET",
        f"{base_url}/config",
        headers=_auth_headers(api_key),
        timeout_s=timeout_s,
    )
    probes.append(
        {
            "name": "config_accepts_valid_key",
            "path": "/config",
            "status_code": authed_config["status_code"],
        }
    )
    _add_check(
        checks,
        "config_accepts_valid_key",
        authed_config["status_code"] == 200,
        category="security",
        expected=200,
        actual=authed_config["status_code"],
        remediation="Verify the deploy-proof --api-key matches the running server key.",
    )
    return {"enabled": True, "checks": checks, "probes": probes}


def _run_metrics_check(base_url: str, *, require_metrics: bool, timeout_s: float) -> dict[str, Any]:
    response = _read_text_url(f"{base_url}/metrics", timeout_s=timeout_s)
    body = str(response["body"])
    metric_names = [
        name
        for name in ("tether_act_latency_seconds", "tether_in_flight_requests", "tether_server_up")
        if name in body
    ]
    ok = response["status_code"] == 200 and bool(metric_names)
    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "metrics_scrape",
        ok or not require_metrics,
        category="observability",
        expected="Prometheus text with Tether metric families" if require_metrics else "optional",
        actual={"status_code": response["status_code"], "metric_names": metric_names},
        remediation="Install fastcrest-tether[serve] with prometheus-client and keep /metrics enabled.",
    )
    return {
        "status_code": response["status_code"],
        "metric_names": metric_names,
        "body_prefix": body[:500],
        "checks": checks,
    }


def _trace_files(record_dir: str | Path | None) -> list[dict[str, Any]]:
    if not record_dir:
        return []
    root = Path(record_dir).expanduser()
    if not root.exists():
        return []
    files = []
    for path in sorted(root.glob("*.jsonl*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return files


def _normalize_policy_diff_fail_on(value: str | None) -> str:
    fail_on = (value or "any").strip().lower()
    valid = {"none", "actions", "latency", "guard", "shape", "any"}
    if fail_on not in valid:
        raise DeployProofError(
            f"policy_diff_fail_on must be one of {sorted(valid)}, got {value!r}"
        )
    return fail_on


def _run_policy_diff_evidence(
    *,
    baseline_trace: str | Path | None,
    candidate_trace: str | Path | None,
    shadow: bool,
    fail_on: str,
    min_action_cos: float,
    max_action_delta: float,
    max_latency_regression_pct: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if not baseline_trace:
        return {
            "enabled": False,
            "baseline_trace": "",
            "candidate_trace": "",
            "shadow": bool(shadow),
            "fail_on": fail_on,
            "report_artifact": "",
            "report": None,
            "error": None,
            "checks": checks,
        }

    evidence = {
        "enabled": True,
        "baseline_trace": str(Path(baseline_trace).expanduser()),
        "candidate_trace": str(Path(candidate_trace).expanduser()) if candidate_trace else "",
        "shadow": bool(shadow),
        "fail_on": fail_on,
        "report_artifact": "",
        "report": None,
        "error": None,
        "checks": checks,
    }

    if shadow and candidate_trace:
        evidence["error"] = "candidate_trace is not allowed when shadow=True"
        _add_check(
            checks,
            "policy_diff_inputs_valid",
            False,
            category="promotion",
            expected="shadow trace only",
            actual=evidence["error"],
            remediation="Remove --policy-diff-candidate or disable --policy-diff-shadow.",
        )
        return evidence
    if not shadow and not candidate_trace:
        evidence["error"] = "candidate_trace is required unless shadow=True"
        _add_check(
            checks,
            "policy_diff_inputs_valid",
            False,
            category="promotion",
            expected="baseline and candidate traces",
            actual=evidence["error"],
            remediation="Pass --policy-diff-candidate or use --policy-diff-shadow.",
        )
        return evidence

    try:
        from tether.policy_diff import diff_policy_traces, should_fail

        report = diff_policy_traces(
            baseline_trace=baseline_trace,
            candidate_trace=candidate_trace,
            shadow=shadow,
            min_action_cos=min_action_cos,
            max_action_delta=max_action_delta,
            max_latency_regression_pct=max_latency_regression_pct,
        )
        evidence["report"] = report
        evidence["report_artifact"] = "policy-diff.json"
        _add_check(
            checks,
            "policy_diff_runs",
            True,
            category="promotion",
            expected="policy diff report generated",
            actual={
                "mode": report.get("mode"),
                "verdict": (report.get("summary") or {}).get("verdict"),
            },
        )
        gate_failed = should_fail(report, fail_on)  # type: ignore[arg-type]
        _add_check(
            checks,
            "policy_diff_gate",
            not gate_failed,
            category="promotion",
            expected=f"no {fail_on} policy diff failures",
            actual=report.get("summary"),
            remediation="Inspect policy-diff.json before promoting the candidate policy.",
        )
    except Exception as exc:  # noqa: BLE001
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        _add_check(
            checks,
            "policy_diff_runs",
            False,
            category="promotion",
            expected="policy diff report generated",
            actual=evidence["error"],
            remediation="Verify trace paths and run `tether policy diff` directly for details.",
        )
    return evidence


def _run_guard_stress(
    *,
    safety_config: str | None,
    embodiment: str,
    custom_embodiment_config: str | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        from tether.safety.guard import ActionGuard, SafetyLimits
    except Exception as exc:  # noqa: BLE001
        _add_check(
            checks,
            "guard_stress_importable",
            False,
            category="safety",
            expected="ActionGuard importable",
            actual=f"{type(exc).__name__}: {exc}",
        )
        return {"enabled": False, "checks": checks}

    guard = None
    source = ""
    try:
        if safety_config:
            guard = ActionGuard(SafetyLimits.from_json(safety_config), max_consecutive_clamps=2)
            source = f"safety_config:{safety_config}"
        elif custom_embodiment_config:
            from tether.embodiments import EmbodimentConfig

            cfg = EmbodimentConfig.load_custom(custom_embodiment_config)
            guard = ActionGuard.from_embodiment_config(cfg, max_consecutive_clamps=2)
            source = f"custom_embodiment_config:{custom_embodiment_config}"
        elif embodiment and embodiment != "custom":
            from tether.embodiments import EmbodimentConfig

            cfg = EmbodimentConfig.load_preset(embodiment)
            guard = ActionGuard.from_embodiment_config(cfg, max_consecutive_clamps=2)
            source = f"embodiment:{embodiment}"
    except Exception as exc:  # noqa: BLE001
        _add_check(
            checks,
            "guard_stress_config_load",
            False,
            category="safety",
            expected="loadable safety or embodiment config",
            actual=f"{type(exc).__name__}: {exc}",
        )
        return {"enabled": False, "checks": checks, "source": source}

    if guard is None:
        _add_check(
            checks,
            "guard_stress_available",
            True,
            category="safety",
            expected="guard stress optional",
            actual="no safety config or embodiment supplied",
        )
        return {"enabled": False, "checks": checks}

    dim = max(1, len(guard.limits.position_max) or len(guard.limits.velocity_max) or 1)
    high = []
    for idx in range(dim):
        if idx < len(guard.limits.position_max):
            high.append(float(guard.limits.position_max[idx]) + 10.0)
        else:
            high.append(10.0)

    import numpy as np

    bad = np.asarray([high], dtype=np.float32)
    safe, results = guard.check(bad)
    clamp_ok = any(r.clamped for r in results) and not np.array_equal(bad, safe)
    _add_check(
        checks,
        "guard_clamps_out_of_range",
        clamp_ok,
        category="safety",
        expected="out-of-range action is clamped",
        actual={"clamped": [r.clamped for r in results], "violations": [v for r in results for v in r.violations][:5]},
    )

    finite = np.zeros((1, dim), dtype=np.float32)
    finite[0, 0] = np.nan
    safe_nan, results_nan = guard.check(finite)
    non_finite_ok = bool(results_nan and not results_nan[0].safe and np.allclose(safe_nan, 0.0))
    _add_check(
        checks,
        "guard_rejects_non_finite",
        non_finite_ok,
        category="safety",
        expected="NaN/Inf chunk zeroed",
        actual={"violations": [v for r in results_nan for v in r.violations][:5]},
    )

    guard.check(bad)
    trip_ok = guard.tripped
    _add_check(
        checks,
        "guard_consecutive_clamp_trip",
        trip_ok,
        category="safety",
        expected="guard trips after repeated clamps",
        actual={"tripped": guard.tripped, "trip_reason": guard.trip_reason},
    )
    return {"enabled": True, "source": source, "checks": checks}


def _threshold(profile: dict[str, Any], key: str) -> Any:
    return (profile.get("thresholds") or {}).get(key)


def _evaluate_thresholds(
    *,
    checks: list[dict[str, Any]],
    profile: dict[str, Any],
    doctor: dict[str, Any] | None,
    latency: dict[str, Any] | None,
    act_samples: list[dict[str, Any]],
    record_trace_files: list[dict[str, Any]],
    record_dir: str | None,
    guard_stress: dict[str, Any],
    policy_diff: dict[str, Any] | None,
) -> None:
    doctor_fails = int(((doctor or {}).get("summary") or {}).get("fail", 0))
    max_doctor = _threshold(profile, "max_doctor_failures")
    if max_doctor is not None:
        _add_check(
            checks,
            "doctor_failures_within_profile",
            doctor_fails <= int(max_doctor),
            category="diagnostics",
            expected=f"<= {max_doctor}",
            actual=doctor_fails,
            remediation="Fix failing deploy diagnostics before production rollout.",
        )

    errors = sum(1 for sample in act_samples if sample.get("error"))
    max_errors = _threshold(profile, "max_act_errors")
    if max_errors is not None:
        _add_check(
            checks,
            "act_errors_within_profile",
            errors <= int(max_errors),
            category="runtime",
            expected=f"<= {max_errors}",
            actual=errors,
            remediation="Investigate /act error samples and server log tail.",
        )

    if _threshold(profile, "require_active_providers"):
        providers_ok = all(bool(sample.get("active_providers")) for sample in act_samples)
        _add_check(
            checks,
            "active_providers_present",
            providers_ok,
            category="runtime",
            expected="every /act sample reports active providers",
            actual=[sample.get("active_providers", []) for sample in act_samples[:3]],
            remediation="Check ONNX Runtime provider setup and strict provider flags.",
        )

    if latency:
        threshold_map = (
            ("max_first_roundtrip_ms", latency["first_sample"]["roundtrip_ms"], "latency_first_roundtrip"),
            ("max_roundtrip_p95_ms", latency["roundtrip_ms"]["p95_ms"], "latency_roundtrip_p95"),
            ("max_warm_roundtrip_p95_ms", latency["warm_roundtrip_ms"]["p95_ms"], "latency_warm_roundtrip_p95"),
            ("max_jitter_p95_minus_p50_ms", latency["jitter"]["p95_minus_p50_ms"], "latency_jitter"),
            ("max_deadline_misses", latency["deadline_misses"], "deadline_misses"),
            (
                "max_missed_control_budget",
                latency["control_budget"]["missed_samples"],
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
                float(actual) <= float(limit),
                category="realtime",
                expected=f"<= {limit}",
                actual=actual,
                remediation="Lower model latency, enable RTC/AAC, reduce load, or relax the deployment profile.",
            )

    require_record = _threshold(profile, "require_record_trace")
    if require_record == "auto":
        require_record = bool(record_dir)
    if require_record:
        _add_check(
            checks,
            "record_trace_written",
            bool(record_trace_files),
            category="forensics",
            expected="at least one JSONL trace file",
            actual=record_trace_files,
            remediation="Run with --record-dir and verify the recorder can write to that path.",
        )

    if _threshold(profile, "require_policy_diff"):
        policy_diff_ok = bool((policy_diff or {}).get("enabled")) and bool(
            (policy_diff or {}).get("report")
        )
        _add_check(
            checks,
            "policy_diff_required",
            policy_diff_ok,
            category="promotion",
            expected="policy diff report present",
            actual={
                "enabled": bool((policy_diff or {}).get("enabled")),
                "error": (policy_diff or {}).get("error"),
            },
            remediation="Pass --policy-diff-baseline plus --policy-diff-candidate or --policy-diff-shadow.",
        )

    if _threshold(profile, "require_guard"):
        guard_ok = bool(guard_stress.get("enabled")) and all(
            check["status"] == "pass" for check in guard_stress.get("checks", [])
        )
        _add_check(
            checks,
            "guard_stress_required",
            guard_ok,
            category="safety",
            expected="safety config or embodiment guard stress passes",
            actual={"enabled": guard_stress.get("enabled"), "source": guard_stress.get("source")},
            remediation="Pass --safety-config or --embodiment and fix ActionGuard stress failures.",
        )


def run_deploy_proof(
    *,
    export_dir: str | Path,
    output_dir: str | Path | None = None,
    profile_path: str | Path | None = None,
    offline: bool = True,
    port: int = 0,
    timeout_s: float = 30.0,
    act_samples: int = 20,
    device: str = "cpu",
    providers: str = "",
    no_strict_providers: bool = False,
    embodiment: str = "custom",
    custom_embodiment_config: str | None = None,
    safety_config: str | None = None,
    api_key: str | None = None,
    deadline_ms: float = 0.0,
    control_hz: float | None = None,
    max_concurrent: int = 0,
    record_dir: str | Path | None = None,
    record_images: str = "hash_only",
    prewarm: bool = True,
    instruction: str = "reach",
    state_dim: int = 6,
    policy_diff_baseline_trace: str | Path | None = None,
    policy_diff_candidate_trace: str | Path | None = None,
    policy_diff_shadow: bool = False,
    policy_diff_fail_on: str | None = None,
    policy_diff_min_action_cos: float = 0.995,
    policy_diff_max_action_delta: float = 0.10,
    policy_diff_max_latency_regression_pct: float = 0.10,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Run a real-export deployment proof and optionally write a packet."""

    if act_samples < 1:
        raise DeployProofError(f"act_samples must be >= 1, got {act_samples}")
    if state_dim < 0:
        raise DeployProofError(f"state_dim must be >= 0, got {state_dim}")
    if control_hz is not None and control_hz <= 0:
        raise DeployProofError(f"control_hz must be > 0, got {control_hz}")

    export_path = Path(export_dir).expanduser().resolve()
    if not export_path.exists():
        raise DeployProofError(f"export directory not found: {export_path}")
    if not list(export_path.glob("*.onnx")):
        raise DeployProofError(f"export directory has no .onnx files: {export_path}")

    profile = load_deploy_profile(profile_path)
    if control_hz is not None:
        profile.setdefault("thresholds", {})["control_hz"] = float(control_hz)
    out_path = Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir()
    if port == 0:
        port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    started = time.monotonic()
    server_process: subprocess.Popen[str] | None = None
    log_path: Path | None = None
    checks: list[dict[str, Any]] = []

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "tether.deployment_proof",
        "timestamp": _now_iso(),
        "passed": False,
        "tether_version": __version__,
        "python": sys.version.split()[0],
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "offline": bool(offline),
        "export_dir": str(export_path),
        "output_dir": str(out_path),
        "profile": profile,
        "server": {
            "url": base_url,
            "port": port,
            "started": False,
            "exit_code": None,
            "log_tail": [],
        },
        "doctor": None,
        "health": None,
        "config": None,
        "security": None,
        "metrics": None,
        "safety_stress": None,
        "trace": None,
        "policy_diff": None,
        "act": None,
        "act_samples": [],
        "latency": None,
        "export_manifest": None,
        "checks": checks,
        "duration_ms": 0.0,
        "error": None,
    }

    try:
        _require_serve_runtime_deps()
        receipt["export_manifest"] = build_file_manifest(export_path)
        _add_check(
            checks,
            "export_manifest_hashed",
            bool(receipt["export_manifest"]["files"]),
            category="reproducibility",
            expected="at least one export file hashed",
            actual=len(receipt["export_manifest"]["files"]),
        )

        doctor = _run_deploy_doctor(export_path, offline)
        receipt["doctor"] = doctor

        log_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w",
            encoding="utf-8",
            prefix="tether-deploy-proof-serve-",
            suffix=".log",
            delete=False,
        )
        log_path = Path(log_file.name)
        env = os.environ.copy()
        env.update(_offline_env(offline))
        cmd = [
            python_executable or sys.executable,
            "-m",
            "tether.cli",
            "serve",
            str(export_path),
            "--device",
            device,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        if providers:
            cmd.extend(["--providers", providers])
        if no_strict_providers:
            cmd.append("--no-strict-providers")
        if not prewarm:
            cmd.append("--no-prewarm")
        if api_key:
            cmd.extend(["--api-key", api_key])
        if embodiment and embodiment != "custom":
            cmd.extend(["--embodiment", embodiment])
        if custom_embodiment_config:
            cmd.extend(["--custom-embodiment-config", custom_embodiment_config])
        if safety_config:
            cmd.extend(["--safety-config", safety_config])
        if deadline_ms > 0:
            cmd.extend(["--deadline-ms", str(deadline_ms)])
        if max_concurrent > 0:
            cmd.extend(["--max-concurrent", str(max_concurrent)])
        if record_dir:
            cmd.extend(["--record", str(Path(record_dir).expanduser())])
            cmd.extend(["--record-images", record_images])

        server_process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        log_file.close()
        receipt["server"]["command"] = _redact_command(cmd)
        receipt["server"]["started"] = True

        health = _wait_for_health(server_process, base_url, timeout_s=timeout_s)
        receipt["health"] = health
        _add_check(
            checks,
            "server_health_ready",
            health.get("status") == "ok",
            category="runtime",
            expected="health status ok",
            actual=health,
        )

        security = _run_security_checks(
            base_url,
            api_key=api_key,
            require_auth=bool(_threshold(profile, "require_auth")),
            timeout_s=timeout_s,
        )
        receipt["security"] = security
        checks.extend(security["checks"])

        config_response = _request(
            "GET",
            f"{base_url}/config",
            headers=_auth_headers(api_key),
            timeout_s=timeout_s,
        )
        receipt["config"] = {
            "status_code": config_response["status_code"],
            "body": config_response["body"],
        }
        _add_check(
            checks,
            "config_readable",
            config_response["status_code"] == 200,
            category="runtime",
            expected=200,
            actual=config_response["status_code"],
        )

        request_body = {
            "instruction": instruction,
            "state": [0.0] * state_dim,
            "episode_id": "deploy-proof",
        }
        for idx in range(act_samples):
            act_started = time.monotonic()
            act_response = _request(
                "POST",
                f"{base_url}/act",
                body=request_body,
                headers=_auth_headers(api_key),
                timeout_s=timeout_s,
            )
            roundtrip_ms = (time.monotonic() - act_started) * 1000.0
            body = act_response["body"] if isinstance(act_response["body"], dict) else {}
            sample = {
                "sample": idx + 1,
                "status_code": act_response["status_code"],
                "num_actions": body.get("num_actions"),
                "action_dim": body.get("action_dim"),
                "latency_ms": body.get("latency_ms"),
                "roundtrip_ms": round(roundtrip_ms, 1),
                "inference_mode": body.get("inference_mode"),
                "provider_mode": body.get("provider_mode"),
                "active_providers": body.get("active_providers", []),
                "denoising_steps": body.get("denoising_steps"),
                "deadline_exceeded": body.get("deadline_exceeded"),
                "guard_margin": body.get("guard_margin"),
                "guard_clamped": body.get("guard_clamped"),
                "guard_violations": body.get("guard_violations", []),
                "error": body.get("error") or (
                    f"http_{act_response['status_code']}"
                    if act_response["status_code"] >= 400
                    else None
                ),
            }
            if isinstance(body.get("actions"), list):
                sample["actions"] = body["actions"]
            for key in (
                "action_execution",
                "execution",
                "adaptive_action_chunking",
                "rtc",
            ):
                if isinstance(body.get(key), dict):
                    sample[key] = body[key]
            receipt["act_samples"].append(sample)
            receipt["act"] = sample

        control_hz = _threshold(profile, "control_hz")
        receipt["latency"] = summarize_deploy_latency(
            receipt["act_samples"],
            control_hz=float(control_hz) if control_hz not in (None, "") else None,
        )

        metrics = _run_metrics_check(
            base_url,
            require_metrics=bool(_threshold(profile, "require_metrics")),
            timeout_s=timeout_s,
        )
        receipt["metrics"] = metrics
        checks.extend(metrics["checks"])

        guard_stress = _run_guard_stress(
            safety_config=safety_config,
            embodiment=embodiment,
            custom_embodiment_config=custom_embodiment_config,
        )
        receipt["safety_stress"] = guard_stress
        checks.extend(guard_stress["checks"])

        trace_files = _trace_files(record_dir)
        receipt["trace"] = {
            "record_dir": str(Path(record_dir).expanduser()) if record_dir else "",
            "files": trace_files,
        }

        resolved_policy_diff_fail_on = _normalize_policy_diff_fail_on(
            policy_diff_fail_on
            or str(_threshold(profile, "policy_diff_fail_on") or "any")
        )
        policy_diff = _run_policy_diff_evidence(
            baseline_trace=policy_diff_baseline_trace,
            candidate_trace=policy_diff_candidate_trace,
            shadow=policy_diff_shadow,
            fail_on=resolved_policy_diff_fail_on,
            min_action_cos=policy_diff_min_action_cos,
            max_action_delta=policy_diff_max_action_delta,
            max_latency_regression_pct=policy_diff_max_latency_regression_pct,
        )
        receipt["policy_diff"] = policy_diff
        checks.extend(policy_diff.get("checks") or [])

        _evaluate_thresholds(
            checks=checks,
            profile=profile,
            doctor=receipt["doctor"],
            latency=receipt["latency"],
            act_samples=receipt["act_samples"],
            record_trace_files=trace_files,
            record_dir=str(record_dir) if record_dir else None,
            guard_stress=guard_stress,
            policy_diff=policy_diff,
        )
    except Exception as exc:  # noqa: BLE001
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        _add_check(
            checks,
            "deploy_proof",
            False,
            category="runtime",
            expected="proof run completes",
            actual=receipt["error"],
        )
    finally:
        receipt["server"]["exit_code"] = _stop_process(server_process)
        if server_process is not None:
            _add_check(
                checks,
                "server_exit_clean",
                receipt["server"]["exit_code"] == 0,
                category="runtime",
                expected=0,
                actual=receipt["server"]["exit_code"],
                remediation="Inspect server.log in the proof packet.",
            )
        receipt["server"]["log_tail"] = _log_tail(log_path)
        receipt["duration_ms"] = round((time.monotonic() - started) * 1000.0, 1)
        receipt["passed"] = all(check.get("status") != "fail" for check in checks)
        write_deploy_proof_packet(receipt, out_path)

    return receipt


def write_deploy_proof_packet(receipt: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Write JSON, Markdown, logs, profile, export manifest, and MANIFEST."""

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "deployment-proof.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (out / "deployment-proof.md").write_text(format_deploy_proof_markdown(receipt))
    if receipt.get("server", {}).get("log_tail"):
        (out / "server.log").write_text("\n".join(receipt["server"]["log_tail"]) + "\n")
    (out / "profile.json").write_text(json.dumps(receipt.get("profile") or {}, indent=2, sort_keys=True) + "\n")
    (out / "export-manifest.json").write_text(
        json.dumps(receipt.get("export_manifest") or {}, indent=2, sort_keys=True) + "\n"
    )
    policy_diff = receipt.get("policy_diff") or {}
    if policy_diff.get("report"):
        (out / "policy-diff.json").write_text(
            json.dumps(policy_diff["report"], indent=2, sort_keys=True) + "\n"
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


def _summary_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "pass": sum(1 for check in checks if check.get("status") == "pass"),
        "fail": sum(1 for check in checks if check.get("status") == "fail"),
    }


def format_deploy_proof_human(receipt: dict[str, Any]) -> str:
    status = "PASS" if receipt.get("passed") else "FAIL"
    checks = receipt.get("checks") or []
    counts = _summary_counts(checks)
    latency = receipt.get("latency") or {}
    roundtrip = latency.get("roundtrip_ms") or {}
    warm = latency.get("warm_roundtrip_ms") or {}
    doctor = receipt.get("doctor") or {}
    doctor_summary = doctor.get("summary") or {}
    lines = [
        f"tether deploy-proof - {status}",
        f"export:  {receipt.get('export_dir')}",
        f"packet:  {receipt.get('output_dir')}",
        f"server:  {receipt.get('server', {}).get('url')}",
        f"checks:  {counts['pass']} pass, {counts['fail']} fail",
    ]
    if doctor_summary:
        lines.append(
            "doctor:  "
            f"{doctor_summary.get('pass', 0)} pass, "
            f"{doctor_summary.get('fail', 0)} fail, "
            f"{doctor_summary.get('warn', 0)} warn, "
            f"{doctor_summary.get('skip', 0)} skip"
        )
    if latency:
        lines.append(
            "latency: "
            f"n={latency.get('samples', 0)} "
            f"ttfa={latency.get('ttfa_ms', 0.0)}ms, "
            f"roundtrip p50/p95/p99={roundtrip.get('p50_ms', 0.0)}/"
            f"{roundtrip.get('p95_ms', 0.0)}/{roundtrip.get('p99_ms', 0.0)}ms, "
            f"warm p95={warm.get('p95_ms', 0.0)}ms, "
            f"jitter={latency.get('jitter', {}).get('p95_minus_p50_ms', 0.0)}ms"
        )
    security = receipt.get("security") or {}
    if security:
        security_counts = _summary_counts(security.get("checks") or [])
        lines.append(
            f"security: {'enabled' if security.get('enabled') else 'not-required'} "
            f"({security_counts['pass']} pass, {security_counts['fail']} fail)"
        )
    metrics = receipt.get("metrics") or {}
    if metrics:
        lines.append(
            "metrics: "
            f"{metrics.get('status_code')} "
            f"{', '.join(metrics.get('metric_names') or []) or 'none'}"
        )
    trace = receipt.get("trace") or {}
    if trace.get("record_dir"):
        lines.append(f"traces:  {len(trace.get('files') or [])} file(s) in {trace.get('record_dir')}")
    policy_diff = receipt.get("policy_diff") or {}
    if policy_diff.get("enabled"):
        report = policy_diff.get("report") or {}
        summary = report.get("summary") or {}
        if summary:
            lines.append(
                "policy diff: "
                f"{str(summary.get('verdict', 'unknown')).upper()} "
                f"(compared={summary.get('compared', 0)}, fail_on={policy_diff.get('fail_on')})"
            )
        else:
            lines.append(f"policy diff: ERROR ({policy_diff.get('error')})")
    if receipt.get("error"):
        lines.append(f"error:   {receipt['error']}")
    return "\n".join(lines)


def format_deploy_proof_markdown(receipt: dict[str, Any]) -> str:
    status = "PASS" if receipt.get("passed") else "FAIL"
    checks = receipt.get("checks") or []
    counts = _summary_counts(checks)
    latency = receipt.get("latency") or {}
    roundtrip = latency.get("roundtrip_ms") or {}
    warm = latency.get("warm_roundtrip_ms") or {}
    doctor = receipt.get("doctor") or {}
    doctor_summary = doctor.get("summary") or {}
    security = receipt.get("security") or {}
    metrics = receipt.get("metrics") or {}
    trace = receipt.get("trace") or {}
    policy_diff = receipt.get("policy_diff") or {}
    policy_report = policy_diff.get("report") or {}
    policy_summary = policy_report.get("summary") or {}

    lines = [
        "# Tether Deployment Proof",
        "",
        f"- Status: {status}",
        f"- Tether version: {receipt.get('tether_version')}",
        f"- Python: {receipt.get('python')}",
        f"- Export dir: `{receipt.get('export_dir')}`",
        f"- Output dir: `{receipt.get('output_dir')}`",
        f"- Profile: `{(receipt.get('profile') or {}).get('name', 'default')}`",
        f"- Duration: {receipt.get('duration_ms')} ms",
        "",
        "## Checks",
        "",
        f"- Pass: {counts['pass']}",
        f"- Fail: {counts['fail']}",
        "",
        "## Doctor",
        "",
        f"- Pass: {doctor_summary.get('pass', 0)}",
        f"- Fail: {doctor_summary.get('fail', 0)}",
        f"- Warn: {doctor_summary.get('warn', 0)}",
        f"- Skip: {doctor_summary.get('skip', 0)}",
        "",
        "## Real-Time Runtime",
        "",
        f"- Samples: {latency.get('samples', 0)}",
        f"- TTFA: {latency.get('ttfa_ms', 0.0)} ms",
        f"- Roundtrip p50: {roundtrip.get('p50_ms', 0.0)} ms",
        f"- Roundtrip p95: {roundtrip.get('p95_ms', 0.0)} ms",
        f"- Roundtrip p99: {roundtrip.get('p99_ms', 0.0)} ms",
        f"- Warm roundtrip p95: {warm.get('p95_ms', 0.0)} ms",
        f"- Jitter p95-p50: {latency.get('jitter', {}).get('p95_minus_p50_ms', 0.0)} ms",
        f"- Deadline misses: {latency.get('deadline_misses', 0)}",
        f"- Control budget misses: {latency.get('control_budget', {}).get('missed_samples', 0)}",
        "",
        "## Security",
        "",
        f"- API-key checks enabled: {bool(security.get('enabled'))}",
        f"- Probe count: {len(security.get('probes') or [])}",
        "",
        "## Observability",
        "",
        f"- Metrics status: {metrics.get('status_code', 'n/a')}",
        f"- Metric families: `{metrics.get('metric_names', [])}`",
        "",
        "## Forensics",
        "",
        f"- Record dir: `{trace.get('record_dir', '')}`",
        f"- Trace files: {len(trace.get('files') or [])}",
        "",
        "## Policy Diff",
        "",
        f"- Enabled: {bool(policy_diff.get('enabled'))}",
        f"- Fail on: `{policy_diff.get('fail_on', '')}`",
        f"- Verdict: `{policy_summary.get('verdict', 'n/a')}`",
        f"- Compared: {policy_summary.get('compared', 0)}",
        f"- Artifact: `{policy_diff.get('report_artifact', '')}`",
    ]
    if policy_diff.get("error"):
        lines.append(f"- Error: `{policy_diff['error']}`")
    failed = [check for check in checks if check.get("status") == "fail"]
    if failed:
        lines.extend(["", "## Failed Checks", ""])
        for check in failed:
            lines.append(
                f"- `{check.get('name')}`: expected {check.get('expected')}, "
                f"actual {check.get('actual')}"
            )
    if receipt.get("error"):
        lines.extend(["", "## Error", "", f"`{receipt['error']}`"])
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_PROFILE",
    "DeployProofError",
    "build_file_manifest",
    "format_deploy_proof_human",
    "format_deploy_proof_markdown",
    "load_deploy_profile",
    "run_deploy_proof",
    "summarize_deploy_latency",
    "write_deploy_proof_packet",
]
