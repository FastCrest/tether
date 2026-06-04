"""Bounded local readiness and resource collection for Agent heartbeats."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.parse
from collections.abc import Callable, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

JsonDict = dict[str, Any]
HttpGetJson = Callable[..., tuple[int, Any]]

DEFAULT_SERVE_URL = "http://127.0.0.1:8000"
DEFAULT_SERVE_TIMEOUT_SECONDS = 3.0
MAX_OUTPUT_BYTES = 16 * 1024
MAX_STRING_CHARS = 512

_TOKEN_RE = re.compile(r"\b(?:dvc|fca|rc)_(?:live|test|dev)_[A-Za-z0-9._-]{6,}\b")
_ABS_PATH_RE = re.compile(r"(?<!:)\/(?:[\w .-]+\/){1,}[\w .-]+")


def collect_serve_status(
    serve_url: str | None = None,
    *,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_SERVE_TIMEOUT_SECONDS,
    http_get_json: HttpGetJson | None = None,
) -> JsonDict:
    """Probe local ``tether serve`` health/config without leaking secrets."""
    base_url = _normalize_serve_url(serve_url)
    get_json = http_get_json or _http_get_json
    headers = _auth_headers(api_key)

    health = _probe_json(get_json, f"{base_url}/health", timeout_seconds)
    config_probe = _probe_json(get_json, f"{base_url}/config", timeout_seconds, headers=headers)
    reachable = bool(health["ok"] or config_probe["ok"])
    ready = bool(
        health["ok"]
        and isinstance(health.get("body"), Mapping)
        and (
            health["body"].get("status") == "ok"
            or health["body"].get("state") in {"ready", "ok", "healthy"}
        )
    )

    return _bounded_mapping(
        {
            "url": _safe_url(base_url),
            "reachable": reachable,
            "ready": ready,
            "health": health,
            "config": config_probe,
        },
        max_bytes=MAX_OUTPUT_BYTES,
    )


def collect_route_readiness(
    config: Any | None = None,
    *,
    observed_at: float | None = None,
    http_get_json: HttpGetJson | None = None,
) -> JsonDict:
    """Collect route-readiness evidence for Fleet telemetry."""
    timestamp = time.time() if observed_at is None else float(observed_at)
    serve_url = _config_value(config, "local_serve_url", "serve_url") or DEFAULT_SERVE_URL
    api_key = _config_value(config, "local_serve_api_key", "serve_api_key")
    timeout = _optional_float(
        _config_value(config, "local_serve_timeout_seconds", "serve_timeout_seconds")
    ) or DEFAULT_SERVE_TIMEOUT_SECONDS

    serve_status = collect_serve_status(
        str(serve_url),
        api_key=str(api_key) if api_key else None,
        timeout_seconds=timeout,
        http_get_json=http_get_json,
    )
    health_body = _body_mapping(serve_status.get("health"))
    config_body = _body_mapping(serve_status.get("config"))
    artifact_identity = _artifact_identity(config, health_body, config_body)
    runtime = _runtime_identity(config, health_body, config_body)
    resource_pressure = collect_resource_pressure()

    readiness: JsonDict = {
        "schema_version": 1,
        "producer": "tether-agent",
        "observed_at": timestamp,
        "serve_url_kind": _serve_url_kind(str(serve_url)),
        "serve_reachable": bool(serve_status.get("reachable")),
        "serve_ready": bool(serve_status.get("ready")),
        "serve_status": _serve_status_summary(serve_status),
        "resource_pressure": resource_pressure,
        "errors": _serve_errors(serve_status),
    }
    if artifact_identity:
        readiness["artifact_identity"] = artifact_identity
    if runtime:
        readiness["runtime"] = runtime
    return _bounded_mapping(readiness, max_bytes=MAX_OUTPUT_BYTES)


def fleet_readiness_heartbeat_payload(
    config: Any | None,
    *,
    route_readiness: Mapping[str, Any] | None = None,
    observed_at: float | None = None,
    http_get_json: HttpGetJson | None = None,
) -> JsonDict:
    readiness = dict(
        route_readiness
        or collect_route_readiness(config, observed_at=observed_at, http_get_json=http_get_json)
    )
    resource_pressure = readiness.get("resource_pressure")
    if not isinstance(resource_pressure, Mapping):
        resource_pressure = {}
    artifact_identity = readiness.get("artifact_identity")
    if not isinstance(artifact_identity, Mapping):
        artifact_identity = {}

    payload: JsonDict = {
        "action_chunks_completed": int(
            _config_value(config, "action_chunks_completed", "fleet_action_chunks_completed") or 0
        ),
        "failure_count": int(_config_value(config, "failure_count", "fleet_failure_count") or 0),
        "extra": {"route_readiness": readiness},
    }
    for key in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms"):
        value = _optional_float(_config_value(config, key))
        if value is not None:
            payload[key] = value
    mem_used = _first_number(resource_pressure, "mem_used_mb", "memory_used_mb")
    gpu_util = _first_number(resource_pressure, "gpu_util_pct", "gpu_utilization_pct")
    artifact_version = _first_string(
        artifact_identity,
        "artifact_version",
        "artifact_id",
        "registry_artifact_id",
        "id",
    ) or _config_value(config, "artifact_version", "current_artifact_version")
    if mem_used is not None:
        payload["mem_used_mb"] = mem_used
    if gpu_util is not None:
        payload["gpu_util_pct"] = gpu_util
    if artifact_version:
        payload["artifact_version"] = str(artifact_version)
    return _bounded_mapping(payload, max_bytes=MAX_OUTPUT_BYTES)


def collect_resource_pressure() -> JsonDict:
    pressure: JsonDict = {}
    _add_memory_pressure(pressure)
    _add_gpu_pressure(pressure)
    _add_thermal_pressure(pressure)
    pressure["pressure"] = _has_pressure(pressure)
    return _bounded_mapping(pressure, max_bytes=MAX_OUTPUT_BYTES)


def _http_get_json(url: str, timeout: float, headers: Mapping[str, str] | None = None) -> tuple[int, Any]:
    request = Request(url, headers={"Accept": "application/json", **dict(headers or {})})
    with urlopen(request, timeout=timeout) as response:
        body = response.read(MAX_OUTPUT_BYTES + 1)
        text = body.decode("utf-8", errors="replace")
        return int(response.status), json.loads(text)


def _probe_json(
    get_json: HttpGetJson,
    url: str,
    timeout: float,
    *,
    headers: Mapping[str, str] | None = None,
) -> JsonDict:
    try:
        status_code, body = _call_get_json(get_json, url, timeout, headers)
    except HTTPError as exc:
        raw = exc.read(MAX_OUTPUT_BYTES + 1).decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = raw
        return {
            "ok": False,
            "status_code": exc.code,
            "body": _sanitize_value(body),
            "error": {"reason": "http_error"},
        }
    except (URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False,
            "status_code": None,
            "body": None,
            "error": {"reason": "unreachable", "message": _sanitize_text(str(exc))},
        }
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "status_code": None,
            "body": None,
            "error": {"reason": "malformed_json", "message": _sanitize_text(str(exc))},
        }
    return {
        "ok": 200 <= int(status_code) < 300,
        "status_code": int(status_code),
        "body": _sanitize_value(body),
    }


def _call_get_json(
    get_json: HttpGetJson,
    url: str,
    timeout: float,
    headers: Mapping[str, str] | None,
) -> tuple[int, Any]:
    try:
        return get_json(url, timeout, dict(headers or {}))
    except TypeError:
        return get_json(url, timeout)


def _normalize_serve_url(value: str | None) -> str:
    return str(value or DEFAULT_SERVE_URL).rstrip("/")


def _auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}", "X-Tether-Key": api_key}


def _body_mapping(probe: Any) -> Mapping[str, Any]:
    if isinstance(probe, Mapping) and isinstance(probe.get("body"), Mapping):
        return probe["body"]
    return {}


def _serve_status_summary(serve_status: Mapping[str, Any]) -> JsonDict:
    health = serve_status.get("health") if isinstance(serve_status.get("health"), Mapping) else {}
    config = serve_status.get("config") if isinstance(serve_status.get("config"), Mapping) else {}
    health_body = _body_mapping(health)
    config_body = _body_mapping(config)
    summary: JsonDict = {
        "reachable": bool(serve_status.get("reachable")),
        "ready": bool(serve_status.get("ready")),
        "health_status_code": health.get("status_code") if isinstance(health, Mapping) else None,
        "config_status_code": config.get("status_code") if isinstance(config, Mapping) else None,
    }
    for key in ("status", "state", "model_loaded", "inference_mode", "robot_id"):
        if key in health_body:
            summary[key] = health_body[key]
    for key in ("robot_id", "runtime", "runtime_name", "artifact_version"):
        if key in config_body:
            summary.setdefault(key, config_body[key])
    return {key: value for key, value in _sanitize_value(summary).items() if value is not None}


def _serve_errors(serve_status: Mapping[str, Any]) -> list[JsonDict]:
    errors: list[JsonDict] = []
    for name in ("health", "config"):
        probe = serve_status.get(name)
        if not isinstance(probe, Mapping) or probe.get("ok"):
            continue
        error = probe.get("error")
        errors.append(
            {
                "probe": name,
                "status_code": probe.get("status_code"),
                "reason": error.get("reason") if isinstance(error, Mapping) else None,
            }
        )
    return [error for error in errors if error.get("reason") or error.get("status_code") is not None]


def _artifact_identity(
    config: Any | None,
    health_body: Mapping[str, Any],
    config_body: Mapping[str, Any],
) -> JsonDict:
    identity: JsonDict = {}
    aliases = {
        "artifact_id": ("artifact_id", "registry_artifact_id"),
        "artifact_version": ("artifact_version", "current_artifact_version", "model_version"),
        "optimized_artifact_digest": ("optimized_artifact_digest", "artifact_digest", "digest"),
        "runtime": ("runtime", "runtime_name", "inference_mode"),
        "target_hardware_class": ("target_hardware_class", "target_sku", "target"),
    }
    for out_key, keys in aliases.items():
        value = _first_source_value(config, health_body, config_body, keys)
        if value:
            identity[out_key] = str(value)
    return _bounded_mapping(identity, max_bytes=MAX_OUTPUT_BYTES)


def _runtime_identity(
    config: Any | None,
    health_body: Mapping[str, Any],
    config_body: Mapping[str, Any],
) -> JsonDict:
    runtime: JsonDict = {}
    name = _first_source_value(
        config,
        health_body,
        config_body,
        ("runtime", "runtime_name", "inference_mode"),
    )
    if name:
        runtime["name"] = str(name)
    for key in ("robot_id", "model_loaded", "vlm_loaded"):
        value = _first_source_value(config, health_body, config_body, (key,))
        if value is not None:
            runtime[key] = value
    return _bounded_mapping(runtime, max_bytes=MAX_OUTPUT_BYTES)


def _first_source_value(
    config: Any | None,
    health_body: Mapping[str, Any],
    config_body: Mapping[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        value = _config_value(config, key)
        if value not in (None, ""):
            return value
    for source in (config_body, health_body):
        value = _first_value(source, keys)
        if value not in (None, ""):
            return value
    return None


def _first_value(source: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return None


def _config_value(config: Any | None, *names: str) -> Any:
    if config is None:
        return None
    for name in names:
        value = getattr(config, name, None)
        if value not in (None, ""):
            return value
    return None


def _add_memory_pressure(out: JsonDict) -> None:
    meminfo = _read_proc_meminfo()
    total_kb = meminfo.get("MemTotal")
    available_kb = meminfo.get("MemAvailable")
    if total_kb and available_kb is not None:
        total_mb = total_kb / 1024.0
        available_mb = available_kb / 1024.0
        used_mb = max(0.0, total_mb - available_mb)
        out["mem_used_mb"] = round(used_mb, 3)
        out["mem_total_mb"] = round(total_mb, 3)
        out["memory_pressure_pct"] = round((used_mb / total_mb) * 100.0, 3)


def _read_proc_meminfo() -> dict[str, float]:
    path = "/proc/meminfo"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return {}
    values: dict[str, float] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if parts:
            try:
                values[key] = float(parts[0])
            except ValueError:
                continue
    return values


def _add_gpu_pressure(out: JsonDict) -> None:
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return
    first = output.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) >= 1:
        out["gpu_util_pct"] = _optional_float(parts[0])
    if len(parts) >= 3:
        used = _optional_float(parts[1])
        total = _optional_float(parts[2])
        if used is not None:
            out["gpu_mem_used_mb"] = used
        if total:
            out["gpu_mem_total_mb"] = total
            if used is not None:
                out["gpu_memory_pressure_pct"] = round((used / total) * 100.0, 3)
    if len(parts) >= 4:
        out["gpu_temperature_c"] = _optional_float(parts[3])


def _add_thermal_pressure(out: JsonDict) -> None:
    thermal_root = "/sys/class/thermal"
    try:
        entries = os.listdir(thermal_root)
    except OSError:
        return
    max_temp_c: float | None = None
    for entry in entries[:32]:
        temp_path = os.path.join(thermal_root, entry, "temp")
        try:
            with open(temp_path, "r", encoding="utf-8") as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        value = _optional_float(raw)
        if value is None:
            continue
        temp_c = value / 1000.0 if value > 1000 else value
        max_temp_c = temp_c if max_temp_c is None else max(max_temp_c, temp_c)
    if max_temp_c is not None:
        out["max_thermal_c"] = round(max_temp_c, 3)
        out["thermal_throttled"] = max_temp_c >= 85.0


def _has_pressure(data: Mapping[str, Any]) -> bool:
    for key in ("memory_pressure_pct", "gpu_memory_pressure_pct"):
        value = _optional_float(data.get(key))
        if value is not None and value >= 90.0:
            return True
    gpu_util = _optional_float(data.get("gpu_util_pct"))
    if gpu_util is not None and gpu_util >= 98.0:
        return True
    for key in ("gpu_temperature_c", "max_thermal_c"):
        value = _optional_float(data.get(key))
        if value is not None and value >= 85.0:
            return True
    return bool(data.get("thermal_throttled"))


def _run(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _serve_url_kind(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    host = (parsed.hostname or "").lower()
    if host in {"127.0.0.1", "localhost", "::1"}:
        return "loopback"
    if host.startswith(("10.", "192.168.")) or (
        host.startswith("172.") and _host_second_octet_in_private_range(host)
    ):
        return "private"
    return "remote" if host else "unknown"


def _host_second_octet_in_private_range(host: str) -> bool:
    try:
        second = int(host.split(".")[1])
    except (IndexError, ValueError):
        return False
    return 16 <= second <= 31


def _safe_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    netloc = parsed.hostname or ""
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _bounded_mapping(value: Mapping[str, Any], *, max_bytes: int) -> JsonDict:
    sanitized = _sanitize_value(dict(value))
    if not isinstance(sanitized, Mapping):
        return {}
    data = dict(sanitized)
    encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    if len(encoded) <= max_bytes:
        return data
    return {
        "truncated": True,
        "schema_version": data.get("schema_version", 1),
        "producer": data.get("producer", "tether-agent"),
    }


def _sanitize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_value(inner)
            for key, inner in value.items()
            if _safe_key(str(key), inner)
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(inner) for inner in value[:32]]
    return _sanitize_text(str(value))


def _safe_key(key: str, value: Any) -> bool:
    lowered = key.lower()
    blocked = (
        "token",
        "secret",
        "password",
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "signed_url",
        "stdout",
        "stderr",
        "traceback",
        "env",
        "path",
        "uri",
    )
    if any(part in lowered for part in blocked):
        return False
    return not (isinstance(value, str) and _looks_like_unsafe_path(value))


def _sanitize_text(text: str) -> str:
    if _looks_like_url(text):
        cleaned = _safe_url(text)
    else:
        cleaned = _ABS_PATH_RE.sub("[redacted-path]", text)
    cleaned = _TOKEN_RE.sub("[redacted-token]", cleaned)
    if len(cleaned) > MAX_STRING_CHARS:
        return cleaned[:MAX_STRING_CHARS] + "...[truncated]"
    return cleaned


def _looks_like_url(value: str) -> bool:
    parsed = urllib.parse.urlsplit(value)
    return bool(parsed.scheme and parsed.netloc)


def _looks_like_unsafe_path(value: str) -> bool:
    if _looks_like_url(value):
        return False
    return bool(_ABS_PATH_RE.search(value))


def _first_number(source: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _optional_float(source.get(key))
        if value is not None:
            return value
    return None


def _first_string(source: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = source.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
