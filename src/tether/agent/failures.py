"""Failure-event producer helpers for Tether Agent."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from tether.agent.models import FailureEventPayload

METADATA_SOFT_MAX_BYTES = 8 * 1024
DIAGNOSTIC_SOFT_MAX_BYTES = 16 * 1024
MAX_STRING_CHARS = 512

_TOKEN_RE = re.compile(r"\b(?:dvc|fca|rc)_(?:live|test|dev)_[A-Za-z0-9._-]{6,}\b")
_ABS_PATH_RE = re.compile(r"(?<!:)\/(?:[\w .-]+\/){1,}[\w .-]+")


def build_failure_from_command_result(
    command: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    config: Any | None = None,
) -> FailureEventPayload | None:
    command_type = _command_type(command, result)
    if command_type == "serve_status":
        payload = _serve_status_failure(command, result, config=config)
        if payload is not None:
            return payload

    if result.get("succeeded") is True:
        return None

    reason = _error_reason(result)
    event_type = "unsupported_command" if reason == "unsupported_command" else "diagnostic_failure"
    severity = "warning" if event_type == "unsupported_command" else "critical"
    return FailureEventPayload(
        event_type=event_type,
        severity=severity,
        started_at=_optional_float(result.get("started_at")),
        ended_at=_optional_float(result.get("finished_at")),
        do_not_train=True,
        metadata=_bounded_json(
            _base_metadata(command, result, config=config),
            max_bytes=METADATA_SOFT_MAX_BYTES,
        ),
        diagnostic=_bounded_json(
            {
                "schema_version": 1,
                "producer": "tether-agent",
                "trigger": event_type,
                "error_code": _sanitize_value(reason or event_type),
                "summary": _summary_for(command_type, reason),
                "command": _command_diagnostic(command, result),
                "doctor": _doctor_diagnostic(result),
                "redaction": _redaction_summary(),
            },
            max_bytes=DIAGNOSTIC_SOFT_MAX_BYTES,
        ),
    )


def _serve_status_failure(
    command: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    config: Any | None = None,
) -> FailureEventPayload | None:
    output = result.get("output")
    if not isinstance(output, Mapping):
        return None if result.get("succeeded") is True else _fallback_failed_serve(command, result, config=config)

    reachable = bool(output.get("reachable"))
    ready = bool(output.get("ready"))
    if reachable and ready:
        return None

    error_code = "serve_unreachable" if not reachable else "serve_not_ready"
    return FailureEventPayload(
        event_type="serve_unavailable",
        severity="critical" if not reachable else "warning",
        started_at=_optional_float(result.get("started_at")),
        ended_at=_optional_float(result.get("finished_at")),
        do_not_train=True,
        metadata=_bounded_json(
            _base_metadata(command, result, config=config),
            max_bytes=METADATA_SOFT_MAX_BYTES,
        ),
        diagnostic=_bounded_json(
            {
                "schema_version": 1,
                "producer": "tether-agent",
                "trigger": "serve_unavailable",
                "error_code": error_code,
                "summary": "local serve endpoint is unreachable"
                if not reachable
                else "local serve endpoint is reachable but not ready",
                "serve": _serve_diagnostic(output),
                "command": _command_diagnostic(command, result),
                "redaction": _redaction_summary(),
            },
            max_bytes=DIAGNOSTIC_SOFT_MAX_BYTES,
        ),
    )


def _fallback_failed_serve(
    command: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    config: Any | None = None,
) -> FailureEventPayload:
    return FailureEventPayload(
        event_type="serve_unavailable",
        severity="critical",
        started_at=_optional_float(result.get("started_at")),
        ended_at=_optional_float(result.get("finished_at")),
        do_not_train=True,
        metadata=_bounded_json(_base_metadata(command, result, config=config), max_bytes=METADATA_SOFT_MAX_BYTES),
        diagnostic=_bounded_json(
            {
                "schema_version": 1,
                "producer": "tether-agent",
                "trigger": "serve_unavailable",
                "error_code": _sanitize_value(_error_reason(result) or "serve_failed"),
                "summary": "serve status command failed",
                "command": _command_diagnostic(command, result),
                "redaction": _redaction_summary(),
            },
            max_bytes=DIAGNOSTIC_SOFT_MAX_BYTES,
        ),
    )


def _base_metadata(command: Mapping[str, Any], result: Mapping[str, Any], *, config: Any | None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "command_id": _command_id(command, result),
        "command_type": _command_type(command, result),
        "status": result.get("status"),
    }
    for attr in ("agent_version", "tether_version"):
        value = getattr(config, attr, None) if config is not None else None
        if value:
            data[attr] = str(value)
    return {key: _sanitize_value(value) for key, value in data.items() if value is not None}


def _command_diagnostic(command: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {
        "id": _command_id(command, result),
        "type": _command_type(command, result),
    }
    reason = _error_reason(result)
    if reason:
        diagnostic["reason"] = reason
    exit_code = result.get("exit_code")
    if exit_code is not None:
        diagnostic["exit_code"] = exit_code
    return {key: _sanitize_value(value) for key, value in diagnostic.items() if value is not None}


def _doctor_diagnostic(result: Mapping[str, Any]) -> dict[str, Any] | None:
    output = result.get("output")
    if not isinstance(output, Mapping):
        return None
    summary = output.get("summary")
    if not isinstance(summary, Mapping):
        doctor = output.get("doctor")
        summary = doctor.get("summary") if isinstance(doctor, Mapping) else None
    if not isinstance(summary, Mapping):
        return None
    return {"summary": {key: int(summary.get(key, 0) or 0) for key in ("pass", "fail", "warn", "skip")}}


def _serve_diagnostic(output: Mapping[str, Any]) -> dict[str, Any]:
    health = output.get("health") if isinstance(output.get("health"), Mapping) else {}
    config = output.get("config") if isinstance(output.get("config"), Mapping) else {}
    return {
        "reachable": bool(output.get("reachable")),
        "ready": bool(output.get("ready")),
        "health_status_code": health.get("status_code") if isinstance(health, Mapping) else None,
        "config_status_code": config.get("status_code") if isinstance(config, Mapping) else None,
        "health_error": _probe_error_reason(health),
        "config_error": _probe_error_reason(config),
    }


def _probe_error_reason(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    error = value.get("error")
    if isinstance(error, Mapping):
        reason = error.get("reason")
        return str(reason) if reason is not None else None
    return None


def _error_reason(result: Mapping[str, Any]) -> str | None:
    error = result.get("error")
    if isinstance(error, Mapping):
        reason = error.get("reason") or error.get("error_code")
        return str(reason) if reason is not None else None
    if error:
        return str(error)
    return None


def _summary_for(command_type: str, reason: str | None) -> str:
    if command_type == "doctor":
        return "doctor command failed"
    if reason == "unsupported_command":
        return "unsupported agent command failed"
    return "agent command failed"


def _redaction_summary() -> dict[str, str]:
    return {
        "stdout": "omitted",
        "stderr": "omitted",
        "traceback": "omitted",
        "images": "none",
        "instructions": "none",
        "actions": "none",
    }


def _bounded_json(value: Mapping[str, Any], *, max_bytes: int) -> dict[str, Any]:
    sanitized = _sanitize_value(value)
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
        "trigger": data.get("trigger") or data.get("command_type") or "failure",
        "error_code": data.get("error_code") or "payload_truncated",
        "redaction": _redaction_summary(),
    }


def _sanitize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, Mapping):
        return {str(key): _sanitize_value(inner) for key, inner in value.items() if _safe_key(str(key))}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(inner) for inner in value[:32]]
    return _sanitize_text(str(value))


def _safe_key(key: str) -> bool:
    lowered = key.lower()
    blocked = ("token", "secret", "password", "api_key", "authorization", "stdout", "stderr", "traceback")
    return not any(part in lowered for part in blocked)


def _sanitize_text(text: str) -> str:
    cleaned = _TOKEN_RE.sub("[redacted-token]", text)
    cleaned = _ABS_PATH_RE.sub("[redacted-path]", cleaned)
    if len(cleaned) > MAX_STRING_CHARS:
        return cleaned[:MAX_STRING_CHARS] + "...[truncated]"
    return cleaned


def _command_type(command: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    value = result.get("command_type") or command.get("type") or command.get("command_type") or command.get("name")
    return str(value or "")


def _command_id(command: Mapping[str, Any], result: Mapping[str, Any]) -> str | None:
    value = result.get("command_id") or command.get("command_id") or command.get("id")
    return str(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
