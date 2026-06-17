from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Mapping
from typing import Any

from tether.agent.resources import DEFAULT_SERVE_URL, collect_serve_status

DEFAULT_DOCTOR_TIMEOUT_SECONDS = 60.0
MAX_OUTPUT_BYTES = 16 * 1024

Runner = Callable[..., subprocess.CompletedProcess[str]]
HttpGetJson = Callable[..., tuple[int, Any]]


def execute_command(
    command: Mapping[str, Any],
    *,
    config: Any | None = None,
    runner: Runner | None = None,
    http_get_json: HttpGetJson | None = None,
    now: Callable[[], float] | None = None,
) -> dict[str, Any]:
    command_type = _command_type(command)
    started_at = _timestamp(now)
    command_id = _command_id(command)

    if command_type == "noop":
        result = _result(
            command_id=command_id,
            command_type=command_type,
            succeeded=True,
            started_at=started_at,
            finished_at=_timestamp(now),
            output={"message": "ok"},
        )
    elif command_type == "doctor":
        result = run_doctor_command(
            command,
            command_id=command_id,
            started_at=started_at,
            runner=runner,
            now=now,
        )
    elif command_type == "serve_status":
        result = run_serve_status_command(
            command,
            config=config,
            command_id=command_id,
            started_at=started_at,
            http_get_json=http_get_json,
            now=now,
        )
    else:
        result = _result(
            command_id=command_id,
            command_type=command_type,
            succeeded=False,
            started_at=started_at,
            finished_at=_timestamp(now),
            error={"reason": "unsupported_command", "command_type": command_type},
        )
    return bound_result(result)


def run_doctor_command(
    command: Mapping[str, Any],
    *,
    command_id: str | None = None,
    started_at: float | None = None,
    runner: Runner | None = None,
    now: Callable[[], float] | None = None,
) -> dict[str, Any]:
    options = _command_options(command)
    timeout = float(options.get("timeout_seconds", DEFAULT_DOCTOR_TIMEOUT_SECONDS))
    argv = list(options.get("argv") or ["tether", "doctor", "--format", "json"])
    run = runner or _subprocess_runner

    try:
        completed = run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return _result(
            command_id=command_id,
            command_type="doctor",
            succeeded=False,
            started_at=started_at,
            finished_at=_timestamp(now),
            stdout=_to_text(exc.stdout),
            stderr=_to_text(exc.stderr),
            error={"reason": "timeout", "timeout_seconds": timeout},
        )
    except OSError as exc:
        return _result(
            command_id=command_id,
            command_type="doctor",
            succeeded=False,
            started_at=started_at,
            finished_at=_timestamp(now),
            error={"reason": "process_start_failed", "message": str(exc), "argv": argv},
        )

    stdout = _to_text(getattr(completed, "stdout", ""))
    stderr = _to_text(getattr(completed, "stderr", ""))
    returncode = int(getattr(completed, "returncode", 1))

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return _result(
            command_id=command_id,
            command_type="doctor",
            succeeded=False,
            started_at=started_at,
            finished_at=_timestamp(now),
            stdout=stdout,
            stderr=stderr,
            exit_code=returncode,
            error={"reason": "malformed_json", "message": str(exc)},
        )

    summary = _normalize_doctor_summary(payload)
    if returncode != 0:
        return _result(
            command_id=command_id,
            command_type="doctor",
            succeeded=False,
            started_at=started_at,
            finished_at=_timestamp(now),
            stdout=stdout,
            stderr=stderr,
            exit_code=returncode,
            output={"summary": summary, "doctor": payload},
            error={"reason": "doctor_exit_nonzero", "exit_code": returncode},
        )

    return _result(
        command_id=command_id,
        command_type="doctor",
        succeeded=True,
        started_at=started_at,
        finished_at=_timestamp(now),
        stdout=stdout,
        stderr=stderr,
        exit_code=returncode,
        output={"summary": summary, "doctor": payload},
    )


def run_serve_status_command(
    command: Mapping[str, Any],
    *,
    config: Any | None = None,
    command_id: str | None = None,
    started_at: float | None = None,
    http_get_json: HttpGetJson | None = None,
    now: Callable[[], float] | None = None,
) -> dict[str, Any]:
    base_url = _serve_url(command, config)
    timeout = float(_command_options(command).get("timeout_seconds", 3.0))
    output = collect_serve_status(
        base_url,
        api_key=_serve_api_key(command, config),
        timeout_seconds=timeout,
        http_get_json=http_get_json,
    )

    return _result(
        command_id=command_id,
        command_type="serve_status",
        succeeded=True,
        started_at=started_at,
        finished_at=_timestamp(now),
        output=output,
    )


def bound_result(result: Mapping[str, Any], *, max_bytes: int = MAX_OUTPUT_BYTES) -> dict[str, Any]:
    bounded = dict(result)
    for key in ("stdout", "stderr"):
        if key in bounded:
            bounded[key] = _truncate_text(_to_text(bounded[key]), max_bytes=max_bytes)
    for key in ("output", "error"):
        if key in bounded and bounded[key] is not None:
            bounded[key] = _truncate_jsonish(bounded[key], max_bytes=max_bytes)
    return bounded


def _subprocess_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, **kwargs)


def _normalize_doctor_summary(payload: Mapping[str, Any]) -> dict[str, int]:
    raw_summary = payload.get("summary") if isinstance(payload, Mapping) else None
    if isinstance(raw_summary, Mapping):
        return {key: int(raw_summary.get(key, 0) or 0) for key in ("pass", "fail", "warn", "skip")}

    counts = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}
    checks = payload.get("checks") if isinstance(payload, Mapping) else []
    if isinstance(checks, list):
        for check in checks:
            if isinstance(check, Mapping):
                status = str(check.get("status", "")).lower()
                if status in counts:
                    counts[status] += 1
    return counts


def _result(
    *,
    command_id: str | None,
    command_type: str,
    succeeded: bool,
    started_at: float | None,
    finished_at: float,
    output: Any | None = None,
    error: Any | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    exit_code: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "command_id": command_id,
        "command_type": command_type,
        "succeeded": succeeded,
        "status": "succeeded" if succeeded else "failed",
        "started_at": started_at,
        "finished_at": finished_at,
    }
    if output is not None:
        result["output"] = output
    if error is not None:
        result["error"] = error
    if stdout is not None:
        result["stdout"] = stdout
    if stderr is not None:
        result["stderr"] = stderr
    if exit_code is not None:
        result["exit_code"] = exit_code
    return result


def _command_type(command: Mapping[str, Any]) -> str:
    return str(command.get("type") or command.get("command_type") or command.get("name") or "")


def _command_id(command: Mapping[str, Any]) -> str | None:
    value = command.get("id") or command.get("command_id")
    return str(value) if value is not None else None


def _serve_url(command: Mapping[str, Any], config: Any | None) -> str:
    options = _command_options(command)
    value = options.get("serve_url") or options.get("local_serve_url")
    if not value and config is not None:
        value = getattr(config, "serve_url", None) or getattr(config, "local_serve_url", None)
    return str(value or DEFAULT_SERVE_URL).rstrip("/")


def _serve_api_key(command: Mapping[str, Any], config: Any | None) -> str | None:
    options = _command_options(command)
    value = options.get("local_serve_api_key") or options.get("serve_api_key")
    if not value and config is not None:
        value = getattr(config, "local_serve_api_key", None) or getattr(config, "serve_api_key", None)
    return str(value) if value else None


def _command_options(command: Mapping[str, Any]) -> dict[str, Any]:
    options = dict(command)
    payload = command.get("payload")
    if isinstance(payload, Mapping):
        options.update(payload)
    return options


def _timestamp(now: Callable[[], float] | None) -> float:
    if now is None:
        return time.time()
    return float(now())


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _truncate_text(text: str, *, max_bytes: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore") + "...[truncated]"


def _truncate_jsonish(value: Any, *, max_bytes: int) -> Any:
    text = json.dumps(value, sort_keys=True, default=str)
    if len(text.encode("utf-8")) <= max_bytes:
        return value
    return {"truncated": True, "text": _truncate_text(text, max_bytes=max_bytes)}
