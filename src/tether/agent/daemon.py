from __future__ import annotations

import inspect
import signal
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from tether.agent.commands import execute_command

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0
MAX_SLEEP_SECONDS = 5.0

CommandRunner = Callable[[Mapping[str, Any]], dict[str, Any]]


def run_once(
    config: Any,
    client: Any,
    command_runner: CommandRunner | None = None,
    now: Callable[[], float] | None = None,
) -> dict[str, Any]:
    timestamp = _timestamp(now)
    heartbeat_payload = _heartbeat_payload(config, timestamp)
    heartbeat_response = _client_heartbeat(client, config, heartbeat_payload)

    raw_commands = _client_poll_commands(client, config)
    commands = _normalize_commands(raw_commands)
    runner = command_runner or (lambda command: execute_command(command, config=config, now=now))

    results: list[dict[str, Any]] = []
    for command in commands:
        result = runner(command)
        if "command_id" not in result or result.get("command_id") is None:
            command_id = _command_id(command)
            if command_id is not None:
                result = dict(result)
                result["command_id"] = command_id
        _ack_command(client, config, command, result)
        results.append(result)

    return {
        "heartbeat": heartbeat_response,
        "commands_polled": len(commands),
        "commands_executed": len(results),
        "results": results,
    }


def run_forever(
    config: Any,
    client: Any,
    interval: float | None = None,
    stop_event: Any | None = None,
    command_runner: CommandRunner | None = None,
    now: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> None:
    heartbeat_interval = _resolve_interval(config, interval)
    sleeper = sleep or time.sleep
    should_stop = False
    old_handler = None

    def _handle_sigint(signum: int, frame: Any) -> None:
        nonlocal should_stop
        should_stop = True

    if threading.current_thread() is threading.main_thread():
        old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _handle_sigint)
    try:
        while not should_stop and not _event_is_set(stop_event):
            started = _timestamp(now)
            run_once(config, client, command_runner=command_runner, now=now)
            elapsed = max(0.0, _timestamp(now) - started)
            remaining = max(0.0, heartbeat_interval - elapsed)
            _bounded_sleep(remaining, sleeper, stop_event, lambda: should_stop)
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)


def _heartbeat_payload(config: Any, observed_at: float) -> dict[str, Any]:
    payload = {
        "device_id": getattr(config, "device_id", None),
        "observed_at": observed_at,
    }
    hardware_profile = _collect_hardware_profile()
    if hardware_profile is not None:
        payload["hardware_profile"] = hardware_profile
    for attr in ("cloud_url", "agent_version", "tether_version"):
        value = getattr(config, attr, None)
        if value is not None:
            payload[attr] = value
    return payload


def _collect_hardware_profile() -> Any | None:
    try:
        from tether.agent.hardware import collect_hardware_profile
    except ImportError:
        return None
    return collect_hardware_profile()


def _normalize_commands(raw: Any) -> list[Mapping[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        raw = raw.get("commands", [])
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        return []
    commands: list[Mapping[str, Any]] = []
    for command in raw:
        if isinstance(command, Mapping):
            commands.append(command)
        elif hasattr(command, "to_dict"):
            commands.append(command.to_dict())
    return commands


def _ack_command(client: Any, config: Any, command: Mapping[str, Any], result: Mapping[str, Any]) -> Any:
    command_id = _command_id(command) or result.get("command_id")
    device_id = _config_device_id(config) or command.get("device_id") or result.get("device_id")
    if device_id is not None:
        ack_model = _command_ack(command_id, result)
        try:
            return client.ack_command(str(device_id), str(command_id), ack_model)
        except TypeError:
            pass
    try:
        return client.ack_command(command_id, result)
    except TypeError:
        return _call_flexible(client.ack_command, {"command_id": command_id, "result": result})


def _client_heartbeat(client: Any, config: Any, payload: Mapping[str, Any]) -> Any:
    device_id = _config_device_id(config)
    if device_id is not None:
        try:
            return client.heartbeat(device_id, _heartbeat_model(payload))
        except TypeError:
            pass
    return _call_flexible(client.heartbeat, payload)


def _client_poll_commands(client: Any, config: Any) -> Any:
    device_id = _config_device_id(config)
    if device_id is not None:
        try:
            return client.poll_commands(device_id)
        except TypeError:
            pass
    return _call_flexible(client.poll_commands)


def _heartbeat_model(payload: Mapping[str, Any]) -> Any:
    try:
        from tether.agent.models import HeartbeatPayload
    except ImportError:
        return dict(payload)
    return HeartbeatPayload.from_dict(payload)


def _command_ack(command_id: Any, result: Mapping[str, Any]) -> Any:
    try:
        from tether.agent.models import CommandAck
    except ImportError:
        return dict(result)
    payload = dict(result)
    payload["command_id"] = str(command_id)
    return CommandAck.from_dict(payload)


def _config_device_id(config: Any) -> str | None:
    value = getattr(config, "device_id", None)
    return str(value) if value is not None else None


def _call_flexible(method: Callable[..., Any], payload: Any | None = None) -> Any:
    signature = inspect.signature(method)
    required_positionals = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    variadic = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if payload is not None and (required_positionals or variadic):
        return method(payload)
    return method()


def _resolve_interval(config: Any, interval: float | None) -> float:
    if interval is not None:
        return float(interval)
    return float(
        getattr(config, "heartbeat_interval_seconds", DEFAULT_HEARTBEAT_INTERVAL_SECONDS)
        or DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    )


def _bounded_sleep(
    seconds: float,
    sleep: Callable[[float], None],
    stop_event: Any | None,
    should_stop: Callable[[], bool],
) -> None:
    deadline = time.monotonic() + seconds
    while not should_stop() and not _event_is_set(stop_event):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        sleep(min(remaining, MAX_SLEEP_SECONDS))


def _event_is_set(stop_event: Any | None) -> bool:
    return bool(stop_event is not None and stop_event.is_set())


def _timestamp(now: Callable[[], float] | None) -> float:
    if now is None:
        return time.time()
    return float(now())


def _command_id(command: Mapping[str, Any]) -> str | None:
    value = command.get("id") or command.get("command_id")
    return str(value) if value is not None else None
