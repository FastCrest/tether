"""Local Tether Agent configuration storage."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from tether.agent.models import CommandResult, JsonDict

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30
DEFAULT_LOCAL_SERVE_TIMEOUT_SECONDS = 3.0


@dataclass(slots=True)
class AgentConfig:
    device_id: str | None = None
    cloud_url: str | None = None
    workspace_id: str | None = None
    device_token: str | None = field(default=None, repr=False)
    fleet_device_id: str | None = None
    fleet_device_token: str | None = field(default=None, repr=False)
    local_serve_url: str | None = None
    local_serve_api_key: str | None = field(default=None, repr=False)
    local_serve_timeout_seconds: float = DEFAULT_LOCAL_SERVE_TIMEOUT_SECONDS
    heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    last_heartbeat_at: str | None = None
    last_command_id: str | None = None
    last_command_result: CommandResult | JsonDict | None = None

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        result = self.last_command_result
        if isinstance(result, CommandResult):
            data["last_command_result"] = result.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentConfig":
        result = data.get("last_command_result")
        parsed_result: CommandResult | JsonDict | None
        if isinstance(result, Mapping) and "succeeded" in result:
            parsed_result = CommandResult.from_dict(result)
        elif isinstance(result, Mapping):
            parsed_result = dict(result)
        else:
            parsed_result = None
        return cls(
            device_id=data.get("device_id"),
            cloud_url=data.get("cloud_url"),
            workspace_id=data.get("workspace_id"),
            device_token=data.get("device_token"),
            fleet_device_id=data.get("fleet_device_id"),
            fleet_device_token=data.get("fleet_device_token"),
            local_serve_url=data.get("local_serve_url", data.get("serve_url")),
            local_serve_api_key=data.get("local_serve_api_key", data.get("serve_api_key")),
            local_serve_timeout_seconds=float(
                data.get("local_serve_timeout_seconds", DEFAULT_LOCAL_SERVE_TIMEOUT_SECONDS)
            ),
            heartbeat_interval_seconds=int(
                data.get("heartbeat_interval_seconds", DEFAULT_HEARTBEAT_INTERVAL_SECONDS)
            ),
            last_heartbeat_at=data.get("last_heartbeat_at"),
            last_command_id=data.get("last_command_id"),
            last_command_result=parsed_result,
        )


def default_config_path() -> Path:
    return Path.home() / ".tether" / "agent.json"


def load_config(path: str | os.PathLike[str] | None = None) -> AgentConfig | None:
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as fh:
        return AgentConfig.from_dict(json.load(fh))


def save_config(config: AgentConfig, path: str | os.PathLike[str] | None = None) -> Path:
    config_path = Path(path) if path is not None else default_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_if_posix(config_path.parent, 0o700)

    payload = json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    tmp_path = config_path.with_name(f".{config_path.name}.tmp")
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        _chmod_if_posix(tmp_path, 0o600)
        os.replace(tmp_path, config_path)
        _chmod_if_posix(config_path, 0o600)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return config_path


def _chmod_if_posix(path: Path, mode: int) -> None:
    if os.name != "posix":
        return
    try:
        os.chmod(path, mode)
    except OSError:
        pass
