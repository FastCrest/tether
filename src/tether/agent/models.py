"""Dataclass models for the Tether Agent control-plane contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

JsonDict = dict[str, Any]


def _dict(value: Mapping[str, Any] | None) -> JsonDict:
    return dict(value or {})


@dataclass(slots=True)
class EnrollRequest:
    enroll_token: str
    hostname: str | None = None
    agent_version: str | None = None
    tether_version: str | None = None
    hardware_profile: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnrollRequest":
        return cls(
            enroll_token=str(data["enroll_token"]),
            hostname=data.get("hostname"),
            agent_version=data.get("agent_version"),
            tether_version=data.get("tether_version"),
            hardware_profile=_dict(data.get("hardware_profile")),
        )


@dataclass(slots=True)
class EnrollResponse:
    device_id: str
    device_token: str
    workspace_id: str
    heartbeat_interval_seconds: int = 30

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnrollResponse":
        return cls(
            device_id=str(data["device_id"]),
            device_token=str(data["device_token"]),
            workspace_id=str(data["workspace_id"]),
            heartbeat_interval_seconds=int(data.get("heartbeat_interval_seconds", 30)),
        )


@dataclass(slots=True)
class CommandResult:
    succeeded: bool
    output: JsonDict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CommandResult":
        return cls(
            succeeded=bool(data["succeeded"]),
            output=_dict(data.get("output")),
            error=data.get("error"),
        )


@dataclass(slots=True)
class HeartbeatPayload:
    device_id: str
    workspace_id: str | None = None
    agent_version: str | None = None
    tether_version: str | None = None
    hardware_profile: JsonDict = field(default_factory=dict)
    last_command_id: str | None = None
    last_command_result: CommandResult | JsonDict | None = None
    last_heartbeat_at: str | None = None

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        result = self.last_command_result
        if isinstance(result, CommandResult):
            data["last_command_result"] = result.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HeartbeatPayload":
        result = data.get("last_command_result")
        parsed_result: CommandResult | JsonDict | None
        if isinstance(result, Mapping) and "succeeded" in result:
            parsed_result = CommandResult.from_dict(result)
        elif isinstance(result, Mapping):
            parsed_result = dict(result)
        else:
            parsed_result = None
        return cls(
            device_id=str(data["device_id"]),
            workspace_id=data.get("workspace_id"),
            agent_version=data.get("agent_version"),
            tether_version=data.get("tether_version"),
            hardware_profile=_dict(data.get("hardware_profile")),
            last_command_id=data.get("last_command_id"),
            last_command_result=parsed_result,
            last_heartbeat_at=data.get("last_heartbeat_at"),
        )


@dataclass(slots=True)
class AgentCommand:
    command_id: str
    type: str
    payload: JsonDict = field(default_factory=dict)
    created_at: str | None = None
    cursor: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentCommand":
        command_id = data.get("command_id", data.get("id"))
        if command_id is None:
            raise ValueError("command is missing command_id")
        command_type = data.get("type", data.get("command_type"))
        if command_type is None:
            raise ValueError("command is missing type")
        return cls(
            command_id=str(command_id),
            type=str(command_type),
            payload=_dict(data.get("payload")),
            created_at=data.get("created_at"),
            cursor=data.get("cursor"),
        )


@dataclass(slots=True)
class CommandAck:
    command_id: str
    status: str
    succeeded: bool | None = None
    started_at: str | None = None
    finished_at: str | None = None
    output: JsonDict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_result(
        cls,
        command_id: str,
        result: CommandResult,
        *,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> "CommandAck":
        return cls(
            command_id=command_id,
            status="succeeded" if result.succeeded else "failed",
            succeeded=result.succeeded,
            started_at=started_at,
            finished_at=finished_at,
            output=result.output,
            error=result.error,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CommandAck":
        return cls(
            command_id=str(data["command_id"]),
            status=str(data["status"]),
            succeeded=data.get("succeeded"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            output=_dict(data.get("output")),
            error=data.get("error"),
        )
