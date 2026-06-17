"""Dataclass models for the Tether Agent control-plane contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

JsonDict = dict[str, Any]


def _dict(value: Mapping[str, Any] | None) -> JsonDict:
    return dict(value or {})


def _optional_float(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    return float(value)


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
    fleet_device_id: str | None = None
    fleet_device_token: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnrollResponse":
        return cls(
            device_id=str(data["device_id"]),
            device_token=str(data["device_token"]),
            workspace_id=str(data["workspace_id"]),
            heartbeat_interval_seconds=int(data.get("heartbeat_interval_seconds", 30)),
            fleet_device_id=data.get("fleet_device_id"),
            fleet_device_token=data.get("fleet_device_token"),
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
    serve_status: JsonDict = field(default_factory=dict)
    doctor_summary: JsonDict = field(default_factory=dict)
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
            serve_status=_dict(data.get("serve_status")),
            doctor_summary=_dict(data.get("doctor_summary")),
            last_command_id=data.get("last_command_id"),
            last_command_result=parsed_result,
            last_heartbeat_at=data.get("last_heartbeat_at"),
        )


@dataclass(slots=True)
class FleetHeartbeatPayload:
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    mem_used_mb: float | None = None
    gpu_util_pct: float | None = None
    action_chunks_completed: int = 0
    failure_count: int = 0
    artifact_version: str | None = None
    extra: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FleetHeartbeatPayload":
        return cls(
            latency_p50_ms=_optional_float(data.get("latency_p50_ms")),
            latency_p95_ms=_optional_float(data.get("latency_p95_ms")),
            latency_p99_ms=_optional_float(data.get("latency_p99_ms")),
            mem_used_mb=_optional_float(data.get("mem_used_mb")),
            gpu_util_pct=_optional_float(data.get("gpu_util_pct")),
            action_chunks_completed=int(data.get("action_chunks_completed", 0) or 0),
            failure_count=int(data.get("failure_count", 0) or 0),
            artifact_version=data.get("artifact_version"),
            extra=_dict(data.get("extra")),
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


@dataclass(slots=True)
class FailureEventPayload:
    event_type: str
    severity: str = "warning"
    started_at: float | None = None
    ended_at: float | None = None
    artifact_id: str | None = None
    assignment_id: str | None = None
    rollout_id: str | None = None
    rollout_stage_id: str | None = None
    rollout_device_step_id: str | None = None
    operator_note: str | None = None
    do_not_train: bool = True
    metadata: JsonDict = field(default_factory=dict)
    diagnostic: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FailureEventPayload":
        return cls(
            event_type=str(data["event_type"]),
            severity=str(data.get("severity", "warning")),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            artifact_id=data.get("artifact_id"),
            assignment_id=data.get("assignment_id"),
            rollout_id=data.get("rollout_id"),
            rollout_stage_id=data.get("rollout_stage_id"),
            rollout_device_step_id=data.get("rollout_device_step_id"),
            operator_note=data.get("operator_note"),
            do_not_train=bool(data.get("do_not_train", True)),
            metadata=_dict(data.get("metadata")),
            diagnostic=_dict(data.get("diagnostic")),
        )
