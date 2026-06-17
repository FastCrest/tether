from __future__ import annotations

import pytest

from tether.agent.client import AgentClient, AgentClientError
from tether.agent.models import CommandAck, EnrollRequest, FailureEventPayload, HeartbeatPayload


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def request(self, method, url, headers=None, json=None, timeout=None):
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers or {}),
                "json": json,
                "timeout": timeout,
            }
        )
        return self.responses.pop(0)


def test_enroll_posts_to_contract_endpoint_without_auth():
    session = FakeSession(
        [
            FakeResponse(
                payload={
                    "device_id": "dev_1",
                    "device_token": "tok_1",
                    "workspace_id": "ws_1",
                    "heartbeat_interval_seconds": 30,
                }
            )
        ]
    )
    client = AgentClient("https://cloud.example.test/", session=session, timeout_seconds=3)

    response = client.enroll(EnrollRequest(enroll_token="enroll_1", hostname="host-a"))

    request = session.requests[0]
    assert response.device_id == "dev_1"
    assert request["method"] == "POST"
    assert request["url"] == "https://cloud.example.test/agent/enroll"
    assert "Authorization" not in request["headers"]
    assert request["json"]["enroll_token"] == "enroll_1"
    assert request["timeout"] == 3


def test_heartbeat_uses_device_url_and_bearer_token():
    session = FakeSession([FakeResponse(payload={"ok": True})])
    client = AgentClient("https://cloud.example.test", device_token="tok_1", session=session)

    client.heartbeat("dev/with space", HeartbeatPayload(device_id="dev/with space", workspace_id="ws_1"))

    request = session.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "https://cloud.example.test/agent/devices/dev%2Fwith%20space/heartbeat"
    assert request["headers"]["Authorization"] == "Bearer tok_1"
    assert request["json"]["workspace_id"] == "ws_1"


def test_poll_commands_adds_after_cursor_and_parses_commands():
    session = FakeSession(
        [
            FakeResponse(
                payload={
                    "commands": [
                        {
                            "command_id": "cmd_1",
                            "type": "noop",
                            "payload": {"x": 1},
                            "created_at": "2026-06-04T12:00:00Z",
                        }
                    ],
                    "next_cursor": "cmd_1",
                }
            )
        ]
    )
    client = AgentClient("https://cloud.example.test", device_token="tok_1", session=session)

    commands = client.poll_commands("dev_1", after="cmd_0")

    assert session.requests[0]["method"] == "GET"
    assert session.requests[0]["url"] == "https://cloud.example.test/agent/devices/dev_1/commands?after=cmd_0"
    assert commands[0].command_id == "cmd_1"
    assert commands[0].type == "noop"
    assert commands[0].payload == {"x": 1}


def test_ack_command_posts_ack_payload():
    session = FakeSession([FakeResponse(payload={"ok": True})])
    client = AgentClient("https://cloud.example.test", device_token="tok_1", session=session)

    client.ack_command("dev_1", "cmd_1", CommandAck(command_id="cmd_1", status="succeeded", succeeded=True))

    request = session.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "https://cloud.example.test/agent/devices/dev_1/commands/cmd_1/ack"
    assert request["headers"]["Authorization"] == "Bearer tok_1"
    assert request["json"]["status"] == "succeeded"
    assert request["json"]["succeeded"] is True


def test_create_failure_posts_to_fleet_failure_endpoint_with_fleet_token():
    session = FakeSession([FakeResponse(payload={"failure": {"id": "fail_1"}})])
    client = AgentClient(
        "https://cloud.example.test",
        device_token="fca_dev_control",
        fleet_device_token="dvc_test_failure",
        session=session,
    )

    client.create_failure(
        "dev/fleet",
        FailureEventPayload(
            event_type="diagnostic_failure",
            diagnostic={"schema_version": 1, "producer": "tether-agent"},
        ),
    )

    request = session.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "https://cloud.example.test/fleet/devices/dev%2Ffleet/failures"
    assert request["headers"]["Authorization"] == "Bearer dvc_test_failure"
    assert request["json"]["event_type"] == "diagnostic_failure"
    assert request["json"]["do_not_train"] is True
    assert "workspace_id" not in request["json"]
    assert "device_id" not in request["json"]


def test_fleet_heartbeat_posts_to_fleet_endpoint_with_fleet_device_token():
    session = FakeSession([FakeResponse(payload={"ok": True})])
    client = AgentClient(
        "https://cloud.example.test",
        device_token="fca_dev_control",
        fleet_device_token="dvc_test_heartbeat",
        session=session,
    )

    client.fleet_heartbeat(
        "dev/fleet",
        {
            "latency_p99_ms": 42.0,
            "mem_used_mb": 1024.0,
            "artifact_version": "art_1",
            "extra": {"route_readiness": {"serve_ready": True}},
        },
    )

    request = session.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "https://cloud.example.test/fleet/devices/dev%2Ffleet/heartbeat"
    assert request["headers"]["Authorization"] == "Bearer dvc_test_heartbeat"
    assert request["json"]["latency_p99_ms"] == 42.0
    assert request["json"]["artifact_version"] == "art_1"
    assert request["json"]["extra"]["route_readiness"]["serve_ready"] is True


def test_authenticated_calls_require_device_token():
    client = AgentClient("https://cloud.example.test", session=FakeSession([]))

    with pytest.raises(AgentClientError, match="device token"):
        client.poll_commands("dev_1")


def test_failure_upload_requires_fleet_device_token():
    client = AgentClient(
        "https://cloud.example.test",
        device_token="fca_dev_control",
        session=FakeSession([]),
    )

    with pytest.raises(AgentClientError, match="fleet device token"):
        client.create_failure("dev_1", {"event_type": "diagnostic_failure"})


def test_fleet_heartbeat_requires_fleet_device_token():
    client = AgentClient(
        "https://cloud.example.test",
        device_token="fca_dev_control",
        session=FakeSession([]),
    )

    with pytest.raises(AgentClientError, match="fleet device token"):
        client.fleet_heartbeat("dev_1", {"extra": {"route_readiness": {}}})


def test_http_errors_raise_client_error():
    client = AgentClient(
        "https://cloud.example.test",
        device_token="tok_1",
        session=FakeSession([FakeResponse(status_code=401, payload={}, text="unauthorized")]),
    )

    with pytest.raises(AgentClientError, match="HTTP 401"):
        client.poll_commands("dev_1")
