from __future__ import annotations

from dataclasses import dataclass

from tether.agent.daemon import run_once


@dataclass
class Config:
    device_id: str = "dev_1"
    cloud_url: str = "https://cloud.example"
    device_token: str = "tok"
    fleet_device_id: str = "dev_fleet_1"
    fleet_device_token: str = "dvc_test_1"
    heartbeat_interval_seconds: float = 30.0


class FakeClient:
    def __init__(self, commands=None, fail_create_failure=False):
        self.commands = commands or []
        self.heartbeats = []
        self.acks = []
        self.failures = []
        self.fail_create_failure = fail_create_failure

    def heartbeat(self, payload):
        self.heartbeats.append(payload)
        return {"ok": True}

    def poll_commands(self):
        return {"commands": self.commands}

    def ack_command(self, command_id, result):
        self.acks.append((command_id, result))
        return {"ok": True}

    def create_failure(self, device_id, payload, device_token=None):
        if self.fail_create_failure:
            raise RuntimeError("cloud unavailable")
        body = payload.to_dict() if hasattr(payload, "to_dict") else dict(payload)
        self.failures.append((device_id, body, device_token))
        return {"failure": {"id": "fail_1"}}


def test_run_once_heartbeat_and_no_commands():
    client = FakeClient()

    result = run_once(Config(), client, now=lambda: 100.0)

    assert result["commands_polled"] == 0
    assert result["commands_executed"] == 0
    assert len(client.heartbeats) == 1
    assert client.heartbeats[0]["device_id"] == "dev_1"
    assert client.heartbeats[0]["observed_at"] == 100.0
    assert client.heartbeats[0]["cloud_url"] == "https://cloud.example"
    assert client.acks == []


def test_run_once_executes_noop_and_acks():
    client = FakeClient(commands=[{"id": "cmd_1", "type": "noop"}])

    result = run_once(Config(), client, now=lambda: 100.0)

    assert result["commands_polled"] == 1
    assert result["commands_executed"] == 1
    assert len(client.acks) == 1
    command_id, ack_result = client.acks[0]
    assert command_id == "cmd_1"
    assert ack_result["command_id"] == "cmd_1"
    assert ack_result["succeeded"] is True
    assert client.failures == []


def test_run_once_uploads_failure_after_failed_command_ack():
    client = FakeClient(commands=[{"id": "cmd_1", "type": "doctor"}])

    def runner(command):
        return {
            "command_id": "cmd_1",
            "command_type": "doctor",
            "succeeded": False,
            "status": "failed",
            "error": {"reason": "doctor_exit_nonzero"},
            "output": {"summary": {"pass": 0, "fail": 1, "warn": 0, "skip": 0}},
        }

    result = run_once(Config(), client, command_runner=runner, now=lambda: 100.0)

    assert len(client.acks) == 1
    assert len(client.failures) == 1
    device_id, body, token = client.failures[0]
    assert device_id == "dev_fleet_1"
    assert token == "dvc_test_1"
    assert body["event_type"] == "diagnostic_failure"
    assert body["do_not_train"] is True
    assert result["results"][0]["failure_upload"]["status"] == "uploaded"


def test_run_once_failure_upload_error_does_not_block_ack():
    client = FakeClient(commands=[{"id": "cmd_1", "type": "doctor"}], fail_create_failure=True)

    def runner(command):
        return {
            "command_id": "cmd_1",
            "command_type": "doctor",
            "succeeded": False,
            "status": "failed",
            "error": {"reason": "doctor_exit_nonzero"},
        }

    result = run_once(Config(), client, command_runner=runner, now=lambda: 100.0)

    assert len(client.acks) == 1
    assert result["results"][0]["failure_upload"]["status"] == "failed"
