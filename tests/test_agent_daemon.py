from __future__ import annotations

from dataclasses import dataclass

from tether.agent.daemon import run_once


@dataclass
class Config:
    device_id: str = "dev_1"
    cloud_url: str = "https://cloud.example"
    device_token: str = "tok"
    heartbeat_interval_seconds: float = 30.0


class FakeClient:
    def __init__(self, commands=None):
        self.commands = commands or []
        self.heartbeats = []
        self.acks = []

    def heartbeat(self, payload):
        self.heartbeats.append(payload)
        return {"ok": True}

    def poll_commands(self):
        return {"commands": self.commands}

    def ack_command(self, command_id, result):
        self.acks.append((command_id, result))
        return {"ok": True}


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
