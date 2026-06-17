from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from tether.agent.daemon import run_once


@dataclass
class Config:
    device_id: str = "dev_1"
    cloud_url: str = "https://cloud.example"
    device_token: str = "tok"
    fleet_device_id: str = "dev_fleet_1"
    fleet_device_token: str = "dvc_test_1"
    local_serve_url: str = "http://127.0.0.1:1"
    local_serve_api_key: str | None = None
    heartbeat_interval_seconds: float = 30.0


class FakeClient:
    def __init__(self, commands=None, fail_create_failure=False, fail_fleet_heartbeat=False):
        self.commands = commands or []
        self.heartbeats = []
        self.fleet_heartbeats = []
        self.acks = []
        self.failures = []
        self.fail_create_failure = fail_create_failure
        self.fail_fleet_heartbeat = fail_fleet_heartbeat

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

    def fleet_heartbeat(self, device_id, payload, device_token=None):
        if self.fail_fleet_heartbeat:
            raise RuntimeError("fleet unavailable")
        body = payload.to_dict() if hasattr(payload, "to_dict") else dict(payload)
        self.fleet_heartbeats.append((device_id, body, device_token))
        return {"telemetry": {"id": "tel_1"}}


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
    assert len(client.fleet_heartbeats) == 1


def test_run_once_posts_agent_and_fleet_readiness_from_fake_ready_serve():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "status": "ok",
                            "state": "ready",
                            "inference_mode": "onnx",
                            "robot_id": "robot_1",
                            "export_dir": "/tmp/secret/model",
                        }
                    ).encode()
                )
                return
            if self.path == "/config":
                assert self.headers["Authorization"] == "Bearer serve_secret"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "robot_id": "robot_1",
                            "artifact_version": "art_1",
                            "optimized_artifact_digest": "sha256:abc",
                            "target_sku": "orin",
                            "signed_url": "https://example.test/model?token=secret",
                            "device_token": "dvc_test_should_not_leak",
                        }
                    ).encode()
                )
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = Config(
            local_serve_url=f"http://127.0.0.1:{server.server_port}",
            local_serve_api_key="serve_secret",
        )
        client = FakeClient()

        result = run_once(config, client, now=lambda: 100.0)
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert len(client.heartbeats) == 1
    assert client.heartbeats[0]["serve_status"]["ready"] is True
    assert len(client.fleet_heartbeats) == 1
    device_id, payload, token = client.fleet_heartbeats[0]
    readiness = payload["extra"]["route_readiness"]
    rendered = json.dumps(payload, sort_keys=True)
    assert result["fleet_heartbeat"]["status"] == "posted"
    assert device_id == "dev_fleet_1"
    assert token == "dvc_test_1"
    assert payload["artifact_version"] == "art_1"
    assert readiness["serve_ready"] is True
    assert readiness["runtime"]["name"] == "onnx"
    assert readiness["artifact_identity"]["optimized_artifact_digest"] == "sha256:abc"
    assert "signed_url" not in rendered
    assert "dvc_test_should_not_leak" not in rendered
    assert "/tmp/secret/model" not in rendered


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


def test_run_once_fleet_readiness_error_does_not_block_command_ack():
    client = FakeClient(commands=[{"id": "cmd_1", "type": "noop"}], fail_fleet_heartbeat=True)

    result = run_once(Config(), client, now=lambda: 100.0)

    assert len(client.acks) == 1
    assert result["fleet_heartbeat"]["status"] == "failed"
    assert result["fleet_heartbeat"]["reason"] == "fleet_heartbeat_failed"
    assert result["commands_executed"] == 1
