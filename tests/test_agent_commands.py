from __future__ import annotations

import json
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from tether.agent.commands import execute_command


def test_noop_succeeds():
    result = execute_command({"id": "cmd_1", "type": "noop"}, now=lambda: 10.0)

    assert result["command_id"] == "cmd_1"
    assert result["succeeded"] is True
    assert result["status"] == "succeeded"


def test_unknown_command_fails_unsupported():
    result = execute_command({"id": "cmd_1", "type": "deploy"}, now=lambda: 10.0)

    assert result["succeeded"] is False
    assert result["error"]["reason"] == "unsupported_command"


def test_doctor_success_parses_summary():
    payload = {"summary": {"pass": 2, "fail": 0, "warn": 1, "skip": 0}, "checks": []}

    def runner(argv, **kwargs):
        assert argv == ["tether", "doctor", "--format", "json"]
        assert kwargs["timeout"] == 60.0
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload), stderr="")

    result = execute_command({"id": "cmd_1", "type": "doctor"}, runner=runner, now=lambda: 10.0)

    assert result["succeeded"] is True
    assert result["output"]["summary"] == {"pass": 2, "fail": 0, "warn": 1, "skip": 0}


def test_doctor_accepts_payload_options():
    payload = {"summary": {"pass": 1, "fail": 0, "warn": 0, "skip": 0}, "checks": []}

    def runner(argv, **kwargs):
        assert argv == ["custom-doctor"]
        assert kwargs["timeout"] == 2.0
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload), stderr="")

    result = execute_command(
        {"id": "cmd_1", "type": "doctor", "payload": {"argv": ["custom-doctor"], "timeout_seconds": 2}},
        runner=runner,
        now=lambda: 10.0,
    )

    assert result["succeeded"] is True


def test_doctor_nonzero_fails_with_summary():
    payload = {"summary": {"pass": 1, "fail": 1, "warn": 0, "skip": 0}, "checks": []}

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout=json.dumps(payload), stderr="failed")

    result = execute_command({"id": "cmd_1", "type": "doctor"}, runner=runner, now=lambda: 10.0)

    assert result["succeeded"] is False
    assert result["error"]["reason"] == "doctor_exit_nonzero"
    assert result["output"]["summary"]["fail"] == 1


def test_doctor_malformed_json_fails():
    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="{not json", stderr="")

    result = execute_command({"id": "cmd_1", "type": "doctor"}, runner=runner, now=lambda: 10.0)

    assert result["succeeded"] is False
    assert result["error"]["reason"] == "malformed_json"


def test_doctor_timeout_fails():
    def runner(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, timeout=kwargs["timeout"], output="partial")

    result = execute_command(
        {"id": "cmd_1", "type": "doctor", "timeout_seconds": 1},
        runner=runner,
        now=lambda: 10.0,
    )

    assert result["succeeded"] is False
    assert result["error"]["reason"] == "timeout"
    assert result["stdout"] == "partial"


def test_doctor_process_start_failure_is_acked():
    def runner(argv, **kwargs):
        raise FileNotFoundError("missing tether")

    result = execute_command({"id": "cmd_1", "type": "doctor"}, runner=runner, now=lambda: 10.0)

    assert result["succeeded"] is False
    assert result["status"] == "failed"
    assert result["error"]["reason"] == "process_start_failed"
    assert result["error"]["argv"] == ["tether", "doctor", "--format", "json"]


def test_serve_status_no_server_is_graceful():
    def get_json(url, timeout):
        raise OSError("connection refused")

    result = execute_command(
        {"id": "cmd_1", "type": "serve_status"},
        http_get_json=get_json,
        now=lambda: 10.0,
    )

    assert result["succeeded"] is True
    assert result["output"]["reachable"] is False
    assert result["output"]["ready"] is False
    assert result["output"]["url"] == "http://127.0.0.1:8000"


def test_serve_status_ready_fake_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "state": "ready"}).encode())
                return
            if self.path == "/config":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"robot_id": "robot_1"}).encode())
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        result = execute_command(
            {"id": "cmd_1", "type": "serve_status", "serve_url": url},
            now=lambda: 10.0,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert result["succeeded"] is True
    assert result["output"]["reachable"] is True
    assert result["output"]["ready"] is True
    assert result["output"]["health"]["body"]["state"] == "ready"
    assert result["output"]["config"]["body"]["robot_id"] == "robot_1"


def test_serve_status_uses_local_api_key_and_sanitizes_response():
    calls = []

    def get_json(url, timeout, headers=None):
        calls.append((url, dict(headers or {})))
        if url.endswith("/health"):
            return 200, {"status": "ok", "state": "ready", "export_dir": "/tmp/secret/model"}
        assert headers["Authorization"] == "Bearer serve_secret"
        return 200, {
            "robot_id": "robot_1",
            "api_key": "serve_secret",
            "signed_url": "https://example.test/blob?token=secret",
        }

    result = execute_command(
        {
            "id": "cmd_1",
            "type": "serve_status",
            "serve_url": "http://127.0.0.1:9999",
            "local_serve_api_key": "serve_secret",
        },
        http_get_json=get_json,
        now=lambda: 10.0,
    )
    rendered = json.dumps(result, sort_keys=True)

    assert result["output"]["ready"] is True
    assert calls[1][1]["Authorization"] == "Bearer serve_secret"
    assert "serve_secret" not in rendered
    assert "signed_url" not in rendered
    assert "/tmp/secret/model" not in rendered
