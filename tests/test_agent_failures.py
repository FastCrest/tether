from __future__ import annotations

from tether.agent.failures import build_failure_from_command_result


class Config:
    agent_version = "0.1.0"
    tether_version = "0.12.0"


def test_failed_doctor_result_builds_bounded_failure_payload():
    command = {"id": "cmd_1", "type": "doctor"}
    result = {
        "command_id": "cmd_1",
        "command_type": "doctor",
        "succeeded": False,
        "status": "failed",
        "started_at": 10.0,
        "finished_at": 12.0,
        "stdout": "raw stdout with dvc_test_SECRETSECRET should not ship",
        "stderr": "/Users/example/private/path should not ship",
        "output": {"summary": {"pass": 2, "fail": 1, "warn": 0, "skip": 0}},
        "error": {"reason": "doctor_exit_nonzero", "message": "failed at /Users/example/private/path"},
    }

    payload = build_failure_from_command_result(command, result, config=Config())

    assert payload is not None
    body = payload.to_dict()
    assert body["event_type"] == "diagnostic_failure"
    assert body["severity"] == "critical"
    assert body["do_not_train"] is True
    assert body["started_at"] == 10.0
    assert body["ended_at"] == 12.0
    assert "workspace_id" not in body
    assert "device_id" not in body
    assert "retention_days" not in body
    assert body["metadata"] == {
        "command_id": "cmd_1",
        "command_type": "doctor",
        "status": "failed",
        "agent_version": "0.1.0",
        "tether_version": "0.12.0",
    }
    assert body["diagnostic"]["error_code"] == "doctor_exit_nonzero"
    assert body["diagnostic"]["doctor"]["summary"]["fail"] == 1
    assert "stdout" not in str(body)
    assert "stderr" not in str(body)
    assert "SECRETSECRET" not in str(body)
    assert "/Users/example" not in str(body)


def test_successful_noop_does_not_emit_failure():
    payload = build_failure_from_command_result(
        {"id": "cmd_1", "type": "noop"},
        {"command_id": "cmd_1", "command_type": "noop", "succeeded": True, "status": "succeeded"},
    )

    assert payload is None


def test_unready_serve_status_emits_serve_unavailable():
    payload = build_failure_from_command_result(
        {"id": "cmd_2", "type": "serve_status"},
        {
            "command_id": "cmd_2",
            "command_type": "serve_status",
            "succeeded": True,
            "status": "succeeded",
            "output": {
                "reachable": True,
                "ready": False,
                "health": {"ok": True, "status_code": 200},
                "config": {"ok": False, "status_code": 503, "error": {"reason": "not_ready"}},
                "url": "http://127.0.0.1:8000",
            },
        },
    )

    assert payload is not None
    body = payload.to_dict()
    assert body["event_type"] == "serve_unavailable"
    assert body["severity"] == "warning"
    assert body["diagnostic"]["error_code"] == "serve_not_ready"
    assert "127.0.0.1" not in str(body)


def test_oversized_diagnostic_is_truncated():
    payload = build_failure_from_command_result(
        {"id": "cmd_3", "type": "doctor"},
        {
            "command_id": "cmd_3",
            "command_type": "doctor",
            "succeeded": False,
            "status": "failed",
            "error": {"reason": "x" * 100_000},
        },
    )

    assert payload is not None
    body = payload.to_dict()
    encoded = str(body["diagnostic"]).encode("utf-8")
    assert len(encoded) < 16 * 1024
