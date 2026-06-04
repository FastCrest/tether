"""Tests for the `tether agent` CLI surface."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tether.cli import app


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    monkeypatch.setenv("TETHER_NO_UPGRADE_CHECK", "1")
    return CliRunner()


@pytest.fixture
def fake_agent_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    state: dict[str, object] = {
        "config": None,
        "saved": [],
        "client_inits": [],
        "enroll_tokens": [],
        "run_once": [],
        "run_loop": [],
    }

    agent_pkg = types.ModuleType("tether.agent")
    config_mod = types.ModuleType("tether.agent.config")
    client_mod = types.ModuleType("tether.agent.client")
    daemon_mod = types.ModuleType("tether.agent.daemon")

    def default_config_path() -> Path:
        return Path("/tmp/tether-agent.json")

    def load_config(path: Path | None = None) -> dict[str, object]:
        if state["config"] is None:
            raise FileNotFoundError(path or default_config_path())
        return state["config"]  # type: ignore[return-value]

    def save_config(cfg: dict[str, object], path: Path | None = None) -> None:
        state["config"] = cfg
        state["saved"].append((cfg, path))  # type: ignore[union-attr]

    class AgentClient:
        def __init__(
            self,
            cloud_url: str,
            device_token: str | None = None,
        ) -> None:
            state["client_inits"].append((cloud_url, device_token))  # type: ignore[union-attr]
            self.cloud_url = cloud_url

        def enroll(self, enroll_token: object) -> dict[str, object]:
            token = getattr(enroll_token, "enroll_token", enroll_token)
            state["enroll_tokens"].append(token)  # type: ignore[union-attr]
            return {
                "device_id": "dev_123",
                "device_token": "tok_123",
                "cloud_url": self.cloud_url,
                "workspace_id": "ws_123",
                "heartbeat_interval_seconds": 30,
            }

    def run_once(cfg: dict[str, object]) -> None:
        state["run_once"].append(cfg)  # type: ignore[union-attr]

    def run_loop(cfg: dict[str, object]) -> None:
        state["run_loop"].append(cfg)  # type: ignore[union-attr]

    config_mod.default_config_path = default_config_path
    config_mod.load_config = load_config
    config_mod.save_config = save_config
    client_mod.AgentClient = AgentClient
    daemon_mod.run_once = run_once
    daemon_mod.run_loop = run_loop
    agent_pkg.config = config_mod
    agent_pkg.client = client_mod
    agent_pkg.daemon = daemon_mod

    monkeypatch.setitem(sys.modules, "tether.agent", agent_pkg)
    monkeypatch.setitem(sys.modules, "tether.agent.config", config_mod)
    monkeypatch.setitem(sys.modules, "tether.agent.client", client_mod)
    monkeypatch.setitem(sys.modules, "tether.agent.daemon", daemon_mod)
    return state


def test_agent_status_missing_config(
    runner: CliRunner,
    fake_agent_modules: dict[str, object],
) -> None:
    result = runner.invoke(app, ["agent", "status"])

    assert result.exit_code == 1
    assert "No agent config found" in result.output


def test_agent_status_json(
    runner: CliRunner,
    fake_agent_modules: dict[str, object],
) -> None:
    fake_agent_modules["config"] = {
        "device_id": "dev_123",
        "cloud_url": "https://cloud.example.test",
        "workspace_id": "ws_123",
        "heartbeat_interval_seconds": 30,
        "last_heartbeat_at": "2026-06-04T12:00:00Z",
        "last_command_id": "cmd_123",
        "last_command_result": "succeeded",
    }

    result = runner.invoke(app, ["agent", "status", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "device_id": "dev_123",
        "cloud_url": "https://cloud.example.test",
        "workspace_id": "ws_123",
        "heartbeat_interval_seconds": 30,
        "last_heartbeat_at": "2026-06-04T12:00:00Z",
        "last_command_id": "cmd_123",
        "last_command_result": "succeeded",
    }


def test_agent_start_once_enrollment_path(
    runner: CliRunner,
    fake_agent_modules: dict[str, object],
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "agent.json"

    result = runner.invoke(
        app,
        [
            "agent",
            "start",
            "--cloud",
            "https://cloud.example.test",
            "--enroll-token",
            "enroll_123",
            "--config",
            str(config_path),
            "--once",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_agent_modules["enroll_tokens"] == ["enroll_123"]
    assert fake_agent_modules["saved"] == [
        (
            {
                "device_id": "dev_123",
                "device_token": "tok_123",
                "cloud_url": "https://cloud.example.test",
                "workspace_id": "ws_123",
                "heartbeat_interval_seconds": 30,
            },
            config_path,
        )
    ]
    assert fake_agent_modules["run_once"] == [fake_agent_modules["config"]]
    assert "Enrolled device dev_123" in result.output
    assert "Agent cycle complete" in result.output


def test_agent_run_once_path(
    runner: CliRunner,
    fake_agent_modules: dict[str, object],
    tmp_path: Path,
) -> None:
    cfg = {
        "device_id": "dev_existing",
        "cloud_url": "https://cloud.example.test",
        "device_token": "tok_existing",
    }
    fake_agent_modules["config"] = cfg

    result = runner.invoke(
        app,
        ["agent", "run-once", "--config", str(tmp_path / "agent.json")],
    )

    assert result.exit_code == 0, result.output
    assert fake_agent_modules["run_once"] == [cfg]
    assert "Agent cycle complete" in result.output
