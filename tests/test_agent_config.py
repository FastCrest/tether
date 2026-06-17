from __future__ import annotations

import os
import stat

from tether.agent.config import AgentConfig, load_config, save_config
from tether.agent.models import CommandResult


def test_config_round_trip_with_override_path(tmp_path):
    path = tmp_path / "nested" / "agent.json"
    config = AgentConfig(
        device_id="dev_123",
        cloud_url="https://cloud.example.test",
        workspace_id="ws_123",
        device_token="tok_secret",
        fleet_device_id="dev_fleet_123",
        fleet_device_token="dvc_test_secret",
        local_serve_url="http://127.0.0.1:8181",
        local_serve_api_key="serve_secret",
        local_serve_timeout_seconds=2.5,
        heartbeat_interval_seconds=45,
        last_heartbeat_at="2026-06-04T12:00:00Z",
        last_command_id="cmd_1",
        last_command_result=CommandResult(succeeded=True, output={"kind": "noop"}),
    )

    saved_path = save_config(config, path)
    loaded = load_config(path)

    assert saved_path == path
    assert loaded == config


def test_config_repr_omits_token_like_values():
    config = AgentConfig(
        device_token="tok_secret",
        fleet_device_token="dvc_test_secret",
        local_serve_api_key="serve_secret",
    )

    rendered = repr(config)

    assert "tok_secret" not in rendered
    assert "dvc_test_secret" not in rendered
    assert "serve_secret" not in rendered


def test_load_config_returns_none_when_missing(tmp_path):
    assert load_config(tmp_path / "missing.json") is None


def test_config_permissions_are_restricted_on_posix(tmp_path):
    if os.name != "posix":
        return
    path = tmp_path / "agent.json"

    save_config(AgentConfig(device_token="tok_secret"), path)

    file_mode = stat.S_IMODE(path.stat().st_mode)
    dir_mode = stat.S_IMODE(path.parent.stat().st_mode)
    assert file_mode & 0o077 == 0
    assert dir_mode & 0o077 == 0
