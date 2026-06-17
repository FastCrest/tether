"""Tests for the integration framework (tether connect)."""
from __future__ import annotations

import pytest

from tether.integrations.registry import (
    Integration,
    get_integration,
    list_integrations,
)


class TestRegistry:
    def test_list_non_empty(self):
        assert len(list_integrations()) >= 1

    def test_rtsm_exists(self):
        rtsm = get_integration("rtsm")
        assert rtsm is not None
        assert rtsm.name == "rtsm"

    def test_unknown_returns_none(self):
        assert get_integration("nonexistent_xyz") is None

    def test_rtsm_pip_spec(self):
        rtsm = get_integration("rtsm")
        assert rtsm.pip_spec == "rtsm[gpu]"

    def test_rtsm_mcp_tools(self):
        rtsm = get_integration("rtsm")
        assert len(rtsm.mcp_tools) == 6
        assert "rtsm.semantic_query" in rtsm.mcp_tools
        assert "rtsm.spatial_query" in rtsm.mcp_tools

    def test_rtsm_health_fails_when_not_running(self):
        rtsm = get_integration("rtsm")
        assert rtsm.health_check(timeout=0.5) is False

    def test_integration_frozen(self):
        rtsm = get_integration("rtsm")
        with pytest.raises(AttributeError):
            rtsm.name = "something_else"


class TestConnector:
    def test_connect_unknown_raises(self):
        from tether.integrations.connector import connect
        with pytest.raises(ValueError, match="Unknown integration"):
            connect("nonexistent_xyz")

    def test_disconnect_not_running(self):
        from tether.integrations.connector import disconnect
        result = disconnect("rtsm")
        assert result["status"] == "not_running"


class TestCli:
    @pytest.fixture
    def runner(self):
        typer_testing = pytest.importorskip("typer.testing")
        return typer_testing.CliRunner()

    @pytest.fixture
    def cli_app(self):
        from tether.cli import app
        return app

    def test_connect_list(self, runner, cli_app):
        result = runner.invoke(cli_app, ["connect", "list"])
        assert result.exit_code == 0
        assert "rtsm" in result.output

    def test_connect_status_rtsm(self, runner, cli_app):
        result = runner.invoke(cli_app, ["connect", "status", "rtsm"])
        assert result.exit_code == 0
        assert "Installed" in result.output

    def test_connect_status_unknown(self, runner, cli_app):
        result = runner.invoke(cli_app, ["connect", "status", "nonexistent_xyz"])
        assert result.exit_code == 2

    def test_connect_up_unknown(self, runner, cli_app):
        result = runner.invoke(cli_app, ["connect", "up", "nonexistent_xyz"])
        assert result.exit_code == 2
