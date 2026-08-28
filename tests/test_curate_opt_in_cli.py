"""Tests for src/tether/curate/opt_in_cli.py — `tether contribute` CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tether.curate.opt_in_cli import contribute_app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate $HOME so each test gets its own ~/.tether/."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TETHER_NO_UPGRADE_CHECK", "1")
    return tmp_path


def test_status_when_not_opted_in(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(contribute_app, ["--status"])
    assert result.exit_code == 0
    assert "not opted in" in result.output


def test_opt_in_creates_receipt(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(contribute_app, ["--opt-in"])
    assert result.exit_code == 0, result.output
    assert "Opted in" in result.output
    receipt = home / ".tether" / "consent.json"
    assert receipt.exists()
    data = json.loads(receipt.read_text())
    assert data["tier"] == "free"
    assert data["contributor_id"].startswith("ctr_")
    credentials = home / ".tether" / "contributor-auth-v1.json"
    assert credentials.exists()
    assert json.loads(credentials.read_text())["contributor_id"] == data["contributor_id"]


def test_opt_in_idempotent(runner: CliRunner, home: Path) -> None:
    runner.invoke(contribute_app, ["--opt-in"])
    result = runner.invoke(contribute_app, ["--opt-in"])
    assert result.exit_code == 0
    assert "Already opted in" in result.output


def test_status_shows_after_opt_in(runner: CliRunner, home: Path) -> None:
    runner.invoke(contribute_app, ["--opt-in"])
    result = runner.invoke(contribute_app, ["--status"])
    assert result.exit_code == 0
    assert "opted in" in result.output
    assert "free" in result.output


def test_opt_out_removes_receipt(runner: CliRunner, home: Path) -> None:
    runner.invoke(contribute_app, ["--opt-in"])
    receipt = home / ".tether" / "consent.json"
    assert receipt.exists()
    result = runner.invoke(contribute_app, ["--opt-out"])
    assert result.exit_code == 0
    assert not receipt.exists()
    assert "Opted out" in result.output


def test_opt_out_idempotent(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(contribute_app, ["--opt-out"])
    assert result.exit_code == 0
    assert "Already opted out" in result.output


def test_revoke_with_yes_flag(runner: CliRunner, home: Path) -> None:
    runner.invoke(contribute_app, ["--opt-in"])
    receipt = home / ".tether" / "consent.json"
    assert receipt.exists()
    result = runner.invoke(contribute_app, ["--revoke", "--yes"])
    assert result.exit_code == 0
    assert not receipt.exists()
    assert "Local consent receipt removed" in result.output
    assert "No server-side purge request was submitted" in result.output
    assert "Revocation submitted" not in result.output
    assert "will complete within 30 days" not in result.output


def test_revoke_preserves_legacy_and_authenticated_ids_for_admin_handoff(
    runner: CliRunner, home: Path,
) -> None:
    runner.invoke(contribute_app, ["--opt-in"])
    receipt_path = home / ".tether" / "consent.json"
    receipt = json.loads(receipt_path.read_text())
    authenticated_id = receipt["contributor_id"]
    historical_id = "free_legacy_history_12345678"
    receipt["contributor_id"] = historical_id
    receipt_path.write_text(json.dumps(receipt))

    result = runner.invoke(contribute_app, ["--revoke", "--yes"])
    assert result.exit_code == 0, result.output
    assert not receipt_path.exists()
    assert f"Contributor Auth ID: {authenticated_id}" in result.output
    assert f"Historical receipt ID: {historical_id}" in result.output
    assert "send both IDs" in result.output
    assert "No server-side purge request was submitted" in result.output


def test_revoke_when_not_opted_in(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(contribute_app, ["--revoke", "--yes"])
    assert result.exit_code == 0
    assert "nothing to revoke" in result.output


def test_info_shows_privacy(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(contribute_app, ["--info"])
    assert result.exit_code == 0
    assert "privacy" in result.output.lower()
    assert "GDPR" in result.output or "revoke" in result.output.lower()


def test_mutually_exclusive_flags_rejected(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(contribute_app, ["--opt-in", "--opt-out"])
    assert result.exit_code != 0
    assert "Pick one" in result.output
