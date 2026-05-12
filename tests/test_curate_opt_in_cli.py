"""Tests for src/reflex/curate/opt_in_cli.py — `reflex contribute` CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from reflex.curate import consent as curate_consent
from reflex.curate.opt_in_cli import contribute_app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate $HOME so each test gets its own ~/.reflex/."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("REFLEX_NO_UPGRADE_CHECK", "1")
    return tmp_path


def test_status_when_not_opted_in(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(contribute_app, ["--status"])
    assert result.exit_code == 0
    assert "not opted in" in result.stdout


def test_opt_in_creates_receipt(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(contribute_app, ["--opt-in"])
    assert result.exit_code == 0, result.stdout
    assert "Opted in" in result.stdout
    receipt = home / ".reflex" / "consent.json"
    assert receipt.exists()
    data = json.loads(receipt.read_text())
    assert data["tier"] == "free"
    assert data["contributor_id"].startswith("free_")


def test_opt_in_idempotent(runner: CliRunner, home: Path) -> None:
    runner.invoke(contribute_app, ["--opt-in"])
    result = runner.invoke(contribute_app, ["--opt-in"])
    assert result.exit_code == 0
    assert "Already opted in" in result.stdout


def test_status_shows_after_opt_in(runner: CliRunner, home: Path) -> None:
    runner.invoke(contribute_app, ["--opt-in"])
    result = runner.invoke(contribute_app, ["--status"])
    assert result.exit_code == 0
    assert "opted in" in result.stdout
    assert "free" in result.stdout


def test_opt_out_removes_receipt(runner: CliRunner, home: Path) -> None:
    runner.invoke(contribute_app, ["--opt-in"])
    receipt = home / ".reflex" / "consent.json"
    assert receipt.exists()
    result = runner.invoke(contribute_app, ["--opt-out"])
    assert result.exit_code == 0
    assert not receipt.exists()
    assert "Opted out" in result.stdout


def test_opt_out_idempotent(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(contribute_app, ["--opt-out"])
    assert result.exit_code == 0
    assert "Already opted out" in result.stdout


def test_revoke_with_yes_flag(runner: CliRunner, home: Path) -> None:
    runner.invoke(contribute_app, ["--opt-in"])
    receipt = home / ".reflex" / "consent.json"
    assert receipt.exists()
    result = runner.invoke(contribute_app, ["--revoke", "--yes"])
    assert result.exit_code == 0
    assert not receipt.exists()
    assert "Revocation submitted" in result.stdout


def test_revoke_when_not_opted_in(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(contribute_app, ["--revoke", "--yes"])
    assert result.exit_code == 0
    assert "nothing to revoke" in result.stdout


def test_info_shows_privacy(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(contribute_app, ["--info"])
    assert result.exit_code == 0
    assert "privacy" in result.stdout.lower()
    assert "GDPR" in result.stdout or "revoke" in result.stdout.lower()


def test_mutually_exclusive_flags_rejected(runner: CliRunner, home: Path) -> None:
    result = runner.invoke(contribute_app, ["--opt-in", "--opt-out"])
    assert result.exit_code != 0
    assert "Pick one" in result.stdout
