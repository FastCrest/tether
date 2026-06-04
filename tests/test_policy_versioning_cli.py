"""Tests for the `tether serve --policy-a/--policy-b/--split` flags (Day 5).

Per ADR 2026-04-25-policy-versioning-architecture: validation-only Phase 1.
Full 2-policy serving lands Days 9-10 integration; this commit ships the
flag surface + mutually-exclusive validation + memory-refuse-to-load
hook.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tether.cli import app


runner = CliRunner()


@pytest.fixture
def fake_export(tmp_path):
    p = tmp_path / "export"
    p.mkdir()
    (p / "model.onnx").write_bytes(b"fake")
    (p / "tether_config.json").write_text("{}")
    return p


@pytest.fixture
def fake_export_b(tmp_path):
    p = tmp_path / "export_b"
    p.mkdir()
    (p / "model.onnx").write_bytes(b"fake")
    (p / "tether_config.json").write_text("{}")
    return p


# ---------------------------------------------------------------------------
# Mutually-exclusive validation
# ---------------------------------------------------------------------------


def test_policy_a_alone_fails(fake_export, fake_export_b):
    """--policy-a without --policy-b is invalid."""
    result = runner.invoke(
        app, ["serve", str(fake_export), "--policy-a", str(fake_export)],
    )
    assert result.exit_code == 1
    assert "must be set together" in result.output


def test_policy_b_alone_fails(fake_export, fake_export_b):
    """--policy-b without --policy-a is invalid."""
    result = runner.invoke(
        app, ["serve", str(fake_export), "--policy-b", str(fake_export_b)],
    )
    assert result.exit_code == 1
    assert "must be set together" in result.output


def test_2policy_mutually_exclusive_with_shadow(fake_export, fake_export_b, tmp_path):
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    result = runner.invoke(
        app, [
            "serve", str(fake_export),
            "--policy-a", str(fake_export),
            "--policy-b", str(fake_export_b),
            "--shadow-policy", str(shadow),
            "--no-rtc",
        ],
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


# ---------------------------------------------------------------------------
# --no-rtc enforcement in 2-policy mode
# ---------------------------------------------------------------------------


def test_2policy_requires_no_rtc(fake_export, fake_export_b):
    """2-policy mode without --no-rtc must fail."""
    result = runner.invoke(
        app, [
            "serve", str(fake_export),
            "--policy-a", str(fake_export),
            "--policy-b", str(fake_export_b),
        ],
    )
    assert result.exit_code == 1
    assert "--no-rtc" in result.output


# ---------------------------------------------------------------------------
# --split bounds
# ---------------------------------------------------------------------------


def test_split_negative_rejected(fake_export, fake_export_b):
    result = runner.invoke(
        app, [
            "serve", str(fake_export),
            "--policy-a", str(fake_export),
            "--policy-b", str(fake_export_b),
            "--split", "-1",
            "--no-rtc",
        ],
    )
    assert result.exit_code == 1
    assert "split_a_percent" in result.output


def test_split_over_hundred_rejected(fake_export, fake_export_b):
    result = runner.invoke(
        app, [
            "serve", str(fake_export),
            "--policy-a", str(fake_export),
            "--policy-b", str(fake_export_b),
            "--split", "150",
            "--no-rtc",
        ],
    )
    assert result.exit_code == 1
    assert "split_a_percent" in result.output


# ---------------------------------------------------------------------------
# Missing export path detection
# ---------------------------------------------------------------------------


def test_missing_policy_a_path_rejected(fake_export, fake_export_b, tmp_path):
    result = runner.invoke(
        app, [
            "serve", str(fake_export),
            "--policy-a", str(tmp_path / "does-not-exist"),
            "--policy-b", str(fake_export_b),
            "--no-rtc",
        ],
    )
    assert result.exit_code == 1
    assert "--policy-a" in result.output
    assert "not found" in result.output


def test_missing_policy_b_path_rejected(fake_export, fake_export_b, tmp_path):
    result = runner.invoke(
        app, [
            "serve", str(fake_export),
            "--policy-a", str(fake_export),
            "--policy-b", str(tmp_path / "does-not-exist"),
            "--no-rtc",
        ],
    )
    assert result.exit_code == 1
    assert "--policy-b" in result.output
    assert "not found" in result.output


# ---------------------------------------------------------------------------
# Shadow flag (Phase 1.5; shipped inert)
# ---------------------------------------------------------------------------


def test_shadow_policy_logs_phase15_warning(fake_export, tmp_path, monkeypatch):
    """--shadow-policy is accepted in single-policy mode but logs a 'shipped
    inert' warning. Skipping actual server-load (would need full deps)."""
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    # Force the server-load path to fail fast so we just observe the
    # validation + banner output. Patch TetherServer.load to raise.

    def _fake_load(self):
        raise SystemExit(0)  # short-circuit before real model load

    monkeypatch.setattr(
        "tether.runtime.server.TetherServer.load", _fake_load,
    )
    result = runner.invoke(
        app, [
            "serve", str(fake_export),
            "--shadow-policy", str(shadow),
        ],
    )
    # The warning text should appear before SystemExit takes us out
    assert "shadow-policy" in result.output.lower()
    assert "phase 1.5" in result.output.lower() or "inert" in result.output.lower()


# ---------------------------------------------------------------------------
# Happy-path 2-policy validation accepted (deferred actual serve)
# ---------------------------------------------------------------------------


def test_2policy_valid_combo_surfaces_active_banner(fake_export, fake_export_b, monkeypatch):
    """--policy-a + --policy-b + --no-rtc + --split=50 -> validation passes,
    surfaces the '2-policy mode active' banner, then proceeds to load.
    Per ADR Day 9-10 integration: full 2-policy serving is now wired
    via setup_two_policy_serving + create_app. TetherServer.load is
    stubbed to short-circuit before the actual model load."""
    def _fake_load(self):
        raise SystemExit(0)

    monkeypatch.setattr(
        "tether.runtime.server.TetherServer.load", _fake_load,
    )
    result = runner.invoke(
        app, [
            "serve", str(fake_export),
            "--policy-a", str(fake_export),
            "--policy-b", str(fake_export_b),
            "--split", "50",
            "--no-rtc",
        ],
    )
    assert "2-policy mode active" in result.output
    assert "--no-rtc enforced" in result.output
