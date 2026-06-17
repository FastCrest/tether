"""Tests for the `tether eval` CLI verb — Phase 1 eval-as-a-service Day 3."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from tether.cli import app
from tether.eval.preflight import PreflightResult


runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers — stub the preflight subprocess so we don't actually fork python
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_preflight_pass(monkeypatch):
    """Force PreflightSmokeTest.run to return passed=True."""
    def _stub(*args, **kwargs):
        return PreflightResult(
            passed=True, failure_mode="ok", elapsed_s=0.5,
            stdout="PREFLIGHT_OK\n", stderr="", remediation="",
        )

    monkeypatch.setattr(
        "tether.eval.preflight.PreflightSmokeTest.run", _stub,
    )


@pytest.fixture
def stub_preflight_fail(monkeypatch):
    """Force PreflightSmokeTest.run to return failed."""
    def _stub(*args, **kwargs):
        return PreflightResult(
            passed=False, failure_mode="dep-version-conflict",
            elapsed_s=2.1, stdout="",
            stderr="PREFLIGHT_FAILURE_MODE=dep-version-conflict: bad pin\n",
            remediation="Pin: robosuite==1.4.1, bddl==1.0.1, mujoco==3.3.2",
        )

    monkeypatch.setattr(
        "tether.eval.preflight.PreflightSmokeTest.run", _stub,
    )


@pytest.fixture
def fake_export(tmp_path):
    """Create a stub export directory (just an empty dir is fine for the CLI)."""
    p = tmp_path / "export"
    p.mkdir()
    (p / "model.onnx").write_bytes(b"fake")
    return p


@pytest.fixture
def stub_modal_runner(monkeypatch):
    """Stub modal_runner.run_libero_on_modal so tests don't hit real Modal.
    Returns 1 episode per task with terminal_reason='adapter_error' to
    preserve the Day 3 "exit 5 on all-adapter-error" CLI semantics."""
    from tether.eval.libero import EpisodeResult

    def _stub(*, config, export_dir, **kwargs):
        # Synthesize 1 EpisodeResult per task in config.tasks
        out = []
        for task in config.tasks:
            out.append(EpisodeResult(
                task_id=task, episode_index=0,
                success=False, terminal_reason="adapter_error",
                wall_clock_s=0.0, n_steps=0,
                video_path=None, error_message="stubbed in tests",
            ))
        return out

    monkeypatch.setattr(
        "tether.eval.modal_runner.run_libero_on_modal", _stub,
    )


# ---------------------------------------------------------------------------
# Validation — fail-loud on bad inputs
# ---------------------------------------------------------------------------


def test_eval_rejects_missing_export_dir(tmp_path):
    result = runner.invoke(
        app, ["eval", str(tmp_path / "does-not-exist")],
    )
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_eval_rejects_unknown_suite(fake_export):
    result = runner.invoke(
        app, ["eval", str(fake_export), "--suite", "fictional"],
    )
    assert result.exit_code == 2
    assert "Unknown suite" in result.output


def test_eval_rejects_unknown_runtime(fake_export):
    result = runner.invoke(
        app, ["eval", str(fake_export), "--runtime", "kubernetes"],
    )
    assert result.exit_code == 2
    assert "Unknown runtime" in result.output


def test_eval_rejects_zero_episodes(fake_export):
    result = runner.invoke(
        app, ["eval", str(fake_export), "--num-episodes", "0"],
    )
    assert result.exit_code == 2
    assert "Invalid configuration" in result.output


def test_eval_rejects_zero_max_parallel(fake_export):
    result = runner.invoke(
        app, ["eval", str(fake_export), "--max-parallel", "0"],
    )
    assert result.exit_code == 2
    assert "Invalid configuration" in result.output


# ---------------------------------------------------------------------------
# Banner echo
# ---------------------------------------------------------------------------


def test_eval_prints_banner_with_inputs(fake_export, stub_preflight_pass, stub_modal_runner):
    result = runner.invoke(
        app, ["eval", str(fake_export), "--suite", "libero",
              "--runtime", "modal", "--num-episodes", "2",
              "--seed", "42", "--tasks", "libero_spatial,libero_object"],
    )
    # Stub runners produce all-adapter-error → exit 5; that's expected Day 3
    assert "Tether Eval" in result.output
    assert "libero_spatial" in result.output
    assert "libero_object" in result.output
    assert "Seed:" in result.output
    assert "42" in result.output


def test_eval_banner_marks_video_when_set(fake_export, stub_preflight_pass, stub_modal_runner):
    """--video is honest about Phase 2 deferral (encoder ready; frame
    capture wires Phase 2 once modal_libero scripts surface frames)."""
    result = runner.invoke(
        app, ["eval", str(fake_export), "--video"],
    )
    assert "Video:" in result.output
    assert "Phase 2" in result.output


# ---------------------------------------------------------------------------
# --cost-preview short-circuit
# ---------------------------------------------------------------------------


def test_cost_preview_exits_clean_without_running_preflight(fake_export, monkeypatch):
    """Cost-preview must NOT invoke preflight (no real run = no need to probe)."""
    called = {"n": 0}

    def _stub(*args, **kwargs):
        called["n"] += 1
        return PreflightResult(
            passed=True, failure_mode="ok", elapsed_s=0.5,
            stdout="PREFLIGHT_OK\n", stderr="", remediation="",
        )

    monkeypatch.setattr(
        "tether.eval.preflight.PreflightSmokeTest.run", _stub,
    )
    result = runner.invoke(
        app, ["eval", str(fake_export), "--cost-preview", "--num-episodes", "5"],
    )
    assert result.exit_code == 0
    assert "Cost preview" in result.output
    assert called["n"] == 0  # preflight skipped


def test_cost_preview_shows_dollar_total(fake_export):
    """Day 4 cost_model wire — surfaces total $ cost, not episode count."""
    result = runner.invoke(
        app, ["eval", str(fake_export), "--cost-preview",
              "--num-episodes", "10", "--tasks", "task_a,task_b,task_c"],
    )
    assert result.exit_code == 0
    # 3 tasks × (10 × $0.025 + $0.10) = $1.05; surfaces as "$1.05"
    assert "$1.05" in result.output or "$1.0" in result.output
    assert "Total estimate" in result.output


def test_cost_preview_uses_default_tasks_when_unspecified(fake_export):
    """Default LIBERO task list (4 families Phase 1) — Day 4 output."""
    result = runner.invoke(
        app, ["eval", str(fake_export), "--cost-preview", "--num-episodes", "5"],
    )
    assert result.exit_code == 0
    # 4 tasks × (5 × $0.025 + $0.10) = $0.90
    assert "$0.90" in result.output or "$0.9" in result.output
    # Banner echoes 4 tasks
    assert "4 tasks" in result.output


def test_cost_preview_local_runtime_is_zero(fake_export):
    result = runner.invoke(
        app, ["eval", str(fake_export), "--cost-preview",
              "--runtime", "local", "--num-episodes", "100"],
    )
    assert result.exit_code == 0
    assert "$0.00" in result.output
    assert "Local" in result.output


def test_cost_preview_warns_above_guardrail(fake_export):
    """Massive --num-episodes should trigger the >$50 warning."""
    result = runner.invoke(
        app, ["eval", str(fake_export), "--cost-preview",
              "--num-episodes", "1000",
              "--tasks", "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t"],
    )
    assert result.exit_code == 0
    # 20 tasks × (1000 × $0.025 + $0.10) = 20 × $25.10 = $502.00
    assert "guardrail" in result.output.lower() or "$50" in result.output


# ---------------------------------------------------------------------------
# Pre-flight integration
# ---------------------------------------------------------------------------


def test_eval_aborts_on_preflight_failure(fake_export, stub_preflight_fail):
    result = runner.invoke(app, ["eval", str(fake_export)])
    assert result.exit_code == 4
    assert "Pre-flight FAILED" in result.output
    assert "dep-version-conflict" in result.output
    assert "robosuite==1.4.1" in result.output  # remediation surfaced


def test_eval_continues_on_preflight_pass(fake_export, stub_preflight_pass, stub_modal_runner):
    result = runner.invoke(app, ["eval", str(fake_export)])
    # Pre-flight OK + Day 3 stub runners → exit 5 (all adapter_error)
    assert "Pre-flight OK" in result.output
    assert result.exit_code == 5


def test_eval_passes_preflight_timeout_through(fake_export, monkeypatch, stub_modal_runner):
    captured = {}

    def _stub(*, timeout_s, **kwargs):
        captured["timeout_s"] = timeout_s
        return PreflightResult(
            passed=True, failure_mode="ok", elapsed_s=0.1,
            stdout="PREFLIGHT_OK\n", stderr="", remediation="",
        )

    monkeypatch.setattr(
        "tether.eval.preflight.PreflightSmokeTest.run", _stub,
    )
    runner.invoke(
        app, ["eval", str(fake_export), "--preflight-timeout", "120"],
    )
    assert captured["timeout_s"] == 120.0


# ---------------------------------------------------------------------------
# Day 3 stub runner honesty — exit 5 when all-adapter-error
# ---------------------------------------------------------------------------


def test_day3_stub_runner_exits_5_on_all_adapter_error(fake_export, stub_preflight_pass, stub_modal_runner):
    """Day 3 ships stub runners — every episode is adapter_error. CLI exits 5
    so CI doesn't mistake substrate-only for success."""
    result = runner.invoke(
        app, ["eval", str(fake_export),
              "--tasks", "libero_spatial", "--num-episodes", "2"],
    )
    assert result.exit_code == 5
    assert "adapter_error" in result.output.lower()
    assert "Day" in result.output  # mentions deferral


def test_day3_emits_per_task_table(fake_export, stub_preflight_pass, stub_modal_runner):
    result = runner.invoke(
        app, ["eval", str(fake_export),
              "--tasks", "libero_spatial,libero_object", "--num-episodes", "1"],
    )
    # Per-task table should render even on all-adapter-error
    assert "Per-task results" in result.output
    assert "libero_spatial" in result.output
    assert "libero_object" in result.output


def test_day3_creates_output_directory(fake_export, stub_preflight_pass, stub_modal_runner, tmp_path):
    out = tmp_path / "my-eval-output"
    runner.invoke(
        app, ["eval", str(fake_export),
              "--tasks", "libero_spatial", "--num-episodes", "1",
              "--output", str(out)],
    )
    assert out.exists() and out.is_dir()


def test_day4_writes_json_envelope_to_output(fake_export, stub_preflight_pass, stub_modal_runner, tmp_path):
    """Day 4 wires build_envelope + write_json — report.json appears in --output."""
    import json
    out = tmp_path / "eval-out"
    runner.invoke(
        app, ["eval", str(fake_export),
              "--tasks", "libero_spatial", "--num-episodes", "1",
              "--output", str(out)],
    )
    envelope_path = out / "report.json"
    assert envelope_path.exists()
    parsed = json.loads(envelope_path.read_text())
    assert parsed["schema_version"] == 1
    assert parsed["suite"] == "libero"
    assert parsed["runtime"] == "modal"
    assert "cost" in parsed
    assert parsed["cost"]["total_usd"] >= 0
    assert "env" in parsed
    assert "modal" in parsed
    assert parsed["modal"] is not None  # runtime=modal → block populated


# ---------------------------------------------------------------------------
# Comma-separated tasks parsing
# ---------------------------------------------------------------------------


def test_tasks_csv_strips_whitespace(fake_export, stub_preflight_pass, stub_modal_runner):
    result = runner.invoke(
        app, ["eval", str(fake_export),
              "--tasks", " libero_spatial , libero_object ",
              "--num-episodes", "1"],
    )
    assert "libero_spatial" in result.output
    assert "libero_object" in result.output


def test_tasks_csv_drops_empty_entries(fake_export, stub_preflight_pass, stub_modal_runner):
    result = runner.invoke(
        app, ["eval", str(fake_export),
              "--tasks", "libero_spatial,,libero_object,",
              "--num-episodes", "1"],
    )
    # Both real tasks present in per-task table; no empty rows
    assert "libero_spatial" in result.output
    assert "libero_object" in result.output


# ---------------------------------------------------------------------------
# resolve_task_runner unit tests
# ---------------------------------------------------------------------------


def test_resolve_task_runner_rejects_unknown_runtime(tmp_path):
    from tether.eval.runner_dispatch import resolve_task_runner
    with pytest.raises(ValueError, match="runtime must be one of"):
        resolve_task_runner(runtime="kubernetes", export_dir=tmp_path)


def test_resolve_task_runner_modal_returns_callable(tmp_path):
    from tether.eval.runner_dispatch import resolve_task_runner
    fn = resolve_task_runner(runtime="modal", export_dir=tmp_path)
    assert callable(fn)


def test_resolve_task_runner_local_returns_callable(tmp_path):
    from tether.eval.runner_dispatch import resolve_task_runner
    fn = resolve_task_runner(runtime="local", export_dir=tmp_path)
    assert callable(fn)


def test_modal_stub_runner_emits_adapter_error(tmp_path):
    """Modal per-episode stub is back-compat only -- real Modal callers
    use resolve_suite_runner. Verifies the per-episode shape still
    surfaces a structured adapter_error row."""
    from tether.eval.libero import LiberoSuiteConfig
    from tether.eval.runner_dispatch import resolve_task_runner

    fn = resolve_task_runner(runtime="modal", export_dir=tmp_path)
    config = LiberoSuiteConfig(num_episodes=1, tasks=("libero_spatial",))
    result = fn("libero_spatial", 0, config)
    assert result.terminal_reason == "adapter_error"
    assert not result.success
    assert "deprecated" in result.error_message
    assert "resolve_suite_runner" in result.error_message


def test_local_stub_runner_emits_adapter_error(tmp_path):
    from tether.eval.libero import LiberoSuiteConfig
    from tether.eval.runner_dispatch import resolve_task_runner

    fn = resolve_task_runner(runtime="local", export_dir=tmp_path)
    config = LiberoSuiteConfig(num_episodes=1, tasks=("libero_spatial",), runtime="local")
    result = fn("libero_spatial", 0, config)
    assert result.terminal_reason == "adapter_error"
    assert not result.success
    assert "Phase 1 follow-up" in result.error_message


def test_default_libero_tasks_returns_phase1_list():
    from tether.eval.runner_dispatch import (
        LIBERO_DEFAULT_TASKS_PHASE1,
        default_libero_tasks,
    )
    tasks = default_libero_tasks()
    assert tasks == list(LIBERO_DEFAULT_TASKS_PHASE1)
    # Phase 1 ships the 4 LIBERO families
    assert len(tasks) == 4
