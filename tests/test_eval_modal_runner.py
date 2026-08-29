"""Tests for src/tether/eval/modal_runner.py — Modal subprocess wrapper."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tether.eval.libero import LiberoSuiteConfig
from tether.eval.modal_runner import (
    TASK_SUITE_MAX_STEPS,
    ModalInvocationResult,
    ModalNotInstalledError,
    _modal_onnx_subdir_for_export,
    _parse_invocation_to_episodes,
    _parse_modal_stdout,
    run_libero_on_modal,
)


# ---------------------------------------------------------------------------
# _parse_modal_stdout — pinned against the existing script's output format
# ---------------------------------------------------------------------------


def test_parse_returns_none_for_empty_stdout():
    assert _parse_modal_stdout("", suite="libero_spatial") is None


def test_parse_returns_none_when_no_summary_header():
    stdout = "Some logs but no end-of-suite summary"
    assert _parse_modal_stdout(stdout, suite="libero_spatial") is None


def test_parse_returns_none_when_header_present_but_no_success_line():
    stdout = "====== libero_spatial (ONNX monolithic) ======\n  Model: foo\n"
    assert _parse_modal_stdout(stdout, suite="libero_spatial") is None


def test_parse_extracts_aggregate_counts():
    stdout = (
        "[onnx] task 0 done: 3/5\n"
        "[onnx] task 1 done: 4/5\n"
        "====== libero_spatial (ONNX monolithic) ======\n"
        "  Model: HuggingFaceVLA/smolvla_libero\n"
        "  Success: 7/10 = 70.0%\n"
    )
    parsed = _parse_modal_stdout(stdout, suite="libero_spatial")
    assert parsed is not None
    assert parsed["suite"] == "libero_spatial"
    assert parsed["total_success"] == 7
    assert parsed["total_eps"] == 10
    assert parsed["success_rate_pct"] == 70.0
    assert len(parsed["per_task"]) == 2
    assert parsed["per_task"][0] == {"task_idx": 0, "success": 3, "total": 5}
    assert parsed["per_task"][1] == {"task_idx": 1, "success": 4, "total": 5}


def test_parse_handles_zero_per_task_lines_when_aggregate_present():
    """Aggregate Success line is present but no [onnx] task lines (early failure)."""
    stdout = (
        "====== libero_spatial (ONNX monolithic) ======\n"
        "  Success: 0/0 = 0.0%\n"
    )
    parsed = _parse_modal_stdout(stdout, suite="libero_spatial")
    assert parsed is not None
    assert parsed["per_task"] == []


# ---------------------------------------------------------------------------
# _parse_invocation_to_episodes
# ---------------------------------------------------------------------------


def _make_invocation(*, returncode=0, stdout="", stderr="", parsed=None, suite="libero_spatial"):
    return ModalInvocationResult(
        suite=suite, returncode=returncode,
        stdout=stdout, stderr=stderr, parsed_result=parsed,
        elapsed_s=10.0,
    )


def test_episodes_from_clean_parsed_result():
    parsed = {
        "suite": "libero_spatial", "total_success": 4, "total_eps": 6,
        "success_rate_pct": 66.7,
        "per_task": [
            {"task_idx": 0, "success": 2, "total": 3},
            {"task_idx": 1, "success": 2, "total": 3},
        ],
    }
    invocation = _make_invocation(parsed=parsed)
    eps = _parse_invocation_to_episodes(invocation)
    # 2 tasks × 3 eps = 6 EpisodeResults
    assert len(eps) == 6
    # Per-task task_id format: <suite>_task_<idx>
    task_ids = {e.task_id for e in eps}
    assert task_ids == {"libero_spatial_task_0", "libero_spatial_task_1"}
    # First N are successes, rest failures
    task_0 = [e for e in eps if e.task_id == "libero_spatial_task_0"]
    assert sum(1 for e in task_0 if e.success) == 2
    assert sum(1 for e in task_0 if not e.success) == 1


def test_episodes_carry_n_steps_from_suite_max():
    parsed = {
        "suite": "libero_10", "total_success": 1, "total_eps": 1,
        "success_rate_pct": 100.0,
        "per_task": [{"task_idx": 0, "success": 1, "total": 1}],
    }
    invocation = _make_invocation(parsed=parsed, suite="libero_10")
    eps = _parse_invocation_to_episodes(invocation)
    assert eps[0].n_steps == TASK_SUITE_MAX_STEPS["libero_10"]


def test_episodes_failure_carries_phase1_limit_message():
    parsed = {
        "suite": "libero_spatial", "total_success": 0, "total_eps": 1,
        "success_rate_pct": 0.0,
        "per_task": [{"task_idx": 0, "success": 0, "total": 1}],
    }
    invocation = _make_invocation(parsed=parsed)
    eps = _parse_invocation_to_episodes(invocation)
    assert not eps[0].success
    assert "Phase 1 limit" in eps[0].error_message


def test_episodes_modal_returncode_nonzero_yields_failure_row():
    invocation = _make_invocation(
        returncode=1, stderr="modal app crashed",
    )
    eps = _parse_invocation_to_episodes(invocation)
    assert len(eps) == 1
    assert not eps[0].success
    assert eps[0].terminal_reason == "adapter_error"
    assert "exited 1" in eps[0].error_message
    assert "modal app crashed" in eps[0].error_message


def test_episodes_unparseable_stdout_yields_failure_row():
    invocation = _make_invocation(returncode=0, stdout="garbage", parsed=None)
    eps = _parse_invocation_to_episodes(invocation)
    assert len(eps) == 1
    assert not eps[0].success
    assert "did not contain expected summary" in eps[0].error_message


def test_episodes_fail_status_marker_surfaces_reason():
    """Per 2026-04-25 modal smoke validation: when the script prints
    status: FAIL + reason: ..., wrapper surfaces the reason directly
    instead of the generic 'no summary marker' fallback."""
    stdout = (
        "[onnx] action_dim orig=7 max=32 chunk=50\n"
        "\n=== RESULT ===\n"
        "  status: FAIL\n"
        "  reason: /onnx_out/smolvla_libero_monolithic/model.onnx not found\n"
        "Stopping app - local entrypoint completed.\n"
    )
    invocation = _make_invocation(returncode=0, stdout=stdout, parsed=None)
    eps = _parse_invocation_to_episodes(invocation)
    assert len(eps) == 1
    assert not eps[0].success
    assert "status=FAIL" in eps[0].error_message
    assert "model.onnx not found" in eps[0].error_message


def test_episodes_fail_status_without_reason_is_handled():
    """Edge case: status: FAIL printed but reason: line absent."""
    stdout = "  status: FAIL\n"
    invocation = _make_invocation(returncode=0, stdout=stdout, parsed=None)
    eps = _parse_invocation_to_episodes(invocation)
    assert len(eps) == 1
    assert "status=FAIL" in eps[0].error_message
    assert "(no reason printed)" in eps[0].error_message


def test_episodes_empty_per_task_yields_failure_row():
    invocation = _make_invocation(parsed={
        "suite": "libero_spatial", "total_success": 0, "total_eps": 0,
        "success_rate_pct": 0.0, "per_task": [],
    })
    eps = _parse_invocation_to_episodes(invocation)
    assert len(eps) == 1
    assert not eps[0].success
    assert "per_task list empty" in eps[0].error_message


# ---------------------------------------------------------------------------
# run_libero_on_modal — full path with stubbed invoker
# ---------------------------------------------------------------------------


def _success_invoker(*, returncode=0, stdout=""):
    """Build a stub invoker that returns a fake CompletedProcess."""
    def _invoker(cmd, timeout_s):
        return subprocess.CompletedProcess(
            args=cmd, returncode=returncode, stdout=stdout, stderr="",
        )

    return _invoker


def test_run_libero_raises_when_modal_missing(tmp_path, monkeypatch):
    """No `modal` on PATH + no invoker injected → ModalNotInstalledError."""
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: None)
    config = LiberoSuiteConfig(num_episodes=1, tasks=("libero_spatial",))
    with pytest.raises(ModalNotInstalledError, match="modal.*CLI"):
        run_libero_on_modal(
            config=config, export_dir=tmp_path,
            # invoker omitted on purpose -- triggers PATH check
        )


def test_run_libero_raises_when_script_missing(tmp_path):
    """Script not found at <repo_root>/scripts/... → FileNotFoundError."""
    config = LiberoSuiteConfig(num_episodes=1, tasks=("libero_spatial",))
    with pytest.raises(FileNotFoundError, match="Modal script not found"):
        run_libero_on_modal(
            config=config, export_dir=tmp_path,
            repo_root=tmp_path,  # tmp_path doesn't have scripts/
            modal_invoker=_success_invoker(),
        )


def test_run_libero_returns_empty_for_empty_tasks(tmp_path):
    """Empty config.tasks → empty EpisodeResult list (no Modal call)."""
    # Need a fake script + repo_root that exists
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "modal_libero_monolithic_onnx.py").write_text("# stub")

    config = LiberoSuiteConfig(num_episodes=3, tasks=())  # empty
    eps = run_libero_on_modal(
        config=config, export_dir=tmp_path,
        repo_root=tmp_path,
        modal_invoker=_success_invoker(),
    )
    assert eps == []


def test_run_libero_one_suite_returns_per_episode_rows(tmp_path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "modal_libero_monolithic_onnx.py").write_text("# stub")

    stdout = (
        "[onnx] task 0 done: 2/3\n"
        "[onnx] task 1 done: 3/3\n"
        "====== libero_spatial (ONNX monolithic) ======\n"
        "  Success: 5/6 = 83.3%\n"
    )
    config = LiberoSuiteConfig(num_episodes=3, tasks=("libero_spatial",))
    eps = run_libero_on_modal(
        config=config, export_dir=tmp_path,
        repo_root=tmp_path,
        modal_invoker=_success_invoker(stdout=stdout),
    )
    # 2 sub-tasks × 3 eps = 6 EpisodeResults
    assert len(eps) == 6
    # Aggregate matches parsed result
    n_success = sum(1 for e in eps if e.success)
    assert n_success == 5


def test_run_libero_passes_correct_cli_args(tmp_path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    fake_script = scripts_dir / "modal_libero_monolithic_onnx.py"
    fake_script.write_text("# stub")
    export_dir = tmp_path / "customer-smolvla-export"
    export_dir.mkdir()

    captured = []

    def _spy_invoker(cmd, timeout_s):
        captured.append(cmd)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="", stderr="",
        )

    config = LiberoSuiteConfig(
        num_episodes=5, tasks=("libero_object",), seed=42,
    )
    run_libero_on_modal(
        config=config, export_dir=export_dir,
        repo_root=tmp_path, modal_invoker=_spy_invoker,
    )
    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[0] == "modal"
    assert cmd[1] == "run"
    assert "--suite" in cmd
    assert "libero_object" in cmd
    assert "--num-episodes" in cmd
    assert "5" in cmd
    assert "--tasks" in cmd
    assert "all" in cmd
    assert "--onnx-subdir" in cmd
    assert cmd[cmd.index("--onnx-subdir") + 1] == "customer-smolvla-export"


def test_run_libero_invokes_per_suite_for_multiple_tasks(tmp_path):
    """Each suite in config.tasks → one Modal invocation."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "modal_libero_monolithic_onnx.py").write_text("# stub")

    invocations = []

    def _counting_invoker(cmd, timeout_s):
        # Track which suite is being invoked
        suite_idx = cmd.index("--suite") + 1
        invocations.append(cmd[suite_idx])
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="", stderr="",
        )

    config = LiberoSuiteConfig(
        num_episodes=1,
        tasks=("libero_spatial", "libero_object", "libero_goal"),
    )
    run_libero_on_modal(
        config=config, export_dir=tmp_path,
        repo_root=tmp_path, modal_invoker=_counting_invoker,
    )
    assert invocations == ["libero_spatial", "libero_object", "libero_goal"]


def test_modal_onnx_subdir_for_local_export_uses_basename(tmp_path):
    export_dir = tmp_path / "my-export"
    export_dir.mkdir()

    assert _modal_onnx_subdir_for_export(export_dir) == "my-export"


def test_modal_onnx_subdir_for_modal_volume_path_preserves_relative_subdir():
    export_dir = Path("/onnx_out/runs/customer-a/export-42")

    assert _modal_onnx_subdir_for_export(export_dir) == "runs/customer-a/export-42"
