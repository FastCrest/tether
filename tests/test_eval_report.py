"""Tests for src/tether/eval/report.py — Phase 1 eval-as-a-service Day 4.

Schema v1 LOCKED per ADR 2026-04-25-eval-as-a-service-architecture
decision #3. Customers grep on these fields in CI; renaming = breakage.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tether.eval.cost_model import CostEstimate, estimate_cost
from tether.eval.libero import (
    EpisodeResult,
    EvalReport,
    LiberoSuiteConfig,
    TaskResult,
)
from tether.eval.report import (
    EVAL_ENVELOPE_SCHEMA_VERSION,
    EvalEnvelope,
    EvalEnvironment,
    build_envelope,
    capture_environment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(task_id: str, ep_idx: int, success: bool = True) -> EpisodeResult:
    return EpisodeResult(
        task_id=task_id,
        episode_index=ep_idx,
        success=success,
        terminal_reason="success" if success else "timeout",
        wall_clock_s=1.0,
        n_steps=10,
        video_path=None,
        error_message=None,
    )


def _make_report(*, n_tasks: int = 2, n_eps: int = 3) -> EvalReport:
    started = datetime.now(timezone.utc)
    finished = started
    task_results = []
    for ti in range(n_tasks):
        task_id = f"task_{ti}"
        episodes = [_make_episode(task_id, ei, success=(ei % 2 == 0))
                    for ei in range(n_eps)]
        task_results.append(TaskResult.from_episodes(task_id, episodes))
    return EvalReport.from_task_results(
        suite="libero", runtime="modal", seed=0,
        started_at=started, finished_at=finished, results=task_results,
    )


def _make_env() -> EvalEnvironment:
    return EvalEnvironment(
        timestamp_utc="2026-04-25T00:00:00.000000Z",
        tether_version="0.1.0+dev",
        git_sha="abc123",
        git_dirty=False,
        python_version="3.13.11",
        platform="Darwin-25.3.0-arm64",
        export_dir="/tmp/export",
        onnx_files=[{"name": "model.onnx", "sha256": "deadbeef", "bytes": 100}],
    )


def _make_cost(total: float = 1.0) -> CostEstimate:
    return CostEstimate(
        total_usd=total,
        suite="libero", runtime="modal",
        num_episodes_per_task=3, n_tasks=1,
        usd_per_episode=0.025, usd_per_task_startup=0.10,
        by_task={"task_0": total},
        cost_table_schema_version=1,
        notes="test",
    )


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


def test_schema_version_is_1():
    assert EVAL_ENVELOPE_SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# EvalEnvironment
# ---------------------------------------------------------------------------


def test_env_to_dict_round_trips():
    env = _make_env()
    d = env.to_dict()
    assert d["tether_version"] == "0.1.0+dev"
    assert d["git_sha"] == "abc123"
    assert d["onnx_files"][0]["name"] == "model.onnx"


def test_env_is_frozen():
    env = _make_env()
    with pytest.raises(AttributeError):
        env.git_sha = "different"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EvalEnvelope construction
# ---------------------------------------------------------------------------


def test_envelope_rejects_wrong_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        EvalEnvelope(
            schema_version=99, tether_version="x", suite="libero", runtime="modal",
            seed=0, started_at="now", finished_at="now", wall_clock_s=0.0,
            tasks=(), num_episodes_per_task=0, aggregate={}, results=(),
            episodes=(), cost={}, modal=None, env={}, video_paths=(), notes=(),
        )


def test_envelope_rejects_negative_episodes():
    with pytest.raises(ValueError, match="num_episodes_per_task"):
        EvalEnvelope(
            schema_version=EVAL_ENVELOPE_SCHEMA_VERSION,
            tether_version="x", suite="libero", runtime="modal",
            seed=0, started_at="now", finished_at="now", wall_clock_s=0.0,
            tasks=(), num_episodes_per_task=-1, aggregate={}, results=(),
            episodes=(), cost={}, modal=None, env={}, video_paths=(), notes=(),
        )


def test_envelope_rejects_negative_wall_clock():
    with pytest.raises(ValueError, match="wall_clock_s"):
        EvalEnvelope(
            schema_version=EVAL_ENVELOPE_SCHEMA_VERSION,
            tether_version="x", suite="libero", runtime="modal",
            seed=0, started_at="now", finished_at="now", wall_clock_s=-1.0,
            tasks=(), num_episodes_per_task=0, aggregate={}, results=(),
            episodes=(), cost={}, modal=None, env={}, video_paths=(), notes=(),
        )


# ---------------------------------------------------------------------------
# build_envelope
# ---------------------------------------------------------------------------


def test_build_envelope_carries_all_required_fields():
    """Schema v1 customers grep on these — rename = breakage."""
    env = build_envelope(
        report=_make_report(n_tasks=2, n_eps=3),
        cost=_make_cost(),
        env=_make_env(),
        num_episodes_per_task=3,
    )
    d = env.to_dict()
    required = {
        "schema_version", "tether_version", "suite", "runtime", "seed",
        "started_at", "finished_at", "wall_clock_s", "tasks",
        "num_episodes_per_task", "aggregate", "results", "episodes",
        "cost", "modal", "env", "video_paths", "notes",
    }
    assert required.issubset(set(d.keys()))


def test_build_envelope_aggregate_block_shape():
    env = build_envelope(
        report=_make_report(n_tasks=2, n_eps=4),
        cost=_make_cost(),
        env=_make_env(),
        num_episodes_per_task=4,
    )
    agg = env.to_dict()["aggregate"]
    assert "success_rate" in agg
    assert "n_success" in agg
    assert "n_total" in agg


def test_build_envelope_per_task_results_block():
    env = build_envelope(
        report=_make_report(n_tasks=3, n_eps=2),
        cost=_make_cost(),
        env=_make_env(),
        num_episodes_per_task=2,
    )
    results = env.to_dict()["results"]
    assert len(results) == 3
    for r in results:
        assert "task_id" in r
        assert "n_success" in r
        assert "n_total" in r
        assert "success_rate" in r


def test_build_envelope_flattens_episodes():
    """Episodes block is a flat list across all tasks."""
    env = build_envelope(
        report=_make_report(n_tasks=2, n_eps=3),
        cost=_make_cost(),
        env=_make_env(),
        num_episodes_per_task=3,
    )
    episodes = env.to_dict()["episodes"]
    # 2 tasks × 3 eps = 6 flattened episodes
    assert len(episodes) == 6
    for ep in episodes:
        assert "task_id" in ep
        assert "episode_index" in ep
        assert "terminal_reason" in ep


def test_build_envelope_modal_block_when_modal():
    env = build_envelope(
        report=_make_report(),
        cost=_make_cost(),
        env=_make_env(),
        num_episodes_per_task=1,
        modal_block={"image_digest": "sha256:abc", "provider": "modal.com"},
    )
    assert env.to_dict()["modal"] == {"image_digest": "sha256:abc", "provider": "modal.com"}


def test_build_envelope_modal_block_none_when_local():
    env = build_envelope(
        report=_make_report(),
        cost=_make_cost(),
        env=_make_env(),
        num_episodes_per_task=1,
        modal_block=None,
    )
    assert env.to_dict()["modal"] is None


def test_build_envelope_carries_cost_dict():
    cost = estimate_cost(
        suite="libero", runtime="modal",
        tasks=["task_0", "task_1"], num_episodes_per_task=3,
    )
    env = build_envelope(
        report=_make_report(n_tasks=2, n_eps=3),
        cost=cost,
        env=_make_env(),
        num_episodes_per_task=3,
    )
    cost_dict = env.to_dict()["cost"]
    assert cost_dict["suite"] == "libero"
    assert cost_dict["runtime"] == "modal"
    assert cost_dict["total_usd"] > 0


def test_build_envelope_carries_video_paths_and_notes():
    env = build_envelope(
        report=_make_report(),
        cost=_make_cost(),
        env=_make_env(),
        num_episodes_per_task=1,
        video_paths=("/tmp/ep1.mp4", "/tmp/ep2.mp4"),
        notes=("warmed-up cold container",),
    )
    d = env.to_dict()
    assert d["video_paths"] == ["/tmp/ep1.mp4", "/tmp/ep2.mp4"]
    assert d["notes"] == ["warmed-up cold container"]


# ---------------------------------------------------------------------------
# write_json
# ---------------------------------------------------------------------------


def test_write_json_creates_file_and_round_trips(tmp_path):
    env = build_envelope(
        report=_make_report(),
        cost=_make_cost(),
        env=_make_env(),
        num_episodes_per_task=3,
    )
    out = tmp_path / "out" / "report.json"
    written = env.write_json(out)
    assert written.exists()
    parsed = json.loads(written.read_text())
    assert parsed["schema_version"] == 1
    assert parsed["suite"] == "libero"


def test_write_json_creates_parent_dirs(tmp_path):
    env = build_envelope(
        report=_make_report(),
        cost=_make_cost(),
        env=_make_env(),
        num_episodes_per_task=3,
    )
    deep = tmp_path / "a" / "b" / "c" / "report.json"
    env.write_json(deep)
    assert deep.exists()


def test_write_json_is_pretty_indented(tmp_path):
    env = build_envelope(
        report=_make_report(),
        cost=_make_cost(),
        env=_make_env(),
        num_episodes_per_task=1,
    )
    out = tmp_path / "report.json"
    env.write_json(out)
    text = out.read_text()
    # Pretty-printed → contains newlines + 2-space indent
    assert "\n" in text
    assert "  " in text


# ---------------------------------------------------------------------------
# capture_environment
# ---------------------------------------------------------------------------


def test_capture_environment_returns_evalenvironment(tmp_path):
    env = capture_environment(export_dir=tmp_path)
    assert isinstance(env, EvalEnvironment)
    assert env.export_dir == str(tmp_path.resolve())


def test_capture_environment_handles_non_git_dir(tmp_path):
    """Falling back to '' git_sha when not in a repo doesn't crash."""
    env = capture_environment(export_dir=tmp_path, repo_dir=tmp_path)
    # No git repo at tmp_path → empty sha + dirty=False
    assert env.git_sha == "" or len(env.git_sha) == 12


def test_capture_environment_hashes_onnx_files(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"fakemodel")
    env = capture_environment(export_dir=tmp_path)
    assert len(env.onnx_files) == 1
    assert env.onnx_files[0]["name"] == "model.onnx"
    assert env.onnx_files[0]["bytes"] == 9
    assert len(env.onnx_files[0]["sha256"]) == 64


def test_capture_environment_handles_no_onnx_files(tmp_path):
    env = capture_environment(export_dir=tmp_path)
    assert env.onnx_files == []


def test_capture_environment_handles_missing_export_dir(tmp_path):
    env = capture_environment(export_dir=tmp_path / "does-not-exist")
    assert env.onnx_files == []
