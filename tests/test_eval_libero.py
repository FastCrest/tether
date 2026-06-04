"""Tests for src/tether/eval/libero.py — Phase 1 eval-as-a-service Day 1.

Per ADR 2026-04-25-eval-as-a-service-architecture: substrate is pure
orchestration (config + dispatcher + per-task fan-out). Day 2 wires
the pre-flight smoke test; Day 3 wires the CLI verb.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tether.eval.libero import (
    ALL_RUNTIMES,
    ALL_TERMINAL_REASONS,
    DEFAULT_NUM_EPISODES,
    DEFAULT_SEED,
    EpisodeResult,
    EvalReport,
    LiberoSuite,
    LiberoSuiteConfig,
    TaskResult,
)


def _mk_episode(
    *,
    task_id: str = "pick_block",
    episode_index: int = 0,
    success: bool = True,
    terminal_reason: str = "success",
    wall_clock_s: float = 30.0,
    n_steps: int = 100,
    video_path: str | None = None,
    error_message: str | None = None,
) -> EpisodeResult:
    return EpisodeResult(
        task_id=task_id, episode_index=episode_index,
        success=success, terminal_reason=terminal_reason,
        wall_clock_s=wall_clock_s, n_steps=n_steps,
        video_path=video_path, error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Bounded enums
# ---------------------------------------------------------------------------


def test_runtimes_bounded_enum():
    assert set(ALL_RUNTIMES) == {"modal", "local"}


def test_terminal_reasons_bounded_enum():
    expected = {"success", "timeout", "bddl_failure", "rendering_failure", "adapter_error"}
    assert set(ALL_TERMINAL_REASONS) == expected


def test_default_seed_matches_bench_convention():
    """seed=0 default per ADR (matches tether bench cli.py:608)."""
    assert DEFAULT_SEED == 0


def test_default_num_episodes_is_smoke():
    """3-episode default for smoke; published benches pass 100+."""
    assert DEFAULT_NUM_EPISODES == 3


# ---------------------------------------------------------------------------
# LiberoSuiteConfig validation
# ---------------------------------------------------------------------------


def test_config_default_construction():
    cfg = LiberoSuiteConfig()
    assert cfg.num_episodes == DEFAULT_NUM_EPISODES
    assert cfg.runtime == "modal"
    assert cfg.seed == DEFAULT_SEED


def test_config_rejects_zero_num_episodes():
    with pytest.raises(ValueError, match="num_episodes"):
        LiberoSuiteConfig(num_episodes=0)


def test_config_rejects_unknown_runtime():
    with pytest.raises(ValueError, match="runtime"):
        LiberoSuiteConfig(runtime="bogus")


def test_config_accepts_both_runtimes():
    LiberoSuiteConfig(runtime="modal")
    LiberoSuiteConfig(runtime="local")


def test_config_rejects_zero_max_parallel():
    with pytest.raises(ValueError, match="max_parallel"):
        LiberoSuiteConfig(max_parallel=0)


def test_config_rejects_zero_episode_timeout():
    with pytest.raises(ValueError, match="episode_timeout_s"):
        LiberoSuiteConfig(episode_timeout_s=0)


def test_config_rejects_empty_string_in_tasks():
    with pytest.raises(ValueError, match="task"):
        LiberoSuiteConfig(tasks=("valid", "", "other"))


def test_config_accepts_empty_tasks_tuple():
    """Empty tasks = use all (resolved by tasks_provider in run)."""
    cfg = LiberoSuiteConfig(tasks=())
    assert cfg.tasks == ()


# ---------------------------------------------------------------------------
# EpisodeResult validation + cross-field invariants
# ---------------------------------------------------------------------------


def test_episode_rejects_unknown_terminal_reason():
    with pytest.raises(ValueError, match="terminal_reason"):
        _mk_episode(terminal_reason="bogus")


def test_episode_rejects_negative_episode_index():
    with pytest.raises(ValueError, match="episode_index"):
        _mk_episode(episode_index=-1)


def test_episode_rejects_negative_n_steps():
    with pytest.raises(ValueError, match="n_steps"):
        _mk_episode(n_steps=-1)


def test_episode_cross_field_success_implies_success_reason():
    """success=True with terminal_reason!='success' is forbidden."""
    with pytest.raises(ValueError, match="success.*terminal_reason"):
        _mk_episode(success=True, terminal_reason="timeout")


def test_episode_cross_field_failure_implies_non_success_reason():
    """success=False with terminal_reason='success' is forbidden."""
    with pytest.raises(ValueError, match="success.*terminal_reason"):
        _mk_episode(success=False, terminal_reason="success")


def test_episode_failure_with_timeout_reason_valid():
    """Failure can carry any non-success reason."""
    e = _mk_episode(success=False, terminal_reason="timeout")
    assert not e.success
    assert e.terminal_reason == "timeout"


# ---------------------------------------------------------------------------
# TaskResult aggregation
# ---------------------------------------------------------------------------


def test_task_result_from_episodes_aggregates_correctly():
    episodes = [
        _mk_episode(task_id="t", episode_index=0, success=True),
        _mk_episode(task_id="t", episode_index=1, success=False, terminal_reason="bddl_failure"),
        _mk_episode(task_id="t", episode_index=2, success=True),
    ]
    result = TaskResult.from_episodes("t", episodes)
    assert result.task_id == "t"
    assert result.n_success == 2
    assert result.n_total == 3
    assert result.success_rate == pytest.approx(2 / 3)


def test_task_result_empty_episodes_zero_rate():
    result = TaskResult.from_episodes("t", [])
    assert result.success_rate == 0.0


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------


def test_eval_report_aggregates_across_tasks():
    started = datetime(2026, 4, 25, 10, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 4, 25, 10, 30, tzinfo=timezone.utc)
    task_results = [
        TaskResult.from_episodes("t1", [
            _mk_episode(task_id="t1", episode_index=0, success=True),
            _mk_episode(task_id="t1", episode_index=1, success=True),
        ]),
        TaskResult.from_episodes("t2", [
            _mk_episode(task_id="t2", episode_index=0, success=True),
            _mk_episode(task_id="t2", episode_index=1, success=False, terminal_reason="rendering_failure"),
        ]),
    ]
    report = EvalReport.from_task_results(
        suite="libero", runtime="modal", seed=0,
        started_at=started, finished_at=finished,
        results=task_results,
    )
    assert report.aggregate_n_success == 3
    assert report.aggregate_n_total == 4
    assert report.aggregate_success_rate == 0.75
    assert report.wall_clock_s == 1800.0  # 30 min
    assert len(report.results) == 2
    assert report.tasks == ("t1", "t2")


# ---------------------------------------------------------------------------
# LiberoSuite.run dispatch
# ---------------------------------------------------------------------------


def test_run_rejects_missing_export_dir(tmp_path):
    cfg = LiberoSuiteConfig(num_episodes=1)
    with pytest.raises(FileNotFoundError):
        LiberoSuite.run(
            export_dir=tmp_path / "missing",
            config=cfg,
            task_runner=lambda task, ep, c: _mk_episode(),
        )


def test_run_returns_empty_report_when_no_tasks(tmp_path):
    """Empty config.tasks + no tasks_provider → empty report (Day 1)."""
    cfg = LiberoSuiteConfig(num_episodes=3, tasks=())
    report = LiberoSuite.run(
        export_dir=tmp_path,  # exists
        config=cfg,
        task_runner=lambda task, ep, c: _mk_episode(),
    )
    assert report.aggregate_n_total == 0
    assert report.results == ()


def test_run_invokes_task_runner_per_episode(tmp_path):
    cfg = LiberoSuiteConfig(num_episodes=3, tasks=("t1", "t2"))
    calls = []

    def runner(task_id, ep_idx, config):
        calls.append((task_id, ep_idx))
        return _mk_episode(task_id=task_id, episode_index=ep_idx)

    report = LiberoSuite.run(
        export_dir=tmp_path, config=cfg, task_runner=runner,
    )
    # 2 tasks × 3 episodes = 6 invocations
    assert len(calls) == 6
    assert calls == [
        ("t1", 0), ("t1", 1), ("t1", 2),
        ("t2", 0), ("t2", 1), ("t2", 2),
    ]
    assert report.aggregate_n_total == 6


def test_run_uses_tasks_provider_when_config_tasks_empty(tmp_path):
    cfg = LiberoSuiteConfig(num_episodes=1, tasks=())
    runner = lambda task, ep, c: _mk_episode(task_id=task)
    report = LiberoSuite.run(
        export_dir=tmp_path, config=cfg, task_runner=runner,
        tasks_provider=lambda: ["task_a", "task_b"],
    )
    assert report.tasks == ("task_a", "task_b")
    assert report.aggregate_n_total == 2


def test_run_config_tasks_take_precedence_over_provider(tmp_path):
    """Explicit config.tasks overrides tasks_provider."""
    cfg = LiberoSuiteConfig(num_episodes=1, tasks=("explicit",))
    runner = lambda task, ep, c: _mk_episode(task_id=task)
    report = LiberoSuite.run(
        export_dir=tmp_path, config=cfg, task_runner=runner,
        tasks_provider=lambda: ["should_not_be_used"],
    )
    assert report.tasks == ("explicit",)


def test_run_converts_runner_exception_to_adapter_error_episode(tmp_path):
    """A bad runner raises Exception; the dispatcher converts to an
    adapter_error EpisodeResult so one bad task doesn't kill the suite."""
    cfg = LiberoSuiteConfig(num_episodes=2, tasks=("t1",))

    def bad_runner(task, ep, c):
        raise RuntimeError("simulated adapter crash")

    report = LiberoSuite.run(
        export_dir=tmp_path, config=cfg, task_runner=bad_runner,
    )
    assert report.aggregate_n_success == 0
    assert report.aggregate_n_total == 2
    for ep in report.results[0].episodes:
        assert ep.terminal_reason == "adapter_error"
        assert "simulated adapter crash" in (ep.error_message or "")


def test_run_propagates_seed_into_report(tmp_path):
    cfg = LiberoSuiteConfig(num_episodes=1, tasks=("t1",), seed=42)
    runner = lambda task, ep, c: _mk_episode(task_id=task)
    report = LiberoSuite.run(
        export_dir=tmp_path, config=cfg, task_runner=runner,
    )
    assert report.seed == 42


def test_run_propagates_runtime_into_report(tmp_path):
    cfg = LiberoSuiteConfig(num_episodes=1, tasks=("t1",), runtime="local")
    runner = lambda task, ep, c: _mk_episode(task_id=task)
    report = LiberoSuite.run(
        export_dir=tmp_path, config=cfg, task_runner=runner,
    )
    assert report.runtime == "local"


# ---------------------------------------------------------------------------
# Frozen-dataclass invariants
# ---------------------------------------------------------------------------


def test_config_is_frozen():
    cfg = LiberoSuiteConfig()
    with pytest.raises(AttributeError):
        cfg.num_episodes = 100  # type: ignore[misc]


def test_episode_is_frozen():
    e = _mk_episode()
    with pytest.raises(AttributeError):
        e.success = False  # type: ignore[misc]


def test_task_result_is_frozen():
    r = TaskResult.from_episodes("t", [_mk_episode()])
    with pytest.raises(AttributeError):
        r.task_id = "other"  # type: ignore[misc]


def test_eval_report_is_frozen():
    started = datetime(2026, 4, 25, 10, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 4, 25, 10, 5, tzinfo=timezone.utc)
    report = EvalReport.from_task_results(
        suite="libero", runtime="modal", seed=0,
        started_at=started, finished_at=finished, results=[],
    )
    with pytest.raises(AttributeError):
        report.suite = "other"  # type: ignore[misc]
