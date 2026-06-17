"""Task-runner resolver for `tether eval`.

Per ADR 2026-04-25-eval-as-a-service-architecture decisions #2 + #8:
- Wrap, not rebuild — production callers route through the existing
  Modal image + osmesa/MuJoCo recipe (lifted from scripts/modal_libero_*.py)
  + the 441-LOC PredictModelServer adapter at
  src/tether/runtime/adapters/vla_eval.py
- Local fallback is Linux-only (osmesa + MuJoCo + lerobot dep stack);
  NEVER silently falls back to Modal — avoids surprise bills + masks
  real env config issues

Two dispatch shapes:

1. **Per-(task, episode) `TaskRunner`** — used by `LiberoSuite.run()`'s
   per-episode inner loop. Local runtime uses this (Day 5 stub still in
   place pending a real Linux OffScreenRenderEnv runner).

2. **Full-suite `SuiteRunner`** — used when the runtime returns
   aggregate results (Modal). Composes the existing Modal-script's
   per-suite loop instead of fanning out per-episode at the Tether
   layer (saves N cold-starts per suite).

The CLI picks shape (2) for `--runtime modal` and shape (1) for
`--runtime local`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tether.eval.libero import (
    ALL_RUNTIMES,
    EpisodeResult,
    EvalReport,
    LiberoSuiteConfig,
    TaskResult,
    TaskRunner,
)

logger = logging.getLogger(__name__)


# Phase 1 ships the 4-suite default. Each "suite" wraps ~10 LIBERO
# tasks (the existing modal_libero_*.py runs per-task within a suite).
LIBERO_DEFAULT_TASKS_PHASE1: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


def default_libero_tasks() -> list[str]:
    """Default LIBERO task list when --tasks not specified."""
    return list(LIBERO_DEFAULT_TASKS_PHASE1)


# ---------------------------------------------------------------------------
# Per-(task, episode) TaskRunner -- used by LiberoSuite.run() inner loop
# (--runtime local path)
# ---------------------------------------------------------------------------


def resolve_task_runner(
    *,
    runtime: str,
    export_dir: Path,
) -> TaskRunner:
    """Return a per-(task, episode) TaskRunner.

    Used by LiberoSuite.run() for runtimes that need per-episode
    dispatch. --runtime local uses this; --runtime modal uses
    resolve_suite_runner() instead.

    Raises:
        ValueError: runtime not in ALL_RUNTIMES.
    """
    if runtime not in ALL_RUNTIMES:
        raise ValueError(
            f"runtime must be one of {ALL_RUNTIMES}, got {runtime!r}"
        )

    if runtime == "modal":
        # Modal callers should use resolve_suite_runner(); this branch
        # exists for back-compat with old call sites + as a safety net.
        return _make_modal_per_episode_stub(export_dir)
    # runtime == "local"
    return _make_local_per_episode_stub(export_dir)


def _make_modal_per_episode_stub(export_dir: Path) -> TaskRunner:
    """Per-episode stub for --runtime modal. Real callers should route
    through resolve_suite_runner(); this exists as a back-compat
    safety net + for tests pinning the old shape."""
    msg = (
        "Modal per-episode dispatch is deprecated; use "
        "resolve_suite_runner() for full-suite Modal invocations. See "
        "src/tether/eval/modal_runner.py."
    )

    def _runner(task_id: str, episode_index: int, config: LiberoSuiteConfig) -> EpisodeResult:
        return EpisodeResult(
            task_id=task_id, episode_index=episode_index,
            success=False, terminal_reason="adapter_error",
            wall_clock_s=0.0, n_steps=0,
            video_path=None, error_message=msg,
        )

    return _runner


def _make_local_per_episode_stub(export_dir: Path) -> TaskRunner:
    """Per-episode stub for --runtime local. Real Linux OffScreenRenderEnv
    runner is Phase 1 follow-up (Linux-only, gated on the [eval-local]
    extra)."""
    msg = (
        "Local task runner not yet wired (Phase 1 follow-up). For now "
        "use --runtime modal which ships LIBERO in the bundled image."
    )

    def _runner(task_id: str, episode_index: int, config: LiberoSuiteConfig) -> EpisodeResult:
        return EpisodeResult(
            task_id=task_id, episode_index=episode_index,
            success=False, terminal_reason="adapter_error",
            wall_clock_s=0.0, n_steps=0,
            video_path=None, error_message=msg,
        )

    return _runner


# ---------------------------------------------------------------------------
# Full-suite SuiteRunner -- used by CLI for runtimes that aggregate
# (--runtime modal path)
# ---------------------------------------------------------------------------


# Type alias for the full-suite dispatch shape. Returns the EvalReport
# directly (caller doesn't need to compose per-task aggregates).
SuiteRunner = Callable[[LiberoSuiteConfig, Path], EvalReport]


def resolve_suite_runner(
    *,
    runtime: str,
    export_dir: Path,
    repo_root: Path | None = None,
) -> SuiteRunner:
    """Return a full-suite SuiteRunner for runtimes that aggregate.

    --runtime modal: wires tether.eval.modal_runner.run_libero_on_modal.
    --runtime local: returns a stub that emits adapter_error rows
        (matches resolve_task_runner shape -- local always uses
        per-episode dispatch).

    Raises:
        ValueError: runtime not in ALL_RUNTIMES.
    """
    if runtime not in ALL_RUNTIMES:
        raise ValueError(
            f"runtime must be one of {ALL_RUNTIMES}, got {runtime!r}"
        )

    if runtime == "modal":
        return _make_modal_suite_runner(export_dir=export_dir, repo_root=repo_root)
    # runtime == "local" — local should use resolve_task_runner; this
    # path exists for symmetry + so callers can branch consistently.
    return _make_local_suite_stub(export_dir)


def _make_modal_suite_runner(
    *,
    export_dir: Path,
    repo_root: Path | None,
) -> SuiteRunner:
    """Real Modal suite runner. Wraps modal_runner.run_libero_on_modal
    + builds the EvalReport from the flat EpisodeResult list."""

    def _runner(config: LiberoSuiteConfig, _export_dir: Path) -> EvalReport:
        # Lazy import keeps modal SDK out of the hot path for local-only
        # callers.
        from tether.eval.modal_runner import run_libero_on_modal

        started_at = datetime.now(timezone.utc)
        episodes = run_libero_on_modal(
            config=config,
            export_dir=_export_dir,
            repo_root=repo_root,
        )
        finished_at = datetime.now(timezone.utc)
        return _build_report_from_flat_episodes(
            episodes=episodes,
            config=config,
            started_at=started_at,
            finished_at=finished_at,
        )

    return _runner


def _make_local_suite_stub(export_dir: Path) -> SuiteRunner:
    """Local suite runner stub. Phase 1 local always per-episode via
    LiberoSuite.run; this returns a single-row error report so callers
    that misroute see a structured failure."""

    def _runner(config: LiberoSuiteConfig, _export_dir: Path) -> EvalReport:
        started_at = datetime.now(timezone.utc)
        finished_at = started_at
        return EvalReport.from_task_results(
            suite="libero", runtime="local", seed=config.seed,
            started_at=started_at, finished_at=finished_at,
            results=[],
        )

    return _runner


def _build_report_from_flat_episodes(
    *,
    episodes: list[EpisodeResult],
    config: LiberoSuiteConfig,
    started_at: datetime,
    finished_at: datetime,
) -> EvalReport:
    """Group flat EpisodeResults by task_id; build TaskResult per task;
    aggregate into EvalReport."""
    by_task: dict[str, list[EpisodeResult]] = {}
    for ep in episodes:
        by_task.setdefault(ep.task_id, []).append(ep)

    task_results: list[TaskResult] = []
    for task_id, eps in by_task.items():
        # Re-index episode_index to be sequential within the task
        renum = [
            EpisodeResult(
                task_id=ep.task_id,
                episode_index=i,
                success=ep.success,
                terminal_reason=ep.terminal_reason,
                wall_clock_s=ep.wall_clock_s,
                n_steps=ep.n_steps,
                video_path=ep.video_path,
                error_message=ep.error_message,
            )
            for i, ep in enumerate(eps)
        ]
        task_results.append(TaskResult.from_episodes(task_id, renum))

    return EvalReport.from_task_results(
        suite="libero", runtime=config.runtime, seed=config.seed,
        started_at=started_at, finished_at=finished_at,
        results=task_results,
    )


__all__ = [
    "LIBERO_DEFAULT_TASKS_PHASE1",
    "SuiteRunner",
    "default_libero_tasks",
    "resolve_suite_runner",
    "resolve_task_runner",
]
