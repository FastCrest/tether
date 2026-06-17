"""LIBERO suite dispatcher for `tether eval`.

Per ADR 2026-04-25-eval-as-a-service-architecture:
- Wrap, not rebuild — lifts the existing Modal image + osmesa/MuJoCo
  recipe from scripts/modal_libero_*.py + the 441-LOC adapter at
  src/tether/runtime/adapters/vla_eval.py
- New top-level `tether eval` verb (NOT subcommand of `bench`); this
  module is the dispatcher invoked by cli.py's eval handler
- LIBERO-only Phase 1; SimplerEnv is Phase 2

Day 1 ships the substrate (config + dispatcher + per-task fan-out).
Day 2 wires the pre-flight smoke test. Day 3 wires the CLI verb.
Day 4 wires the cost model + JSON envelope. Day 5 wires video.

The dispatcher is PURE — given a config + an api_caller (production
hits Modal; tests stub it), it returns an EvalReport. No I/O outside
the api_caller.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)


# Bounded enum of supported runtimes. Stable across minor releases —
# surfaced in CLI flags + JSON envelope + telemetry labels.
RuntimeMode = Literal["modal", "local"]
ALL_RUNTIMES: tuple[str, ...] = ("modal", "local")

# Bounded enum of terminal reasons — locked per ADR. Customers grep on
# these in CI scripts; renaming = breakage.
TerminalReason = Literal[
    "success",
    "timeout",
    "bddl_failure",
    "rendering_failure",
    "adapter_error",
]
ALL_TERMINAL_REASONS: tuple[str, ...] = (
    "success", "timeout", "bddl_failure",
    "rendering_failure", "adapter_error",
)

# Default seed matches `tether bench` (cli.py:608) for cross-verb
# consistency. Customers reproducing prior modal_libero_*.py results
# pass --seed 7.
DEFAULT_SEED = 0

# Default num_episodes is "smoke" (3); customers running published
# benches pass 100+. Documented in customer doc.
DEFAULT_NUM_EPISODES = 3

# Wall-clock cap per episode. Beyond this, the caller marks the
# episode as terminal_reason="timeout" rather than waiting forever
# (LIBERO occasionally hangs in osmesa scene-compilation).
DEFAULT_EPISODE_TIMEOUT_S = 300


@dataclass(frozen=True)
class LiberoSuiteConfig:
    """Frozen config for one LIBERO eval run. Constructed at CLI layer
    from --num-episodes / --tasks / --runtime / etc.; passed verbatim
    to LiberoSuite.run()."""

    num_episodes: int = DEFAULT_NUM_EPISODES
    tasks: tuple[str, ...] = ()  # empty = all 90 LIBERO tasks
    runtime: str = "modal"
    video: bool = False
    output_dir: str = "./eval_output"
    seed: int = DEFAULT_SEED
    max_parallel: int = 1
    cost_preview: bool = False
    episode_timeout_s: float = DEFAULT_EPISODE_TIMEOUT_S

    def __post_init__(self) -> None:
        if self.num_episodes < 1:
            raise ValueError(
                f"num_episodes must be >= 1, got {self.num_episodes}"
            )
        if self.runtime not in ALL_RUNTIMES:
            raise ValueError(
                f"runtime must be one of {ALL_RUNTIMES}, got {self.runtime!r}"
            )
        if self.max_parallel < 1:
            raise ValueError(
                f"max_parallel must be >= 1, got {self.max_parallel}"
            )
        if self.episode_timeout_s <= 0:
            raise ValueError(
                f"episode_timeout_s must be > 0, got {self.episode_timeout_s}"
            )
        # Normalize tasks: empty tuple OR tuple of non-empty strings
        for task in self.tasks:
            if not task or not isinstance(task, str):
                raise ValueError(
                    f"each task must be a non-empty string, got {task!r}"
                )


@dataclass(frozen=True)
class EpisodeResult:
    """One episode's outcome. Frozen — the caller serializes into the
    JSON envelope (Day 4 wiring)."""

    task_id: str
    episode_index: int
    success: bool
    terminal_reason: str  # bounded enum: ALL_TERMINAL_REASONS
    wall_clock_s: float
    n_steps: int
    video_path: str | None  # None when --video unset OR encode failed
    error_message: str | None  # populated when terminal_reason != "success"

    def __post_init__(self) -> None:
        if self.terminal_reason not in ALL_TERMINAL_REASONS:
            raise ValueError(
                f"terminal_reason must be one of {ALL_TERMINAL_REASONS}, "
                f"got {self.terminal_reason!r}"
            )
        if self.episode_index < 0:
            raise ValueError(
                f"episode_index must be >= 0, got {self.episode_index}"
            )
        if self.n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {self.n_steps}")
        # Cross-field invariant: success=True iff terminal_reason="success"
        if self.success and self.terminal_reason != "success":
            raise ValueError(
                f"success=True but terminal_reason={self.terminal_reason!r}; "
                f"these must agree"
            )
        if not self.success and self.terminal_reason == "success":
            raise ValueError(
                "success=False but terminal_reason='success'; these must agree"
            )


@dataclass(frozen=True)
class TaskResult:
    """One task's aggregated outcome — built from per-episode results."""

    task_id: str
    n_success: int
    n_total: int
    success_rate: float
    episodes: tuple[EpisodeResult, ...]

    @classmethod
    def from_episodes(cls, task_id: str, episodes: list[EpisodeResult]) -> "TaskResult":
        n_success = sum(1 for e in episodes if e.success)
        n_total = len(episodes)
        rate = n_success / n_total if n_total > 0 else 0.0
        return cls(
            task_id=task_id, n_success=n_success, n_total=n_total,
            success_rate=rate, episodes=tuple(episodes),
        )


@dataclass(frozen=True)
class EvalReport:
    """Full result of one LiberoSuite.run(). Phase 1 schema v1 LOCKED;
    Phase 2 evolution is additive-only.

    The full JSON envelope (Day 4 wiring) wraps this with cost + modal
    blocks; Day 1 ships the per-task / per-episode core."""

    suite: str
    runtime: str
    seed: int
    started_at: str  # ISO 8601
    finished_at: str  # ISO 8601
    wall_clock_s: float
    tasks: tuple[str, ...]
    results: tuple[TaskResult, ...]
    aggregate_success_rate: float
    aggregate_n_success: int
    aggregate_n_total: int

    @classmethod
    def from_task_results(
        cls,
        *,
        suite: str,
        runtime: str,
        seed: int,
        started_at: datetime,
        finished_at: datetime,
        results: list[TaskResult],
    ) -> "EvalReport":
        n_success = sum(r.n_success for r in results)
        n_total = sum(r.n_total for r in results)
        rate = n_success / n_total if n_total > 0 else 0.0
        wall = (finished_at - started_at).total_seconds()
        return cls(
            suite=suite,
            runtime=runtime,
            seed=seed,
            started_at=started_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            finished_at=finished_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            wall_clock_s=wall,
            tasks=tuple(r.task_id for r in results),
            results=tuple(results),
            aggregate_success_rate=rate,
            aggregate_n_success=n_success,
            aggregate_n_total=n_total,
        )


# Type alias for the dispatcher's per-task callback. Production wires
# this to the vla-eval adapter on Modal; tests stub it.
TaskRunner = Callable[[str, int, LiberoSuiteConfig], EpisodeResult]


class LiberoSuite:
    """Suite dispatcher. Pure orchestration — owns no model, no Modal
    state. The api_caller injection point lets tests stub the actual
    per-episode runner without spinning up Modal.

    Usage (Day 3+ wiring from cli.py):
        config = LiberoSuiteConfig(num_episodes=3, runtime="modal")
        suite = LiberoSuite()
        report = suite.run(
            export_dir="./my-export",
            config=config,
            task_runner=_modal_task_runner,  # production
        )
    """

    @classmethod
    def run(
        cls,
        *,
        export_dir: str | Path,
        config: LiberoSuiteConfig,
        task_runner: TaskRunner,
        tasks_provider: Callable[[], list[str]] | None = None,
    ) -> EvalReport:
        """Dispatch + aggregate. Production callers pass `task_runner`
        bound to a Modal app's per-episode function; tests pass a stub
        that returns canned EpisodeResult values.

        `tasks_provider` returns the LIBERO task list when config.tasks
        is empty (default = all 90). Tests bypass this with explicit
        config.tasks.

        Phase 1 fan-out is sequential (max_parallel honored only when
        the api_caller supports it; the wrapper layer is sequential).
        Phase 2 wires concurrent.futures.ThreadPoolExecutor.
        """
        export_path = Path(export_dir)
        if not export_path.exists():
            raise FileNotFoundError(
                f"export_dir does not exist: {export_path}"
            )

        # Resolve task list
        if config.tasks:
            tasks = list(config.tasks)
        elif tasks_provider is not None:
            tasks = tasks_provider()
        else:
            # Default: caller didn't pin tasks AND didn't supply a provider.
            # Day 1 substrate: empty list → empty report (no tasks to run).
            # Day 2+ wiring will fall back to the canonical LIBERO-90 list.
            tasks = []

        if not tasks:
            logger.warning(
                "LiberoSuite.run: no tasks resolved (config.tasks empty + "
                "tasks_provider returned empty). Returning empty report."
            )

        started_at = datetime.now(timezone.utc)
        task_results: list[TaskResult] = []

        for task_id in tasks:
            episode_results: list[EpisodeResult] = []
            for ep_idx in range(config.num_episodes):
                try:
                    result = task_runner(task_id, ep_idx, config)
                except Exception as exc:  # noqa: BLE001
                    # Any unhandled exception in the runner becomes an
                    # adapter_error episode result rather than crashing
                    # the whole suite (one bad task shouldn't kill the rest)
                    logger.error(
                        "LiberoSuite.run: task=%s episode=%d adapter_error: "
                        "%s: %s",
                        task_id, ep_idx, type(exc).__name__, exc,
                    )
                    result = EpisodeResult(
                        task_id=task_id,
                        episode_index=ep_idx,
                        success=False,
                        terminal_reason="adapter_error",
                        wall_clock_s=0.0,
                        n_steps=0,
                        video_path=None,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                episode_results.append(result)
            task_results.append(
                TaskResult.from_episodes(task_id, episode_results)
            )

        finished_at = datetime.now(timezone.utc)
        return EvalReport.from_task_results(
            suite="libero",
            runtime=config.runtime,
            seed=config.seed,
            started_at=started_at,
            finished_at=finished_at,
            results=task_results,
        )


__all__ = [
    "ALL_RUNTIMES",
    "ALL_TERMINAL_REASONS",
    "DEFAULT_EPISODE_TIMEOUT_S",
    "DEFAULT_NUM_EPISODES",
    "DEFAULT_SEED",
    "EpisodeResult",
    "EvalReport",
    "LiberoSuite",
    "LiberoSuiteConfig",
    "RuntimeMode",
    "TaskResult",
    "TaskRunner",
    "TerminalReason",
]
