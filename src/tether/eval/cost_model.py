"""Per-suite x per-runtime cost model for `tether eval`.

Per ADR 2026-04-25-eval-as-a-service-architecture decision #6
(--cost-preview before the run) + risk #6 (cost-table baked + audited;
CI quarterly job has cost cap):

The cost table is baked at ship time. Estimates are conservative
(round up to avoid surprise bills). Quarterly refresh job audits the
table against actual Modal billing logs.

Customers run `tether eval --cost-preview` before invoking a real
run to see a $-per-suite estimate. When the estimate exceeds
COST_PREVIEW_GUARDRAIL_USD (default $50), the CLI shows an extra
"are you sure?" warning per ADR risk #4.

Cost components:
- $/episode: per-episode compute (A10G GPU-second × episode wall-clock)
- $/task-startup: cold container warm-up + image pull amortized
  per-task (Modal only; local has no startup cost)

Local runtime is $0 — customer's own hardware. Phase 1 is Linux x86_64
only per ADR decision #8.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


# Cost-table schema version. Bump when fields change (additive
# evolution preferred). Customers can pin to a specific version in CI.
COST_TABLE_SCHEMA_VERSION = 1

# Threshold above which the CLI surfaces an extra "this run will cost
# $X — confirm with --yes-i-know-its-expensive or run with smaller
# --num-episodes" warning. Conservative default.
COST_PREVIEW_GUARDRAIL_USD = 50.0


# Per-suite x per-runtime baked table. Keys: (suite, runtime).
# Values: (usd_per_episode, usd_per_task_startup).
#
# LIBERO on Modal A10G (per scripts/modal_libero_*.py production runs):
# - Average episode wall-clock: ~30s on A10G ($0.000625/s × 30s = $0.019)
# - Cold-container warm-up + image pull amortized per-task: $0.10
#   (covers ~3 minutes of A10G boot at $0.000625/s + osmesa scene compile)
#
# Conservative round-up: $0.025/episode (covers OOM-retry overhead).
#
# Local runtime: $0 — customer's hardware. Episode wall-clock varies by
# customer GPU; not Tether's cost surface.
_COST_TABLE: dict[tuple[str, str], tuple[float, float]] = {
    ("libero", "modal"): (0.025, 0.10),
    ("libero", "local"): (0.0, 0.0),
}


@dataclass(frozen=True)
class CostEstimate:
    """Frozen cost estimate. Returned by estimate_cost(); serialized
    into the JSON envelope cost block."""

    total_usd: float
    suite: str
    runtime: str
    num_episodes_per_task: int
    n_tasks: int
    usd_per_episode: float
    usd_per_task_startup: float
    by_task: dict[str, float]
    cost_table_schema_version: int
    notes: str  # operator-readable explanation

    def __post_init__(self) -> None:
        if self.total_usd < 0:
            raise ValueError(f"total_usd must be >= 0, got {self.total_usd}")
        if self.n_tasks < 0:
            raise ValueError(f"n_tasks must be >= 0, got {self.n_tasks}")
        if self.num_episodes_per_task < 0:
            raise ValueError(
                f"num_episodes_per_task must be >= 0, got {self.num_episodes_per_task}"
            )
        # Cross-field invariant: total_usd ~ sum(by_task)
        if self.by_task:
            sum_by_task = sum(self.by_task.values())
            if abs(sum_by_task - self.total_usd) > 0.01:
                raise ValueError(
                    f"total_usd={self.total_usd} disagrees with sum(by_task)="
                    f"{sum_by_task} (must agree to within 1 cent)"
                )

    def to_dict(self) -> dict:
        return {
            "total_usd": round(self.total_usd, 4),
            "suite": self.suite,
            "runtime": self.runtime,
            "num_episodes_per_task": self.num_episodes_per_task,
            "n_tasks": self.n_tasks,
            "usd_per_episode": self.usd_per_episode,
            "usd_per_task_startup": self.usd_per_task_startup,
            "by_task": {k: round(v, 4) for k, v in self.by_task.items()},
            "cost_table_schema_version": self.cost_table_schema_version,
            "notes": self.notes,
        }

    @property
    def exceeds_guardrail(self) -> bool:
        """True if total_usd > COST_PREVIEW_GUARDRAIL_USD."""
        return self.total_usd > COST_PREVIEW_GUARDRAIL_USD


def estimate_cost(
    *,
    suite: str,
    runtime: str,
    tasks: Iterable[str],
    num_episodes_per_task: int,
) -> CostEstimate:
    """Estimate $ cost for a given (suite, runtime, tasks, num_episodes).

    Raises:
        ValueError: (suite, runtime) not in the baked cost table.
    """
    if num_episodes_per_task < 0:
        raise ValueError(
            f"num_episodes_per_task must be >= 0, got {num_episodes_per_task}"
        )

    key = (suite, runtime)
    if key not in _COST_TABLE:
        raise ValueError(
            f"No cost-table entry for (suite={suite!r}, runtime={runtime!r}). "
            f"Known keys: {sorted(_COST_TABLE.keys())}"
        )

    usd_per_ep, usd_per_startup = _COST_TABLE[key]
    task_list = list(tasks)
    n_tasks = len(task_list)

    by_task: dict[str, float] = {}
    for task_id in task_list:
        episode_cost = num_episodes_per_task * usd_per_ep
        task_cost = episode_cost + usd_per_startup
        by_task[task_id] = task_cost

    total = sum(by_task.values())

    notes = _build_notes(
        suite=suite, runtime=runtime, n_tasks=n_tasks,
        num_episodes_per_task=num_episodes_per_task,
        usd_per_ep=usd_per_ep, usd_per_startup=usd_per_startup,
    )

    return CostEstimate(
        total_usd=total,
        suite=suite,
        runtime=runtime,
        num_episodes_per_task=num_episodes_per_task,
        n_tasks=n_tasks,
        usd_per_episode=usd_per_ep,
        usd_per_task_startup=usd_per_startup,
        by_task=by_task,
        cost_table_schema_version=COST_TABLE_SCHEMA_VERSION,
        notes=notes,
    )


def _build_notes(
    *,
    suite: str, runtime: str, n_tasks: int,
    num_episodes_per_task: int,
    usd_per_ep: float, usd_per_startup: float,
) -> str:
    if runtime == "local":
        return (
            "Local runtime: $0 (customer hardware). Phase 1 Linux x86_64 only."
        )
    return (
        f"Modal runtime: {n_tasks} tasks × ({num_episodes_per_task} eps × "
        f"${usd_per_ep:.3f}/ep + ${usd_per_startup:.2f} cold-startup). "
        f"Cost table schema v{COST_TABLE_SCHEMA_VERSION}; refreshed "
        f"quarterly against Modal billing logs."
    )


def known_suite_runtime_pairs() -> tuple[tuple[str, str], ...]:
    """Returns the (suite, runtime) keys present in the baked cost table.

    Useful for CLI validation + test enumeration.
    """
    return tuple(sorted(_COST_TABLE.keys()))


__all__ = [
    "COST_PREVIEW_GUARDRAIL_USD",
    "COST_TABLE_SCHEMA_VERSION",
    "CostEstimate",
    "estimate_cost",
    "known_suite_runtime_pairs",
]
