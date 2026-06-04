"""Orchestrates the preflight checks.

Callable from `run_finetune()` (as the first step before GPU time) and
from the `--dry-run` CLI flag (runs only preflight, exits with the
report).

Check ordering: cheap local checks first, network checks second. If any
blocking check fails, later checks still run — customer gets the full
picture in one shot instead of fix-one-then-retry.
"""
from __future__ import annotations

import logging

from tether.finetune.config import FinetuneConfig
from tether.finetune.preflight.dataset_size import check_dataset_size
from tether.finetune.preflight.result import PreflightReport
from tether.finetune.preflight.schema import check_schema

logger = logging.getLogger(__name__)


def run_preflight(cfg: FinetuneConfig) -> PreflightReport:
    """Run all enabled preflight checks. Return a report.

    Checks run in sequence but none short-circuit — we gather every
    issue so the customer can fix them together.

    v0.5 checks implemented:
      * schema (action-dim dataset vs base)
      * dataset_size (episode-count floor per policy type)

    v0.6+ pending:
      * memory (VRAM budget estimate)
      * norm_stats (base-checkpoint stats reuse vs recompute)
    """
    report = PreflightReport()

    for check_fn in (check_schema, check_dataset_size):
        try:
            result = check_fn(cfg)
            report.add(result)
        except Exception as exc:
            # A check crashing should not take down the whole run. Log
            # it as a warning and keep going.
            logger.warning(
                "[preflight] %s crashed: %s — treating as warn",
                check_fn.__name__, exc,
            )
            from tether.finetune.preflight.result import PreflightCheck
            report.add(PreflightCheck(
                name=check_fn.__name__.replace("check_", ""),
                severity="warn",
                summary=f"check crashed: {type(exc).__name__}: {exc}",
            ))

    return report


__all__ = ["run_preflight"]
