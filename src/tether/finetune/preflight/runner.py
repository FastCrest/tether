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
from tether.finetune.preflight.result import PreflightCheck, PreflightReport
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
    from tether.finetune.local_dataset import (
        effective_imagenet_stats, validate_local_config, verify_local_export,
    )
    # Local integrity is mandatory. Even an unexpected verifier exception blocks
    # before any remote metadata lookup or later checks.
    if cfg.dataset_root is not None or cfg.dataset_manifest_sha256 is not None:
        try:
            validate_local_config(cfg)
            verified = verify_local_export(cfg.dataset_root, cfg.dataset_manifest_sha256)
            report.add(PreflightCheck(
                "local_dataset", "ok", "Local export bytes and retained admission evidence verified.",
                {"dataset_root": verified["dataset_root"],
                 "manifest_sha256": verified["manifest_sha256"],
                 "statistics_policy": "verified-moments-v1",
                 "normalization_profile": "smolvla",
                 "visual_normalization": "IDENTITY",
                 "export_processor_used_image_statistics": False,
                 "trainer_use_imagenet_stats_default": True,
                 "effective_use_imagenet_stats": effective_imagenet_stats(cfg),
                 "boundary": "Receipt/byte verification; no trainer or model was run. Keep the export immutable."},
            ))
        except Exception as exc:
            report.add(PreflightCheck("local_dataset", "fail", str(exc),
                                      {"code": getattr(exc, "code", "local-verification-error")}))
            return report

    for check_fn in (check_schema, check_dataset_size):
        try:
            result = check_fn(cfg)
            report.add(result)
        except Exception as exc:
            # Remote checks retain warning behavior. Local admission is
            # mandatory, including unexpected schema/count check failures.
            severity = "fail" if cfg.dataset_root is not None else "warn"
            logger.warning(
                "[preflight] %s crashed: %s — treating as %s",
                check_fn.__name__, exc, severity,
            )
            report.add(PreflightCheck(
                name=check_fn.__name__.replace("check_", ""),
                severity=severity,
                summary=f"check crashed: {type(exc).__name__}: {exc}",
            ))

    return report


__all__ = ["run_preflight"]
