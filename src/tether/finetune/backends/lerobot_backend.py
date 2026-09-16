"""Lerobot-train subprocess backend.

The backend keeps ordinary fine-tuning on the existing command builder. Resume is
special: LeRobot v0.5.1 requires the saved checkpoint's own
`pretrained_model/train_config.json` via `--config_path`; repeating the original
`--policy.path` path bypasses the resume branch in LeRobot's config validator.
"""
from __future__ import annotations

import logging
from pathlib import Path

from tether.finetune.backends.base import (
    Backend,
    CheckpointResult,
    TrainerContext,
)

logger = logging.getLogger(__name__)


class LerobotBackend:
    """Adapter for LeRobot fine-tuning and exact checkpoint resume."""

    def fit(self, ctx: TrainerContext) -> CheckpointResult:
        from tether.finetune.run import (
            _locate_checkpoint,
            _run_lerobot_training,
        )

        cfg = ctx.config
        if cfg.resume:
            from tether.finetune.resume import ResumeConfigurationError, run_lerobot_resume

            try:
                rc, resumed_from = run_lerobot_resume(cfg.output, ctx.training_log_path)
            except ResumeConfigurationError as exc:
                return CheckpointResult(
                    final_checkpoint_path=Path(cfg.output),
                    training_steps_completed=0,
                    status="training_failed",
                    error=f"resume configuration invalid: {exc}",
                )
            logger.info("[finetune] resumed from exact checkpoint %s", resumed_from)
        else:
            rc = _run_lerobot_training(cfg, ctx.training_log_path)

        if rc != 0:
            return CheckpointResult(
                final_checkpoint_path=Path(cfg.output),
                training_steps_completed=0,
                status="training_failed",
                error=f"lerobot-train exited with code {rc}",
            )
        checkpoint = _locate_checkpoint(cfg.output)
        if checkpoint is None:
            return CheckpointResult(
                final_checkpoint_path=Path(cfg.output),
                training_steps_completed=0,
                status="training_failed",
                error=(
                    f"no checkpoint found under {cfg.output / 'training' / 'checkpoints'}; "
                    f"training reported success but produced no output"
                ),
            )
        ctx.hooks.run(
            "on_end",
            ctx,
            status="ok",
            steps_completed=cfg.num_steps,
        )
        return CheckpointResult(
            final_checkpoint_path=checkpoint,
            training_steps_completed=cfg.num_steps,
            status="ok",
        )


__all__ = ["LerobotBackend"]
