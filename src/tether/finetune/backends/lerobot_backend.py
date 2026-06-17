"""Lerobot-train subprocess backend.

Migrates the existing `run.py` subprocess logic behind the `Backend`
Protocol so the orchestrator can treat fine-tune and distill
uniformly. The underlying mechanics (_build_lerobot_command,
_run_lerobot_training, _locate_checkpoint) stay in run.py so the v0.3
CLI path is unchanged — LerobotBackend is a thin adapter.

Phase A: this backend ships alongside the existing run.py path, BUT
run.py doesn't go through `resolve_backend()` yet. run.py stays as
the `phase='train'` default path until we're confident the adapter
is semantically identical. Phase B (when SnapFlowBackend lands) is
when the orchestrator flips to `resolve_backend` + `Backend.fit()`
end-to-end.

Meaning: this file exists today to unblock the distill architecture's
Backend Protocol, without rewriting the fine-tune happy path. Treat it
as a contract declaration for now.
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
    """Adapter for the existing lerobot-train subprocess path.

    Wraps run.py's `_run_lerobot_training` + `_locate_checkpoint` so
    they're reachable via the Backend protocol for distill tests and
    for the future orchestrator refactor.
    """

    def fit(self, ctx: TrainerContext) -> CheckpointResult:
        # Lazy-import run.py internals to keep Backend import cheap.
        from tether.finetune.run import (
            _locate_checkpoint,
            _run_lerobot_training,
        )

        cfg = ctx.config
        rc = _run_lerobot_training(cfg, ctx.training_log_path)
        if rc != 0:
            return CheckpointResult(
                final_checkpoint_path=Path(cfg.output),  # placeholder
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
        # Fire on_end (lerobot-train doesn't give us per-step hooks via
        # subprocess; on_step is left to in-process backends like SnapFlow).
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
