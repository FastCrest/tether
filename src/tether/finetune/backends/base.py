"""Backend Protocol + shared context types.

`Backend` is the abstraction that lets one orchestrator (run.py) drive
both fine-tune (lerobot-train subprocess) and distillation (SnapFlow
in-process) without per-phase branches in the orchestration code.

Per architecture doc Section B + C.2 (distill_architecture.md):
  TrainerContext carries: config + hooks + teacher_path + training_log_path
  CheckpointResult carries: final_checkpoint_path + steps_completed + metrics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class TrainerContext:
    """Input to Backend.fit(). Carries everything a trainer needs from the
    orchestrator without implying a specific training paradigm.

    `teacher_path` is None for fine-tune (no teacher), set for distill.
    `hooks` is the registry of callbacks the trainer should fire at the
    documented lifecycle points (see hooks/__init__.py for the contract).
    """

    config: Any
    """The FinetuneConfig (avoided import cycle by using Any)."""

    hooks: Any
    """The HookRegistry. Avoided import cycle here too."""

    training_log_path: Path
    """Where the trainer should write stdout / step logs. run.py creates
    this before calling fit() so hooks can tail it during training."""

    teacher_path: Path | None = None
    """For phase='distill' only: path to the teacher's tether-export dir
    (the merged PyTorch checkpoint, same format as `_auto_export`
    consumes today). None when phase='train'."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Escape hatch for backend-specific inputs that don't belong on the
    FinetuneConfig. Kept out of the main struct so the Protocol doesn't
    grow per-backend fields."""


@dataclass
class CheckpointResult:
    """Output of Backend.fit(). The orchestrator consumes this to drive
    postprocess (export → validate → verification) regardless of which
    backend ran.

    `final_checkpoint_path` is where the trained/distilled student
    weights live. For lerobot-train: the standard
    `<output>/training/checkpoints/<step>/pretrained_model/` dir. For
    SnapFlow: `<output>/training/student/` (self-contained full
    weights, no adapter merge step — see architecture C.4).
    """

    final_checkpoint_path: Path
    training_steps_completed: int
    status: str = "ok"
    """One of: 'ok' | 'training_failed' | 'aborted'."""

    error: str | None = None
    """Populated when status != 'ok'."""

    intermediate_metrics: dict[str, Any] = field(default_factory=dict)
    """Per-backend metrics (loss curve, grad norms, teacher-student
    consistency scores for distill, etc.). Written into VERIFICATION.md
    by postprocess.finalize()."""


@runtime_checkable
class Backend(Protocol):
    """Training backend interface.

    Implementations:
      - LerobotBackend: subprocess-invokes lerobot-train (fine-tune)
      - SnapFlowBackend: in-process SnapFlow loop (distill v0.3)
      - (future) OpenPIBackend, HFTrainerBackend, ConsistencyBackend
    """

    def fit(self, ctx: TrainerContext) -> CheckpointResult:
        """Run the training loop to completion. Return a CheckpointResult.

        Contract:
          - Must honor ctx.config.num_steps (or equivalent)
          - Must write training logs to ctx.training_log_path
          - Must fire ctx.hooks at documented lifecycle points
          - Must not write outside cfg.output/ (orchestrator owns that dir)
          - Must return status='ok' only if final checkpoint is on disk
        """
        ...


__all__ = ["Backend", "CheckpointResult", "TrainerContext"]
