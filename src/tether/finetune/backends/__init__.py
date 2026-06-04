"""Training backend registry for tether.finetune.

The Backend protocol normalizes over different training code paths
(lerobot-train subprocess, SnapFlow in-process, future openpi-JAX)
so the orchestrator (run.py) stays paradigm-agnostic.

`phase="train"` routes to the lerobot-subprocess backend by default.
`phase="distill"` routes to the SnapFlow backend (once Phase B lands).
"""
from __future__ import annotations

from tether.finetune.backends.base import (
    Backend,
    CheckpointResult,
    TrainerContext,
)

__all__ = [
    "Backend",
    "CheckpointResult",
    "TrainerContext",
    "resolve_backend",
]


def resolve_backend(cfg) -> Backend:
    """Pick a Backend instance based on FinetuneConfig.phase + cfg.backend.

    Dispatch table:
      phase="train", backend="lerobot"  → LerobotBackend  (default fine-tune)
      phase="distill", method="snapflow" → SnapFlowBackend (v0.3)
      phase="train", backend="openpi"    → NotImplementedError (v0.5+)
      phase="train", backend="hf"        → NotImplementedError (v0.5+)
      phase="distill", method="consistency" → NotImplementedError (v0.5+ GR00T)

    Backends are imported lazily so an engineer running `tether finetune`
    with the default phase doesn't import SnapFlow's teacher-loader
    machinery (pulls in peft + lerobot policy classes).
    """
    phase = getattr(cfg, "phase", "train")
    if phase == "train":
        from tether.finetune.backends.lerobot_backend import LerobotBackend
        return LerobotBackend()
    if phase == "distill":
        # SnapFlow is the only v0.3 distillation method. Consistency
        # Policy (DDPM) lands in v0.5+ for GR00T.
        method = getattr(cfg, "distillation_method", "snapflow")
        if method == "snapflow":
            from tether.finetune.backends.snapflow_backend import SnapFlowBackend
            return SnapFlowBackend()
        raise NotImplementedError(
            f"distillation_method={method!r} not supported in v0.3. "
            f"Only 'snapflow' is available. 'consistency' lands in v0.5+ "
            f"when GR00T DDPM support ships."
        )
    raise ValueError(
        f"Unknown phase={phase!r}. Valid: 'train' | 'distill'."
    )
