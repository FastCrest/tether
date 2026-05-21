"""VisionBackbone abstract base — vision-encoder slot for the BaseVLA spine.

Per decision S-2 + S-3 in `01_decisions/2026-05-19-fluxvla-lift-program-
decisions.md`. Subclasses register via `@VISION_BACKBONES.register` from
`reflex.registry.components` and are referenced from VLA spec dicts as
`{"type": "<ClassName>", ...}`.

The `prepare_triton(prefix="")` method is a concrete no-op default per
decision S-2 — bundling the lift #3 (inference-only-weights) interface
into the spine refactor frees lift #3 from per-component retrofit. Real
implementations land in lift #3 Day 1.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class VisionBackbone(ABC):
    """Abstract vision encoder — image(s) → embedding tensor.

    Subclass contract:

    - `__init__(self, config: Any)` — accept VLA-spec config; subclasses
      decide what config fields they need.
    - `forward(self, images, ...)` — abstract; the call shape is
      VLA-specific (single image vs multi-camera; tensor vs PIL etc).
    - `prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]` —
      concrete no-op default; override in lift #3 to flatten weights.

    Concrete examples (added in lift #1 Days 4-7):
    - `SigLIPBackbone` — PaliGemma's vision tower
    - `DinoSigLIPBackbone` — fused Dino + SigLIP
    - `SmolVLAVisionBackbone` — SmolVLM2 vision tower
    """

    @abstractmethod
    def forward(self, images: Any) -> Any:
        """Image → vision embeddings. VLA-specific input/output shape."""
        ...

    def prepare_triton(self, prefix: str = "") -> dict[str, "torch.Tensor"]:
        """Flatten this backbone's weights for `--inference-only-weights` mode.

        No-op default — subclasses override in lift #3 to return
        `{f"{prefix}<param_path>": tensor}` for every parameter.
        """
        return {}


__all__ = ["VisionBackbone"]
