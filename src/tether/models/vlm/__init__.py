"""VLMBackbone abstract base — fused vision-language slot for the BaseVLA spine.

The 6th slot added per decision S-2 specifically for **fused VLMs** that
don't decompose into separate vision_backbone + llm_backbone (GR00T's
Eagle = SigLIP+Llama with internal cross-attention; future fused VLMs).

For 2-tower VLAs, use `VisionBackbone` + `LLMBackbone` separately. For
text-only encoders (DreamZero's T5), use `TextEncoder`.

Subclasses register via `@VLM_BACKBONES.register` from
`tether.registry.components`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class VLMBackbone(ABC):
    """Abstract fused vision-language model — (images, tokens) → joint embeddings.

    Subclass contract:

    - `__init__(self, config: Any)` — accept VLA-spec config.
    - `forward(self, images, input_ids, attention_mask, ...)` — abstract;
      VLA-specific. Eagle, for instance, returns image+text joint
      embeddings; future fused VLMs may differ.
    - `prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]` —
      no-op default; lift #3 fills in.

    Concrete examples (added in lift #1 Day 7):
    - `EagleBackbone` — GR00T's SigLIP+Qwen3+mlp1 fused stack
    """

    @abstractmethod
    def forward(
        self,
        images: Any,
        input_ids: Any,
        attention_mask: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """(images, tokens) → joint embeddings. VLA-specific output shape."""
        ...

    def prepare_triton(self, prefix: str = "") -> dict[str, "torch.Tensor"]:
        return {}


__all__ = ["VLMBackbone"]
