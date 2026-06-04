"""LLMBackbone abstract base — language-model slot for the BaseVLA spine.

Per decision S-2 + S-3. Subclasses register via `@LLM_BACKBONES.register`.

This slot is for **2-tower** VLAs where the language model is a distinct
component (PaliGemma + Gemma expert; SmolLM2 + action expert). For
**fused** VLMs like GR00T's Eagle, use `VLMBackbone` instead (the spine's
6th slot, decision S-2).

For text-only encoders without RoPE'd attention (DreamZero's T5), use
`TextEncoder` instead.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class LLMBackbone(ABC):
    """Abstract language model — tokens → contextualized embeddings + KV.

    Subclass contract:

    - `__init__(self, config: Any)` — accept VLA-spec config.
    - `forward(self, input_ids, attention_mask, ...)` — abstract; the
      exact signature is VLA-specific (single-tower vs dual-tower with
      action expert, with/without past_kv).
    - `prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]` —
      no-op default; lift #3 fills in.

    Concrete examples (added in lift #1 Days 4-7):
    - `PaliGemmaBackbone` — Pi0/Pi0.5's language tower (without expert)
    - `PaliGemmaWithExpert` — Pi0.5's fused paligemma + gemma_expert
    - `SmolLM2Backbone` — SmolVLA's language tower
    - `GemmaExpertBackbone` — standalone Gemma action expert (Pi0 path)
    """

    @abstractmethod
    def forward(
        self,
        input_ids: Any,
        attention_mask: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Token IDs → embeddings + (optional) KV cache. VLA-specific shape."""
        ...

    def prepare_triton(self, prefix: str = "") -> dict[str, "torch.Tensor"]:
        return {}


__all__ = ["LLMBackbone"]
