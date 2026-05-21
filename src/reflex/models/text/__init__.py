"""TextEncoder abstract base — pure text encoder slot for the BaseVLA spine.

The 7th slot per decision S-2 — specifically for VLAs where the text
encoder is structurally separate from the language model (DreamZero's T5
is a text-only encoder, NOT a RoPE'd LLM). For 2-tower VLAs where the
language model handles both encoding and attention, use `LLMBackbone`.

Subclasses register via `@TEXT_ENCODERS.register` from
`reflex.registry.components`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class TextEncoder(ABC):
    """Abstract text-only encoder — input_ids → text embeddings.

    Subclass contract:

    - `__init__(self, config: Any)` — accept VLA-spec config.
    - `forward(self, input_ids, attention_mask, ...)` — abstract;
      VLA-specific output shape (T5 returns encoder hidden states; future
      encoders may differ).
    - `prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]` —
      no-op default; lift #3 fills in.

    Concrete examples (added in lift #7 DreamZero):
    - `T5Encoder` — DreamZero's T5 text encoder
    """

    @abstractmethod
    def forward(
        self,
        input_ids: Any,
        attention_mask: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ...

    def prepare_triton(self, prefix: str = "") -> dict[str, "torch.Tensor"]:
        return {}


__all__ = ["TextEncoder"]
