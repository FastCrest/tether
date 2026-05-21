"""VLAHead abstract base — action-prediction head for the BaseVLA spine.

Per decision S-2 + S-3. Subclasses register via `@VLA_HEADS.register`.

The head is the one component every VLA has — it produces actions. Heads
differ markedly across VLA families (flow-matching vs DiT diffusion vs
autoregressive argmax-over-bins) so the base ABC stays minimal.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class VLAHead(ABC):
    """Abstract action head — context embeddings → action chunk.

    Subclass contract:

    - `__init__(self, config: Any)` — accept VLA-spec config.
    - `forward(self, context, ...)` — abstract; the input shape depends
      on the head type. Flow-matching heads take VLM context + noise +
      timestep; DiT heads take VLM KV + noise + timestep; OpenVLA's
      argmax head takes Llama hidden states.
    - `prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]` —
      no-op default; lift #3 fills in.

    Concrete examples (added in lift #1 Days 4-7):
    - `FlowMatchingHead` — pi0/pi0.5/smolvla shared flow-matching head
    - `DiTHead` — GR00T's diffusion transformer head
    - `OpenVLAHead` — Llama-2 argmax-over-bins (lives in `exporters/openvla.py`
      shim per decision S-4, NOT on the spine)
    """

    @abstractmethod
    def forward(self, context: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    def prepare_triton(self, prefix: str = "") -> dict[str, "torch.Tensor"]:
        return {}


__all__ = ["VLAHead"]
