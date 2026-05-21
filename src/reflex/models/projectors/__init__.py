"""Projector abstract base — cross-modal projection slot for the BaseVLA spine.

Per decision S-2 + S-3. Subclasses register via `@PROJECTORS.register`.

Projectors handle dim/shape adaptation between components — e.g. mapping
robot state vector to VLM hidden space, or projecting action embeddings
from expert hidden to action_dim. Most VLAs need at least one; some
(DreamZero) embed projection inside the head.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class Projector(ABC):
    """Abstract cross-modal projection — input embeddings → output embeddings.

    Subclass contract:

    - `__init__(self, config: Any)` — accept VLA-spec config.
    - `forward(self, x, ...)` — abstract; the input shape depends on
      what's being projected (state vector, action embeddings, etc.).
    - `prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]` —
      no-op default; lift #3 fills in.

    Concrete examples (added in lift #1 Days 4-7):
    - `LinearProjector` — single nn.Linear, the common case
    - `StateProjector` — robot-state → VLM-hidden
    """

    @abstractmethod
    def forward(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    def prepare_triton(self, prefix: str = "") -> dict[str, "torch.Tensor"]:
        return {}


__all__ = ["Projector"]
