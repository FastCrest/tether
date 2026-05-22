"""LinearProjector — single nn.Linear, the most common projection.

The first concrete component on the BaseVLA spine (lift #1 Day 4a per
`features/03_export/basevla-spine_plan.md`). Proves the spine + Registry
+ Projector ABC compose end-to-end with a real registered class.

Most VLAs need at least one linear projection — common shapes:

- robot state vector → VLM hidden dim (e.g. `state_proj: 32 → 960` for SmolVLA)
- action embedding → action dim (`action_out_proj: expert_hidden → 32`)
- expert hidden → projected hidden (`action_in_proj: 32 → expert_hidden`)

The class wraps a single `nn.Linear` with optional bias. Subclasses can
extend for fused matmul-bias-activation patterns in lift #3.

Registered under `PROJECTORS` per decision S-3 hybrid-registration pattern.
The class registers automatically at import time via the decorator.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from reflex.models.projectors import Projector
from reflex.registry.components import PROJECTORS

if TYPE_CHECKING:
    pass


@PROJECTORS.register
class LinearProjector(Projector, nn.Module):
    """Single nn.Linear with optional bias.

    Args:
        in_dim: input feature dimension
        out_dim: output feature dimension
        bias: whether to include a bias term (default True — matches PyTorch's
            nn.Linear default)

    The class inherits from both Projector (the spine ABC, declares contract)
    and nn.Module (PyTorch state-dict + .to(device) machinery). The dual
    inheritance is the load-bearing trick that lets spine components work
    with both BaseVLA's load_state_dict + standard PyTorch tooling.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        # Initialize nn.Module first to set up _parameters dict.
        nn.Module.__init__(self)
        # Projector is ABC with no __init__ state, so no super call needed for it.
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply the linear projection. Extra args/kwargs ignored for ABC
        signature compatibility — concrete Projector subclasses can use
        them in subclassed forward()."""
        return self.linear(x)

    def prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """Flatten weights for `--inference-only-weights` mode (lift #3).

        Returns keys matching standard nn.Linear naming so the runtime's
        functional forward path can bind to them:

            {prefix}linear.weight  (out_dim, in_dim)
            {prefix}linear.bias    (out_dim,)  — only present when bias=True
        """
        out: dict[str, torch.Tensor] = {
            f"{prefix}linear.weight": self.linear.weight.detach().clone()
        }
        if self.linear.bias is not None:
            out[f"{prefix}linear.bias"] = self.linear.bias.detach().clone()
        return out


__all__ = ["LinearProjector"]
