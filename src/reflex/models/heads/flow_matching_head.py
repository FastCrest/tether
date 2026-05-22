"""FlowMatchingHead — pi0/pi05/smolvla shared flow-matching head wrapped for the BaseVLA spine.

Per the Day 4 design (lift #1 plan): pi0/pi05/smolvla all share a single
expert-decoder + suffix architecture for action prediction. The `ExpertStack`
primitive lives at `reflex.models.heads.expert_stack` (moved there in
Day 4g cleanup; previously lived in `exporters/smolvla_exporter.py` and
the spine reached into the exporters package — now cleanly separated).

FlowMatchingHead is a thin spine-compatible wrapper around an `ExpertStack`
instance. It:

- Provides the `VLAHead` ABC contract (`forward()` + `prepare_triton()`)
- Delegates the actual denoising computation to the wrapped ExpertStack
- Does NOT reimplement the expert — preserves bit-identical behavior with
  the existing pi0/pi05/smolvla exporters

Construction is via a pre-built ExpertStack or via state_dict + family
dispatch (which still uses the exporter-side build helpers — those stay
in exporters/* because they're tightly coupled to checkpoint-key parsing).

Registered under `VLA_HEADS` per decision S-3 hybrid-registration.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from reflex.models.heads import VLAHead
from reflex.registry.components import VLA_HEADS

if TYPE_CHECKING:
    pass


@VLA_HEADS.register
class FlowMatchingHead(VLAHead, nn.Module):
    """Spine wrapper for pi0/pi05/smolvla's ExpertStack.

    Args (exactly one of expert_stack / state_dict required):
        expert_stack: A pre-built `ExpertStack` instance from
            `src/reflex/exporters/smolvla_exporter.py`. Produced by
            `build_pi0_expert_stack(state_dict)` /
            `build_pi05_expert_stack(state_dict)` /
            `build_expert_stack(state_dict, head_dim)` per VLA family.
        state_dict: Raw checkpoint state_dict + `vla_family` (one of
            "pi0", "pi05", "smolvla") — head dispatches to the right
            builder. Convenience constructor; same shape as direct
            expert_stack=.
        vla_family: Required when `state_dict` is provided. Picks the
            build function. Must be one of "pi0" / "pi05" / "smolvla".
        head_dim: Required for SmolVLA's build_expert_stack (which needs
            the VLM head_dim — typically 64 for SmolLM2). Ignored for
            pi0/pi05.

    Raises:
        ValueError: if neither or both of expert_stack / state_dict
            provided; if state_dict provided without vla_family.
    """

    SUPPORTED_FAMILIES: tuple[str, ...] = ("pi0", "pi05", "smolvla")

    def __init__(
        self,
        *,
        expert_stack: Any = None,
        state_dict: dict[str, torch.Tensor] | None = None,
        vla_family: str | None = None,
        head_dim: int | None = None,
    ) -> None:
        nn.Module.__init__(self)
        if (expert_stack is None) == (state_dict is None):
            raise ValueError(
                "Provide exactly one of `expert_stack` or `state_dict` "
                f"(got expert_stack={type(expert_stack).__name__ if expert_stack else None!r}, "
                f"state_dict={'<dict>' if state_dict else None!r})."
            )

        if state_dict is not None:
            if vla_family not in self.SUPPORTED_FAMILIES:
                raise ValueError(
                    f"vla_family must be one of {self.SUPPORTED_FAMILIES} when "
                    f"state_dict is provided (got {vla_family!r})."
                )
            expert_stack = self._build_from_state_dict(
                state_dict, vla_family=vla_family, head_dim=head_dim,
            )

        self.expert_stack = expert_stack

    @staticmethod
    def _build_from_state_dict(
        state_dict: dict[str, torch.Tensor],
        *,
        vla_family: str,
        head_dim: int | None,
    ) -> Any:
        """Dispatch to the right exporter-side builder by VLA family.

        Imports are lazy — they pull in heavy model code (PyTorch +
        transformers) that tests don't want at import time.
        """
        if vla_family == "pi0":
            from reflex.exporters.pi0_exporter import build_pi0_expert_stack
            stack, _meta = build_pi0_expert_stack(state_dict)
            return stack
        elif vla_family == "pi05":
            from reflex.exporters.pi0_exporter import build_pi05_expert_stack
            stack, _meta = build_pi05_expert_stack(state_dict)
            return stack
        elif vla_family == "smolvla":
            from reflex.exporters.smolvla_exporter import build_expert_stack
            if head_dim is None:
                raise ValueError(
                    "head_dim required for SmolVLA (the VLM head_dim — "
                    "typically 64 for SmolLM2). pi0/pi05 don't need it."
                )
            stack, _meta = build_expert_stack(state_dict, head_dim=head_dim)
            return stack
        else:
            # Unreachable per SUPPORTED_FAMILIES validation above; defensive.
            raise ValueError(f"Unknown vla_family: {vla_family!r}")

    # ── ABC contract ────────────────────────────────────────────────────

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        *args: Any,
        vlm_k: torch.Tensor | None = None,
        vlm_v: torch.Tensor | None = None,
        prefix_k: torch.Tensor | None = None,
        prefix_v: torch.Tensor | None = None,
        prefix_offset: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
        state_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """One denoising step — delegates to the wrapped expert module.

        Supports both expert flavors:

        - `ExpertStack` (SmolVLA / pi0.5 cross-attn-only or expert-only path):
          pass `vlm_k`/`vlm_v` for the cross-attn layers, plus optional
          `prefix_offset` + `kv_mask`.
        - `Pi0ExpertStackWithPrefix` (pi0 prefix-concat-on-every-layer path):
          pass `prefix_k`/`prefix_v` (per-layer K/V from PaliGemma's
          past_key_values). No `vlm_k`/`vlm_v`, no `kv_mask`.

        Dispatch is by signature inspection — the wrapped expert's `forward`
        signature determines which kwargs are forwarded. This keeps the
        wrapped class blissfully unaware of FlowMatchingHead.

        Args:
            noisy_actions: `[batch, chunk, action_dim]` noised action tensor.
            timestep: `[batch]` flow-matching timestep.
            position_ids: `[batch, chunk]` action position indices.
            vlm_k: SmolVLA cross-attn per-layer K cache (mutually exclusive
                with prefix_k).
            vlm_v: paired V cache.
            prefix_k: pi0 prefix-concat per-layer K
                `[L, B, prefix_len, nkv, hd]` (mutually exclusive with vlm_k).
            prefix_v: paired V.
            prefix_offset: VLM prefix length per batch element (for position-id
                offsetting in ExpertStack's self-attention layers; ignored by
                Pi0ExpertStackWithPrefix).
            kv_mask: optional KV-attention mask (ExpertStack cross-attn only).

        Returns:
            `[batch, chunk, action_dim]` denoised actions.
        """
        # Detect prefix-aware expert variants — Pi0 (with state) or Pi05
        # (action-only suffix + AdaRMSNorm time conditioning). Both consume
        # per-layer prefix K/V and route through this head.
        from reflex.exporters.pi0_prefix import (
            Pi0ExpertStackWithPrefix,
            Pi05ExpertStackWithPrefix,
        )
        if isinstance(self.expert_stack, Pi05ExpertStackWithPrefix):
            if prefix_k is None or prefix_v is None:
                raise ValueError(
                    "Pi05ExpertStackWithPrefix requires prefix_k + prefix_v "
                    "(per-layer VLM prefix K/V from PaliGemma's past_key_values)."
                )
            # pi0.5 has NO state_emb — silently drop if passed
            return self.expert_stack(
                noisy_actions=noisy_actions,
                timestep=timestep,
                position_ids=position_ids,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                attn_mask=attn_mask,
            )
        if isinstance(self.expert_stack, Pi0ExpertStackWithPrefix):
            if prefix_k is None or prefix_v is None:
                raise ValueError(
                    "Pi0ExpertStackWithPrefix requires prefix_k + prefix_v "
                    "(per-layer VLM prefix K/V from PaliGemma's past_key_values)."
                )
            return self.expert_stack(
                noisy_actions=noisy_actions,
                timestep=timestep,
                position_ids=position_ids,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                state_emb=state_emb,
                attn_mask=attn_mask,
            )

        # Default path — ExpertStack (cross-attn or self-attn).
        return self.expert_stack(
            noisy_actions=noisy_actions,
            timestep=timestep,
            position_ids=position_ids,
            vlm_k=vlm_k,
            vlm_v=vlm_v,
            prefix_offset=prefix_offset,
            kv_mask=kv_mask,
        )

    def prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """Flatten every parameter in the wrapped ExpertStack under prefix."""
        return {
            f"{prefix}expert_stack.{name}": param.detach()
            for name, param in self.expert_stack.named_parameters()
        }


__all__ = ["FlowMatchingHead"]
