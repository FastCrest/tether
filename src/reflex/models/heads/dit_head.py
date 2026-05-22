"""DITHead — GR00T's diffusion transformer head, wrapped as a spine VLAHead.

GR00T uses a **diffusion** (DiT) head rather than the flow-matching head
that pi0/pi05/SmolVLA share. The head ingests:

- noisy action chunk ``[B, chunk, raw_action_dim]``
- timestep ``[B]`` (scalar in [0, 1])
- position_ids ``[B, chunk]``
- optional state ``[B, state_dim]`` (per-embodiment state encoder)
- optional VLM-KV ``[B, vlm_seq, vlm_hidden]`` (cross-attn to Eagle's output)

And emits velocity tokens ``[B, chunk, raw_action_dim]`` (the diffusion
step's predicted velocity at the current timestep).

This wraps the existing ``GR00TFullStack`` from
``reflex.exporters.gr00t_exporter`` — that module encapsulates:

- ``GR00TActionEncoder`` (per-embodiment, embeds raw actions + time)
- ``GR00TStateEncoder`` (per-embodiment, embeds state as a token)
- ``GR00TExpertStack`` (32 DiT blocks with alternating cross/self attn)
- ``GR00TActionDecoder`` (per-embodiment, decodes velocity tokens → raw)

No behavior change vs the legacy exporter — same builder, same numerics.
Day 11 sunsets the legacy direct-build path.

Registered under ``VLA_HEADS`` per decision S-3 hybrid-registration.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from reflex.models.heads import VLAHead
from reflex.registry.components import VLA_HEADS


@VLA_HEADS.register
class DITHead(VLAHead, nn.Module):
    """GR00T diffusion transformer head — wraps ``GR00TFullStack``.

    Args (exactly one of full_stack / state_dict required):
        full_stack: A pre-built ``GR00TFullStack`` (or ``GR00TExpertStack``
            for the expert-only path). Used by tests + by
            ``GR00TVLA.from_pretrained`` when sharing weights across slots.
        state_dict: Raw GR00T N1.6 checkpoint. Builds the full stack via
            ``build_gr00t_full_stack`` (existing path from gr00t_exporter).
        embodiment_id: Which embodiment's encoder/decoder weights to bind
            (per-embodiment leading-dim 32). N1.6 default 0.
    """

    def __init__(
        self,
        *,
        full_stack: Any = None,
        state_dict: dict[str, torch.Tensor] | None = None,
        embodiment_id: int = 0,
    ) -> None:
        nn.Module.__init__(self)
        if (full_stack is None) == (state_dict is None):
            raise ValueError(
                "Provide exactly one of `full_stack` or `state_dict` "
                "(got full_stack=%r, state_dict=%s)."
                % (full_stack, "None" if state_dict is None else
                   f"<dict {len(state_dict)} keys>")
            )

        if full_stack is not None:
            self.full_stack = full_stack
            self.metadata: dict[str, Any] = {}
        else:
            from reflex.exporters.gr00t_exporter import build_gr00t_full_stack
            self.full_stack, self.metadata = build_gr00t_full_stack(
                state_dict=state_dict,
                embodiment_id=embodiment_id,
            )

        self.embodiment_id = embodiment_id

    # ── Convenience accessors ──────────────────────────────────────────

    @property
    def dit_stack(self) -> nn.Module:
        """The 32-block DiT expert (``GR00TExpertStack``)."""
        return self.full_stack.dit

    @property
    def action_encoder(self) -> nn.Module:
        return self.full_stack.action_encoder

    @property
    def action_decoder(self) -> nn.Module:
        return self.full_stack.action_decoder

    @property
    def state_encoder(self) -> nn.Module | None:
        return getattr(self.full_stack, "state_encoder", None)

    # ── ABC contract ────────────────────────────────────────────────────

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        *args: Any,
        state: torch.Tensor | None = None,
        vlm_kv: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """One DiT step — delegates to ``GR00TFullStack.forward``.

        Args:
            noisy_actions: ``[B, chunk, raw_action_dim]`` noised action.
            timestep: ``[B]`` scalar timestep (≈ flow-matching's t).
            position_ids: ``[B, chunk]`` action positions; used by the
                DiT's internal pos_embed when state is None.
            state: ``[B, state_dim]`` raw robot state (optional; per-embodiment
                state encoder embeds it to a 1536-dim token).
            vlm_kv: ``[B, vlm_seq, vlm_hidden]`` Eagle's hidden states from
                ``EagleBackbone.forward`` — cross-attn target for even DiT
                blocks. None → zero placeholder (export-only fallback).

        Returns:
            ``[B, chunk, raw_action_dim]`` predicted velocity.
        """
        if timestep is None or position_ids is None:
            raise ValueError(
                "DITHead.forward requires both `timestep` and `position_ids`."
            )
        return self.full_stack(
            noisy_actions,
            timestep,
            position_ids,
            state=state,
            vlm_kv=vlm_kv,
        )

    def prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """Flatten weights for ``--inference-only-weights`` mode (lift #3)."""
        return {
            f"{prefix}{name}": param.detach()
            for name, param in self.named_parameters()
        }


__all__ = ["DITHead"]
