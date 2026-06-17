"""EagleBackbone — GR00T N1.6's fused Eagle VLM wrapped as a spine VLMBackbone.

Eagle is the fused SigLIP-so400m + Qwen2-0.5B + mlp1 model that NVIDIA's
GR00T N1.6 uses as its vision-language backbone. Unlike pi0/pi05/SmolVLA
which split vision + language into two slots, Eagle ships as **one fused
module** — it has internal cross-attention between the SigLIP image
embeddings and the Qwen2 text decoder, so the spine treats it as a single
``vlm_backbone`` slot.

This is THE proof that the BaseVLA spine's 6-slot design works for fused
VLMs (per decision S-2 motivation: "GR00T's Eagle (fused SigLIP + Llama)
doesn't decompose into the standard vision_backbone + llm_backbone
pattern, so it gets its own `vlm_backbone` slot").

This backbone wraps the existing ``EagleExportStack`` (which lives in
``tether.exporters.eagle_export_stack`` and is built from the vendored
``Eagle25VLForConditionalGeneration`` in ``tether.exporters.eagle_vendor``).
No new model loading code — same vendored path the exporter has used since
Day 0.

Registered under ``VLM_BACKBONES`` per decision S-3 hybrid-registration.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from tether.models.vlm import VLMBackbone
from tether.registry.components import VLM_BACKBONES


@VLM_BACKBONES.register
class EagleBackbone(VLMBackbone, nn.Module):
    """GR00T's fused Eagle VLM wrapper.

    Args (exactly one of state_dict / stack required):
        state_dict: Raw GR00T N1.6 checkpoint state_dict. Eagle is
            extracted via ``build_eagle_export_stack`` (existing path
            from ``tether.exporters.eagle_export_stack``).
        stack: A pre-built ``EagleExportStack`` instance — used by tests
            + by ``GR00TVLA.from_pretrained`` to avoid rebuilding when
            the same state_dict is shared across slots.
        select_layer: Which Qwen2 hidden layer to surface as the VLM-KV
            input for the DiT cross-attn (default -1 = last).
    """

    def __init__(
        self,
        *,
        state_dict: dict[str, torch.Tensor] | None = None,
        stack: Any = None,
        select_layer: int = -1,
    ) -> None:
        nn.Module.__init__(self)
        if (state_dict is None) == (stack is None):
            raise ValueError(
                "Provide exactly one of `state_dict` or `stack` "
                "(got state_dict=%s, stack=%s)."
                % ("None" if state_dict is None else f"<dict {len(state_dict)} keys>", stack)
            )

        if stack is not None:
            self.stack = stack
            self.metadata: dict[str, Any] = getattr(stack, "metadata", {})
        else:
            from tether.exporters.eagle_export_stack import build_eagle_export_stack
            self.stack, self.metadata = build_eagle_export_stack(
                state_dict=state_dict,
                select_layer=select_layer,
            )

        self.select_layer = select_layer

    # ── Convenience accessors ──────────────────────────────────────────

    @property
    def llm_hidden(self) -> int:
        """Qwen2 hidden size (2048 for N1.6's Qwen2-0.5B)."""
        return int(self.metadata.get("llm_hidden", 2048))

    @property
    def vit_hidden(self) -> int:
        """SigLIP vision hidden (1152 for SigLIP-so400m)."""
        return int(self.metadata.get("vit_hidden", 1152))

    # ── ABC contract ────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *args: Any,
        image_flags: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """(images, tokens) → Qwen2 hidden_states at select_layer.

        Args:
            images: ``[batch, 3, H, W]`` SigLIP pixel inputs.
            input_ids: ``[batch, seq]`` Qwen2 token ids (with image-token
                placeholders at the front per the export contract).
            attention_mask: ``[batch, seq]`` 1=valid, 0=padding. Required
                for non-trivial inference; passed as 1s if None.
            image_flags: ``[batch]`` 1=image present, 0=drop. Defaults
                to all-ones (image always present).

        Returns:
            ``[batch, seq, llm_hidden]`` — the VLM-KV input that
            ``DITHead`` cross-attends to.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if image_flags is None:
            image_flags = torch.ones(images.shape[0], dtype=torch.long, device=images.device)
        return self.stack(images, input_ids, attention_mask, image_flags)

    def prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """Flatten weights for ``--inference-only-weights`` mode (lift #3)."""
        return {
            f"{prefix}{name}": param.detach().clone()
            for name, param in self.named_parameters()
        }


__all__ = ["EagleBackbone"]
