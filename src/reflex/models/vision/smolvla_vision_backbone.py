"""SmolVLAVisionBackbone — SmolVLM2 vision tower wrapped as a spine VisionBackbone.

SmolVLA loads its VLM via
``AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")``,
giving a ``SmolVLMForConditionalGeneration`` instance whose vision tower lives
at ``vlm.model.vision_model``. Per lerobot ``smolvlm_with_expert.py:191-204``,
the canonical call sequence is::

    image_embed = vlm.model.vision_model(pixel_values=img).last_hidden_state
    image_embed = vlm.model.connector(image_embed)  # → text-hidden space

This backbone owns step 1 (vision tower). The ``connector`` lives on
``SmolVLALLMBackbone`` so the slot boundary mirrors the actual data flow.

Registered under ``VISION_BACKBONES`` per decision S-3 hybrid-registration.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from reflex.models.vision import VisionBackbone
from reflex.registry.components import VISION_BACKBONES


@VISION_BACKBONES.register
class SmolVLAVisionBackbone(VisionBackbone, nn.Module):
    """SmolVLM2 vision tower wrapper.

    Args (exactly one of model_id / model required):
        model_id: HF repo id to load the full VLM via
            ``AutoModelForImageTextToText.from_pretrained``; vision tower is
            extracted from it.
        model: A pre-built vision module (typically
            ``vlm.model.vision_model``) — used by
            ``SmolVLA.from_pretrained`` to share weights with the LLM
            backbone slot.
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        model: Any = None,
    ) -> None:
        nn.Module.__init__(self)
        if (model_id is None) == (model is None):
            raise ValueError(
                "Provide exactly one of `model_id` or `model` "
                f"(got model_id={model_id!r}, model={model!r})."
            )

        if model is not None:
            self.model = model
        else:
            from transformers import AutoModelForImageTextToText
            vlm = AutoModelForImageTextToText.from_pretrained(model_id)
            self.model = vlm.model.vision_model

    def forward(
        self,
        images: torch.Tensor,
        *args: Any,
        patch_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Images → patch embeddings (pre-connector).

        Args:
            images: ``[batch, channels, height, width]``. SmolVLM2's default
                image size is 512×512.
            patch_attention_mask: optional ``[batch, num_patches]`` bool mask;
                lerobot passes ``None`` in ``embed_image``.

        Returns:
            ``last_hidden_state`` ``[batch, num_patches, vision_hidden]``.
            Caller (typically SmolVLALLMBackbone.connector) projects to
            text-hidden space.
        """
        outputs = self.model(
            pixel_values=images.to(dtype=self.model.dtype),
            patch_attention_mask=patch_attention_mask,
        )
        return outputs.last_hidden_state

    def prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """Flatten weights for ``--inference-only-weights`` mode (lift #3)."""
        return {
            f"{prefix}{name}": param.detach()
            for name, param in self.named_parameters()
        }


__all__ = ["SmolVLAVisionBackbone"]
