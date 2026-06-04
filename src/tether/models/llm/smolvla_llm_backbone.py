"""SmolVLALLMBackbone — SmolVLM2 language path wrapped as a spine LLMBackbone.

Mirrors ``PaliGemmaBackbone`` but for SmolVLM2's
``SmolVLMForConditionalGeneration``. The vision tower has been split out to
``SmolVLAVisionBackbone`` (see ``models/vision/smolvla_vision_backbone.py``);
this backbone owns the **connector** + **text_model** + token embeddings.

Per lerobot ``smolvlm_with_expert.py``::

    vlm.model.vision_model     → SmolVLAVisionBackbone
    vlm.model.connector        → SmolVLALLMBackbone.connector
    vlm.model.text_model       → SmolVLALLMBackbone.text_model

The vision_model attribute is **kept** rather than deleted so the model's
state_dict matches the upstream HF checkpoint naming (avoids state_dict load
drift). It's never called through this class.

Registered under ``LLM_BACKBONES`` per decision S-3 hybrid-registration.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from tether.models.llm import LLMBackbone
from tether.registry.components import LLM_BACKBONES


@LLM_BACKBONES.register
class SmolVLALLMBackbone(LLMBackbone, nn.Module):
    """SmolVLM2 language path wrapper (vision_model NOT called at runtime).

    Args (exactly one of model_id / model required):
        model_id: HF repo id loaded via
            ``AutoModelForImageTextToText.from_pretrained`` (e.g.
            ``"HuggingFaceTB/SmolVLM2-500M-Video-Instruct"``).
        model: A pre-built ``SmolVLMForConditionalGeneration`` instance —
            used by ``SmolVLA.from_pretrained`` to share weights with the
            vision backbone slot.
        dtype: Optional dtype to cast the loaded model to (e.g.
            ``torch.bfloat16``).
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        model: Any = None,
        dtype: torch.dtype | None = None,
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
            self.model = AutoModelForImageTextToText.from_pretrained(model_id)

        if dtype is not None:
            self.model = self.model.to(dtype=dtype)

    # ── Convenience accessors for SmolVLA orchestrator ────────────────

    @property
    def text_model(self) -> nn.Module:
        """SmolLM2 decoder. Used directly by SmolVLA's forward orchestrator
        to call ``text_model(inputs_embeds=...)``."""
        return self.model.model.text_model

    @property
    def connector(self) -> nn.Module:
        """Image-embed → text-hidden projection. SmolVLA calls this on
        SmolVLAVisionBackbone's output before merging with text embeds.
        Equivalent to PaliGemma's ``multi_modal_projector``."""
        return self.model.model.connector

    @property
    def embed_tokens(self) -> nn.Module:
        """Token embedding table. SmolVLA uses this to embed input_ids
        before merging in the projected image embeds."""
        return self.text_model.get_input_embeddings()

    @property
    def text_hidden_size(self) -> int:
        """Hidden dim of the language tower."""
        return int(self.model.config.text_config.hidden_size)

    @property
    def text_head_dim(self) -> int:
        """VLM head_dim — used to derive the expert's GQA shapes."""
        return int(self.model.config.text_config.head_dim)

    @property
    def num_text_layers(self) -> int:
        """Number of layers in the text decoder (16 for SmolVLM2-500M)."""
        return int(self.model.config.text_config.num_hidden_layers)

    # ── ABC contract ────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *args: Any,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the language path.

        Two call shapes (same contract as PaliGemmaBackbone):

        1. ``forward(input_ids, attention_mask)`` — embed tokens internally.
        2. ``forward(inputs_embeds=<pre-merged>, attention_mask=...)`` —
           caller pre-merged image embeds (connector output) with text embeds.

        Returns the raw text_model output.
        """
        if inputs_embeds is None and input_ids is None:
            raise ValueError("Must provide either input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    def prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """Flatten weights for ``--inference-only-weights`` mode (lift #3).

        Excludes vision_model weights — those belong to SmolVLAVisionBackbone.
        Includes text_model + connector + the rest of SmolVLM2.
        """
        out: dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            if ".vision_model." in name:
                continue
            out[f"{prefix}{name}"] = param.detach().clone()
        return out


__all__ = ["SmolVLALLMBackbone"]
