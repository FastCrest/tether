"""PaliGemmaBackbone — PaliGemma's language path wrapped as a spine LLMBackbone.

Per Romir's 2026-05-20 design fork: SigLIP is split out as the
`vision_backbone` slot (see `models/vision/siglip_backbone.py`). This
backbone wraps **the rest of PaliGemma** — language_model (Gemma2) +
multi_modal_projector + token embeddings — for use in the BaseVLA spine's
`llm_backbone` slot.

How the slots split at runtime:

    images       → SigLIPBackbone        → image embeds [B, 256, vision_hidden]
                                         ↓
                   PaliGemmaBackbone     → multi_modal_projector
                                         → merge with text embeds
                                         → language_model.forward(inputs_embeds=…)
                                         → hidden states [B, seq, text_hidden]
                                         ↓
                   (action head consumes hidden states)

The PaliGemma model object is held in full (vision_tower attribute included)
but vision_tower is **never called** through this class — that's
SigLIPBackbone's responsibility. The vision_tower attribute is kept rather
than deleted so the model's state_dict matches the upstream HF checkpoint
naming (avoids state_dict load drift).

Loads via either:

- A HuggingFace model id (`PaliGemmaForConditionalGeneration.from_pretrained`)
- A pre-built `PaliGemmaForConditionalGeneration` instance (for tests; also
  used by `Pi0VLA.from_pretrained` in Day 4f where the same PaliGemma model
  is split between SigLIPBackbone(model=paligemma.model.vision_tower) and
  this class)

Registered under `LLM_BACKBONES` per decision S-3 hybrid-registration.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from reflex.models.llm import LLMBackbone
from reflex.registry.components import LLM_BACKBONES

if TYPE_CHECKING:
    pass


@LLM_BACKBONES.register
class PaliGemmaBackbone(LLMBackbone, nn.Module):
    """PaliGemma language path wrapper (vision_tower NOT called at runtime).

    Args (exactly one of model_id / model required):
        model_id: HF repo id to load via
            `PaliGemmaForConditionalGeneration.from_pretrained` (e.g.
            `"google/paligemma-3b-pt-224"`).
        model: A pre-built `PaliGemmaForConditionalGeneration` instance. Used
            by Day 4f Pi0VLA where the same model is split between
            SigLIPBackbone(model=paligemma.model.vision_tower) and this class.
        dtype: Optional dtype to cast the loaded model to (e.g. torch.bfloat16).
            None → leave at whatever the checkpoint stores (typically float32).

    Raises:
        ValueError: if neither or both of `model_id` / `model` are provided.
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
            from transformers import PaliGemmaForConditionalGeneration
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)

        if dtype is not None:
            self.model = self.model.to(dtype=dtype)

    # ── Convenience accessors that downstream composition code (Day 4f
    # Pi0VLA orchestrator) uses to compose the multimodal forward. Each
    # is a thin property so callers don't have to reach into `.model.model`.

    @property
    def language_model(self) -> nn.Module:
        """Gemma2 decoder (the language tower). Used directly by Pi0VLA's
        forward orchestrator to call `language_model(inputs_embeds=...)`."""
        return self.model.model.language_model

    @property
    def multi_modal_projector(self) -> nn.Module:
        """Image-embed → language-hidden-dim projection. Pi0VLA calls this
        on SigLIPBackbone's output before merging with text embeds."""
        return self.model.model.multi_modal_projector

    @property
    def embed_tokens(self) -> nn.Module:
        """Token embedding table. Pi0VLA uses this to embed input_ids
        before merging in the projected image embeds."""
        return self.language_model.embed_tokens

    @property
    def text_hidden_size(self) -> int:
        """Hidden dim of the language tower (1152 for PaliGemma-3B-pt-224)."""
        return self.model.config.text_config.hidden_size

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

        Two call shapes:

        1. `forward(input_ids, attention_mask)` — embed tokens internally,
           run through language_model. No image merging. Used for text-only
           probes + the chat model surface.

        2. `forward(inputs_embeds=<pre-merged>, attention_mask=...)` — caller
           pre-merged image embeds into the text-embed sequence. Used by
           Pi0VLA orchestrator at Day 4f.

        Returns the raw `language_model` output (typically a `BaseModelOutput`
        with `last_hidden_state` and `past_key_values`).
        """
        if inputs_embeds is None and input_ids is None:
            raise ValueError("Must provide either input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    def prepare_triton(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """Flatten weights for `--inference-only-weights` mode (lift #3).

        Excludes the vision_tower weights — those belong to SigLIPBackbone's
        prepare_triton output. Includes language_model + multi_modal_projector
        + the rest of PaliGemma.
        """
        out: dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            # Skip vision_tower — SigLIPBackbone owns those weights.
            # The dotted-path under self.model is e.g.
            # `model.model.vision_tower.vision_model.embeddings.patch_embedding.weight`.
            if ".vision_tower." in name:
                continue
            out[f"{prefix}{name}"] = param.detach().clone()
        return out


__all__ = ["PaliGemmaBackbone"]
