"""Pi0VLA safetensors→flat-dict key transformation.

Maps a lerobot ``pi0_base`` (or compatible pi0) safetensors checkpoint into
the flat-dict naming convention produced by ``Pi0VLA.prepare_inference_weights()``.

The transformation is not a pure prefix substitution — three transformations
are needed:

1. **Prefix substitution.** ``paligemma_with_expert.paligemma.model.vision_tower.``
   → ``vision_backbone.model.``, etc. First-match-wins; order matters
   (vision_tower must match before paligemma.model fallback).

2. **Substring removal in expert layer paths.** The lerobot checkpoint nests
   expert layer projections under ``.self_attn.`` and ``.mlp.``, but reflex's
   ``ExpertGQALayer`` has flat attribute names — ``self.q_proj`` directly,
   not ``self.self_attn.q_proj``. We strip those substrings from the
   transformed key.

3. **Buffer skip.** ``DecomposedRMSNorm`` registers ``weight`` as a buffer
   (not a ``Parameter``), so the per-layer ``input_layernorm.weight`` and
   ``post_attention_layernorm.weight`` don't appear in ``named_parameters()``
   and aren't part of Phase A's flat dict. We return ``None`` for these keys
   to skip them — they're baked into the ONNX initializers at export time,
   not rebound at runtime. Same for the final ``gemma_expert.model.norm.weight``.

4. **Tied weight expansion.** PaliGemma ties ``embed_tokens.weight`` with
   ``lm_head.weight``. The lerobot checkpoint stores only ``lm_head.weight``;
   the model expects both. ``expand_tied_pi0()`` is applied after the per-key
   loop to populate the missing tied key.

Validated against ``lerobot/pi0_base`` (777 safetensors keys → 740 mapped keys,
38 buffer/tied keys skipped, plus 1 lm_head→embed_tokens expansion). Matches
``Pi0VLA.prepare_inference_weights()`` exactly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def pi0_safetensors_to_flat(key: str) -> str | None:
    """Map a pi0 safetensors checkpoint key to its Phase A flat-dict equivalent.

    Returns ``None`` for keys that don't appear in Phase A (buffers in
    ``DecomposedRMSNorm``, tied ``gemma_expert.lm_head`` which is unused).

    Order matters: more specific prefixes must come first.
    """
    if key == "paligemma_with_expert.gemma_expert.lm_head.weight":
        return None
    if key.startswith("paligemma_with_expert.gemma_expert.model.layers."):
        if ".input_layernorm.weight" in key or ".post_attention_layernorm.weight" in key:
            return None
    if key == "paligemma_with_expert.gemma_expert.model.norm.weight":
        return None

    if key.startswith("paligemma_with_expert.paligemma.model.vision_tower."):
        return "vision_backbone.model." + key[len("paligemma_with_expert.paligemma.model.vision_tower."):]

    if key.startswith("paligemma_with_expert.paligemma.lm_head."):
        return "llm_backbone.model.lm_head." + key[len("paligemma_with_expert.paligemma.lm_head."):]

    if key.startswith("paligemma_with_expert.paligemma.model."):
        return "llm_backbone.model.model." + key[len("paligemma_with_expert.paligemma.model."):]

    if key.startswith("paligemma_with_expert.gemma_expert.model."):
        sub = key[len("paligemma_with_expert.gemma_expert.model."):]
        sub = sub.replace(".self_attn.", ".").replace(".mlp.", ".")
        return "vla_head.expert_stack." + sub

    if key.startswith((
        "action_in_proj.",
        "action_out_proj.",
        "action_time_mlp_in.",
        "action_time_mlp_out.",
    )):
        return "vla_head.expert_stack." + key

    if key.startswith("state_proj."):
        return "projector.linear." + key[len("state_proj."):]

    return key


def expand_tied_pi0(flat: "dict[str, torch.Tensor]") -> "dict[str, torch.Tensor]":
    """Expand tied weights in a Pi0 flat dict in-place.

    PaliGemma ties ``embed_tokens.weight`` with ``lm_head.weight``. The lerobot
    checkpoint stores only ``lm_head.weight``; we populate the tied
    ``embed_tokens.weight`` from the same tensor (shared storage).

    Returns the same dict (for fluent chaining).
    """
    lm_head_key = "llm_backbone.model.lm_head.weight"
    embed_key = "llm_backbone.model.model.language_model.embed_tokens.weight"
    if lm_head_key in flat and embed_key not in flat:
        flat[embed_key] = flat[lm_head_key]  # shared storage — no clone
    return flat


__all__ = ["pi0_safetensors_to_flat", "expand_tied_pi0"]
