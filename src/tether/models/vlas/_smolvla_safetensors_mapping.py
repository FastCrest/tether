"""SmolVLA safetensors→flat-dict key transformation.

Maps a lerobot ``smolvla_base`` (or compatible SmolVLA) safetensors
checkpoint into the flat-dict naming convention produced by
``SmolVLA.prepare_inference_weights()``.

SmolVLA differs from Pi0/Pi0.5:
1. All keys prefixed with ``model.`` (lerobot convention)
2. VLM is ``vlm_with_expert.vlm.*`` (SmolVLM2) not ``paligemma_with_expert.paligemma.*``
3. Expert is ``vlm_with_expert.lm_expert.*`` not ``gemma_expert.*``
4. Expert uses ``ExpertGQALayer`` (same as Pi0) — DecomposedRMSNorm buffer → SKIP
5. VLM depth may differ between checkpoint and runtime (SmolVLA uses
   reduced layers at training time). The flat-dict only contains what's
   in the checkpoint; runtime must match.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def smolvla_safetensors_to_flat(key: str) -> str | None:
    """Map a SmolVLA safetensors checkpoint key to its Phase A flat-dict equivalent."""

    # Expert layer norms (DecomposedRMSNorm buffer — SKIP, same as Pi0)
    if key.startswith("model.vlm_with_expert.lm_expert.layers."):
        if ".input_layernorm.weight" in key or ".post_attention_layernorm.weight" in key:
            return None
    # Expert final norm (buffer — SKIP)
    if key == "model.vlm_with_expert.lm_expert.norm.weight":
        return None

    # Vision model
    if key.startswith("model.vlm_with_expert.vlm.model.vision_model."):
        return "vision_backbone.model." + key[len("model.vlm_with_expert.vlm.model.vision_model."):]

    # VLM lm_head
    if key.startswith("model.vlm_with_expert.vlm.lm_head."):
        return "llm_backbone.model.lm_head." + key[len("model.vlm_with_expert.vlm.lm_head."):]

    # VLM language model (text_model, connector, etc.)
    if key.startswith("model.vlm_with_expert.vlm.model."):
        return "llm_backbone.model.model." + key[len("model.vlm_with_expert.vlm.model."):]

    # Expert layers — flatten .self_attn./.mlp.
    if key.startswith("model.vlm_with_expert.lm_expert.layers."):
        sub = key[len("model.vlm_with_expert.lm_expert."):]
        sub = sub.replace(".self_attn.", ".").replace(".mlp.", ".")
        return f"vla_head.expert_stack.{sub}"

    # Top-level action projections
    if key.startswith(("model.action_in_proj.", "model.action_out_proj.",
                       "model.action_time_mlp_in.", "model.action_time_mlp_out.")):
        return "vla_head.expert_stack." + key[len("model."):]

    # State projection
    if key.startswith("model.state_proj."):
        return "projector.linear." + key[len("model.state_proj."):]

    return key


def expand_tied_smolvla(flat: "dict[str, torch.Tensor]") -> "dict[str, torch.Tensor]":
    """Expand tied weights for SmolVLA (lm_head ↔ embed_tokens)."""
    lm_head_key = "llm_backbone.model.lm_head.weight"
    embed_key = "llm_backbone.model.model.text_model.embed_tokens.weight"
    if lm_head_key in flat and embed_key not in flat:
        flat[embed_key] = flat[lm_head_key]
    return flat


__all__ = ["smolvla_safetensors_to_flat", "expand_tied_smolvla"]
