"""FluxVLA pi0.5 safetensors→flat-dict key transformation.

Maps a ``limxdynamics/FluxVLAEngine`` pi0.5 checkpoint (native FluxVLA
naming) into the flat-dict naming convention produced by
``Pi05VLA.prepare_inference_weights()``.

FluxVLA uses its own naming convention, distinct from lerobot's:
- ``llm_backbone.*`` — PaliGemma LLM (18 layers)
- ``llm_expert.*`` — Gemma expert (18 layers, AdaRMSNorm dense.*)
- ``vision_backbone.vision.*`` — SigLIP vision tower
- ``projector.projector.*`` — linear projector
- ``action_{in,out}_proj.projector.*`` — action projections
- ``time_mlp_{in,out}.projector.*`` — time MLP

Source: huggingface.co/limxdynamics/FluxVLAEngine (Apache-2.0).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def fluxvla_pi05_safetensors_to_flat(key: str) -> str | None:
    """Map a FluxVLA pi0.5 checkpoint key to its flat-dict equivalent.

    Returns ``None`` for keys that should be skipped.
    """
    # Vision tower: strip extra 'vision.' level
    if key.startswith("vision_backbone.vision."):
        return "vision_backbone.model." + key[len("vision_backbone.vision."):]

    # LLM backbone → spine llm_backbone path
    if key.startswith("llm_backbone."):
        return "llm_backbone.model.model.language_model." + key[len("llm_backbone."):]

    # Expert layers — flatten .self_attn./.mlp., keep .dense. norms
    if key.startswith("llm_expert.layers."):
        sub = key[len("llm_expert."):]
        sub = sub.replace(".self_attn.", ".").replace(".mlp.", ".")
        return f"vla_head.expert_stack.{sub}"

    # Expert final norm (AdaRMSNorm dense)
    if key.startswith("llm_expert.norm.dense."):
        return "vla_head.expert_stack.final_norm.dense." + key[len("llm_expert.norm.dense."):]

    # Expert embed_tokens
    if key == "llm_expert.embed_tokens.weight":
        return "vla_head.expert_stack.embed_tokens.weight"

    # Projector: strip inner .projector. level
    if key.startswith("projector.projector."):
        return "llm_backbone.model.model.multi_modal_projector.linear." + key[len("projector.projector."):]

    # Action projections: strip .projector. level, route to expert_stack
    if key.startswith("action_in_proj.projector."):
        return "vla_head.expert_stack.action_in_proj." + key[len("action_in_proj.projector."):]
    if key.startswith("action_out_proj.projector."):
        return "vla_head.expert_stack.action_out_proj." + key[len("action_out_proj.projector."):]

    # Time MLP: strip .projector. level, route to expert_stack
    if key.startswith("time_mlp_in.projector."):
        return "vla_head.expert_stack.time_mlp_in." + key[len("time_mlp_in.projector."):]
    if key.startswith("time_mlp_out.projector."):
        return "vla_head.expert_stack.time_mlp_out." + key[len("time_mlp_out.projector."):]

    return key


def expand_tied_fluxvla_pi05(flat: "dict[str, torch.Tensor]") -> "dict[str, torch.Tensor]":
    """Expand tied weights for FluxVLA pi0.5.

    PaliGemma ties ``embed_tokens.weight`` with ``lm_head.weight``.
    FluxVLA checkpoints don't include lm_head — synthesize it from embed_tokens.
    """
    embed_key = "llm_backbone.model.model.language_model.embed_tokens.weight"
    lm_head_key = "llm_backbone.model.lm_head.weight"
    if embed_key in flat and lm_head_key not in flat:
        flat[lm_head_key] = flat[embed_key]
    return flat


__all__ = ["fluxvla_pi05_safetensors_to_flat", "expand_tied_fluxvla_pi05"]
