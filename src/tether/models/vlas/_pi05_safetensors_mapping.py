"""Pi0.5 safetensors→flat-dict key transformation.

Maps a lerobot ``pi05_libero_finetuned_v044`` (or compatible pi0.5)
safetensors checkpoint into the flat-dict naming convention produced by
``Pi05VLA.prepare_inference_weights()``.

Key difference from Pi0 mapping (``_pi0_safetensors_mapping.py``):

1. **Expert layer norms are KEPT** (not skipped). Pi0's expert uses
   ``DecomposedRMSNorm`` (register_buffer → not in named_parameters).
   Pi0.5's expert uses ``DecomposedAdaRMSNorm`` which has a ``dense``
   ``nn.Linear`` — those weights ARE Parameters and appear in the flat dict.

2. **Final expert norm is KEPT.** ``norm.dense.weight/bias`` is an
   AdaRMSNorm Parameter in Pi0.5.

3. **No state_proj.** Pi0.5 uses state-in-language (knowledge insulation).
   The safetensors checkpoint has ``time_mlp_in/out`` but NOT ``state_proj``.

4. **time_mlp naming.** Pi0.5's expert stack uses ``time_mlp_in/out``
   (not ``action_time_mlp_in/out`` like pi0).

Validated against ``lerobot/pi05_libero_finetuned_v044`` when the
checkpoint becomes available for local parity testing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def pi05_safetensors_to_flat(key: str) -> str | None:
    """Map a pi0.5 safetensors checkpoint key to its Phase A flat-dict equivalent.

    Returns ``None`` for keys that don't appear in Phase A's flat dict.
    """
    # Tied/unused: gemma_expert lm_head
    if key == "paligemma_with_expert.gemma_expert.lm_head.weight":
        return None
    # Final expert norm weight (NOT buffer — AdaRMSNorm uses nn.Linear)
    # Note: Pi0 skips this; Pi0.5 KEEPS it because AdaRMSNorm.dense is a Parameter
    if key == "paligemma_with_expert.gemma_expert.model.norm.weight":
        return None  # The raw norm.weight is a buffer; the dense.weight/bias are the Parameters
    if key.startswith("paligemma_with_expert.gemma_expert.model.norm.dense."):
        sub = key[len("paligemma_with_expert.gemma_expert.model.norm.dense."):]
        return f"vla_head.expert_stack.final_norm.dense.{sub}"

    # Expert layer norms — Pi0.5 KEEPS input_layernorm.dense.* and
    # post_attention_layernorm.dense.* (AdaRMSNorm Parameters)
    if key.startswith("paligemma_with_expert.gemma_expert.model.layers."):
        # AdaRMSNorm dense weights: keep
        if ".input_layernorm.dense." in key or ".post_attention_layernorm.dense." in key:
            sub = key[len("paligemma_with_expert.gemma_expert.model."):]
            sub = sub.replace(".self_attn.", ".").replace(".mlp.", ".")
            return f"vla_head.expert_stack.{sub}"
        # Plain layernorm weight (buffer, not parameter) — skip
        if ".input_layernorm.weight" in key or ".post_attention_layernorm.weight" in key:
            return None
        # Projection weights — flatten .self_attn./.mlp. like Pi0
        sub = key[len("paligemma_with_expert.gemma_expert.model."):]
        sub = sub.replace(".self_attn.", ".").replace(".mlp.", ".")
        return f"vla_head.expert_stack.{sub}"

    # Vision tower (most specific first — same as Pi0)
    if key.startswith("paligemma_with_expert.paligemma.model.vision_tower."):
        return "vision_backbone.model." + key[len("paligemma_with_expert.paligemma.model.vision_tower."):]

    # PaliGemma lm_head
    if key.startswith("paligemma_with_expert.paligemma.lm_head."):
        return "llm_backbone.model.lm_head." + key[len("paligemma_with_expert.paligemma.lm_head."):]

    # PaliGemma language model
    if key.startswith("paligemma_with_expert.paligemma.model."):
        return "llm_backbone.model.model." + key[len("paligemma_with_expert.paligemma.model."):]

    # Action projections at top-level
    if key.startswith(("action_in_proj.", "action_out_proj.")):
        return f"vla_head.expert_stack.{key}"

    # Time MLP — Pi0.5 uses action_time_mlp_in/out at top level
    if key.startswith(("action_time_mlp_in.", "action_time_mlp_out.")):
        # Map to time_mlp_in/out (the Pi0.5 expert stack attr name)
        mapped = key.replace("action_time_mlp_in.", "time_mlp_in.").replace("action_time_mlp_out.", "time_mlp_out.")
        return f"vla_head.expert_stack.{mapped}"

    # No state_proj for pi0.5 (state-in-language)
    # If checkpoint has it anyway, skip
    if key.startswith("state_proj."):
        return None

    return key


def expand_tied_pi05(flat: "dict[str, torch.Tensor]") -> "dict[str, torch.Tensor]":
    """Expand tied weights in a Pi0.5 flat dict in-place.

    PaliGemma ties ``embed_tokens.weight`` with ``lm_head.weight``.
    """
    lm_head_key = "llm_backbone.model.lm_head.weight"
    embed_key = "llm_backbone.model.model.language_model.embed_tokens.weight"
    if lm_head_key in flat and embed_key not in flat:
        flat[embed_key] = flat[lm_head_key]
    return flat


__all__ = ["pi05_safetensors_to_flat", "expand_tied_pi05"]
