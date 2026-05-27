"""MolmoAct2 safetensors→flat-dict key transformation.

Maps an ``allenai/MolmoAct2`` safetensors checkpoint (sharded, 5 files)
into the flat-dict naming convention for the BaseVLA spine.

Architecture: 3 components under ``model.*`` + ``lm_head``:
- ``model.vision_backbone.*`` (414 keys) — SigLIP2 ViT
- ``model.transformer.*`` (292 keys) — Molmo2-ER (Qwen3-based LLM)
- ``model.action_expert.*`` (588 keys) — flow-matching action head
- ``lm_head.weight`` (1 key) — language model head

Source: huggingface.co/allenai/MolmoAct2 (Apache-2.0).
"""
from __future__ import annotations


def molmoact2_safetensors_to_flat(key: str) -> str | None:
    """Map a MolmoAct2 checkpoint key to its flat-dict equivalent.

    Returns ``None`` for keys that should be skipped.
    """
    if key.startswith("model.vision_backbone."):
        return "vision_backbone." + key[len("model.vision_backbone."):]

    if key.startswith("model.transformer."):
        return "vlm_backbone." + key[len("model.transformer."):]

    if key.startswith("model.action_expert."):
        return "vla_head." + key[len("model.action_expert."):]

    if key == "lm_head.weight":
        return "vlm_backbone.lm_head.weight"

    return key


__all__ = ["molmoact2_safetensors_to_flat"]
