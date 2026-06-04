"""GR00T safetensors→flat-dict key transformation.

Maps a ``nvidia/GR00T-N1.6-3B`` safetensors checkpoint (sharded, two files)
into the flat-dict naming convention produced by
``GR00TVLA.prepare_inference_weights()``.

GR00T's architecture is simpler to map than Pi0/Pi0.5:
- ``backbone.*`` → ``vlm_backbone.*`` (Eagle: SigLIP + Llama-3B + MLP projector)
- ``action_head.*`` → ``vla_head.*`` (32-block DiT + action encoder)

No layer norm skip rules — GR00T's DiT uses standard ``nn.Module``
Parameters throughout (no DecomposedRMSNorm buffer vs Parameter distinction).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def groot_safetensors_to_flat(key: str) -> str | None:
    """Map a GR00T safetensors checkpoint key to its Phase A flat-dict equivalent.

    Returns ``None`` for keys that should be skipped (none currently — GR00T
    checkpoints don't have tied/buffer keys to filter).
    """
    if key.startswith("backbone."):
        return "vlm_backbone." + key[len("backbone."):]

    if key.startswith("action_head."):
        return "vla_head." + key[len("action_head."):]

    return key


def expand_tied_groot(flat: "dict[str, torch.Tensor]") -> "dict[str, torch.Tensor]":
    """Expand tied weights for GR00T.

    GR00T's Eagle backbone may tie ``lm_head.weight`` with
    ``embed_tokens.weight`` depending on the Llama config.
    """
    lm_head_key = "vlm_backbone.model.language_model.lm_head.weight"
    embed_key = "vlm_backbone.model.language_model.model.embed_tokens.weight"
    if lm_head_key in flat and embed_key not in flat:
        flat[embed_key] = flat[lm_head_key]
    return flat


__all__ = ["groot_safetensors_to_flat", "expand_tied_groot"]
