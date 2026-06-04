"""Eagle 2.5 VL — vendored from lerobot for ONNX export.

Origin: `huggingface/lerobot` → `src/lerobot/policies/groot/eagle2_hg_model/`.
Vendored 2026-04-19 with modifications:

1. `peft` imports made optional (training-only dep; not needed for export).
2. `_attn_implementation` default flipped `flash_attention_2` → `eager`
   so ONNX/TRT export works out of the box. Training code can still
   pass `flash_attention_2` explicitly.

Why vendored: we need to apply export-time patches (3-patch stack for
Qwen2 decoder: F.pad mask + frozen DynamicLayer.update + past_kv
seq_length) that would otherwise require forking lerobot. Keeping the
modified Eagle here scoped to the export path.

Public API: callers should instantiate `Eagle2_5_VLForConditionalGeneration`
directly and feed it inputs matching `EagleBackbone.forward_eagle` from
lerobot's GR00T wrapper.
"""
from .modeling_eagle2_5_vl import Eagle25VLForConditionalGeneration
from .configuration_eagle2_5_vl import Eagle25VLConfig

__all__ = [
    "Eagle25VLForConditionalGeneration",
    "Eagle25VLConfig",
]
