"""Typed component registries for the BaseVLA spine.

Each registry is a `Registry` instance from `reflex.registry.builder`.
Components self-register via `@<REGISTRY>.register` at import time; the spine
calls `build_from_cfg(spec, <REGISTRY>)` to instantiate them.

The 7 registries (per decision S-2 in `01_decisions/2026-05-19-fluxvla-lift-
program-decisions.md` — the 5-slot pattern + GR00T's 6th vlm_backbone slot +
DreamZero's text_encoder slot):

- `VISION_BACKBONES` — image → embeddings (SigLIP, DinoSigLIP, CLIP, VAE...)
- `LLM_BACKBONES` — language → embeddings + attention (PaliGemma, Gemma, ...)
- `VLM_BACKBONES` — fused vision+language (GR00T's Eagle; not the 2-tower path)
- `PROJECTORS` — cross-modal projection (Linear, etc.)
- `VLA_HEADS` — action prediction (flow-matching, DiT, autoregressive, ...)
- `TEXT_ENCODERS` — text-only encoders (T5 for DreamZero)
- `VLAS` — the top-level model spine (Pi0VLA, Pi05VLA, SmolVLA, GR00TVLA,
  DreamZeroVLA — OpenVLA stays a shim per decision S-4)

Why 7 not 6: original spine plan said 5; decision S-2 added VLM_BACKBONES for
GR00T's Eagle (fused SigLIP+Llama); DreamZero needs TEXT_ENCODERS for T5 (not
the same shape as LLM_BACKBONES, which carry RoPE'd attention).

Import this module to make the registries available; components in
`src/reflex/models/{vision,llm,vlm,projectors,heads,text,vlas}/*.py` register
themselves via decorators at module import time.
"""
from __future__ import annotations

from reflex.registry.builder import Registry

# Component-level registries. Each is a separate namespace — registering a
# class in VISION_BACKBONES doesn't collide with the same class name in
# VLA_HEADS. Spine `from_config` calls resolve through the right registry per
# slot.
VISION_BACKBONES: Registry = Registry("vision_backbones")
LLM_BACKBONES: Registry = Registry("llm_backbones")
VLM_BACKBONES: Registry = Registry("vlm_backbones")
PROJECTORS: Registry = Registry("projectors")
VLA_HEADS: Registry = Registry("vla_heads")
TEXT_ENCODERS: Registry = Registry("text_encoders")

# Top-level spine. Pi0VLA, Pi05VLA, SmolVLA, GR00TVLA, DreamZeroVLA register
# here. `reflex models pull <id>` resolves the ModelEntry's `vla_type` field
# through VLAS to pick the right composition class.
VLAS: Registry = Registry("vlas")


__all__ = [
    "VISION_BACKBONES",
    "LLM_BACKBONES",
    "VLM_BACKBONES",
    "PROJECTORS",
    "VLA_HEADS",
    "TEXT_ENCODERS",
    "VLAS",
]
