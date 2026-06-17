"""Curated model registry for `tether models {list,pull,info}`.

Indexes VLA model checkpoints we have verified work with Tether serving — saves
customers the "5-tab research session" of figuring out which HF repo to use.

The registry is shipped IN-PACKAGE (`src/tether/registry/data.py`) rather than
queried from HF Hub at runtime. Reasons:

- Curation: every entry has been verified against our parity tests; HF tags can
  be applied by anyone, ours can't be spoofed
- Offline: `tether models list` works without internet
- Pinning: each entry has a specific revision (commit sha) for reproducibility
- Zero rate-limit risk

Pull operations (`tether models pull <id>`) DO hit HF Hub via huggingface_hub —
that's where the actual weights live. The registry only stores metadata.
"""

from tether.registry.models import (
    ModelEntry,
    ModelBenchmark,
    REGISTRY,
    by_id,
    filter_models,
    list_families,
    list_devices,
)

# Component + spine registries — added 2026-05-20 for the BaseVLA spine
# refactor (lift #1). Foundation for build_from_cfg-driven model construction
# across vision / llm / vlm / projector / head / text / vla slots.
from tether.registry.builder import (
    Registry,
    RegistryError,
    build_from_cfg,
)
from tether.registry.components import (
    VISION_BACKBONES,
    LLM_BACKBONES,
    VLM_BACKBONES,
    PROJECTORS,
    VLA_HEADS,
    TEXT_ENCODERS,
    VLAS,
)

__all__ = [
    # Curated model registry (preserved API)
    "ModelEntry",
    "ModelBenchmark",
    "REGISTRY",
    "by_id",
    "filter_models",
    "list_families",
    "list_devices",
    # Component registries + builder (new for BaseVLA spine)
    "Registry",
    "RegistryError",
    "build_from_cfg",
    "VISION_BACKBONES",
    "LLM_BACKBONES",
    "VLM_BACKBONES",
    "PROJECTORS",
    "VLA_HEADS",
    "TEXT_ENCODERS",
    "VLAS",
]
