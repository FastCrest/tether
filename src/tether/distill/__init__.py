"""VLA distillation — SnapFlow 1-step self-distillation for flow-matching VLAs.

v0.3: SnapFlow only (pi0 + pi0.5). See
https://github.com/FastCrest/reflex-vault/blob/main/reflex_vla/01_architecture/distill_SYNTHESIS.md
for the scope decision.

## What's in v0.3

- `snapflow.py` — SnapFlow loss math (flow-matching + consistency +
  zero-init target-time embedding). Unit-testable without GPU.
- `teacher_loader.py` — load teacher PyTorch policy from tether-export
  dir; freeze + eval. (Phase B deliverable.)

## What's DEPRECATED from v0.2

- `pi_flow.py` — moved to `archive/v0.2/distill/pi_flow.py`. Scope
  decision rejected pi-Flow in favor of SnapFlow (better 1-NFE
  quality on VLAs per Luan et al. 2604.05656).
- `dmpo.py` — moved to `archive/v0.2/distill/dmpo.py`. DMPO turned out
  to be RL+MeanFlow, not knowledge distillation. Different product
  line; may return as `phase=rl` in v0.5+.

## Public API

Use `tether finetune --phase distill` (the new CLI entry). The
trainer is selected by `distillation_method` in FinetuneConfig;
v0.3 supports only `"snapflow"`.
"""
from __future__ import annotations

__all__ = ["get_method"]


# v0.3 method registry. Extend (NOT replace) when v0.5+ adds Consistency
# Policy for GR00T DDPM.
_SUPPORTED_METHODS: dict[str, str] = {
    "snapflow": "tether.distill.snapflow",
}

_DEPRECATED: dict[str, str] = {
    "dmpo": (
        "DMPO is deprecated in v0.3. It was RL+MeanFlow, not knowledge "
        "distillation — different product line. Archived to "
        "archive/v0.2/distill/dmpo.py. If you want RL-style policy "
        "optimization, watch for `phase=rl` in v0.5+."
    ),
    "pi_flow": (
        "pi_flow is deprecated in v0.3. Scope decision picked SnapFlow "
        "instead (better 1-NFE quality on real VLAs per arxiv "
        "2604.05656). Archived to archive/v0.2/distill/pi_flow.py."
    ),
    "consistency": (
        "Consistency Policy (DDPM distillation for GR00T) is deferred "
        "to v0.5+, conditional on the Eagle VLM TRT-on-Jetson goal "
        "landing first (GR00T denoise is only 35% of E2E latency; "
        "VLM dominates — see distill_SYNTHESIS.md)."
    ),
}


def get_method(name: str):
    """Resolve a distillation-method name to its module.

    Raises ValueError with an actionable message for deprecated or
    unknown names. See `distill_SYNTHESIS.md` for the scope context.
    """
    if name in _SUPPORTED_METHODS:
        import importlib
        return importlib.import_module(_SUPPORTED_METHODS[name])
    if name in _DEPRECATED:
        raise ValueError(f"method={name!r}: {_DEPRECATED[name]}")
    raise ValueError(
        f"Unknown distillation method: {name!r}. Supported in v0.3: "
        f"{sorted(_SUPPORTED_METHODS)}."
    )
