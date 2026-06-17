"""VLA model definitions — both the legacy per-model modules (adapt, smolvla)
AND the BaseVLA spine (lift #1 Day 3+).

The spine is the long-term home for model composition; legacy modules stay
in place until each VLA is decomposed onto the spine (Days 4-9 per
`features/03_export/basevla-spine_plan.md`).
"""
from tether.models._name_mapping import apply_name_mapping
from tether.models.base_vla import ALL_SLOTS, BaseVLA

__all__ = [
    "BaseVLA",
    "ALL_SLOTS",
    "apply_name_mapping",
]
