"""Per-embodiment weight overrides for quality_score.

Different embodiments have different baseline characteristics (per the spec):
SO-100 has noisier actions even on successful demos, so smoothness should
be weighted less. UR5 is industrial → efficiency matters more. Franka is
the precise baseline.

Weights are loaded as `QualityWeights` (defined in composite.py to avoid
circular imports — re-exported here for convenience).
"""
from __future__ import annotations

from reflex.curate.quality.composite import QualityWeights

# Default weights (Franka / generic precise arm).
DEFAULT_WEIGHTS = QualityWeights(
    success=0.4, smoothness=0.3, efficiency=0.2, coverage=0.1,
)

# Per-embodiment overrides. Embodiment slug → QualityWeights.
EMBODIMENT_WEIGHTS: dict[str, QualityWeights] = {
    "franka": DEFAULT_WEIGHTS,
    # SO-100: noisier actions on successful demos → reduce smoothness weight.
    # Compensate by raising efficiency (consistent with low-precision arms
    # taking longer when retrying).
    "so100": QualityWeights(
        success=0.5, smoothness=0.15, efficiency=0.25, coverage=0.1,
    ),
    # UR5: industrial — efficiency is what buyers care about most.
    "ur5": QualityWeights(
        success=0.4, smoothness=0.2, efficiency=0.3, coverage=0.1,
    ),
    # Bimanual setups (aloha, dual-franka) — smoothness is harder to score
    # cleanly across two arms; efficiency more reliable.
    "aloha": QualityWeights(
        success=0.45, smoothness=0.2, efficiency=0.25, coverage=0.1,
    ),
}


def weights_for(embodiment: str | None) -> QualityWeights:
    """Look up weights for an embodiment slug; falls back to DEFAULT_WEIGHTS.

    Slug matching is case-insensitive + tolerant of common synonyms.
    """
    if not embodiment:
        return DEFAULT_WEIGHTS
    key = embodiment.lower().strip()
    if key in EMBODIMENT_WEIGHTS:
        return EMBODIMENT_WEIGHTS[key]
    # Loose-match on prefix (handles franka_panda, franka2, etc.)
    for slug, weights in EMBODIMENT_WEIGHTS.items():
        if key.startswith(slug):
            return weights
    return DEFAULT_WEIGHTS


__all__ = [
    "DEFAULT_WEIGHTS",
    "EMBODIMENT_WEIGHTS",
    "weights_for",
]
