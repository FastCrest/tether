"""Baseline median episode lengths per (embodiment, task) tuple.

Used by the efficiency signal: an episode is rated 1.0 if at-or-below the
baseline median, decaying linearly to 0.0 at 4× median.

The Phase 1 table is hand-tuned from LIBERO baseline traces. Phase 1.5 will
auto-update from the production corpus once 1000+ episodes per (embodiment,
task) flow in (per `quality-scoring_research.md` open-question 1).

Lookup falls back across:
    (embodiment, task) → (embodiment, "*") → ("*", task) → DEFAULT_MEDIAN
"""
from __future__ import annotations

DEFAULT_MEDIAN = 200

# (embodiment, task) → median step count.
# Tasks use the controlled vocabulary from `_curation/metadata-enrichment.md`
# (pick / place / pour / insert / fold / open / close / etc.) plus "*" wildcard.
BASELINE_MEDIAN_STEPS: dict[tuple[str, str], int] = {
    # Franka (precise 7-DoF arm, ~50Hz control)
    ("franka", "pick"): 120,
    ("franka", "place"): 130,
    ("franka", "pour"): 280,
    ("franka", "insert"): 240,
    ("franka", "fold"): 360,
    ("franka", "*"): 200,
    # SO-100 (LeRobot, lower-precision 6-DoF, ~30Hz control)
    ("so100", "pick"): 180,
    ("so100", "place"): 200,
    ("so100", "*"): 250,
    # UR5 (industrial 6-DoF, ~125Hz control — fewer steps per task)
    ("ur5", "pick"): 90,
    ("ur5", "place"): 95,
    ("ur5", "*"): 150,
    # Generic / unknown embodiment
    ("*", "*"): DEFAULT_MEDIAN,
}


def baseline_median(embodiment: str, task: str) -> int:
    """Look up the baseline median episode length with progressive fallback.

    Lookup order:
        1. (embodiment, task)
        2. (embodiment, "*")
        3. ("*", task)
        4. DEFAULT_MEDIAN
    """
    if (embodiment, task) in BASELINE_MEDIAN_STEPS:
        return BASELINE_MEDIAN_STEPS[(embodiment, task)]
    if (embodiment, "*") in BASELINE_MEDIAN_STEPS:
        return BASELINE_MEDIAN_STEPS[(embodiment, "*")]
    if ("*", task) in BASELINE_MEDIAN_STEPS:
        return BASELINE_MEDIAN_STEPS[("*", task)]
    return DEFAULT_MEDIAN


__all__ = [
    "BASELINE_MEDIAN_STEPS",
    "DEFAULT_MEDIAN",
    "baseline_median",
]
