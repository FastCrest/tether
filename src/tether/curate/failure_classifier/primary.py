"""Primary failure-mode selection.

When an episode triggers multiple modes (common — a grasp_miss leads to
pose_error because the robot tried to recover), we still emit ALL detected
modes but flag one as `primary` for buyers who want a single-mode filter.

Selection: highest confidence wins; tiebreaker by priority order
(more-specific modes beat more-general ones).
"""
from __future__ import annotations

from typing import Any

# Lower index = higher priority on tiebreak. More-specific modes first so
# they "claim" the episode when confidences are tied.
_PRIORITY_ORDER = (
    "collision",
    "grasp_miss",
    "gripper_jam",
    "pose_error",
    "action_clamp",
    "timeout",
    "generalization_failure",
)


def primary_failure(
    failure_modes: list[dict[str, Any]],
    *,
    min_confidence: float = 0.3,
) -> dict[str, Any] | None:
    """Pick the primary failure mode from a per-episode mode list.

    Args:
        failure_modes: list of detected mode dicts ({type, confidence, evidence}).
        min_confidence: drop modes below this threshold (per research sidecar
            open question 2 — confidence < 0.3 is too noisy to surface).

    Returns:
        The single mode dict that should be flagged as `primary_failure_mode`,
        or None when no mode meets the confidence threshold.
    """
    if not failure_modes:
        return None
    qualifying = [m for m in failure_modes if float(m.get("confidence", 0.0)) >= min_confidence]
    if not qualifying:
        return None

    def _priority(mode: dict[str, Any]) -> int:
        try:
            return _PRIORITY_ORDER.index(mode.get("type", ""))
        except ValueError:
            return len(_PRIORITY_ORDER)

    # Sort by (-confidence, priority_order). Highest confidence wins;
    # ties broken by priority order (smaller priority index = higher priority).
    qualifying.sort(key=lambda m: (-float(m.get("confidence", 0.0)), _priority(m)))
    return qualifying[0]


__all__ = [
    "primary_failure",
]
