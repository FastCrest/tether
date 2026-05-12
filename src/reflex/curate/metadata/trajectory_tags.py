"""Action-trajectory derived metadata tags.

Reuses the per-step action analysis already done by the quality module's
smoothness signal. These tags are cheap computations off the recorded
action array; computed alongside quality to avoid double-passing the data.
"""
from __future__ import annotations

import numpy as np


def terminal_gripper_state(
    actions: np.ndarray,
    *,
    gripper_dim: int | None = None,
) -> tuple[str, float]:
    """Return ('open' | 'closed' | 'unknown', confidence) from the last
    5 steps of the gripper channel.

    Useful proxy for task-type detection (pick → ends closed; place → ends
    open). When gripper_dim is None, looks at the last channel of `actions`.
    """
    if actions.size == 0:
        return "unknown", 0.0
    if gripper_dim is None:
        gripper_dim = actions.shape[1] - 1
    if not (0 <= gripper_dim < actions.shape[1]):
        return "unknown", 0.0

    last_n = actions[-min(5, len(actions)):, gripper_dim]
    if np.all(last_n > 0.5):
        return "closed", 0.95
    if np.all(last_n < -0.5):
        return "open", 0.95
    if np.mean(last_n) > 0.3:
        return "closed", 0.6
    if np.mean(last_n) < -0.3:
        return "open", 0.6
    return "unknown", 0.4


def action_complexity(actions: np.ndarray) -> dict[str, float]:
    """Per-channel action range + variance summary. Inputs to difficulty
    estimation in Phase 1.5; surface as raw numbers for now."""
    if actions.size == 0:
        return {"action_range_max": 0.0, "action_variance": 0.0}
    return {
        "action_range_max": float(np.ptp(actions, axis=0).max()),
        "action_variance": float(actions.var(axis=0).mean()),
    }


__all__ = [
    "action_complexity",
    "terminal_gripper_state",
]
