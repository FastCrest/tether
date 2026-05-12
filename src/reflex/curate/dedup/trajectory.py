"""DTW-based trajectory similarity for dedup verification (stage 2).

Phase 1 uses pure-numpy DTW. Hand-rolled to avoid taking a new C-extension
dep (`dtaidistance`) that the spec recommended. At Phase 1 scale (≤100K
episodes, average action_dim=7-14, average length=200 steps), pure-numpy
DTW per-pair is ~30-100ms — acceptable since stage-1 phash already cut
the candidate set down to a handful per episode.

If/when we hit volume where DTW-per-pair dominates dedup pass time, we
swap in `dtaidistance` (C-extension) — same function signature, no other
call-site changes.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _dtw_distance_numpy(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Standard DTW dynamic programming on 1-D float arrays.

    Returns the total accumulated distance along the optimal warping path.
    O(N*M) time, O(N*M) memory. For action arrays of length 200, that's
    40K cells — negligible.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("inf")

    # +inf-padded cost matrix; cell [0,0] = 0; otherwise DP.
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(float(seq_a[i - 1]) - float(seq_b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])

    return float(cost[n, m])


def trajectory_similarity(
    actions_a: np.ndarray,
    actions_b: np.ndarray,
) -> float:
    """Return [0, 1] similarity between two action arrays. 1.0 = identical
    trajectories, 0.0 = maximally different.

    Process:
        1. Per-channel z-score normalize each trajectory (so unit-scale
           differences don't dominate the distance).
        2. Flatten to 1-D and run DTW.
        3. Convert distance to similarity via inverse-magnitude scaling.

    Args:
        actions_a, actions_b: shape (T_a, action_dim) and (T_b, action_dim).
            Channel dims must match; lengths can differ.
    """
    if actions_a.size == 0 or actions_b.size == 0:
        return 0.0
    if actions_a.shape[1] != actions_b.shape[1]:
        return 0.0

    # Z-score normalize per channel.
    a_mean = actions_a.mean(axis=0)
    a_std = actions_a.std(axis=0) + 1e-6
    b_mean = actions_b.mean(axis=0)
    b_std = actions_b.std(axis=0) + 1e-6
    a_norm = (actions_a - a_mean) / a_std
    b_norm = (actions_b - b_mean) / b_std

    # Flatten + DTW.
    distance = _dtw_distance_numpy(a_norm.flatten(), b_norm.flatten())

    # Convert to similarity. Use total magnitude as scale.
    max_distance = float(np.linalg.norm(a_norm) + np.linalg.norm(b_norm))
    if max_distance <= 0:
        return 1.0 if distance < 1e-6 else 0.0
    return float(max(0.0, 1.0 - distance / max_distance))


__all__ = [
    "trajectory_similarity",
]
