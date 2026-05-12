"""The 4 quality signals.

Spec: reflex_context/features/08_curate/_curation/quality-scoring.md
Research: reflex_context/features/08_curate/_curation/quality-scoring_research.md

Smoothness adjustments per the research sidecar (Lens 3 findings):
    F3.2  within-chunk only (no per-boundary impulse contamination)
    F3.4  velocity-normalize deltas (handles 50Hz vs 240Hz embodiments)
    F3.5  skip gripper dims (binary toggles drag scores down)
    F3.6  range floor 1e-3 (avoid divide-by-zero on near-static dims)
    F3.1  signal labeled `policy_smoothness` not `trajectory_smoothness`
          (recorded actions are EMITTED, not EXECUTED — until per-step
          executed_action capture lands, the signal measures inference
          output regularity, not robot motion regularity)

All signals return float in [0, 1]. Pure-numpy; no torch.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from reflex.curate.quality.baseline import baseline_median


def success_signal(
    *,
    success_flag: bool | None,
    actions: np.ndarray,
    gripper_dims: Sequence[int] = (),
) -> float:
    """Did the robot complete the task?

    3 sub-signals:
        success_flag (0.5)            harness-recorded ground truth
        terminal gripper state (0.2)  closed at end → likely-pick success;
                                      open at end → likely-place success
        action stability at end (0.3) last 5 steps near-zero magnitude →
                                      reached steady state (likely succeeded)

    When success_flag is None (production deployment), redistribute the 0.5
    weight to the other two signals proportionally.
    """
    if actions.size == 0:
        return 0.0

    # Terminal gripper state: 1.0 if any gripper dim ends at extremum.
    gripper = 0.5  # neutral
    if gripper_dims:
        last_grippers = actions[-1, list(gripper_dims)]
        if np.all(last_grippers > 0.5):
            gripper = 1.0
        elif np.all(last_grippers < -0.5):
            gripper = 1.0
        else:
            gripper = 0.3

    # Action stability at end: scale to [0, 1] from L2 magnitude of last 5 steps.
    last_n = actions[-min(5, len(actions)):]
    deltas = np.diff(last_n, axis=0)
    if len(deltas) == 0:
        stability = 0.5
    else:
        # Per-channel range as scale; clamp at 1e-3 floor (F3.6).
        scale = np.maximum(np.ptp(actions, axis=0), 1e-3)
        relative_motion = float(np.mean(np.abs(deltas) / scale))
        stability = float(np.clip(1.0 - relative_motion * 3.0, 0.0, 1.0))

    if success_flag is True:
        return float(np.clip(0.5 + 0.2 * gripper + 0.3 * stability, 0.0, 1.0))
    if success_flag is False:
        return 0.0
    # success_flag is None → redistribute 0.5 weight proportionally.
    return float(np.clip((0.2 / 0.5) * gripper + (0.3 / 0.5) * stability, 0.0, 1.0))


def policy_smoothness_signal(
    *,
    actions: np.ndarray,
    chunk_ids: np.ndarray | None = None,
    execute_hz: float = 50.0,
    gripper_dims: Sequence[int] = (),
) -> float:
    """How smooth is the policy's action output?

    Per F3.1: this measures POLICY inference regularity, not robot motion
    regularity. The recorded `actions` are emitted by the policy, not
    actually executed (RTC-on overwrites stale tail; RTC-off has chunk
    boundary discontinuities). Renamed from `trajectory_smoothness` for
    honest labeling.

    Per F3.2: when chunk_ids is provided, computes smoothness within each
    chunk independently and averages. Avoids contamination from chunk-
    boundary discontinuities (default chunk_size=50 → 1 spike per 50 steps).

    Per F3.4: deltas are velocity-normalized via execute_hz. Same true
    motion produces 4.8× smaller deltas at 240Hz vs 50Hz; multiplying by
    execute_hz makes the signal scale-invariant across control frequencies.

    Per F3.5: gripper_dims (typically the last 1-2 channels for grippers)
    are excluded from the per-channel mean. Quasi-binary toggles drag
    smoothness down on perfect demos.

    Per F3.6: per-channel range floored at 1e-3 (was 1e-6 in spec). Avoids
    huge jitter values when a dim is nearly static across the episode.

    Returns [0, 1]: 1.0 = smooth, 0.0 = max-jitter.
    """
    if actions.shape[0] < 2:
        return 0.0

    # F3.5: drop gripper dims from the per-channel mean.
    keep_mask = np.ones(actions.shape[1], dtype=bool)
    for d in gripper_dims:
        if 0 <= d < actions.shape[1]:
            keep_mask[d] = False
    kept = actions[:, keep_mask]
    if kept.size == 0:
        return 0.0

    # F3.2: chunk-grouped smoothness. Falls back to single-pass if no chunk_ids.
    if chunk_ids is not None and len(chunk_ids) == len(actions):
        unique_chunks = np.unique(chunk_ids)
        per_chunk = []
        for cid in unique_chunks:
            mask = chunk_ids == cid
            chunk_actions = kept[mask]
            if chunk_actions.shape[0] < 2:
                continue
            per_chunk.append(_chunk_smoothness(chunk_actions, execute_hz))
        if not per_chunk:
            return 0.0
        return float(np.clip(np.mean(per_chunk), 0.0, 1.0))

    return float(np.clip(_chunk_smoothness(kept, execute_hz), 0.0, 1.0))


def _chunk_smoothness(actions: np.ndarray, execute_hz: float) -> float:
    """Smoothness within a single chunk. Helper for policy_smoothness_signal."""
    deltas = np.diff(actions, axis=0)
    # F3.4: velocity-normalize.
    deltas_per_sec = deltas * float(execute_hz)
    delta_std = np.std(deltas_per_sec, axis=0)
    # F3.6: range floor.
    action_range = np.maximum(np.ptp(actions, axis=0), 1e-3)
    jitter_per_channel = delta_std / action_range
    # Cap at 1.0 (jitter > range means truly chaotic — already saturated).
    jitter = float(np.mean(np.minimum(jitter_per_channel, 1.0)))
    return float(np.clip(1.0 - jitter, 0.0, 1.0))


def efficiency_signal(
    *,
    episode_steps: int,
    embodiment: str = "*",
    task: str = "*",
) -> float:
    """Did the episode complete in a reasonable number of steps?

    Returns 1.0 when at or below baseline median; decays linearly to 0.0
    at 4× median. Long episodes are likely retries or failures.
    """
    if episode_steps <= 0:
        return 0.0
    median = baseline_median(embodiment, task)
    if episode_steps <= median:
        return 1.0
    ratio = episode_steps / float(median)
    return float(np.clip(1.0 - (ratio - 1.0) / 3.0, 0.0, 1.0))


def coverage_signal(
    *,
    episode_phash: str | None,
    recent_phashes: Sequence[str] = (),
) -> float:
    """Does this episode add diversity to the corpus?

    Hamming-distance check against the last N recent episode phashes.
    Returns 1.0 if novel (distance > 16), 0.0 if near-duplicate (distance ≤ 4),
    linear in between.

    First episode (empty recent_phashes) is always 1.0. None phash returns
    0.5 (neutral — can't compute, don't penalize or reward).
    """
    if episode_phash is None:
        return 0.5
    if not recent_phashes:
        return 1.0
    distances = [
        _hamming_hex(episode_phash, p) for p in list(recent_phashes)[-50:] if p
    ]
    if not distances:
        return 1.0
    min_distance = min(distances)
    return float(np.clip((min_distance - 4) / 12.0, 0.0, 1.0))


def _hamming_hex(a: str, b: str) -> int:
    """Hamming distance between two hex strings interpreted as bit arrays."""
    if len(a) != len(b):
        # Mismatched-length phashes can't be compared; treat as max distance.
        return 64
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 64


__all__ = [
    "coverage_signal",
    "efficiency_signal",
    "policy_smoothness_signal",
    "success_signal",
]
