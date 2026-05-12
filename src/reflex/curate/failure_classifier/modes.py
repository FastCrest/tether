"""Per-mode detector functions for the failure classifier.

Each detector returns a (confidence, evidence) tuple. confidence in [0, 1];
evidence is a string explaining why the mode triggered (or "not_triggered"
when the mode didn't fire). Per the research sidecar:

  - Detectors degrade gracefully when ActionGuard data is absent
    (Finding 3.1). Emit confidence=0.0 + evidence='action_guard_data_unavailable'.
  - Action arrays are EMITTED, not EXECUTED. Evidence strings prefix
    `commanded_*` or `policy_intent_*` to surface this honestly (Finding 3.2).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


# Threshold defaults (per spec; tunable per-embodiment in Phase 1.5).
GRASP_MISS_LAST_N = 5
POSE_ERROR_THRESHOLD_M = 0.05  # 5cm
GENERALIZATION_MIN_RATIO = 0.10  # episode shorter than 10% of max → likely-generalization-fail
GRIPPER_JAM_OSCILLATION_THRESHOLD = 0.5  # mean abs delta on gripper channel
ACTION_CLAMP_RATIO_THRESHOLD = 0.5  # >50% clamp events = clamp-driven episode


def detect_grasp_miss(
    *,
    actions: np.ndarray,
    state: np.ndarray | None = None,
    gripper_dim: int | None = None,
) -> tuple[float, str]:
    """Gripper closed at end without obvious force feedback.

    v1 heuristic: last 5 steps have gripper > 0.5 AND state vector (when
    present) shows no contact-force-like signature in the gripper neighborhood.
    """
    if actions.size == 0:
        return 0.0, "not_triggered:empty_actions"
    if gripper_dim is None:
        gripper_dim = actions.shape[1] - 1
    if not (0 <= gripper_dim < actions.shape[1]):
        return 0.0, "not_triggered:no_gripper_dim"

    last_n = min(GRASP_MISS_LAST_N, len(actions))
    closed_count = int(np.sum(actions[-last_n:, gripper_dim] > 0.5))
    if closed_count < last_n:
        return 0.0, "not_triggered:gripper_not_closed_at_end"

    # Closed for last N → candidate. State-based force check (when available)
    # would up-vote. Without state, ship the gripper-closed signal alone.
    confidence = 0.6 + 0.07 * (closed_count - last_n + GRASP_MISS_LAST_N)
    confidence = float(min(confidence, 0.85))
    evidence = (
        f"commanded_grasp_miss:gripper_closed_for_last_{closed_count}_of_{last_n}_steps"
    )
    if state is None or state.size == 0:
        evidence += "+no_state_force_signal"
    return confidence, evidence


def detect_pose_error(
    *,
    actions: np.ndarray,
    state: np.ndarray | None = None,
    threshold_m: float = POSE_ERROR_THRESHOLD_M,
) -> tuple[float, str]:
    """Final state pose >threshold_m away from any commanded pose.

    Requires state vector. Without state, returns confidence=0 — pose error
    can't be detected from actions alone.
    """
    if actions.size == 0 or state is None or state.size == 0:
        return 0.0, "not_triggered:state_unavailable"
    if state.shape[1] < 3:
        return 0.0, "not_triggered:state_dim_too_small"

    # Take first 3 dims of state as Cartesian position (per Reflex
    # embodiment convention). Compare against the commanded pose at each
    # step — minimum distance to any commanded pose tells us if the robot
    # ended somewhere reachable.
    final_pos = state[-1, :3]
    if actions.shape[1] < 3:
        return 0.0, "not_triggered:action_dim_too_small_for_pose"
    distances = np.linalg.norm(actions[:, :3] - final_pos, axis=1)
    min_distance = float(np.min(distances))
    if min_distance < threshold_m:
        return 0.0, f"not_triggered:final_pos_within_{threshold_m}m_of_commanded"

    confidence = 0.5 + min(0.4, (min_distance - threshold_m) * 4.0)
    confidence = float(min(confidence, 0.95))
    return (
        confidence,
        f"commanded_pose_error:final_state_{min_distance:.3f}m_from_nearest_commanded",
    )


def detect_collision(
    *,
    guard_events: list[dict] | None,
    last_n: int = 20,
) -> tuple[float, str]:
    """ActionGuard tripped a velocity/torque limit within last_n steps.

    Per the research sidecar Finding 3.1: ActionGuard data is currently
    absent from the JSONL stream. Returns confidence 0 with an explicit
    'unavailable' evidence string when guard_events is None.
    """
    if guard_events is None:
        return 0.0, "action_guard_data_unavailable"
    if not guard_events:
        return 0.0, "not_triggered:no_guard_events"
    recent = guard_events[-last_n:] if len(guard_events) > last_n else guard_events
    has_velocity = any(
        "velocity" in str(e.get("type", "")).lower()
        or "velocity" in " ".join(e.get("violations", []))
        for e in recent
    )
    has_torque = any(
        "torque" in str(e.get("type", "")).lower()
        or "effort" in str(e.get("type", "")).lower()
        for e in recent
    )
    if not (has_velocity or has_torque):
        return 0.0, "not_triggered:no_collision_signal_in_recent_guard_events"
    return 0.9, "collision:guard_velocity_or_torque_limit_within_last_20_steps"


def detect_generalization_failure(
    *,
    episode_steps: int,
    max_steps: int,
    success_flag: bool | None,
) -> tuple[float, str]:
    """Episode terminated early without success_flag.

    Indicates the policy gave up / the harness aborted. Most useful when
    success_flag is explicitly False; degrades when None.
    """
    if max_steps <= 0:
        return 0.0, "not_triggered:no_max_steps"
    if episode_steps >= max_steps:
        return 0.0, "not_triggered:reached_max_steps"
    short_ratio = episode_steps / float(max_steps)
    if short_ratio > 0.85:
        return 0.0, "not_triggered:episode_almost_complete"
    if success_flag is True:
        return 0.0, "not_triggered:success_flag_true"

    base = 0.6 if success_flag is False else 0.4
    confidence = base + 0.25 * (1.0 - short_ratio)
    confidence = float(min(confidence, 0.85))
    evidence = (
        f"generalization_failure:episode_terminated_at_{short_ratio:.2f}_of_max_steps"
    )
    if success_flag is None:
        evidence += "+success_flag_absent"
    return confidence, evidence


def detect_timeout(
    *,
    episode_steps: int,
    max_steps: int,
    success_flag: bool | None,
    actions: np.ndarray | None = None,
) -> tuple[float, str]:
    """Episode reached max_steps without success.

    Indicates the policy ran out of time. Smoothness signal at episode end
    contributes — a smoothly-still robot at timeout suggests stuck / waiting,
    while jittery actions suggest desperate retrying.
    """
    if max_steps <= 0 or episode_steps < max_steps - 5:
        return 0.0, "not_triggered:not_at_max_steps"
    if success_flag is True:
        return 0.0, "not_triggered:success_flag_true"

    confidence = 0.7
    evidence_parts = ["timeout:episode_reached_max_steps"]

    if actions is not None and actions.size > 5:
        last_5 = actions[-5:]
        delta_mag = float(np.mean(np.abs(np.diff(last_5, axis=0))))
        # Action smoothness near-zero → robot stuck (more confident timeout)
        # high jitter at end → desperate retry pattern
        if delta_mag < 0.05:
            confidence += 0.1
            evidence_parts.append("steady_at_end_likely_stuck")
        elif delta_mag > 0.5:
            confidence += 0.05
            evidence_parts.append("jittery_at_end_likely_retrying")

    return float(min(confidence, 0.95)), ":".join(evidence_parts)


def detect_action_clamp(
    *,
    guard_events: list[dict] | None,
    actions: np.ndarray | None = None,
    chunk_size: int = 50,
) -> tuple[float, str]:
    """ActionGuard 'output_saturation' events OR clamp_count > chunk_size/2.

    Per Finding 3.1: degrades when guard data unavailable.
    Fallback heuristic when guard absent: per-channel saturation count
    (actions hitting +/-1.0 boundary) — weak signal but better than nothing.
    """
    if guard_events is None:
        # Fallback: action-boundary saturation count.
        if actions is None or actions.size == 0:
            return 0.0, "action_guard_data_unavailable"
        # Count how many actions saturated to within 1% of the [-1, 1] boundary.
        sat = np.sum(np.abs(actions) > 0.99)
        total = actions.size
        if total == 0:
            return 0.0, "action_guard_data_unavailable"
        ratio = sat / total
        if ratio > 0.3:
            return 0.5, (
                f"policy_intent_clamp:fallback_boundary_saturation_{ratio:.2f}"
                "+action_guard_data_unavailable"
            )
        return 0.0, "action_guard_data_unavailable"

    if not guard_events:
        return 0.0, "not_triggered:no_guard_events"

    saturation_events = [
        e for e in guard_events
        if "saturation" in str(e.get("type", "")).lower()
        or e.get("clamped")
    ]
    if not saturation_events:
        return 0.0, "not_triggered:no_saturation_events"

    clamp_count = sum(int(e.get("clamp_count", 1)) for e in saturation_events)
    threshold = max(1, int(chunk_size * ACTION_CLAMP_RATIO_THRESHOLD))
    if clamp_count < threshold:
        return 0.0, f"not_triggered:clamp_count_{clamp_count}_below_{threshold}"

    ratio = min(clamp_count / float(chunk_size), 1.0)
    confidence = 0.7 + 0.1 * ratio
    return float(min(confidence, 0.95)), (
        f"action_clamp:saturation_events_clamp_count_{clamp_count}"
    )


def detect_gripper_jam(
    *,
    actions: np.ndarray,
    gripper_dim: int | None = None,
) -> tuple[float, str]:
    """Gripper position oscillating without force-feedback change.

    v1 heuristic: high mean absolute delta on the gripper channel suggests
    the policy is rapidly toggling, often a sign of unresolved grasp attempts.
    """
    if actions.size == 0:
        return 0.0, "not_triggered:empty_actions"
    if gripper_dim is None:
        gripper_dim = actions.shape[1] - 1
    if not (0 <= gripper_dim < actions.shape[1]):
        return 0.0, "not_triggered:no_gripper_dim"
    if len(actions) < 10:
        return 0.0, "not_triggered:episode_too_short"

    gripper_track = actions[:, gripper_dim]
    delta = np.diff(gripper_track)
    oscillation_amp = float(np.mean(np.abs(delta)))
    if oscillation_amp < GRIPPER_JAM_OSCILLATION_THRESHOLD:
        return 0.0, f"not_triggered:gripper_oscillation_{oscillation_amp:.3f}_below_threshold"

    confidence = 0.6 + 0.1 * min((oscillation_amp - GRIPPER_JAM_OSCILLATION_THRESHOLD) / 0.5, 1.0)
    return float(min(confidence, 0.85)), (
        f"gripper_jam:gripper_oscillation_amp_{oscillation_amp:.3f}"
    )


__all__ = [
    "GRASP_MISS_LAST_N",
    "POSE_ERROR_THRESHOLD_M",
    "GENERALIZATION_MIN_RATIO",
    "GRIPPER_JAM_OSCILLATION_THRESHOLD",
    "ACTION_CLAMP_RATIO_THRESHOLD",
    "detect_action_clamp",
    "detect_collision",
    "detect_generalization_failure",
    "detect_grasp_miss",
    "detect_gripper_jam",
    "detect_pose_error",
    "detect_timeout",
]
