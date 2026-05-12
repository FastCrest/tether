"""Tests for src/reflex/curate/failure_classifier/ — rule-based v1.

Covers each detector + composite + primary selection + JSONL row helper.
Verifies graceful degradation when ActionGuard data is absent (per
research sidecar Finding 3.1).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from reflex.curate.failure_classifier import (
    CLASSIFIER_VERSION,
    ClassificationResult,
    FailureMode,
    classify_episode,
    classify_from_jsonl_rows,
    detect_action_clamp,
    detect_collision,
    detect_generalization_failure,
    detect_grasp_miss,
    detect_gripper_jam,
    detect_pose_error,
    detect_timeout,
    primary_failure,
)


# ── grasp_miss ──────────────────────────────────────────────────────────────


def test_grasp_miss_gripper_closed_at_end_fires() -> None:
    actions = np.zeros((50, 7))
    actions[-5:, 6] = 1.0
    conf, ev = detect_grasp_miss(actions=actions, gripper_dim=6)
    assert conf > 0.5
    assert "grasp_miss" in ev


def test_grasp_miss_gripper_open_at_end_no_fire() -> None:
    actions = np.zeros((50, 7))
    actions[-5:, 6] = -1.0
    conf, ev = detect_grasp_miss(actions=actions, gripper_dim=6)
    assert conf == 0.0
    assert "not_triggered" in ev


def test_grasp_miss_empty_actions() -> None:
    conf, ev = detect_grasp_miss(actions=np.zeros((0, 7)))
    assert conf == 0.0


# ── pose_error ──────────────────────────────────────────────────────────────


def test_pose_error_state_unavailable_no_fire() -> None:
    actions = np.zeros((50, 7))
    conf, ev = detect_pose_error(actions=actions, state=None)
    assert conf == 0.0
    assert "state_unavailable" in ev


def test_pose_error_far_final_state_fires() -> None:
    actions = np.zeros((50, 7))  # commanded at origin
    state = np.zeros((50, 6))
    state[-1, :3] = [1.0, 1.0, 1.0]  # final pose 1.7m away
    conf, ev = detect_pose_error(actions=actions, state=state, threshold_m=0.05)
    assert conf > 0.5
    assert "pose_error" in ev


def test_pose_error_close_final_state_no_fire() -> None:
    actions = np.zeros((50, 7))
    state = np.zeros((50, 6))
    state[-1, :3] = [0.01, 0.01, 0.01]  # within 0.05m
    conf, _ = detect_pose_error(actions=actions, state=state, threshold_m=0.05)
    assert conf == 0.0


# ── collision (ActionGuard-dependent) ───────────────────────────────────────


def test_collision_no_guard_data_returns_zero() -> None:
    """Per F3.1: graceful degradation when ActionGuard data absent."""
    conf, ev = detect_collision(guard_events=None)
    assert conf == 0.0
    assert "action_guard_data_unavailable" in ev


def test_collision_velocity_violation_fires() -> None:
    events = [{"type": "velocity_limit", "joint": 3, "magnitude": 4.2}]
    conf, ev = detect_collision(guard_events=events)
    assert conf >= 0.85
    assert "collision" in ev


def test_collision_no_relevant_events_no_fire() -> None:
    events = [{"type": "info_only"}]
    conf, _ = detect_collision(guard_events=events)
    assert conf == 0.0


# ── generalization_failure ──────────────────────────────────────────────────


def test_generalization_short_episode_failed_flag_fires() -> None:
    conf, _ = detect_generalization_failure(
        episode_steps=50, max_steps=200, success_flag=False,
    )
    assert conf > 0.5


def test_generalization_short_episode_no_flag_lower_confidence() -> None:
    """Short episode with success_flag=None gets lower confidence than =False."""
    conf_with_false = detect_generalization_failure(
        episode_steps=50, max_steps=200, success_flag=False,
    )[0]
    conf_with_none = detect_generalization_failure(
        episode_steps=50, max_steps=200, success_flag=None,
    )[0]
    assert conf_with_false > conf_with_none


def test_generalization_full_episode_no_fire() -> None:
    conf, _ = detect_generalization_failure(
        episode_steps=200, max_steps=200, success_flag=False,
    )
    assert conf == 0.0


def test_generalization_success_no_fire() -> None:
    conf, _ = detect_generalization_failure(
        episode_steps=50, max_steps=200, success_flag=True,
    )
    assert conf == 0.0


# ── timeout ─────────────────────────────────────────────────────────────────


def test_timeout_at_max_steps_fires() -> None:
    actions = np.zeros((200, 7))
    conf, ev = detect_timeout(
        episode_steps=200, max_steps=200, success_flag=False, actions=actions,
    )
    assert conf >= 0.7
    assert "timeout" in ev


def test_timeout_well_below_max_no_fire() -> None:
    conf, _ = detect_timeout(
        episode_steps=50, max_steps=200, success_flag=False,
    )
    assert conf == 0.0


def test_timeout_success_at_max_no_fire() -> None:
    conf, _ = detect_timeout(
        episode_steps=200, max_steps=200, success_flag=True,
    )
    assert conf == 0.0


def test_timeout_steady_at_end_higher_confidence() -> None:
    """Stuck/steady robot at timeout → higher confidence than jittery."""
    actions_steady = np.zeros((200, 7))
    actions_jitter = np.random.uniform(-1, 1, (200, 7))
    conf_steady = detect_timeout(
        episode_steps=200, max_steps=200, success_flag=False, actions=actions_steady,
    )[0]
    conf_jitter = detect_timeout(
        episode_steps=200, max_steps=200, success_flag=False, actions=actions_jitter,
    )[0]
    assert conf_steady > conf_jitter


# ── action_clamp ────────────────────────────────────────────────────────────


def test_action_clamp_no_guard_returns_unavailable() -> None:
    conf, ev = detect_action_clamp(guard_events=None, actions=None)
    assert conf == 0.0
    assert "action_guard_data_unavailable" in ev


def test_action_clamp_fallback_high_saturation_partial_fire() -> None:
    """Fallback heuristic fires on boundary saturation when guard absent."""
    actions = np.ones((50, 7)) * 0.999
    conf, ev = detect_action_clamp(guard_events=None, actions=actions)
    assert conf > 0.0
    assert "policy_intent_clamp" in ev


def test_action_clamp_saturation_events_fire() -> None:
    events = [{"type": "output_saturation", "clamp_count": 30, "clamped": True}]
    conf, ev = detect_action_clamp(guard_events=events, chunk_size=50)
    assert conf > 0.5
    assert "action_clamp" in ev


# ── gripper_jam ─────────────────────────────────────────────────────────────


def test_gripper_jam_oscillating_fires() -> None:
    """High mean abs delta on the gripper channel."""
    actions = np.zeros((50, 7))
    actions[:, 6] = (-1.0) ** np.arange(50)  # toggles every step
    conf, ev = detect_gripper_jam(actions=actions, gripper_dim=6)
    assert conf > 0.5
    assert "gripper_jam" in ev


def test_gripper_jam_steady_no_fire() -> None:
    actions = np.zeros((50, 7))
    actions[:, 6] = 1.0  # constant
    conf, _ = detect_gripper_jam(actions=actions, gripper_dim=6)
    assert conf == 0.0


# ── primary_failure ─────────────────────────────────────────────────────────


def test_primary_picks_highest_confidence() -> None:
    modes = [
        {"type": "grasp_miss", "confidence": 0.7, "evidence": "..."},
        {"type": "timeout", "confidence": 0.9, "evidence": "..."},
    ]
    primary = primary_failure(modes)
    assert primary["type"] == "timeout"


def test_primary_priority_tiebreak() -> None:
    """When confidences are tied, more-specific modes win."""
    modes = [
        {"type": "generalization_failure", "confidence": 0.7, "evidence": "..."},
        {"type": "collision", "confidence": 0.7, "evidence": "..."},
    ]
    primary = primary_failure(modes)
    assert primary["type"] == "collision"


def test_primary_below_threshold_returns_none() -> None:
    modes = [{"type": "x", "confidence": 0.1, "evidence": "..."}]
    assert primary_failure(modes, min_confidence=0.3) is None


def test_primary_empty_returns_none() -> None:
    assert primary_failure([]) is None


# ── classify_episode (composite) ────────────────────────────────────────────


def test_classify_episode_returns_ClassificationResult() -> None:
    actions = np.zeros((50, 7))
    result = classify_episode(actions=actions, success_flag=True, max_steps=200)
    assert isinstance(result, ClassificationResult)
    assert result.classifier_version == CLASSIFIER_VERSION


def test_classify_episode_clean_success_no_failure() -> None:
    """Smooth-trajectory short successful episode with clean gripper open."""
    actions = np.zeros((50, 7))
    actions[:, :6] = np.linspace(0, 1, 50).reshape(50, 1)
    actions[:, 6] = -1.0  # gripper open at end (place-style success)
    result = classify_episode(
        actions=actions, success_flag=True, max_steps=200, gripper_dim=6,
    )
    assert not result.is_failure
    assert result.primary_failure_mode is None


def test_classify_episode_timeout_fires() -> None:
    actions = np.zeros((200, 7))
    result = classify_episode(actions=actions, success_flag=False, max_steps=200)
    assert result.is_failure
    assert result.primary_failure_mode == "timeout"


def test_classify_episode_to_dict_keys() -> None:
    actions = np.zeros((50, 7))
    result = classify_episode(actions=actions, max_steps=200)
    d = result.to_dict()
    assert "failure_modes" in d
    assert "is_failure" in d
    assert "primary_failure_mode" in d
    assert "classifier_version" in d


def test_classify_episode_collision_when_guard_present() -> None:
    actions = np.zeros((50, 7))
    guard = [{"type": "velocity_limit", "joint": 3}]
    result = classify_episode(
        actions=actions, max_steps=200, guard_events=guard, success_flag=False,
    )
    assert any(m.type == "collision" for m in result.failure_modes)


# ── classify_from_jsonl_rows ────────────────────────────────────────────────


def test_classify_from_rows_empty_input() -> None:
    result = classify_from_jsonl_rows([])
    assert not result.is_failure
    assert result.primary_failure_mode is None


def test_classify_from_rows_extracts_actions_and_state() -> None:
    rows = [
        {
            "timestamp": "2026-05-05T10:00:00Z",
            "episode_id": "ep_1",
            "state_vec": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "action_chunk": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]],
            "metadata": {"chunk_id": 0},
        },
    ] * 50  # replicated 50 times → 50 actions
    result = classify_from_jsonl_rows(rows, max_steps=200)
    # Gripper closed at end (1.0) → grasp_miss should fire
    assert any(m.type == "grasp_miss" for m in result.failure_modes)
