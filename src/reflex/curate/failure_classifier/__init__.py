"""Rule-based failure classifier for the Curate wedge — v1.

Per the PRD, the Failure Corpus is the highest-value Tier-1 data product
($10K-$100K/dataset). This module classifies each episode by failure mode
to power buyer-side filtering ("give me 1000 grasp_miss episodes on
Franka").

7 failure modes (per `_curation/failure-classifier-v1.md`):
    grasp_miss            — gripper closed but no force feedback
    pose_error            — final state >5cm from any commanded pose
    collision             — ActionGuard tripped velocity/torque limit
    generalization_failure — episode terminated early without success
    timeout               — episode hit max_steps without success
    action_clamp          — output saturation events / clamp ratio high
    gripper_jam           — gripper position oscillating

ActionGuard substrate (per Lens 3 Finding 3.1): the recorder now passes a
`guard_summary` dict to write_request when the server has an ActionGuard
configured (see `runtime/server.py` predict() + lifespan write_request
call site). When the guard field is still absent (no URDF / embodiment_
config.constraints absent), detectors that depend on guard data (collision,
action_clamp) degrade gracefully — confidence=0.0 with evidence=
'action_guard_data_unavailable'.

Submodules:
    modes      — per-mode detector functions
    primary    — primary failure-mode selection logic
    composite  — top-level classify_episode() + integration with JSONL rows
"""
from __future__ import annotations

CLASSIFIER_VERSION = "rule-v1"

from reflex.curate.failure_classifier.composite import (
    ClassificationResult,
    FailureMode,
    classify_episode,
    classify_from_jsonl_rows,
)
from reflex.curate.failure_classifier.modes import (
    detect_action_clamp,
    detect_collision,
    detect_generalization_failure,
    detect_grasp_miss,
    detect_gripper_jam,
    detect_pose_error,
    detect_timeout,
)
from reflex.curate.failure_classifier.primary import primary_failure

__all__ = [
    "CLASSIFIER_VERSION",
    "ClassificationResult",
    "FailureMode",
    "classify_episode",
    "classify_from_jsonl_rows",
    "detect_action_clamp",
    "detect_collision",
    "detect_generalization_failure",
    "detect_grasp_miss",
    "detect_gripper_jam",
    "detect_pose_error",
    "detect_timeout",
    "primary_failure",
]
