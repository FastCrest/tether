"""Top-level failure classifier — composes 7 detectors into a per-episode result.

Output shape (matches spec):

    {
      "failure_modes": [{"type": ..., "confidence": ..., "evidence": ...}, ...],
      "is_failure": bool,
      "primary_failure_mode": str | None,
      "classifier_version": "rule-v1"
    }

`is_failure` is True iff any detected mode meets the surfacing-confidence
threshold (default 0.3). Buyers filter datasets by both `is_failure` AND
`primary_failure_mode` per the research sidecar open question 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

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

MIN_SURFACING_CONFIDENCE = 0.3


@dataclass(frozen=True)
class FailureMode:
    """One detected failure mode."""

    type: str
    confidence: float
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "confidence": float(self.confidence),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ClassificationResult:
    """Per-episode classifier output."""

    failure_modes: list[FailureMode]
    is_failure: bool
    primary_failure_mode: str | None
    classifier_version: str
    computed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_modes": [m.to_dict() for m in self.failure_modes],
            "is_failure": bool(self.is_failure),
            "primary_failure_mode": self.primary_failure_mode,
            "classifier_version": self.classifier_version,
            "computed_at": self.computed_at,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def classify_episode(
    *,
    actions: np.ndarray,
    state: np.ndarray | None = None,
    success_flag: bool | None = None,
    max_steps: int | None = None,
    guard_events: list[dict] | None = None,
    gripper_dim: int | None = None,
    chunk_size: int = 50,
    min_surfacing_confidence: float = MIN_SURFACING_CONFIDENCE,
) -> ClassificationResult:
    """Run all 7 detectors, surface modes above the confidence threshold,
    select primary failure mode.

    Args:
        actions: (T, action_dim) recorded action trajectory.
        state: (T, state_dim) recorded state vector. Optional but enables
            pose_error detection.
        success_flag: harness-recorded ground truth. Optional; when None
            the secondary signals carry the load.
        max_steps: configured episode max length (used by timeout +
            generalization_failure). Defaults to len(actions) when None,
            which makes timeout always fire — caller should pass the real
            harness limit when known.
        guard_events: ActionGuard intervention log. Per research sidecar
            Finding 3.1: typically None on production traces today;
            detectors degrade gracefully.
        gripper_dim: index of gripper channel in actions array.
        chunk_size: action chunk size used by the policy (default 50).

    Returns:
        ClassificationResult with all triggered modes + is_failure +
        primary_failure_mode.
    """
    from reflex.curate.failure_classifier import CLASSIFIER_VERSION

    effective_max = max_steps if max_steps is not None else len(actions)

    detector_outputs: list[tuple[str, float, str]] = []

    # Run each of the 7 detectors. Each returns (confidence, evidence).
    conf, ev = detect_grasp_miss(actions=actions, state=state, gripper_dim=gripper_dim)
    detector_outputs.append(("grasp_miss", conf, ev))

    conf, ev = detect_pose_error(actions=actions, state=state)
    detector_outputs.append(("pose_error", conf, ev))

    conf, ev = detect_collision(guard_events=guard_events)
    detector_outputs.append(("collision", conf, ev))

    conf, ev = detect_generalization_failure(
        episode_steps=len(actions),
        max_steps=effective_max,
        success_flag=success_flag,
    )
    detector_outputs.append(("generalization_failure", conf, ev))

    conf, ev = detect_timeout(
        episode_steps=len(actions),
        max_steps=effective_max,
        success_flag=success_flag,
        actions=actions,
    )
    detector_outputs.append(("timeout", conf, ev))

    conf, ev = detect_action_clamp(
        guard_events=guard_events,
        actions=actions,
        chunk_size=chunk_size,
    )
    detector_outputs.append(("action_clamp", conf, ev))

    conf, ev = detect_gripper_jam(actions=actions, gripper_dim=gripper_dim)
    detector_outputs.append(("gripper_jam", conf, ev))

    # Filter to modes above the surfacing threshold.
    surfaced = [
        FailureMode(type=name, confidence=conf, evidence=ev)
        for name, conf, ev in detector_outputs
        if conf >= min_surfacing_confidence
    ]
    is_failure = bool(surfaced)
    primary = primary_failure(
        [m.to_dict() for m in surfaced],
        min_confidence=min_surfacing_confidence,
    )
    primary_type = primary["type"] if primary is not None else None

    return ClassificationResult(
        failure_modes=surfaced,
        is_failure=is_failure,
        primary_failure_mode=primary_type,
        classifier_version=CLASSIFIER_VERSION,
        computed_at=_utc_now_iso(),
    )


def classify_from_jsonl_rows(
    rows: list[dict[str, Any]],
    *,
    max_steps: int | None = None,
) -> ClassificationResult:
    """Convenience: extract the inputs from a list of /act event rows
    (uploader entry point) and run classify_episode.
    """
    if not rows:
        return ClassificationResult(
            failure_modes=[],
            is_failure=False,
            primary_failure_mode=None,
            classifier_version="rule-v1",
            computed_at=_utc_now_iso(),
        )

    md0 = rows[0].get("metadata", {}) or {}

    # Flatten action_chunks.
    flat_actions: list[list[float]] = []
    for r in rows:
        chunk = r.get("action_chunk") or []
        for action in chunk:
            if isinstance(action, list):
                flat_actions.append(action)
    actions = (
        np.asarray(flat_actions, dtype=np.float32) if flat_actions else np.zeros((0, 0))
    )

    # State vector — use the per-row state field (one state per /act call).
    state_rows = [r.get("state_vec") for r in rows if r.get("state_vec")]
    state = (
        np.asarray(state_rows, dtype=np.float32) if state_rows else None
    )

    # success_flag from metadata if present (LIBERO traces have it).
    success_raw = md0.get("success_flag")
    success_flag = bool(success_raw) if success_raw is not None else None

    # Guard events — collect across rows where present.
    guard_events: list[dict] = []
    for r in rows:
        guard = r.get("guard")
        if isinstance(guard, dict):
            guard_events.append(guard)
    guard_input: list[dict] | None = guard_events or None

    # gripper_dim from metadata if present.
    gripper_dims_raw = md0.get("gripper_dims", ())
    gripper_dim = (
        int(gripper_dims_raw[0])
        if gripper_dims_raw and isinstance(gripper_dims_raw, (list, tuple))
        else None
    )

    return classify_episode(
        actions=actions,
        state=state,
        success_flag=success_flag,
        max_steps=max_steps,
        guard_events=guard_input,
        gripper_dim=gripper_dim,
    )


__all__ = [
    "ClassificationResult",
    "FailureMode",
    "MIN_SURFACING_CONFIDENCE",
    "classify_episode",
    "classify_from_jsonl_rows",
]
