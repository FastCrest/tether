import json

import numpy as np
import pytest

from tether.eval.evidence_capture import CaptureLimits, EpisodeEvidenceWriter, validate_episode_evidence


def provenance():
    return {
        "suite": "libero_10", "task_index": 0, "task_description": "put objects in basket",
        "seed": 7, "episode_index": 0, "init_index": 0,
        "policy": {"source": "lerobot/smolvla_base", "revision": "abc", "adapter": None},
    }


def test_capture_writes_aligned_steps_frames_and_validated_manifest(tmp_path):
    writer = EpisodeEvidenceWriter(tmp_path / "episode", provenance=provenance(), limits=CaptureLimits(frame_stride=1))
    writer.record_step(
        step_index=0, phase="policy", observation={"state": [1, 2, 3]}, action=[0.1] * 7,
        policy_request={"started_s": 0.01, "finished_s": 0.03, "duration_ms": 20},
        task_events=[{"kind": "episode_started"}], frame=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    manifest = writer.finish("success")
    assert manifest["complete"] and manifest["counts"] == {"steps": 1, "frames": 1}
    step = json.loads((tmp_path / "episode/steps.jsonl").read_text())
    assert step["camera_frame"] == "frames/000000.jpg"
    assert step["policy_request"]["duration_ms"] == 20
    assert validate_episode_evidence(tmp_path / "episode")["terminal_reason"] == "success"


def test_capture_preserves_partial_evidence_and_reports_limits(tmp_path):
    writer = EpisodeEvidenceWriter(
        tmp_path / "episode", provenance=provenance(),
        limits=CaptureLimits(max_steps=1, max_frames=1, frame_stride=1, flush_steps=1),
    )
    args = dict(phase="policy", observation={"state": []}, action=[0] * 7, policy_request=None, task_events=[])
    assert writer.record_step(step_index=0, frame=np.zeros((4, 4, 3), dtype=np.uint8), **args)
    assert not writer.record_step(step_index=1, frame=None, **args)
    manifest = writer.interrupt()
    assert not manifest["complete"]
    assert manifest["terminal_reason"] == "interrupted"
    assert manifest["truncated"] and "step limit" in manifest["truncation_reasons"]
    assert validate_episode_evidence(tmp_path / "episode")["counts"]["steps"] == 1


def test_validation_rejects_changed_artifact(tmp_path):
    writer = EpisodeEvidenceWriter(tmp_path / "episode", provenance=provenance())
    writer.record_step(step_index=0, phase="settling", observation={}, action=None, policy_request=None, task_events=[], frame=None)
    writer.finish("timeout")
    (tmp_path / "episode/steps.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="artifact validation failed"):
        validate_episode_evidence(tmp_path / "episode")
