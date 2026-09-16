import json

import pytest

from tether.eval.episode_evidence import (
    CAPTURE_SEMANTICS,
    CaptureLimits,
    EpisodeEvidenceWriter,
    validate_episode_evidence,
)


def observation(x):
    return {
        "eef_position": [x, 0.0, 0.0],
        "eef_quaternion": [0.0, 0.0, 0.0, 1.0],
        "gripper_qpos": [0.0, 0.0],
        "agentview_shape": [256, 256, 3],
        "wrist_shape": [256, 256, 3],
    }


def test_schema2_declares_exact_transition_semantics_and_validates_steps(tmp_path):
    writer = EpisodeEvidenceWriter(
        tmp_path,
        provenance={"suite": "libero_10", "task_index": 0, "seed": 7, "episode_index": 0},
        limits=CaptureLimits(max_bytes=100000, max_steps=10, max_frames=10, frame_stride=2, flush_steps=1),
    )
    writer.record_step(
        step_index=0,
        phase="settling",
        observation=observation(1.0),
        action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        policy_request=None,
        task_events=[{"kind": "episode_started"}],
    )
    writer.record_step(
        step_index=1,
        phase="policy",
        observation=observation(2.0),
        action=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        policy_request={"started_s": 0.01, "finished_s": 0.02, "duration_ms": 10.0, "planned_actions": 5, "action_offset": 0},
        task_events=[{"kind": "task_success"}],
    )
    manifest = writer.finish("success")
    assert manifest["schema_version"] == 2
    assert manifest["capture_semantics"] == CAPTURE_SEMANTICS
    assert manifest["terminal_reason"] == "success"
    validated = validate_episode_evidence(tmp_path)
    assert validated["counts"]["steps"] == 2
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert rows[1]["observation_semantics"] == "post-env-step"
    assert rows[1]["action_semantics"] == "applied-before-recorded-observation"
    assert rows[1]["action"][-1] == 0.7
    assert rows[1]["policy_request"]["duration_ms"] == 10.0


def test_schema2_validation_rejects_missing_action_dimension(tmp_path):
    writer = EpisodeEvidenceWriter(tmp_path, provenance={})
    writer.record_step(
        step_index=0,
        phase="policy",
        observation=observation(1.0),
        action=[0.0] * 7,
        policy_request={"duration_ms": 1.0},
        task_events=[],
    )
    writer.finish("timeout")
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    rows[0]["action"] = [0.0] * 6
    (tmp_path / "steps.jsonl").write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n")
    manifest = json.loads((tmp_path / "episode-manifest.json").read_text())
    steps_artifact = next(item for item in manifest["artifacts"] if item["path"] == "steps.jsonl")
    import hashlib
    data = (tmp_path / "steps.jsonl").read_bytes()
    steps_artifact["size_bytes"] = len(data)
    steps_artifact["sha256"] = hashlib.sha256(data).hexdigest()
    (tmp_path / "episode-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="7-D"):
        validate_episode_evidence(tmp_path)


def test_schema1_manifest_remains_readable_for_historical_evidence(tmp_path):
    steps = tmp_path / "steps.jsonl"
    steps.write_text('{"step_index":0,"observation":{},"action":[0,0,0,0,0,0,-1]}\n')
    import hashlib
    data = steps.read_bytes()
    manifest = {
        "schema_version": 1,
        "kind": "tether-checkpoint-episode-evidence",
        "complete": True,
        "terminal_reason": "success",
        "artifacts": [{"path": "steps.jsonl", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
    }
    (tmp_path / "episode-manifest.json").write_text(json.dumps(manifest))
    assert validate_episode_evidence(tmp_path)["schema_version"] == 1
