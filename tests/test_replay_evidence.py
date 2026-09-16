from __future__ import annotations

import json

from tether.replay import cli
from tether.runtime.record import RecordWriter


class _Server:
    def predict_from_base64(self, *, image_b64, instruction, state):
        assert image_b64 == "aGVsbG8="
        return {"actions": [[0.25, -0.5]], "latency_ms": 12.5}


def _trace(tmp_path, *, image_redaction="full"):
    writer = RecordWriter(
        record_dir=tmp_path,
        model_hash="recorded-model",
        config_hash="recorded-config",
        export_dir=str(tmp_path / "recorded"),
        model_type="smolvla",
        export_kind="monolithic",
        providers=["CPUExecutionProvider"],
        image_redaction=image_redaction,
        gzip_output=False,
    )
    writer.write_request(
        chunk_id=7,
        image_b64="aGVsbG8=",
        instruction="move",
        state=[1.0, 2.0],
        actions=[[0.0, 0.0]],
        action_dim=2,
        latency_total_ms=10.0,
        mode="test",
    )
    writer.write_footer({"total_requests": 1})
    writer.close()
    return writer.filepath


def test_replay_report_preserves_request_and_replayed_evidence(monkeypatch, tmp_path):
    trace = _trace(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    report = tmp_path / "report.json"
    monkeypatch.setattr(cli, "_load_target_server", lambda _path: _Server())

    assert cli.run_replay(str(trace), str(model), diff_mode="all", output_json=str(report)) == 0
    payload = json.loads(report.read_text())
    assert payload["summary"]["n_diffed"] == 1
    assert payload["summary"]["n_skipped"] == 0
    row = payload["per_request_diffs"][0]
    assert row["status"] == "replayed"
    assert row["request"] == {
        "instruction": "move",
        "state": [1.0, 2.0],
        "image_sha256": row["request"]["image_sha256"],
    }
    assert row["recorded_actions"] == [[0.0, 0.0]]
    assert row["replayed_actions"] == [[0.25, -0.5]]
    assert row["replayed_latency"] == {"total_ms": 12.5}
    assert row["cache"]["comparable"] is False


def test_replay_report_marks_missing_images_and_fail_on(monkeypatch, tmp_path):
    trace = _trace(tmp_path, image_redaction="hash_only")
    model = tmp_path / "model"
    model.mkdir()
    report = tmp_path / "report.json"
    monkeypatch.setattr(cli, "_load_target_server", lambda _path: _Server())

    assert cli.run_replay(str(trace), str(model), output_json=str(report), fail_on="actions") == 3
    payload = json.loads(report.read_text())
    assert payload["summary"]["n_diffed"] == 0
    assert payload["summary"]["n_skipped"] == 1
    assert payload["per_request_diffs"][0]["status"] == "skipped"
    assert "full recorded image" in payload["per_request_diffs"][0]["reason"]
