from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_export_dir(tmp_path: Path, name: str) -> Path:
    export_dir = tmp_path / name
    export_dir.mkdir()
    (export_dir / "model.onnx").write_bytes(b"stub")
    (export_dir / "tether_config.json").write_text(json.dumps({
        "model_type": "smolvla",
        "export_kind": "monolithic",
        "num_denoising_steps": 1,
        "chunk_size": 2,
        "action_chunk_size": 2,
        "action_dim": 2,
        "max_state_dim": 4,
    }))
    return export_dir


def _stub_ort_session(value: float):
    sess = MagicMock()
    input_names = [
        "img_cam1", "img_cam2", "img_cam3",
        "mask_cam1", "mask_cam2", "mask_cam3",
        "lang_tokens", "lang_masks", "state", "noise",
    ]
    inputs = [MagicMock() for _ in input_names]
    for inp, name in zip(inputs, input_names):
        inp.name = name
    sess.get_inputs.return_value = inputs
    sess.get_providers.return_value = ["CPUExecutionProvider"]
    sess.run.return_value = [np.ones((1, 2, 2), dtype=np.float32) * value]
    return sess


def test_shadow_policy_records_candidate_actions_without_returning_them(
    tmp_path: Path,
    monkeypatch,
) -> None:
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/httpx not installed")

    ort = pytest.importorskip("onnxruntime")
    transformers = pytest.importorskip("transformers")

    prod_export = _make_export_dir(tmp_path, "prod_export")
    shadow_export = _make_export_dir(tmp_path, "shadow_export")
    record_dir = tmp_path / "records"

    prod_session = _stub_ort_session(0.05)
    shadow_session = _stub_ort_session(0.07)

    def _fake_inference_session(path, *args, **kwargs):
        return shadow_session if "shadow_export" in str(path) else prod_session

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", _fake_inference_session)

    tok_stub = MagicMock()
    tok_stub.return_value = {
        "input_ids": np.zeros((1, 16), dtype=np.int64),
        "attention_mask": np.ones((1, 16), dtype=np.int64),
    }
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tok_stub,
    )

    from tether.runtime.server import create_app

    app = create_app(
        str(prod_export),
        device="cpu",
        record_dir=record_dir,
        record_gzip=False,
        prewarm=False,
        shadow_policy=str(shadow_export),
        shadow_sample=1.0,
    )

    with TestClient(app) as client:
        response = client.post(
            "/act",
            json={
                "instruction": "pick",
                "state": [0.0, 0.0],
                "request_id": "req-shadow-1",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["actions"][0][0] == pytest.approx(0.05)
        assert body["shadow_sampled"] is True
        assert body["shadow_pending"] is True
        assert body["shadow_mode"] == "background"
        assert "shadow_actions" not in body

    trace_paths = list(record_dir.glob("*.jsonl"))
    assert len(trace_paths) == 1
    records = [
        json.loads(line)
        for line in trace_paths[0].read_text().splitlines()
        if line.strip()
    ]
    request_record = next(record for record in records if record["kind"] == "request")
    shadow_record = next(
        record for record in records if record["kind"] == "shadow_result"
    )

    assert request_record["response"]["actions"][0][0] == pytest.approx(0.05)
    routing = request_record["routing"]
    assert routing["shadow_sampled"] is True
    assert routing["shadow_pending"] is True
    assert "shadow_actions" not in routing

    shadow_routing = shadow_record["routing"]
    assert shadow_record["seq"] == request_record["seq"]
    assert shadow_routing["shadow_sampled"] is True
    assert shadow_routing["shadow_actions"][0][0] == pytest.approx(0.07)
    assert routing["shadow_policy_export_dir"] == str(shadow_export)

    from tether.policy_diff import diff_policy_traces

    report = diff_policy_traces(
        baseline_trace=trace_paths[0],
        shadow=True,
        max_action_delta=0.05,
    )
    assert report["summary"]["verdict"] == "pass"
    assert report["summary"]["compared"] == 1
