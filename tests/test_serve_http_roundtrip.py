"""Integration test: real FastAPI `/act` roundtrip.

Gate-7 (`serve-act-roundtrip`) was previously closed on evidence that
`Pi0OnnxServer.predict()` works end-to-end via Modal — but that bypasses
the FastAPI layer. This test closes the HTTP gap: it boots the actual
`create_app()` against a monolithic config (SmolVLA or pi0), sends a
POST /act via `fastapi.testclient.TestClient`, and verifies actions
come back through the full HTTP → `predict_from_base64_async` →
monolithic server → ORT stub → JSONResponse chain.

ORT session is stubbed so we don't need a real 12-GB ONNX on disk —
the goal is to verify wiring, not numerical correctness (that's covered
by the parity gates).
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def _stub_ort_session(input_names: list[str], output_shape=(1, 50, 32)):
    sess = MagicMock()
    inputs = [MagicMock() for _ in input_names]
    for inp, name in zip(inputs, input_names):
        inp.name = name
    sess.get_inputs.return_value = inputs
    sess.run.return_value = [np.ones(output_shape, dtype=np.float32) * 0.05]
    return sess


def _make_export_dir(tmp_path: Path, model_type: str = "smolvla") -> Path:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "model.onnx").write_bytes(b"stub")
    (export_dir / "tether_config.json").write_text(json.dumps({
        "model_type": model_type,
        "export_kind": "monolithic",
        "num_denoising_steps": 10,
        "chunk_size": 50,
        "action_chunk_size": 50,
        "action_dim": 32,
        "max_state_dim": 32,
    }))
    return export_dir


def _tiny_jpeg_b64() -> str:
    """Produce a valid 32x32 RGB JPEG as base64 — real enough to decode."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed — needed for base64 image decode")
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(128, 64, 200)).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("model_type,input_names", [
    ("smolvla", [
        "img_cam1", "img_cam2", "img_cam3",
        "mask_cam1", "mask_cam2", "mask_cam3",
        "lang_tokens", "lang_masks", "state", "noise",
    ]),
    ("pi0", [
        "img_base", "img_wrist_l", "img_wrist_r",
        "mask_base", "mask_wrist_l", "mask_wrist_r",
        "lang_tokens", "lang_masks", "state", "noise",
    ]),
])
def test_act_http_roundtrip(tmp_path, monkeypatch, model_type, input_names):
    """POST /act returns a valid action chunk through the full FastAPI stack."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/httpx not installed")

    import onnxruntime as ort

    stub_session = _stub_ort_session(input_names)
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **kw: stub_session)

    # Also stub the tokenizer to avoid network downloads in the test.
    import transformers
    tok_stub = MagicMock()
    tok_stub.return_value = {
        "input_ids": np.zeros((1, 16), dtype=np.int64),
        "attention_mask": np.ones((1, 16), dtype=np.int64),
    }
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        lambda *a, **kw: tok_stub,
    )

    export_dir = _make_export_dir(tmp_path, model_type=model_type)

    from tether.runtime.server import create_app
    app = create_app(str(export_dir), device="cpu")

    # TestClient triggers lifespan automatically on `with` entry.
    with TestClient(app) as client:
        resp = client.post("/act", json={
            "image": _tiny_jpeg_b64(),
            "instruction": "pick up the red cup",
            "state": [0.0] * 6,
        })
        assert resp.status_code == 200, f"status={resp.status_code}, body={resp.text[:500]}"
        body = resp.json()
        # Either we got actions or an informative error — never a crash
        if "error" in body:
            pytest.fail(f"/act returned error: {body}")
        assert "actions" in body
        actions = body["actions"]
        assert isinstance(actions, list)
        assert len(actions) == 50, f"expected 50-step chunk, got {len(actions)}"
        assert all(isinstance(v, (int, float)) for v in actions[0])
        # Metadata sanity
        assert body.get("num_actions") == 50
        assert body.get("num_denoising_steps") == 10


def test_guard_status_endpoint(tmp_path, monkeypatch):
    """GET /guard/status returns enabled=false when no guard is loaded."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/httpx not installed")

    import onnxruntime as ort
    stub = _stub_ort_session([
        "img_cam1", "img_cam2", "img_cam3",
        "mask_cam1", "mask_cam2", "mask_cam3",
        "lang_tokens", "lang_masks", "state", "noise",
    ])
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **kw: stub)

    export_dir = _make_export_dir(tmp_path, "smolvla")

    from tether.runtime.server import create_app
    app = create_app(str(export_dir), device="cpu")
    with TestClient(app) as client:
        resp = client.get("/guard/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("enabled") is False


def test_guard_endpoints_require_api_key_when_configured(tmp_path, monkeypatch):
    """Guard control surfaces must use the same auth boundary as /act."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/httpx not installed")

    import onnxruntime as ort
    stub = _stub_ort_session([
        "img_cam1", "img_cam2", "img_cam3",
        "mask_cam1", "mask_cam2", "mask_cam3",
        "lang_tokens", "lang_masks", "state", "noise",
    ])
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **kw: stub)

    export_dir = _make_export_dir(tmp_path, "smolvla")

    from tether.runtime.server import create_app
    app = create_app(str(export_dir), device="cpu", api_key="secret-key")
    with TestClient(app) as client:
        assert client.get("/guard/status").status_code == 401
        assert client.post("/guard/reset").status_code == 401
        authed_status = client.get(
            "/guard/status",
            headers={"X-Tether-Key": "secret-key"},
        )
        assert authed_status.status_code == 200
        assert authed_status.json()["enabled"] is False
        authed_reset = client.post(
            "/guard/reset",
            headers={"X-Tether-Key": "secret-key"},
        )
        assert authed_reset.status_code == 400
        assert authed_reset.json()["error"] == "guard_not_enabled"


def test_offline_mode_requires_bundled_tokenizer_at_startup(tmp_path, monkeypatch):
    """TETHER_OFFLINE=1 should fail before serving an export that needs HF."""
    import onnxruntime as ort

    from tether.runtime.smolvla_onnx_server import SmolVLAOnnxServer
    from tether.runtime.tokenizers import OfflineTokenizerMissingError

    stub = _stub_ort_session([
        "img_cam1", "img_cam2", "img_cam3",
        "mask_cam1", "mask_cam2", "mask_cam3",
        "lang_tokens", "lang_masks", "state", "noise",
    ])
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **kw: stub)
    monkeypatch.setenv("TETHER_OFFLINE", "1")
    export_dir = _make_export_dir(tmp_path, "smolvla")

    server = SmolVLAOnnxServer(export_dir, device="cpu")

    with pytest.raises(OfflineTokenizerMissingError, match="offline tokenizer assets missing"):
        server.load()
