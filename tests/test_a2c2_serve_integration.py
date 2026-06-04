"""End-to-end integration tests for the A2C2 serve runtime hook.

Verifies the /act → A2C2Hook wire-up against the real FastAPI app
(TestClient) with a stubbed monolithic backend + a tiny A2C2 checkpoint.

Per a2c2-correction execution plan B.5 Day 3 acceptance criteria.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tether.kernels.a2c2_correction import A2C2Config, A2C2Head


def _stub_ort_session(input_names: list[str], output_shape=(1, 50, 32)):
    sess = MagicMock()
    inputs = [MagicMock() for _ in input_names]
    for inp, name in zip(inputs, input_names):
        inp.name = name
    sess.get_inputs.return_value = inputs
    sess.run.return_value = [np.ones(output_shape, dtype=np.float32) * 0.05]
    return sess


def _make_export_dir(tmp_path: Path) -> Path:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "model.onnx").write_bytes(b"stub")
    (export_dir / "tether_config.json").write_text(json.dumps({
        "model_type": "smolvla",
        "export_kind": "monolithic",
        "num_denoising_steps": 10,
        "chunk_size": 50,
        "action_chunk_size": 50,
        "action_dim": 32,
        "max_state_dim": 32,
    }))
    return export_dir


def _tiny_jpeg_b64() -> str:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    img = Image.new("RGB", (224, 224), color=(120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _setup_app(tmp_path, monkeypatch, *, a2c2_ckpt: str | None = None):
    """Builds a FastAPI app via create_app with optional A2C2 checkpoint."""
    try:
        from fastapi.testclient import TestClient  # noqa: F401
    except ImportError:
        pytest.skip("fastapi/httpx not installed")
    import onnxruntime as ort
    import transformers

    input_names = [
        "img_cam1", "img_cam2", "img_cam3",
        "mask_cam1", "mask_cam2", "mask_cam3",
        "lang_tokens", "lang_masks", "state", "noise",
    ]
    stub_session = _stub_ort_session(input_names)
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **kw: stub_session)

    tok_stub = MagicMock()
    tok_stub.return_value = {
        "input_ids": np.zeros((1, 16), dtype=np.int64),
        "attention_mask": np.ones((1, 16), dtype=np.int64),
    }
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        lambda *a, **kw: tok_stub,
    )

    export_dir = _make_export_dir(tmp_path)
    from tether.runtime.server import create_app

    return create_app(
        str(export_dir), device="cpu",
        a2c2_checkpoint=a2c2_ckpt,
    )


def _make_a2c2_checkpoint(tmp_path: Path, action_dim: int) -> str:
    """Save a random-init A2C2 head with action_dim matching the test backend."""
    cfg = A2C2Config(action_dim=action_dim, hidden_dim=64)
    head = A2C2Head.random_init(cfg, seed=42)
    ckpt = tmp_path / "a2c2_test.npz"
    head.save(ckpt)
    return str(ckpt)


def test_a2c2_hook_not_loaded_when_checkpoint_unset(tmp_path, monkeypatch):
    """No --a2c2-checkpoint → server.a2c2_hook is None."""
    from fastapi.testclient import TestClient
    app = _setup_app(tmp_path, monkeypatch, a2c2_ckpt=None)
    with TestClient(app) as client:
        client.get("/health")
        server = app.state.tether_server
        assert getattr(server, "a2c2_hook", "MISSING") is None


def test_a2c2_hook_loaded_when_checkpoint_set(tmp_path, monkeypatch):
    """--a2c2-checkpoint provided → hook is loaded + attached."""
    from fastapi.testclient import TestClient
    ckpt = _make_a2c2_checkpoint(tmp_path, action_dim=32)
    app = _setup_app(tmp_path, monkeypatch, a2c2_ckpt=ckpt)
    with TestClient(app) as client:
        client.get("/health")
        server = app.state.tether_server
        hook = getattr(server, "a2c2_hook", None)
        assert hook is not None
        assert hook.head.config.action_dim == 32


def test_a2c2_hook_load_failure_disables_gracefully(tmp_path, monkeypatch):
    """Bad checkpoint path → server starts with a2c2_hook=None + error log
    instead of crashing."""
    from fastapi.testclient import TestClient
    app = _setup_app(
        tmp_path, monkeypatch,
        a2c2_ckpt=str(tmp_path / "does_not_exist.npz"),
    )
    with TestClient(app) as client:
        client.get("/health")
        assert getattr(app.state.tether_server, "a2c2_hook", "MISSING") is None


def test_act_response_includes_a2c2_telemetry_when_hook_loaded(tmp_path, monkeypatch):
    """When the hook is loaded, /act response includes a2c2_applied,
    a2c2_reason, a2c2_correction_magnitude — even on cold start (skipped)."""
    from fastapi.testclient import TestClient
    ckpt = _make_a2c2_checkpoint(tmp_path, action_dim=32)
    app = _setup_app(tmp_path, monkeypatch, a2c2_ckpt=ckpt)
    with TestClient(app) as client:
        resp = client.post("/act", json={
            "image": _tiny_jpeg_b64(),
            "instruction": "test",
            "state": [0.0] * 6,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "a2c2_applied" in body
        assert "a2c2_reason" in body
        assert "a2c2_correction_magnitude" in body
        # Cold start = skip
        assert body["a2c2_applied"] is False
        assert body["a2c2_reason"] == "cold_start"
        assert body["a2c2_correction_magnitude"] == 0.0


def test_act_response_omits_a2c2_telemetry_when_hook_unset(tmp_path, monkeypatch):
    """No hook → no a2c2_* fields in /act response (back-compat)."""
    from fastapi.testclient import TestClient
    app = _setup_app(tmp_path, monkeypatch, a2c2_ckpt=None)
    with TestClient(app) as client:
        resp = client.post("/act", json={
            "image": _tiny_jpeg_b64(),
            "instruction": "test",
            "state": [0.0] * 6,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "a2c2_applied" not in body


def test_a2c2_hook_records_outcomes_after_each_act(tmp_path, monkeypatch):
    """Each /act updates the hook's rolling windows so steady-state
    decisions reflect real traffic."""
    from fastapi.testclient import TestClient
    ckpt = _make_a2c2_checkpoint(tmp_path, action_dim=32)
    app = _setup_app(tmp_path, monkeypatch, a2c2_ckpt=ckpt)
    with TestClient(app) as client:
        for _ in range(3):
            resp = client.post("/act", json={
                "image": _tiny_jpeg_b64(),
                "instruction": "test",
                "state": [0.0] * 6,
            })
            assert resp.status_code == 200
        hook = app.state.tether_server.a2c2_hook
        n_lat, n_succ = hook.sample_count()
        assert n_lat == 3
        assert n_succ == 3


def test_metrics_endpoint_includes_a2c2_counters(tmp_path, monkeypatch):
    """After /act traffic, /metrics surfaces the a2c2 applied + skipped
    counters (Phase B.5 Day 3 metric requirement)."""
    from fastapi.testclient import TestClient
    ckpt = _make_a2c2_checkpoint(tmp_path, action_dim=32)
    app = _setup_app(tmp_path, monkeypatch, a2c2_ckpt=ckpt)
    with TestClient(app) as client:
        for _ in range(3):
            resp = client.post("/act", json={
                "image": _tiny_jpeg_b64(),
                "instruction": "test",
                "state": [0.0] * 6,
            })
            assert resp.status_code == 200
        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        body = metrics_resp.text
        assert "tether_a2c2_applied_total" in body
        assert "tether_a2c2_skipped_total" in body
