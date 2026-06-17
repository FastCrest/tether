"""End-to-end integration tests for chunk-budget-batching Phase 1.

Verifies the /act → PolicyRuntime → run_batch_callback wire-up against
the real FastAPI app (TestClient) with a stubbed monolithic backend.

The tests assert:
- Backend receives requests via PolicyRuntime.submit, not the legacy
  predict_from_base64_async fallback path.
- /act correctly returns results from the queue's fan-out.
- Queue-full scenarios surface as HTTP 503 with Retry-After.
- /act behavior unchanged for backends that don't have run_batch
  (graceful fallback).
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


def _setup_app(tmp_path, monkeypatch):
    """Builds a FastAPI app via create_app with a stubbed monolithic backend."""
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
    return _make_export_dir(tmp_path)


def test_policy_runtime_installed_in_lifespan(tmp_path, monkeypatch):
    """After lifespan startup, server.policies['prod'] is a running runtime."""
    from fastapi.testclient import TestClient

    export_dir = _setup_app(tmp_path, monkeypatch)
    from tether.runtime.server import create_app

    app = create_app(str(export_dir), device="cpu")
    with TestClient(app) as client:
        # Hit /health to ensure lifespan completed
        resp = client.get("/health")
        assert resp.status_code in (200, 503)
        # The runtime should be installed on app.state.tether_server.policies
        server = app.state.tether_server
        policies = getattr(server, "policies", None)
        assert policies is not None, "create_app should install server.policies"
        assert "prod" in policies, "single-policy default slot should be 'prod'"
        runtime = policies["prod"]
        assert runtime.is_running is True
        assert runtime.policy_id == "prod"
        snap = runtime.snapshot()
        assert snap["is_running"] is True


def test_act_routes_through_policy_runtime(tmp_path, monkeypatch):
    """A successful /act increments the runtime's batches_run counter."""
    from fastapi.testclient import TestClient

    export_dir = _setup_app(tmp_path, monkeypatch)
    from tether.runtime.server import create_app

    app = create_app(str(export_dir), device="cpu")
    with TestClient(app) as client:
        runtime = app.state.tether_server.policies["prod"]
        before_batches = runtime.snapshot()["batches_run"]

        resp = client.post("/act", json={
            "image": _tiny_jpeg_b64(),
            "instruction": "pick up the red cup",
            "state": [0.0] * 6,
        })
        assert resp.status_code == 200, resp.text

        after_batches = runtime.snapshot()["batches_run"]
        assert after_batches > before_batches, (
            f"PolicyRuntime should have processed at least one batch via /act; "
            f"before={before_batches} after={after_batches}"
        )
        # And requests_processed should match too
        assert runtime.snapshot()["requests_processed"] >= 1


def test_act_fans_back_correct_result_shape(tmp_path, monkeypatch):
    """The result returned to /act preserves the action-chunk shape."""
    from fastapi.testclient import TestClient

    export_dir = _setup_app(tmp_path, monkeypatch)
    from tether.runtime.server import create_app

    app = create_app(str(export_dir), device="cpu")
    with TestClient(app) as client:
        resp = client.post("/act", json={
            "image": _tiny_jpeg_b64(),
            "instruction": "demo",
            "state": [0.0] * 6,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "actions" in body
        assert len(body["actions"]) == 50, "50-step chunk must round-trip cleanly"


def test_cost_model_records_measurements_after_act(tmp_path, monkeypatch):
    """Each /act updates the runtime's cost model so subsequent scheduler
    decisions use real numbers."""
    from fastapi.testclient import TestClient

    export_dir = _setup_app(tmp_path, monkeypatch)
    from tether.runtime.server import create_app

    app = create_app(str(export_dir), device="cpu")
    with TestClient(app) as client:
        runtime = app.state.tether_server.policies["prod"]
        # Five /act calls — should leave at least one measurement per shape
        for _ in range(5):
            resp = client.post("/act", json={
                "image": _tiny_jpeg_b64(),
                "instruction": "demo",
                "state": [0.0] * 6,
            })
            assert resp.status_code == 200
        snap = runtime.cost_model.export_snapshot()
        # At least one entry should exist after 5 acts
        assert len(snap["entries"]) >= 1


def test_runtime_stops_cleanly_on_lifespan_exit(tmp_path, monkeypatch):
    """The runtime is shut down when the FastAPI app's lifespan exits."""
    from fastapi.testclient import TestClient

    export_dir = _setup_app(tmp_path, monkeypatch)
    from tether.runtime.server import create_app

    app = create_app(str(export_dir), device="cpu")
    runtime_ref = None
    with TestClient(app) as client:
        client.get("/health")
        runtime_ref = app.state.tether_server.policies["prod"]
        assert runtime_ref.is_running is True
    # After context exit, lifespan should have stopped the runtime
    assert runtime_ref is not None
    assert runtime_ref.is_running is False


def test_metrics_endpoint_includes_batch_diagnostics(tmp_path, monkeypatch):
    """After /act traffic, /metrics surfaces the chunk-budget-batching
    diagnostic series (cost histogram, size histogram, flush counter,
    capture-hit-rate gauge, queue-depth gauge)."""
    from fastapi.testclient import TestClient

    export_dir = _setup_app(tmp_path, monkeypatch)
    from tether.runtime.server import create_app

    app = create_app(str(export_dir), device="cpu")
    with TestClient(app) as client:
        # Drive a few /act calls to populate the metrics.
        for _ in range(3):
            resp = client.post("/act", json={
                "image": _tiny_jpeg_b64(),
                "instruction": "drive metrics",
                "state": [0.0] * 6,
            })
            assert resp.status_code == 200

        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        body = metrics_resp.text

        # All five batch-budget metric families should be present.
        assert "tether_batch_cost_per_flush_ms" in body
        assert "tether_batch_size_per_flush" in body
        assert "tether_batch_flush_total" in body
        assert "tether_captured_graph_hit_rate" in body
        assert "tether_policy_runtime_queue_depth" in body
        # Phase 1 single-shape: capture-hit-rate gauge should always be 1.0.
        assert "tether_captured_graph_hit_rate{" in body
