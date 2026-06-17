"""Integration tests for cuda-graphs (Day 7 of cuda-graphs_plan.md).

Complements tests/test_cuda_graphs.py (wrapper-level unit tests with mocked
sessions) and tests/test_check_cuda_graphs.py (tether doctor surface) by
exercising the integration surface:

- create_app() propagates cuda_graphs_enabled to server._cuda_graphs_enabled
- create_app() with legacy TetherServer + cuda_graphs_enabled=True logs
  the no-op warning (Phase 1 production wire-up gap, tracked by
  chunk-budget-batching's decomposed-dispatch fix)
- Pi05DecomposedInference threads cuda_graphs_enabled into the ORT session
  factory + try_capture_or_fall_back path
- Capture-failed-at-init produces an EagerSessionWrapper + emits the
  capture_failed_at_init metric with reason label
- /metrics endpoint surfaces the cuda-graph counters (zero-valued when
  no captures have occurred — Prometheus convention)
- Prometheus label cardinality stays bounded across embodiment/model/session
  combinations

Real-GPU end-to-end tests are gated on CUDA availability via
pytest.mark.skipif. They run on Modal A10G / A100 (per Day 8-9 of plan)
and are skipped in local dev environments.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _has_cuda() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


@pytest.fixture
def decomposed_export_dir(tmp_path: Path) -> Path:
    """Build a minimal decomposed export dir with a tether_config.json that
    Pi05DecomposedInference's __init__ will accept."""
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "vlm_prefix.onnx").write_bytes(b"stub-prefix")
    (export_dir / "expert_denoise.onnx").write_bytes(b"stub-expert")
    (export_dir / "tether_config.json").write_text(json.dumps({
        "model_type": "pi05",
        "export_kind": "decomposed",
        "num_denoising_steps": 10,
        "chunk_size": 50,
        "action_chunk_size": 50,
        "action_dim": 7,
        "max_state_dim": 32,
        "decomposed": {
            "vlm_prefix_onnx": "vlm_prefix.onnx",
            "expert_denoise_onnx": "expert_denoise.onnx",
            "past_kv_tensor_names": [f"past_kv_{i}" for i in range(2)],
            "paligemma_layers": 2,
        },
    }))
    return export_dir


def _stub_ort_session_with_io(input_names: list[str], output_shape=(1, 50, 7)):
    """Mock ort.InferenceSession with get_inputs() that satisfies
    _make_probe_feed() in cuda_graphs.try_capture_or_fall_back()."""
    sess = MagicMock()
    inputs = []
    for name in input_names:
        inp = MagicMock()
        inp.name = name
        inp.shape = [1, 4]
        inp.type = "tensor(float)"
        inputs.append(inp)
    sess.get_inputs.return_value = inputs
    sess.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess.run.return_value = [np.ones(output_shape, dtype=np.float32) * 0.1]
    return sess


# ---------------------------------------------------------------------------
# create_app integration: flag propagation
# ---------------------------------------------------------------------------


def test_create_app_propagates_cuda_graphs_enabled_to_server_attr(tmp_path, monkeypatch):
    """create_app(cuda_graphs_enabled=True) must set server._cuda_graphs_enabled.
    Catches signature drift between the CLI flag and the server attribute."""
    pytest.importorskip("fastapi")
    import onnxruntime as ort

    stub = _stub_ort_session_with_io(["x"])
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **kw: stub)

    monolithic_export = tmp_path / "export"
    monolithic_export.mkdir()
    (monolithic_export / "model.onnx").write_bytes(b"stub")
    (monolithic_export / "tether_config.json").write_text(json.dumps({
        "model_type": "smolvla",
        "export_kind": "monolithic",
        "num_denoising_steps": 10,
        "chunk_size": 50,
        "action_chunk_size": 50,
        "action_dim": 7,
        "max_state_dim": 32,
    }))

    from tether.runtime.server import create_app

    app = create_app(str(monolithic_export), device="cpu", cuda_graphs_enabled=True)
    server = app.state.tether_server
    assert getattr(server, "_cuda_graphs_enabled", None) is True


def test_create_app_default_cuda_graphs_disabled(tmp_path, monkeypatch):
    """Default behavior: cuda_graphs_enabled defaults to False."""
    pytest.importorskip("fastapi")
    import onnxruntime as ort

    stub = _stub_ort_session_with_io(["x"])
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **kw: stub)

    monolithic_export = tmp_path / "export"
    monolithic_export.mkdir()
    (monolithic_export / "model.onnx").write_bytes(b"stub")
    (monolithic_export / "tether_config.json").write_text(json.dumps({
        "model_type": "smolvla",
        "export_kind": "monolithic",
        "num_denoising_steps": 10,
        "chunk_size": 50,
        "action_chunk_size": 50,
        "action_dim": 7,
        "max_state_dim": 32,
    }))

    from tether.runtime.server import create_app

    app = create_app(str(monolithic_export), device="cpu")
    server = app.state.tether_server
    assert getattr(server, "_cuda_graphs_enabled", None) is False


def test_legacy_tether_server_logs_noop_warning_when_cuda_graphs_set(
    tmp_path, monkeypatch, caplog,
):
    """When cuda_graphs_enabled=True is set on the legacy TetherServer
    (legacy path with no export_kind in tether_config.json — neither
    monolithic nor decomposed — that doesn't currently consume the flag),
    a single info log surfaces the no-op so operators notice. Production
    wire-up for the decomposed path is tracked by chunk-budget-batching's
    decomposed-dispatch fix."""
    pytest.importorskip("fastapi")
    import onnxruntime as ort

    stub = _stub_ort_session_with_io(["x"])
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **kw: stub)

    legacy_export = tmp_path / "export"
    legacy_export.mkdir()
    (legacy_export / "expert_stack.onnx").write_bytes(b"stub")
    (legacy_export / "tether_config.json").write_text(json.dumps({
        "model_type": "pi05",
        "num_denoising_steps": 10,
        "chunk_size": 50,
        "action_chunk_size": 50,
        "action_dim": 7,
        "max_state_dim": 32,
    }))

    from tether.runtime.server import create_app

    with caplog.at_level(logging.INFO, logger="tether.runtime.server"):
        try:
            create_app(str(legacy_export), device="cpu", cuda_graphs_enabled=True)
        except Exception:
            pass  # legacy path may fail to load expert_stack stub; we only care about the log

    assert any("--cuda-graphs was set" in rec.message for rec in caplog.records), (
        f"Expected no-op log on legacy TetherServer; got messages: "
        f"{[rec.message for rec in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Pi05DecomposedInference: cuda-graphs wiring through to ORT
# ---------------------------------------------------------------------------


def test_pi05_decomposed_inference_wires_cuda_graphs_provider_to_session(
    decomposed_export_dir, monkeypatch,
):
    """When cuda_graphs_enabled=True, Pi05DecomposedInference must construct
    sessions with enable_cuda_graph=1 in the CUDA provider options. Verifies
    the provider list passed to ort.InferenceSession includes the flag."""
    import onnxruntime as ort

    captured_provider_lists = []

    def fake_session(model_path, *, providers=None, **kwargs):
        captured_provider_lists.append(providers)
        return _stub_ort_session_with_io(["x"])

    monkeypatch.setattr(ort, "InferenceSession", fake_session)

    from tether.runtime.pi05_decomposed_server import Pi05DecomposedInference

    Pi05DecomposedInference(
        decomposed_export_dir,
        enable_cache=False,
        cuda_graphs_enabled=True,
        cuda_graphs_embodiment="franka",
        cuda_graphs_model_id="pi05-test",
    )

    cuda_entries = [
        p for plist in captured_provider_lists
        for p in plist
        if isinstance(p, tuple) and p[0] == "CUDAExecutionProvider"
    ]
    assert any(
        entry[1].get("enable_cuda_graph") == "1" for entry in cuda_entries
    ), f"cuda_graphs_enabled=True should set enable_cuda_graph=1 on CUDA provider; got {captured_provider_lists}"


def test_pi05_decomposed_capture_failure_falls_back_to_eager(
    decomposed_export_dir, monkeypatch,
):
    """When cuda-graph capture probe raises (e.g., A10G OOM on vlm_prefix per
    2026-04-25-cuda-graphs-ort-spike-modal experiment), Pi05DecomposedInference
    should construct an EagerSessionWrapper for that session and continue
    loading. Verifies the graceful-degrade path lands as expected."""
    import onnxruntime as ort
    from tether.runtime.cuda_graphs import EagerSessionWrapper

    failing_capture_session = _stub_ort_session_with_io(["x"])
    failing_capture_session.run.side_effect = RuntimeError("BFC arena alloc 16MB failed")
    eager_session = _stub_ort_session_with_io(["x"])

    call_log = []

    def fake_session(model_path, *, providers=None, **kwargs):
        cuda_entry = providers[0]
        cg_enabled = (
            isinstance(cuda_entry, tuple)
            and cuda_entry[1].get("enable_cuda_graph") == "1"
        )
        call_log.append((str(model_path), cg_enabled))
        return failing_capture_session if cg_enabled else eager_session

    monkeypatch.setattr(ort, "InferenceSession", fake_session)

    from tether.runtime.pi05_decomposed_server import Pi05DecomposedInference

    inference = Pi05DecomposedInference(
        decomposed_export_dir,
        enable_cache=False,
        cuda_graphs_enabled=True,
        cuda_graphs_embodiment="franka",
        cuda_graphs_model_id="pi05-cgfail",
    )

    assert isinstance(inference._sess_prefix, EagerSessionWrapper)
    assert isinstance(inference._sess_expert, EagerSessionWrapper)

    cg_attempts = [(p, cg) for (p, cg) in call_log if cg]
    eager_builds = [(p, cg) for (p, cg) in call_log if not cg]
    assert len(cg_attempts) == 2, "should have probed capture for both sessions"
    assert len(eager_builds) == 2, "should have built eager sessions after fallback"


# ---------------------------------------------------------------------------
# /metrics endpoint surfaces cuda-graph counters
# ---------------------------------------------------------------------------


def test_metrics_endpoint_includes_cuda_graph_counter_names(tmp_path, monkeypatch):
    """/metrics must expose all 5 cuda-graph metrics by name (Prometheus
    convention: counters appear with zero values until incremented).
    Catches accidental metric removal or rename."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import onnxruntime as ort
    from fastapi.testclient import TestClient

    stub = _stub_ort_session_with_io(["x"])
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **kw: stub)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "model.onnx").write_bytes(b"stub")
    (export_dir / "tether_config.json").write_text(json.dumps({
        "model_type": "smolvla",
        "export_kind": "monolithic",
        "num_denoising_steps": 10,
        "chunk_size": 50,
        "action_chunk_size": 50,
        "action_dim": 7,
        "max_state_dim": 32,
    }))

    from tether.runtime.cuda_graphs import CudaGraphWrapper

    sess = MagicMock()
    sess.run.return_value = ["x"]
    w = CudaGraphWrapper(sess, "vlm_prefix", embodiment="franka", model_id="metrics-warm")
    w.run(None, {})

    from tether.runtime.server import create_app

    app = create_app(str(export_dir), device="cpu")
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    for metric_name in (
        "tether_cuda_graph_captured_total",
        "tether_cuda_graph_replayed_total",
        "tether_cuda_graph_eager_fallback_total",
        "tether_cuda_graph_capture_failed_at_init_total",
        "tether_cuda_graph_capture_seconds",
    ):
        assert metric_name in body, f"missing metric {metric_name} in /metrics output"


# ---------------------------------------------------------------------------
# Label cardinality bound
# ---------------------------------------------------------------------------


def test_prometheus_label_cardinality_stays_bounded():
    """5 embodiments × 3 model_ids × 2 sessions = 30 series. Verify the
    metric registry stays bounded — no accidental high-cardinality dimension
    (request_id, episode_id) that would blow up Prometheus storage."""
    from tether.observability.prometheus import (
        tether_cuda_graph_captured_total,
    )
    from tether.runtime.cuda_graphs import CudaGraphWrapper

    embodiments = ["card_emb_a", "card_emb_b", "card_emb_c", "card_emb_d", "card_emb_e"]
    model_ids = ["card_mid_1", "card_mid_2", "card_mid_3"]
    sessions = ["vlm_prefix", "expert_denoise"]

    series_before = len(tether_cuda_graph_captured_total._metrics)
    for emb in embodiments:
        for mid in model_ids:
            for sname in sessions:
                sess = MagicMock()
                sess.run.return_value = ["x"]
                w = CudaGraphWrapper(sess, sname, embodiment=emb, model_id=mid)
                w.run(None, {})
    series_after = len(tether_cuda_graph_captured_total._metrics)
    delta = series_after - series_before
    expected = len(embodiments) * len(model_ids) * len(sessions)
    assert delta == expected, (
        f"cardinality grew {delta} for {len(embodiments)}×{len(model_ids)}×{len(sessions)}"
        f"={expected} combinations — expected exact match (one series per labelset)"
    )


# ---------------------------------------------------------------------------
# CUDA-gated: real ORT capture+replay (Modal-only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_cuda(), reason="CUDAExecutionProvider not available")
def test_real_cuda_graph_capture_and_replay_parity(tmp_path):
    """End-to-end test gated on CUDA. Builds a tiny ONNX, captures, replays,
    and verifies the captured output matches eager output within atol=1e-4.

    Runs on Modal A10G/A100 per Day 8-9 of cuda-graphs_plan.md. Local dev
    machines without GPU skip this test; the unit + mock-based integration
    tests above provide CI coverage of the wrapper logic."""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16])
    add_node = helper.make_node("Add", ["x", "x"], ["y"], name="dbl")
    graph = helper.make_graph([add_node], "dbl_graph", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_path = tmp_path / "dbl.onnx"
    onnx.save(model, str(onnx_path))

    feed = {"x": np.arange(16, dtype=np.float32).reshape(1, 16)}

    eager = ort.InferenceSession(
        str(onnx_path),
        providers=[("CUDAExecutionProvider", {}), "CPUExecutionProvider"],
    )
    eager_out = eager.run(None, feed)[0]

    from tether.runtime.cuda_graphs import CudaGraphWrapper, build_cuda_graph_providers

    cg_session = ort.InferenceSession(
        str(onnx_path),
        providers=build_cuda_graph_providers(enabled=True),
    )
    wrapped = CudaGraphWrapper(
        cg_session,
        session_name="expert_denoise",
        embodiment="test",
        model_id="cg-real-1",
    )
    capture_out = wrapped.run(None, feed)[0]
    replay_out = wrapped.run(None, feed)[0]

    np.testing.assert_allclose(eager_out, capture_out, atol=1e-4)
    np.testing.assert_allclose(capture_out, replay_out, atol=1e-9)
    assert wrapped.captured is True
