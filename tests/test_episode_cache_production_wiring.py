"""Regression test for the EpisodeCache -> Pi05DecomposedInference wiring.

tests/test_episode_cache_bytes.py covers EpisodeCache's byte tracking and
Prometheus emission in isolation (constructing EpisodeCache directly with
embodiment/model_id). This test covers the one place EpisodeCache is
actually constructed in production: Pi05DecomposedInference.__init__ with
cache_level="episode" (src/tether/runtime/pi05_decomposed_server.py).

Until this fix, that call site built EpisodeCache with no embodiment/model_id,
so tether_episode_cache_bytes_total silently never emitted once episode mode
was enabled (EpisodeCache._emit_bytes_metric is a no-op unless both are set).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def _stub_ort_session_with_io(input_names: list[str], output_shape=(1, 50, 7)):
    """Mock ort.InferenceSession — same shape as test_cuda_graphs_integration.py."""
    sess = MagicMock()
    inputs = []
    for name in input_names:
        inp = MagicMock()
        inp.name = name
        inp.shape = [1, 4]
        inp.type = "tensor(float)"
        inputs.append(inp)
    sess.get_inputs.return_value = inputs
    sess.get_outputs.return_value = []
    sess.get_providers.return_value = ["CPUExecutionProvider"]
    sess.run.return_value = [np.ones(output_shape, dtype=np.float32) * 0.1]
    return sess


@pytest.fixture
def decomposed_export_dir(tmp_path: Path) -> Path:
    """Minimal decomposed export dir Pi05DecomposedInference.__init__ accepts."""
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


def test_episode_cache_level_wires_embodiment_and_model_id(
    decomposed_export_dir, monkeypatch,
):
    """cache_level='episode' must construct EpisodeCache with the same
    embodiment/model_id already threaded in for CUDA-graph metrics, so the
    byte gauge actually emits instead of silently no-opping."""
    import onnxruntime as ort

    monkeypatch.setattr(
        ort, "InferenceSession",
        lambda *a, **kw: _stub_ort_session_with_io(["x"]),
    )

    from tether.runtime.pi05_decomposed_server import Pi05DecomposedInference

    inference = Pi05DecomposedInference(
        decomposed_export_dir,
        enable_cache=False,
        cache_level="episode",
        cuda_graphs_embodiment="franka",
        cuda_graphs_model_id="pi05-test",
    )

    assert inference._episode_cache is not None
    assert inference._episode_cache._embodiment == "franka"
    assert inference._episode_cache._model_id == "pi05-test"


def test_episode_cache_level_defaults_still_wire_through(
    decomposed_export_dir, monkeypatch,
):
    """Even without explicit cuda_graphs_* kwargs, the (default 'unknown')
    values must still reach EpisodeCache — the gauge should emit under an
    'unknown' label rather than not emit at all."""
    import onnxruntime as ort

    monkeypatch.setattr(
        ort, "InferenceSession",
        lambda *a, **kw: _stub_ort_session_with_io(["x"]),
    )

    from tether.runtime.pi05_decomposed_server import Pi05DecomposedInference

    inference = Pi05DecomposedInference(
        decomposed_export_dir,
        enable_cache=False,
        cache_level="episode",
    )

    assert inference._episode_cache is not None
    assert inference._episode_cache._embodiment == "unknown"
    assert inference._episode_cache._model_id == "unknown"
