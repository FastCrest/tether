"""Regression test for `runtime-num-steps-10` GOALS.yaml gate.

The runtime server classes were originally written assuming num_steps=1.
They need to correctly surface num_denoising_steps from reflex_config.json
so customers know what denoise schedule the artifact was exported with.

Verifies:
  1. Pi0OnnxServer.predict() reads num_denoising_steps from config (not
     hardcoded).
  2. SmolVLAOnnxServer.predict() does the same.
  3. create_app dispatches to the monolithic server classes based on
     `export_kind: monolithic` + `model_type` in reflex_config.json.

The real num_steps=10 runtime roundtrip is covered by
`scripts/modal_serve_roundtrip_test.py` (via Pi0OnnxServer); this file
owns the unit-level correctness.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def _mock_ort_session(input_names: list[str], output_shape: tuple = (1, 50, 32)):
    sess = MagicMock()
    inputs = [MagicMock() for _ in input_names]
    for inp, name in zip(inputs, input_names):
        inp.name = name
    sess.get_inputs.return_value = inputs
    sess.run.return_value = [np.zeros(output_shape, dtype=np.float32)]
    return sess


def test_pi0_server_reports_num_steps_from_config(tmp_path):
    """Pi0OnnxServer.predict() returns num_denoising_steps from config,
    not a hardcoded 1."""
    from reflex.runtime.pi0_onnx_server import Pi0OnnxServer

    (tmp_path / "reflex_config.json").write_text(json.dumps({
        "model_type": "pi0",
        "export_kind": "monolithic",
        "num_denoising_steps": 10,
    }))

    srv = Pi0OnnxServer(str(tmp_path))
    # Simulate loaded state
    srv._session = _mock_ort_session([
        "img_base", "img_wrist_l", "img_wrist_r",
        "mask_base", "mask_wrist_l", "mask_wrist_r",
        "lang_tokens", "lang_masks", "state", "noise",
    ])
    srv._input_names = [i.name for i in srv._session.get_inputs()]
    srv._ready = True
    srv.config = json.loads((tmp_path / "reflex_config.json").read_text())

    resp = srv.predict(
        image=np.zeros((224, 224, 3), dtype=np.uint8),
        instruction="pick up",
        state=[0.0] * 14,
        noise=np.zeros((1, 50, 32), dtype=np.float32),
        lang_tokens=np.zeros((1, 16), dtype=np.int64),
        lang_masks=np.ones((1, 16), dtype=np.bool_),
    )
    assert resp["num_denoising_steps"] == 10, (
        f"Expected 10 from config, got {resp['num_denoising_steps']}"
    )
    assert "actions" in resp
    assert len(resp["actions"]) == 50


def test_smolvla_server_reports_num_steps_from_config(tmp_path):
    """SmolVLAOnnxServer.predict() returns num_denoising_steps from config."""
    from reflex.runtime.smolvla_onnx_server import SmolVLAOnnxServer

    (tmp_path / "reflex_config.json").write_text(json.dumps({
        "model_type": "smolvla",
        "export_kind": "monolithic",
        "num_denoising_steps": 10,
        "chunk_size": 50,
        "action_dim": 32,
    }))

    srv = SmolVLAOnnxServer(str(tmp_path))
    srv._session = _mock_ort_session([
        "img_cam1", "img_cam2", "img_cam3",
        "mask_cam1", "mask_cam2", "mask_cam3",
        "lang_tokens", "lang_masks", "state", "noise",
    ])
    srv._input_names = [i.name for i in srv._session.get_inputs()]
    srv._ready = True
    srv.config = json.loads((tmp_path / "reflex_config.json").read_text())

    resp = srv.predict(
        image=np.zeros((512, 512, 3), dtype=np.uint8),
        instruction="pick up",
        state=[0.0] * 6,
        noise=np.zeros((1, 50, 32), dtype=np.float32),
        lang_tokens=np.zeros((1, 16), dtype=np.int64),
        lang_masks=np.ones((1, 16), dtype=np.bool_),
    )
    assert resp["num_denoising_steps"] == 10
    assert "actions" in resp
    assert resp["inference_mode"] == "smolvla_onnx_monolithic"


def test_pi05_server_uses_pi0_shape_without_state_input(tmp_path):
    """Pi05OnnxServer reuses pi0-style camera/lang/noise inputs but should
    not require a state input because pi0.5 carries proprio in language."""
    from reflex.runtime.pi05_onnx_server import Pi05OnnxServer

    (tmp_path / "reflex_config.json").write_text(json.dumps({
        "model_type": "pi05",
        "export_kind": "monolithic",
        "num_denoising_steps": 10,
        "chunk_size": 50,
        "action_dim": 32,
    }))

    srv = Pi05OnnxServer(str(tmp_path))
    srv._session = _mock_ort_session([
        "img_base", "img_wrist_l", "img_wrist_r",
        "mask_base", "mask_wrist_l", "mask_wrist_r",
        "lang_tokens", "lang_masks", "noise",
    ])
    srv._input_names = [i.name for i in srv._session.get_inputs()]
    srv._ready = True
    srv.config = json.loads((tmp_path / "reflex_config.json").read_text())

    resp = srv.predict(
        image=np.zeros((224, 224, 3), dtype=np.uint8),
        instruction="pick up",
        noise=np.zeros((1, 50, 32), dtype=np.float32),
        lang_tokens=np.zeros((1, 16), dtype=np.int64),
        lang_masks=np.ones((1, 16), dtype=np.bool_),
    )

    assert resp["inference_mode"] == "pi05_onnx_monolithic"
    assert resp["num_denoising_steps"] == 10
    assert "state" not in srv._session.run.call_args.args[1]


def test_reflex_server_uses_model_onnx_for_gr00t_monolithic(tmp_path, monkeypatch):
    from reflex.runtime.server import ReflexServer

    (tmp_path / "model.onnx").write_bytes(b"fake")
    (tmp_path / "reflex_config.json").write_text(json.dumps({
        "model_type": "gr00t",
        "export_kind": "monolithic",
        "chunk_size": 50,
        "action_dim": 64,
        "num_denoising_steps": 4,
    }))

    loaded = {}

    def _fake_load_onnx(self, onnx_path):
        loaded["path"] = Path(onnx_path)
        self._ort_session = _mock_ort_session([
            "noisy_actions", "timestep", "position_ids",
        ], output_shape=(1, 50, 64))
        self._inference_mode = "onnx_gpu"

    monkeypatch.setattr(ReflexServer, "_load_onnx", _fake_load_onnx)
    monkeypatch.setattr(ReflexServer, "_load_vlm_orchestrator", lambda self: None)

    srv = ReflexServer(str(tmp_path), device="cpu")
    srv.load()

    assert loaded["path"] == tmp_path / "model.onnx"
    assert srv.ready is True
    assert srv.action_dim == 64


def test_create_app_dispatch_monolithic(tmp_path, monkeypatch):
    """create_app routes to Pi0OnnxServer / SmolVLAOnnxServer when
    reflex_config.json declares `export_kind: monolithic`."""
    pytest.importorskip("fastapi")
    from reflex.runtime import server as server_module

    # Write a fake model.onnx + config so _find_onnx_path succeeds
    (tmp_path / "model.onnx").write_bytes(b"fake")
    (tmp_path / "reflex_config.json").write_text(json.dumps({
        "model_type": "smolvla",
        "export_kind": "monolithic",
        "num_denoising_steps": 10,
    }))

    # Stub SmolVLAOnnxServer so we don't actually load the fake ONNX
    from reflex.runtime import smolvla_onnx_server as smolvla_server_module
    original_server_cls = smolvla_server_module.SmolVLAOnnxServer
    instances = []

    class StubSmolVLA:
        def __init__(self, *a, **kw):
            instances.append((a, kw))
            self.ready = True
            self.export_dir = tmp_path
            self.config = {}
            self._inference_mode = "smolvla_onnx_monolithic"
        def load(self): pass
        async def start_batch_worker(self): pass
        async def stop_batch_worker(self): pass
        def predict(self, **kw): return {"actions": [[0.0] * 6] * 50}

    monkeypatch.setattr(smolvla_server_module, "SmolVLAOnnxServer", StubSmolVLA)

    app = server_module.create_app(str(tmp_path), device="cpu")
    # The dispatch happened — we got a SmolVLA stub instance
    assert len(instances) >= 1, "create_app did not instantiate SmolVLAOnnxServer"
