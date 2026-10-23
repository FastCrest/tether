"""Every ActionGuard caller must fail closed and never synthesize a zero target.

Covers the serving paths (TetherServer.predict, the batched path, the
decomposed pi0.5 server), the deadline fallback, `tether deploy-proof`'s
guard stress check, the `tether guard` CLI and the safety-config vs
embodiment unit cross-check. The /act embodiment-guard path is covered in
tests/test_action_guard_embodiment.py::TestRealActPath.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.export_config_factory import write_test_export_config
from tether.embodiments import EmbodimentConfig
from tether.safety import ActionGuard, GuardUnitsError, SafetyLimits
from tether.safety.guard import check_safety_units_against_embodiment


def _limits(n: int, **kw) -> SafetyLimits:
    base = dict(
        joint_names=[f"j{i}" for i in range(n)],
        position_min=[-100.0] * n,
        position_max=[100.0] * n,
        velocity_max=[1000.0] * n,
        effort_max=[1.0] * n,
        units=["radians"] * n,
    )
    base.update(kw)
    return SafetyLimits(**base)


# ---------------------------------------------------------------------------
# TetherServer.predict / batched path
# ---------------------------------------------------------------------------


@pytest.fixture
def onnx_server(tmp_path):
    from tether.runtime.server import TetherServer

    write_test_export_config(
        tmp_path,
        model_type="smolvla",
        model_id="lerobot/smolvla_base",
        export_kind="decomposed_onnx",
        artifacts=["expert_stack.onnx"],
        action_chunk_size=4,
        expert={"expert_hidden": 720, "action_dim": 3, "num_layers": 16},
    )
    server = TetherServer(tmp_path, device="cpu")
    server.action_dim = 3
    server.chunk_size = 4
    server.expert_hidden = 720
    server.num_denoising_steps = 1
    server._inference_mode = "onnx"
    server._ready = True
    server._action_guard = ActionGuard(_limits(3), max_consecutive_clamps=0)
    session = MagicMock()
    server._ort_session = session

    def set_velocity(value):
        session.run.return_value = [np.full((1, 4, 3), value, dtype=np.float32)]

    server.set_velocity = set_velocity  # type: ignore[attr-defined]
    return server


def test_predict_nan_first_chunk_returns_guard_stop_not_zeros(onnx_server):
    onnx_server.set_velocity(np.nan)
    result = onnx_server.predict()
    assert result["error"] == "guard_stop"
    assert "actions" not in result


def test_predict_nan_after_safe_chunk_holds_last_action(onnx_server):
    onnx_server.set_velocity(0.5)
    ok = onnx_server.predict(noise=np.full((4, 3), 0.25, dtype=np.float32))
    last = ok["actions"][-1]
    onnx_server.set_velocity(np.nan)
    held = onnx_server.predict()
    assert held["guard_hold"] is True
    assert held["actions"] == [pytest.approx(last)] * 4
    assert held["guard_summary"]["hold"] is True


def test_predict_guard_exception_fails_closed(onnx_server):
    onnx_server.set_velocity(0.0)
    onnx_server._action_guard.check = MagicMock(side_effect=RuntimeError("boom"))
    result = onnx_server.predict()
    assert result["error"] == "guard_stop"
    assert "actions" not in result


def test_predict_deadline_miss_without_prior_returns_error_not_zeros(onnx_server):
    onnx_server._action_guard = None
    onnx_server.set_velocity(0.0)
    onnx_server._deadline_ms = 0.0
    result = onnx_server.predict()
    assert result["error"] == "deadline_exceeded_no_safe_action"
    assert "actions" not in result


def test_batched_predict_nan_item_is_guard_stop(onnx_server):
    onnx_server.set_velocity(np.nan)
    session = onnx_server._ort_session
    session.run.return_value = [np.full((2, 4, 3), np.nan, dtype=np.float32)]
    results = onnx_server._predict_batch_sync([(None,), (None,)])
    assert [r.get("error") for r in results] == ["guard_stop", "guard_stop"]
    assert all("actions" not in r for r in results)


# ---------------------------------------------------------------------------
# Pi05DecomposedServer: guard is now actually applied
# ---------------------------------------------------------------------------


def _decomposed(tmp_path, monkeypatch, chunk: np.ndarray, limits: SafetyLimits):
    from tests.test_decomposed_server import _make_export_dir, _StubInference, _StubTokenizer
    from tether.runtime.decomposed_server import Pi05DecomposedServer

    class _Inference(_StubInference):
        # Keep the split-runtime fixture's actual prefix/expert provider interface
        # so startup reaches the guard under test without weakening admission.
        def predict_action_chunk(self, **kwargs):
            return chunk

    monkeypatch.setattr(
        "tether.runtime.pi05_decomposed_server.Pi05DecomposedInference",
        lambda **kwargs: _Inference(**kwargs),
    )
    monkeypatch.setattr(
        "tether.runtime.decomposed_server.Pi05DecomposedServer._infer_teacher_ref",
        lambda self: None,
    )
    safety = tmp_path / "safety.json"
    limits.save(safety)
    server = Pi05DecomposedServer(_make_export_dir(tmp_path), safety_config=safety)
    server.load()
    server._tokenizer = _StubTokenizer()
    return server


def test_decomposed_server_applies_guard_and_clamps(tmp_path, monkeypatch):
    chunk = np.full((1, 50, 32), 5.0, dtype=np.float32)
    lim = _limits(7, position_min=[-1.0] * 7, position_max=[1.0] * 7)
    server = _decomposed(tmp_path, monkeypatch, chunk, lim)
    server._action_mean = server._action_std = None
    result = server.predict_from_base64(image_b64=None, instruction="x", state=None)
    assert np.allclose(result["actions"], 1.0)
    assert result["safety_violations"] > 0


def test_decomposed_server_nan_first_chunk_is_guard_stop(tmp_path, monkeypatch):
    chunk = np.full((1, 50, 32), np.nan, dtype=np.float32)
    server = _decomposed(tmp_path, monkeypatch, chunk, _limits(7))
    result = server.predict_from_base64(image_b64=None, instruction="x", state=None)
    assert result["error"] == "guard_stop"
    assert "actions" not in result


def test_decomposed_server_refuses_legacy_safety_config(tmp_path, monkeypatch):
    with pytest.raises(GuardUnitsError):
        _decomposed(tmp_path, monkeypatch, np.zeros((1, 50, 32), np.float32), _limits(7, units=[]))


# ---------------------------------------------------------------------------
# Safety-config vs embodiment units
# ---------------------------------------------------------------------------


def test_safety_config_degrees_vs_so100_radians_refused():
    cfg = EmbodimentConfig.load_preset("so100")
    with pytest.raises(GuardUnitsError):
        check_safety_units_against_embodiment(
            _limits(6, units=["degrees"] * 5 + ["gripper_0_1"]), cfg
        )
    check_safety_units_against_embodiment(
        _limits(6, units=["radians"] * 5 + ["gripper_0_1"]), cfg
    )
    check_safety_units_against_embodiment(_limits(6), None)


# ---------------------------------------------------------------------------
# deploy-proof guard stress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("embodiment", ["so100", "franka", "ur5", "quadcopter"])
def test_deploy_proof_guard_stress_passes_with_hold_semantics(embodiment):
    from tether.deploy_proof import _run_guard_stress

    out = _run_guard_stress(
        safety_config=None, embodiment=embodiment, custom_embodiment_config=None
    )
    assert out["enabled"] is True
    checks = {c["name"]: c for c in out["checks"]}
    assert checks["guard_rejects_non_finite"]["status"] == "pass", checks
    assert checks["guard_clamps_out_of_range"]["status"] == "pass"
    assert checks["guard_consecutive_clamp_trip"]["status"] == "pass"


def test_deploy_proof_guard_stress_reports_legacy_config_failure(tmp_path):
    from tether.deploy_proof import _run_guard_stress

    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "joint_names": ["j0"], "position_min": [-1.0], "position_max": [1.0],
        "velocity_max": [1.0], "effort_max": [1.0],
    }))
    out = _run_guard_stress(
        safety_config=str(path), embodiment="", custom_embodiment_config=None
    )
    assert out["enabled"] is False
    assert any(
        c["name"] == "guard_stress_config_load" and c["status"] == "fail" for c in out["checks"]
    )


# ---------------------------------------------------------------------------
# `tether guard` CLI
# ---------------------------------------------------------------------------


def test_cli_guard_init_writes_units_and_check_accepts_it(tmp_path):
    from typer.testing import CliRunner

    from tether.cli import app

    out = tmp_path / "safety.json"
    runner = CliRunner()
    r = runner.invoke(app, ["guard", "init", "--output", str(out), "--num-joints", "3"])
    assert r.exit_code == 0, r.output
    data = json.loads(out.read_text())
    assert data["units"] == ["radians"] * 3
    assert data["action_space"] == "position"
    r = runner.invoke(app, ["guard", "check", "--config", str(out), "--num-joints", "3"])
    assert r.exit_code == 0, r.output


def test_cli_guard_check_refuses_legacy_config(tmp_path):
    from typer.testing import CliRunner

    from tether.cli import app

    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "joint_names": ["j0"], "position_min": [-1.0], "position_max": [1.0],
        "velocity_max": [1.0], "effort_max": [1.0],
    }))
    r = CliRunner().invoke(app, ["guard", "check", "--config", str(path), "--num-joints", "1"])
    assert r.exit_code == 1
