"""Policy action units: recorded at export time, enforced at serve time.

- every exporter family writes ``action_units`` into tether_config.json
  when the export command declares them;
- ``tether serve`` refuses a mismatch with the safety config / embodiment,
  and refuses an older export without action_units unless the operator
  passes them explicitly;
- /act latches after an ActionGuard trip and /guard/reset clears both guards.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.export_config_factory import write_test_export_config
from tether.embodiments import EmbodimentConfig
from tether.export_config import (
    ExportConfigError,
    build_producer_config,
    export_action_units,
    load_tether_config,
    write_tether_config,
)
from tether.safety import ActionGuard, GuardUnitsError, SafetyLimits
from tether.safety.guard import parse_action_units

SO100_UNITS = {"kind": "position", "units": ["radians"] * 5 + ["gripper_0_1"]}
FRANKA_UNITS = {"kind": "position", "units": ["radians"] * 6 + ["gripper_0_1"]}


def _identity_onnx(path: Path, action_dim: int = 7, opset: int = 17) -> None:
    import onnx
    from onnx import TensorProto, helper

    path.parent.mkdir(parents=True, exist_ok=True)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, action_dim])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, action_dim])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.save(model, str(path))


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------


def test_parse_action_units_repeat_syntax():
    assert parse_action_units("radians*5,gripper_0_1") == SO100_UNITS
    assert parse_action_units("normalized*7", "velocity") == {
        "kind": "velocity", "units": ["normalized"] * 7,
    }


@pytest.mark.parametrize("text", ["", "radians*x", "newton_meters*2", "parsecs"])
def test_parse_action_units_rejects_bad_specs(text):
    with pytest.raises(GuardUnitsError):
        parse_action_units(text)


def test_export_config_rejects_malformed_action_units(tmp_path):
    root = tmp_path / "e"
    _identity_onnx(root / "model.onnx")
    payload = build_producer_config(
        root, producer="monolithic", model_id="org/m", model_type="pi0",
        action_dim=7, num_denoising_steps=10, opset=17,
    )
    payload["action_units"] = {"kind": "position", "units": ["furlongs"]}
    with pytest.raises(ExportConfigError):
        write_tether_config(root, payload)


# ---------------------------------------------------------------------------
# Exporters write action_units (every family goes through write_tether_config)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("producer", "model_type"),
    [
        ("monolithic", "pi0"),  # monolithic: smolvla / pi0 / pi05 / gr00t
        ("pi05_split", "pi05_decomposed"),
        ("expert_stack", "smolvla"),  # smolvla / pi0 / pi05 / gr00t expert stacks
        ("expert_stack", "gr00t"),
        ("expert_stack", "pi0"),
        ("pi0_prefix", "pi0"),
        ("dreamzero", "dreamzero"),
    ],
)
def test_each_producer_layout_records_action_units(tmp_path, producer, model_type):
    from tether.export_config import PRODUCER_LAYOUTS

    root = tmp_path / producer
    for path, role in PRODUCER_LAYOUTS[producer]["artifacts"]:
        if role == "model":
            _identity_onnx(root / path)
    with export_action_units(FRANKA_UNITS):
        payload = build_producer_config(
            root, producer=producer, model_id=f"org/{model_type}", model_type=model_type,
            action_dim=7, num_denoising_steps=10, opset=17,
        )
        write_tether_config(root, payload)
    assert load_tether_config(root)["action_units"] == FRANKA_UNITS


def test_no_declared_units_writes_no_field(tmp_path):
    root = tmp_path / "e"
    _identity_onnx(root / "model.onnx")
    payload = build_producer_config(
        root, producer="monolithic", model_id="org/m", model_type="pi0",
        action_dim=7, num_denoising_steps=10, opset=17,
    )
    write_tether_config(root, payload)
    assert "action_units" not in load_tether_config(root)


def test_monolithic_exporter_writer_records_action_units(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tether.exporters import monolithic

    monkeypatch.setattr(monolithic, "_bundle_tokenizer", lambda *a, **k: {})
    _identity_onnx(tmp_path / "model.onnx", opset=19)
    with export_action_units(FRANKA_UNITS):
        monolithic._write_tether_config(
            tmp_path, SimpleNamespace(chunk_size=50, max_action_dim=7, max_state_dim=8),
            num_steps=10, model_id="org/pi0", model_type="pi0",
        )
    assert load_tether_config(tmp_path)["action_units"] == FRANKA_UNITS


def test_pi05_decomposed_exporter_writer_records_action_units(tmp_path):
    from tether.exporters import decomposed
    from tether.exporters._export_mode import ExportMode

    _identity_onnx(tmp_path / "vlm_prefix.onnx", opset=19)
    _identity_onnx(tmp_path / "expert_denoise.onnx", opset=19)
    with export_action_units(FRANKA_UNITS):
        decomposed._write_decomposed_export_result(
            model_id="org/pi05", output_dir=tmp_path, num_steps=10, target="desktop",
            student_checkpoint=None, variant="default", past_kv_names=[],
            chunk_size=50, action_dim=7, max_action_dim=32, state_in_prompt=True,
            export_mode=ExportMode.SEQUENTIAL,
            export_mode_reason="test",
        )
    assert load_tether_config(tmp_path)["action_units"] == FRANKA_UNITS


def test_dreamzero_exporter_records_action_units(tmp_path):
    from tether.exporters.dreamzero import export_dreamzero

    with export_action_units(FRANKA_UNITS):
        export_dreamzero("org/dreamzero", tmp_path, action_dim=7)
    assert load_tether_config(tmp_path)["action_units"] == FRANKA_UNITS


def test_smolvla_vlm_manifest_update_keeps_action_units(tmp_path):
    from tether.exporters.vlm_prefix_exporter import _update_vlm_manifest

    _identity_onnx(tmp_path / "expert_stack.onnx")
    with export_action_units(FRANKA_UNITS):
        write_tether_config(tmp_path, build_producer_config(
            tmp_path, producer="expert_stack", model_id="org/smolvla",
            model_type="smolvla", action_dim=7, num_denoising_steps=10, opset=17,
        ))
    for name in ("vision_encoder.onnx", "text_embedder.onnx", "decoder_prefill.onnx"):
        _identity_onnx(tmp_path / name)
    _update_vlm_manifest(
        tmp_path, checkpoint_path_or_id="org/smolvla", image_size=512,
        vlm_hidden_size=960, vlm_kv_dim=320,
    )
    assert load_tether_config(tmp_path)["action_units"] == FRANKA_UNITS


def test_every_exporter_writes_config_only_through_write_tether_config():
    """The stamp lives in write_tether_config; no exporter may bypass it."""
    exporters = Path(__file__).resolve().parents[1] / "src" / "tether" / "exporters"
    offenders = []
    for path in sorted(exporters.glob("*.py")):
        text = path.read_text()
        if "tether_config.json" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "tether_config.json":
                line = text.splitlines()[node.lineno - 1]
                if "write_text" in line or "open(" in line or "json.dump" in line:
                    offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


def test_cli_export_stamps_action_units(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from tether import cli
    from tether.exporters import monolithic

    def fake_export_monolithic(model, output, **kwargs):
        out = Path(output)
        _identity_onnx(out / "model.onnx")
        write_tether_config(out, build_producer_config(
            out, producer="monolithic", model_id=model, model_type="smolvla",
            action_dim=7, num_denoising_steps=10, opset=17,
        ))
        return {"status": "ok", "output_dir": str(out), "files": {}}

    monkeypatch.setattr(monolithic, "export_monolithic", fake_export_monolithic)
    out = tmp_path / "export"
    r = CliRunner().invoke(cli.app, [
        "export", "org/smolvla", "--output", str(out),
        "--action-units", "radians*6,gripper_0_1",
    ])
    assert (out / "tether_config.json").exists(), r.output
    assert json.loads((out / "tether_config.json").read_text())["action_units"] == FRANKA_UNITS

    r = CliRunner().invoke(cli.app, [
        "export", "org/smolvla", "--output", str(out), "--action-units", "parsecs",
    ])
    assert r.exit_code == 2


# ---------------------------------------------------------------------------
# Serve-time gate (real create_app)
# ---------------------------------------------------------------------------


class _Stub:
    responses: list[dict] = []

    def __init__(self, export_dir, *args, **kwargs):
        self.export_dir = Path(export_dir)
        self._ready = True
        self.health_state = "ready"
        self._inference_mode = "stub"
        self._vlm_loaded = False
        self.consecutive_crash_count = 0
        self.max_consecutive_crashes = 0
        self.robot_id = ""
        self._action_guard = None

    @property
    def ready(self):
        return self._ready

    async def load(self):
        self._ready = True

    async def predict_from_base64_async(self, image_b64=None, instruction="", state=None):
        return dict(type(self).responses.pop(0))


def _export(tmp_path, **extras) -> Path:
    root = tmp_path / "export"
    write_test_export_config(root, model_type="gr00t", action_dim=7, **extras)
    return root


def _safety(tmp_path, units) -> Path:
    path = tmp_path / "safety.json"
    SafetyLimits(
        joint_names=[f"j{i}" for i in range(len(units))],
        position_min=[-3.0] * len(units), position_max=[3.0] * len(units),
        velocity_max=[10.0] * len(units), effort_max=[1.0] * len(units),
        units=list(units),
    ).save(path)
    return path


@pytest.fixture
def stub_server(monkeypatch):
    from tether.runtime import server as runtime_server

    monkeypatch.setattr(runtime_server, "TetherServer", _Stub)
    return runtime_server


def test_serve_refuses_export_units_vs_safety_config_mismatch(tmp_path, stub_server):
    export = _export(tmp_path, action_units={"kind": "position", "units": ["degrees"] * 7})
    with pytest.raises(GuardUnitsError):
        stub_server.create_app(
            str(export), device="cpu", safety_config=str(_safety(tmp_path, ["radians"] * 7))
        )


def test_serve_refuses_export_units_vs_embodiment_mismatch(tmp_path, stub_server):
    export = _export(tmp_path, action_units={"kind": "position", "units": ["degrees"] * 7})
    with pytest.raises(GuardUnitsError):
        stub_server.create_app(
            str(export), device="cpu",
            embodiment_config=EmbodimentConfig.load_preset("franka"),
        )


def test_serve_accepts_matching_export_units(tmp_path, stub_server):
    export = _export(tmp_path, action_units=FRANKA_UNITS)
    app = stub_server.create_app(
        str(export), device="cpu",
        safety_config=str(_safety(tmp_path, FRANKA_UNITS["units"])),
        embodiment_config=EmbodimentConfig.load_preset("franka"),
    )
    assert app.state.tether_server.action_units == FRANKA_UNITS


def test_legacy_export_without_units_refused_when_guarded(tmp_path, stub_server):
    export = _export(tmp_path)
    with pytest.raises(GuardUnitsError, match="--action-units"):
        stub_server.create_app(
            str(export), device="cpu",
            embodiment_config=EmbodimentConfig.load_preset("franka"),
        )
    with pytest.raises(GuardUnitsError, match="--action-units"):
        stub_server.create_app(
            str(export), device="cpu", safety_config=str(_safety(tmp_path, ["radians"] * 7)),
        )


def test_legacy_export_accepted_with_explicit_units(tmp_path, stub_server):
    export = _export(tmp_path)
    app = stub_server.create_app(
        str(export), device="cpu",
        embodiment_config=EmbodimentConfig.load_preset("franka"),
        action_units=FRANKA_UNITS,
    )
    assert app.state.tether_server.action_units == FRANKA_UNITS


def test_legacy_export_explicit_units_still_checked(tmp_path, stub_server):
    export = _export(tmp_path)
    with pytest.raises(GuardUnitsError):
        stub_server.create_app(
            str(export), device="cpu",
            embodiment_config=EmbodimentConfig.load_preset("franka"),
            action_units={"kind": "position", "units": ["degrees"] * 6 + ["gripper_0_1"]},
        )


def test_explicit_units_contradicting_export_refused(tmp_path, stub_server):
    export = _export(tmp_path, action_units=FRANKA_UNITS)
    with pytest.raises(GuardUnitsError, match="contradict"):
        stub_server.create_app(
            str(export), device="cpu",
            action_units={"kind": "position", "units": ["degrees"] * 7},
        )


def test_unguarded_legacy_export_still_serves(tmp_path, stub_server):
    app = stub_server.create_app(str(_export(tmp_path)), device="cpu")
    assert app.state.tether_server.action_units is None


def test_cli_serve_passes_action_units(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import uvicorn

    from tether import cli
    from tether.runtime import server as runtime_server

    captured = {}

    def fake_create_app(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(runtime_server, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    export = _export(tmp_path)
    r = CliRunner().invoke(cli.app, [
        "serve", str(export), "--device", "cpu", "--action-units", "radians*6,gripper_0_1",
    ])
    assert captured.get("action_units") == FRANKA_UNITS, r.output


# ---------------------------------------------------------------------------
# /act latch + /guard/reset (real create_app)
# ---------------------------------------------------------------------------


@pytest.fixture
def latched_client(tmp_path, stub_server):
    export = _export(tmp_path, action_units=FRANKA_UNITS)
    app = stub_server.create_app(
        str(export), device="cpu", embodiment_config=EmbodimentConfig.load_preset("franka"),
    )
    with TestClient(app) as client:
        yield client, app.state.tether_server


def _mean():
    return [float(v) for v in EmbodimentConfig.load_preset("franka").normalization["mean_action"]]


def test_act_latches_after_embodiment_guard_trip_and_reset_clears(latched_client):
    client, server = latched_client
    server.embodiment_guard.max_consecutive_clamps = 1
    bad = _mean()
    bad[0] = 99.0
    _Stub.responses = [
        {"actions": [bad], "latency_ms": 1.0},
        {"actions": [_mean()], "latency_ms": 1.0},
    ]
    tripping = client.post("/act", json={"image": None}).json()
    assert tripping["guard_clamped"] is True
    assert server.embodiment_guard.tripped

    latched = client.post("/act", json={"image": None})
    assert latched.status_code == 503
    assert latched.json()["error"] == "guard_stop"
    assert "actions" not in latched.json()
    assert len(_Stub.responses) == 1  # no inference ran while latched

    status = client.get("/guard/status").json()
    assert status["tripped"] is True
    assert status["guards"]["embodiment"]["tripped"] is True

    reset = client.post("/guard/reset").json()
    assert reset["was_tripped"] is True
    assert reset["guards_reset"] == ["embodiment"]
    ok = client.post("/act", json={"image": None}).json()
    assert ok["actions"][0] == pytest.approx(_mean())


def test_reset_clears_both_guards_and_safety_trip_latches(latched_client):
    client, server = latched_client
    safety = ActionGuard(SafetyLimits.default(num_joints=7), max_consecutive_clamps=1)
    safety.check(np.full((1, 7), 10.0))
    assert safety.tripped
    server._action_guard = safety
    server.embodiment_guard.max_consecutive_clamps = 1
    server.embodiment_guard.check(np.full((1, 7), 99.0, dtype=np.float32))
    assert server.embodiment_guard.tripped

    latched = client.post("/act", json={"image": None})
    assert latched.status_code == 503
    assert "safety_config" in latched.json()["reason"]
    assert "embodiment" in latched.json()["reason"]

    reset = client.post("/guard/reset").json()
    assert reset["guards_reset"] == ["safety_config", "embodiment"]
    assert not safety.tripped and not server.embodiment_guard.tripped
    status = client.get("/guard/status").json()
    assert status["tripped"] is False
    assert set(status["guards"]) == {"safety_config", "embodiment"}
