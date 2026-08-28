from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tether.export_config import (
    PRODUCER_LAYOUTS,
    ExportConfigError,
    UnsupportedExportKindError,
    UnsupportedExportPipelineError,
    build_producer_config,
    decomposed_layout,
    load_tether_config,
    normalize_legacy_tether_config,
    require_supported_export_kind,
    require_supported_pipeline,
    validate_tether_config,
    write_tether_config,
)


def _payload(path: str = "model.onnx") -> dict:
    return {
        "schema_version": 1,
        "model_id": "org/model",
        "model_type": "pi0",
        "action_dim": 7,
        "num_denoising_steps": 10,
        "opset": 17,
        "export_kind": "monolithic_onnx",
        "artifacts": [{"path": path, "role": "model"}],
        "io_contract": {
            "inputs": [
                {"name": "state", "dtype": "float32", "shape": ["batch", 7]},
                {"name": "image", "dtype": "uint8", "shape": [None, 224, 224, 3]},
            ],
            "outputs": [{"name": "actions", "dtype": "float32", "shape": ["batch", 50, 7]}],
        },
    }


def _export(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "export"
    root.mkdir()
    artifact = root / "model.onnx"
    artifact.write_bytes(b"not an actual onnx graph")
    payload = _payload()
    return root, payload


def test_canonical_round_trip_checks_digest_and_preserves_extensions(tmp_path):
    root, payload = _export(tmp_path)
    payload["artifacts"][0]["sha256"] = hashlib.sha256(b"not an actual onnx graph").hexdigest()
    payload["extensions"] = {"vendor": {"token_width": 2048}}

    path = write_tether_config(root, payload)
    loaded = load_tether_config(path)

    assert loaded == payload
    assert json.loads(path.read_text())["extensions"] == payload["extensions"]


def test_legacy_kind_is_normalized_without_guessing_required_fields(tmp_path):
    root, payload = _export(tmp_path)
    payload.pop("schema_version")
    payload["export_kind"] = "monolithic"
    (root / "tether_config.json").write_text(json.dumps(payload))

    loaded = load_tether_config(root)

    assert loaded["schema_version"] == 1
    assert loaded["export_kind"] == "monolithic_onnx"


def test_legacy_top_level_num_inference_steps_is_normalized(tmp_path):
    root, payload = _export(tmp_path)
    payload.pop("schema_version")
    payload.pop("num_denoising_steps")
    payload["num_inference_steps"] = 6
    payload["export_kind"] = "monolithic"
    (root / "tether_config.json").write_text(json.dumps(payload))

    loaded = load_tether_config(root)

    assert loaded["num_denoising_steps"] == 6


def test_old_pi0_expert_layout_derives_decomposed_kind(tmp_path):
    root = tmp_path / "pi0-old"
    _write_identity_onnx(root / "expert_stack.onnx", action_dim=7)
    legacy = {
        "model_id": "lerobot/pi0",
        "model_type": "pi0",
        "action_dim": 7,
        "num_inference_steps": 10,
    }
    (root / "tether_config.json").write_text(json.dumps(legacy))

    loaded = load_tether_config(root)

    assert loaded["export_kind"] == "decomposed_onnx"
    assert loaded["num_denoising_steps"] == 10


def test_old_gr00t_model_layout_derives_monolithic_kind(tmp_path):
    root = tmp_path / "gr00t-old"
    _write_identity_onnx(root / "model.onnx", action_dim=14)
    legacy = {
        "model_id": "nvidia/gr00t",
        "model_type": "gr00t",
        "action_dim": 14,
        "num_inference_steps": 4,
    }
    (root / "tether_config.json").write_text(json.dumps(legacy))

    loaded = load_tether_config(root)

    assert loaded["export_kind"] == "monolithic_onnx"


def test_old_dreamzero_export_format_derives_config_only(tmp_path):
    root = tmp_path / "dreamzero-old"
    root.mkdir()
    legacy = {
        "checkpoint_path": "org/dreamzero",
        "model_family": "dreamzero",
        "export_format": "dreamzero_decomposed",
        "action_dim": 32,
        "num_inference_steps": 4,
        "opset_version": 17,
    }
    (root / "tether_config.json").write_text(json.dumps(legacy))

    loaded = load_tether_config(root)

    assert loaded["export_kind"] == "config_only"
    assert loaded["artifacts"] == []
    assert loaded["io_contract"] == {"inputs": [], "outputs": []}


def test_legacy_onnx_metadata_and_external_weights_are_inspected(tmp_path):
    onnx = pytest.importorskip("onnx")
    root = tmp_path / "export"
    root.mkdir()
    x = onnx.helper.make_tensor_value_info("state", onnx.TensorProto.FLOAT, ["batch", 7])
    y = onnx.helper.make_tensor_value_info("actions", onnx.TensorProto.FLOAT, ["batch", 7])
    node = onnx.helper.make_node("Identity", ["state"], ["actions"])
    graph = onnx.helper.make_graph([node], "fixture", [x], [y])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    onnx.save(model, root / "model.onnx")
    (root / "model.onnx.data").write_bytes(b"external weights")
    legacy = {
        "model_id": "org/model",
        "model_type": "pi0",
        "action_dim": 7,
        "num_denoising_steps": 10,
        "export_kind": "monolithic",
    }

    normalized = normalize_legacy_tether_config(legacy, root)
    validated = validate_tether_config(normalized, root=root)

    assert validated["opset"] == 17
    assert validated["io_contract"]["inputs"] == [
        {"name": "state", "dtype": "float32", "shape": ["batch", 7]}
    ]
    assert {item["role"] for item in validated["artifacts"]} == {"model", "weights"}


@pytest.mark.parametrize(
    "missing", ["model_id", "model_type", "action_dim", "num_denoising_steps", "opset"]
)
def test_legacy_migration_never_guesses_required_fields(tmp_path, missing):
    root, payload = _export(tmp_path)
    payload.pop("schema_version")
    payload.pop(missing)
    if missing == "opset":
        (root / "model.onnx").rename(root / "model.bin")
        payload["artifacts"][0]["path"] = "model.bin"
    (root / "tether_config.json").write_text(json.dumps(payload))

    with pytest.raises(ExportConfigError, match=missing):
        load_tether_config(root, inspect_artifacts=False)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("/tmp/model.onnx", "relative path"),
        ("../model.onnx", "without '\\.\\.'"),
        ("parts/../model.onnx", "without '\\.\\.'"),
        ("parts\\model.onnx", "POSIX separators"),
        ("./model.onnx", "normalized"),
        ("parts//model.onnx", "normalized"),
    ],
)
def test_artifact_paths_are_confined_and_normalized(path, message):
    payload = _payload(path)
    with pytest.raises(ExportConfigError, match=message):
        validate_tether_config(payload, inspect_artifacts=False)


@pytest.mark.parametrize("role", ["checkpoint", "onnx", "MODEL", ""])
def test_unknown_artifact_role_fails(role):
    payload = _payload()
    payload["artifacts"][0]["role"] = role
    with pytest.raises(ExportConfigError, match="role must be one of"):
        validate_tether_config(payload, inspect_artifacts=False)


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "z" * 64, 42])
def test_malformed_digest_fails(digest):
    payload = _payload()
    payload["artifacts"][0]["sha256"] = digest
    with pytest.raises(ExportConfigError, match="lowercase 64-hex"):
        validate_tether_config(payload, inspect_artifacts=False)


def test_missing_and_digest_mismatched_artifacts_fail(tmp_path):
    root, payload = _export(tmp_path)
    payload["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(ExportConfigError, match="sha256 mismatch"):
        validate_tether_config(payload, root=root)
    (root / "model.onnx").unlink()
    payload["artifacts"][0].pop("sha256")
    with pytest.raises(ExportConfigError, match="is missing"):
        validate_tether_config(payload, root=root)


@pytest.mark.parametrize(
    "shape",
    [[0], [-1], [True], ["9batch"], ["has-hyphen"], [{}]],
)
def test_invalid_tensor_dimensions_fail(shape):
    payload = _payload()
    payload["io_contract"]["inputs"][0]["shape"] = shape
    with pytest.raises(ExportConfigError, match="shape"):
        validate_tether_config(payload, inspect_artifacts=False)


def test_config_only_can_have_empty_artifacts_and_io_contract():
    payload = _payload()
    payload["export_kind"] = "config_only"
    payload["artifacts"] = []
    payload["io_contract"] = {"inputs": [], "outputs": []}
    assert validate_tether_config(payload)["export_kind"] == "config_only"


def test_non_config_export_requires_an_artifact():
    payload = _payload()
    payload["artifacts"] = []
    with pytest.raises(ExportConfigError, match="must not be empty"):
        validate_tether_config(payload)


def test_unsupported_kind_is_explicit_after_validation():
    payload = _payload()
    payload["export_kind"] = "trt_engine"
    with pytest.raises(UnsupportedExportKindError, match="does not support"):
        require_supported_export_kind(payload, {"monolithic_onnx"}, "local verifier")


def test_symlink_artifact_cannot_escape_export_root(tmp_path):
    root, payload = _export(tmp_path)
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"outside")
    (root / "model.onnx").unlink()
    (root / "model.onnx").symlink_to(outside)
    with pytest.raises(ExportConfigError, match="outside the export directory"):
        validate_tether_config(payload, root=root)


def _write_identity_onnx(path: Path, *, action_dim: int = 7, opset: int = 17) -> None:
    onnx = pytest.importorskip("onnx")
    path.parent.mkdir(parents=True, exist_ok=True)
    x = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 50, action_dim])
    y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 50, action_dim])
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["output"])],
        path.stem,
        [x],
        [y],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", opset)])
    onnx.save(model, path)


def _materialize_producer(
    tmp_path: Path,
    producer: str,
    *,
    model_type: str,
    action_dim: int = 7,
) -> Path:
    root = tmp_path / producer
    for path, role in PRODUCER_LAYOUTS[producer]["artifacts"]:
        if role == "model":
            _write_identity_onnx(root / path, action_dim=action_dim)
    payload = build_producer_config(
        root,
        producer=producer,
        model_id=f"org/{model_type}",
        model_type=model_type,
        action_dim=action_dim,
        num_denoising_steps=10,
        opset=17,
    )
    write_tether_config(root, payload)
    return root


def _materialize_smolvla_full_bundle(tmp_path: Path) -> Path:
    from tether.exporters.vlm_prefix_exporter import _update_vlm_manifest

    root = _materialize_producer(tmp_path, "expert_stack", model_type="smolvla")
    for filename in ("vision_encoder.onnx", "text_embedder.onnx", "decoder_prefill.onnx"):
        _write_identity_onnx(root / filename)
    _update_vlm_manifest(
        root,
        checkpoint_path_or_id="org/smolvla",
        image_size=512,
        vlm_hidden_size=960,
        vlm_kv_dim=320,
    )
    return root


@pytest.mark.parametrize(
    ("producer", "model_type"),
    [
        ("monolithic", "pi0"),
        ("pi05_split", "pi05_decomposed"),
        ("expert_stack", "smolvla"),
        ("pi0_prefix", "pi0"),
        ("dreamzero", "dreamzero"),
    ],
)
def test_each_real_producer_layout_builds_a_valid_config(tmp_path, producer, model_type):
    root = _materialize_producer(tmp_path, producer, model_type=model_type)
    assert load_tether_config(root)["export_kind"] == PRODUCER_LAYOUTS[producer]["export_kind"]


def test_canonical_producer_does_not_discover_stale_files(tmp_path):
    root = tmp_path / "export"
    _write_identity_onnx(root / "model.onnx")
    _write_identity_onnx(root / "stale.onnx")
    (root / "old.bin").write_bytes(b"stale")

    payload = build_producer_config(
        root,
        producer="monolithic",
        model_id="org/model",
        model_type="pi0",
        action_dim=7,
        num_denoising_steps=10,
        opset=17,
    )

    assert [artifact["path"] for artifact in payload["artifacts"]] == ["model.onnx"]


def test_canonical_producer_includes_only_referenced_external_weights(tmp_path):
    import numpy as np

    onnx = pytest.importorskip("onnx")
    root = tmp_path / "export"
    root.mkdir()
    x = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 7])
    y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 7])
    weight = onnx.numpy_helper.from_array(np.eye(7, dtype=np.float32), name="weight")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "external",
        [x],
        [y],
        [weight],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    onnx.save_model(
        model,
        root / "model.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=0,
    )
    (root / "stale.data").write_bytes(b"stale")

    payload = build_producer_config(
        root,
        producer="monolithic",
        model_id="org/model",
        model_type="pi0",
        action_dim=7,
        num_denoising_steps=10,
        opset=17,
    )

    assert [artifact["path"] for artifact in payload["artifacts"]] == [
        "model.onnx",
        "model.onnx.data",
    ]


def test_gr00t_robot_action_dim_comes_from_encoder_not_token_width():
    import torch

    from tether.exporters.gr00t import GR00T_META_KEYS, _raw_action_dim_from_checkpoint

    checkpoint = {GR00T_META_KEYS["action_enc_W1_W"]: torch.empty(32, 14, 1024)}
    assert _raw_action_dim_from_checkpoint(checkpoint) == 14


def test_gr00t_token_expert_is_not_routed_as_raw_action_reader(tmp_path):
    from tether.validate_roundtrip import ValidateRoundTrip

    root = tmp_path / "gr00t-token-expert"
    _write_identity_onnx(root / "expert_stack.onnx", action_dim=1024)
    payload = build_producer_config(
        root,
        producer="expert_stack",
        model_id="nvidia/gr00t",
        model_type="gr00t",
        action_dim=14,
        num_denoising_steps=4,
        opset=17,
        metadata={
            "pipeline": "gr00t_token_expert",
            "extensions": {"token_input_dim": 1024, "token_output_dim": 1024},
        },
    )
    write_tether_config(root, payload)

    with pytest.raises(UnsupportedExportPipelineError, match="does not support pipeline"):
        ValidateRoundTrip(root, num_test_cases=1)
    with pytest.raises(UnsupportedExportPipelineError, match="unsupported decomposed pipeline"):
        decomposed_layout(load_tether_config(root))


@pytest.mark.parametrize(
    "producer_and_model",
    [
        ("expert_stack", "pi0"),
        ("expert_stack", "smolvla"),
        ("expert_stack", "gr00t"),
    ],
)
def test_declared_round_trip_reader_pairs_load_config(tmp_path, producer_and_model):
    from tether.validate_roundtrip import ValidateRoundTrip

    producer, model_type = producer_and_model
    root = _materialize_producer(tmp_path, producer, model_type=model_type)
    reader = ValidateRoundTrip(root, num_test_cases=1)
    assert reader.config == load_tether_config(root)


@pytest.mark.parametrize(
    ("producer", "model_type", "error", "message"),
    [
        ("monolithic", "pi0", UnsupportedExportKindError, "does not support export_kind"),
        (
            "pi05_split",
            "pi05_decomposed",
            UnsupportedExportPipelineError,
            "supports only the expert_stack",
        ),
        ("dreamzero", "dreamzero", UnsupportedExportKindError, "does not support export_kind"),
        ("pi0_prefix", "pi0", UnsupportedExportPipelineError, "does not support pipeline"),
    ],
)
def test_unsupported_reader_pairs_fail_explicitly(
    tmp_path, producer, model_type, error, message
):
    from tether.validate_roundtrip import ValidateRoundTrip

    root = _materialize_producer(tmp_path, producer, model_type=model_type)
    with pytest.raises(error, match=message):
        ValidateRoundTrip(root, num_test_cases=1)


def test_decomposed_reader_gate_distinguishes_supported_layouts(tmp_path):
    split = _materialize_producer(tmp_path, "pi05_split", model_type="pi05_decomposed")
    expert = _materialize_producer(tmp_path, "expert_stack", model_type="smolvla")
    custom = _materialize_producer(tmp_path, "pi0_prefix", model_type="pi0")

    assert decomposed_layout(load_tether_config(split)) == "pi05_split"
    assert decomposed_layout(load_tether_config(expert)) == "expert_stack"
    with pytest.raises(UnsupportedExportPipelineError, match="unsupported decomposed pipeline"):
        decomposed_layout(load_tether_config(custom))


def test_smolvla_full_bundle_is_an_exact_supported_layout(tmp_path):
    root = _materialize_smolvla_full_bundle(tmp_path)

    assert decomposed_layout(load_tether_config(root)) == "smolvla_full_bundle"

    config = load_tether_config(root)
    config["artifacts"].append({"path": "custom.onnx", "role": "model"})
    with pytest.raises(UnsupportedExportPipelineError, match="unsupported decomposed"):
        decomposed_layout(config)


def test_ambiguous_decomposed_manifest_is_rejected():
    payload = _payload()
    payload["export_kind"] = "decomposed_onnx"
    payload["artifacts"] = [
        {"path": "vlm_prefix.onnx", "role": "model"},
        {"path": "expert_denoise.onnx", "role": "model"},
        {"path": "expert_stack.onnx", "role": "model"},
    ]

    with pytest.raises(UnsupportedExportPipelineError, match="unsupported decomposed"):
        decomposed_layout(payload)


def test_declared_expert_onnx_reader_executes_with_exact_inputs(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from tether._onnx_backend import load_onnx_backend

    root = tmp_path / "expert-reader"
    root.mkdir()
    noisy = onnx.helper.make_tensor_value_info(
        "noisy_actions", onnx.TensorProto.FLOAT, [1, 2, 3]
    )
    timestep = onnx.helper.make_tensor_value_info(
        "timestep", onnx.TensorProto.FLOAT, [1]
    )
    position_ids = onnx.helper.make_tensor_value_info(
        "position_ids", onnx.TensorProto.INT64, [1, 2]
    )
    velocity = onnx.helper.make_tensor_value_info(
        "velocity", onnx.TensorProto.FLOAT, [1, 2, 3]
    )
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Sub", ["noisy_actions", "noisy_actions"], ["velocity"])],
        "expert-reader",
        [noisy, timestep, position_ids],
        [velocity],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, root / "expert_stack.onnx")
    payload = build_producer_config(
        root,
        producer="expert_stack",
        model_id="org/pi0",
        model_type="pi0",
        action_dim=3,
        num_denoising_steps=2,
        opset=17,
        metadata={"action_chunk_size": 2},
    )
    write_tether_config(root, payload)
    backend = load_onnx_backend(root)

    real_session = backend.session

    class SessionSpy:
        def __init__(self):
            self.feeds = []

        def run(self, outputs, feed):
            self.feeds.append(dict(feed))
            return real_session.run(outputs, feed)

        def get_inputs(self):
            return real_session.get_inputs()

    spy = SessionSpy()
    backend.session = spy
    initial = np.ones((2, 3), dtype=np.float32)
    result = backend.forward(
        np.zeros((1, 1, 3), dtype=np.float32),
        "pick",
        np.zeros(3, dtype=np.float32),
        initial,
    )

    np.testing.assert_array_equal(result, initial)
    assert len(spy.feeds) == 2
    assert all(
        set(feed) == {"noisy_actions", "timestep", "position_ids"}
        for feed in spy.feeds
    )


def test_declared_pytorch_expert_reader_executes_euler_inputs():
    import torch

    from tether._pytorch_backend import PyTorchBackend

    class ZeroVelocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, actions, timestep, position_ids, vlm_kv=None):
            self.calls.append((actions.shape, timestep.shape, position_ids.shape, vlm_kv))
            return torch.zeros_like(actions)

    model = ZeroVelocity()
    backend = PyTorchBackend(
        "pi0",
        model,
        {
            "action_dim": 3,
            "action_chunk_size": 2,
            "num_denoising_steps": 2,
        },
        "cpu",
    )
    initial = np.ones((2, 3), dtype=np.float32)

    result = backend.forward(
        np.zeros((1, 1, 3), dtype=np.float32),
        "pick",
        np.zeros(3, dtype=np.float32),
        initial,
    )

    np.testing.assert_array_equal(result, initial)
    assert model.calls == [
        (torch.Size([1, 2, 3]), torch.Size([1]), torch.Size([1, 2]), None),
        (torch.Size([1, 2, 3]), torch.Size([1]), torch.Size([1, 2]), None),
    ]


@pytest.mark.parametrize(
    ("producer", "model_type", "expected_class"),
    [
        ("monolithic", "gr00t", "TetherServer"),
        ("pi05_split", "pi05_decomposed", "Pi05DecomposedServer"),
        ("expert_stack", "smolvla", "TetherServer"),
    ],
)
def test_benchmark_dispatch_uses_declared_decomposed_reader(
    tmp_path, producer, model_type, expected_class
):
    from tether.cli import _build_benchmark_server

    root = _materialize_producer(tmp_path, producer, model_type=model_type)
    server = _build_benchmark_server(root, load_tether_config(root), "cpu")

    assert type(server).__name__ == expected_class


def test_benchmark_dispatches_smolvla_full_bundle(tmp_path):
    from tether.cli import _build_benchmark_server

    root = _materialize_smolvla_full_bundle(tmp_path)
    server = _build_benchmark_server(root, load_tether_config(root), "cpu")

    assert type(server).__name__ == "TetherServer"


def test_create_app_dispatches_produced_smolvla_bundle_in_onnx_and_native_modes(
    tmp_path, monkeypatch
):
    from tether.runtime.server import TetherServer, create_app
    from tether.runtime.smolvla_native import SmolVLANativeServer

    root = _materialize_smolvla_full_bundle(tmp_path)
    monkeypatch.delenv("TETHER_NATIVE", raising=False)
    onnx_app = create_app(str(root), device="cpu")
    assert isinstance(onnx_app.state.tether_server, TetherServer)

    monkeypatch.setenv("TETHER_NATIVE", "1")
    native_app = create_app(str(root), device="cpu")
    assert isinstance(native_app.state.tether_server, SmolVLANativeServer)


@pytest.mark.parametrize(
    ("producer", "model_type", "expected_class"),
    [
        ("pi05_split", "pi05_decomposed", "Pi05DecomposedServer"),
        ("expert_stack", "smolvla", "TetherServer"),
    ],
)
def test_replay_dispatch_uses_declared_decomposed_reader(
    tmp_path, producer, model_type, expected_class, monkeypatch
):
    from tether.replay.cli import _load_target_server
    from tether.runtime import decomposed_server, server

    class FakeServer:
        def __init__(self, _root):
            self.ready = False

        def load(self):
            self.ready = True

    FakeServer.__name__ = expected_class
    monkeypatch.setattr(decomposed_server, "Pi05DecomposedServer", FakeServer)
    monkeypatch.setattr(server, "TetherServer", FakeServer)
    root = _materialize_producer(tmp_path, producer, model_type=model_type)

    loaded = _load_target_server(str(root))

    assert type(loaded).__name__ == expected_class
    assert loaded.ready


def test_replay_dispatches_smolvla_full_bundle(tmp_path, monkeypatch):
    from tether.replay.cli import _load_target_server
    from tether.runtime import server

    class FakeServer:
        def __init__(self, _root):
            self.ready = False

        def load(self):
            self.ready = True

    monkeypatch.setattr(server, "TetherServer", FakeServer)
    root = _materialize_smolvla_full_bundle(tmp_path)

    loaded = _load_target_server(str(root))

    assert isinstance(loaded, FakeServer)
    assert loaded.ready


def test_custom_pipeline_requires_reader_opt_in():
    payload = _payload()
    payload["pipeline"] = "prefix_optimum + expert_custom"
    with pytest.raises(UnsupportedExportPipelineError, match="does not support pipeline"):
        require_supported_pipeline(payload, set(), "test reader")
    assert (
        require_supported_pipeline(payload, {"prefix_optimum + expert_custom"}, "test reader")
        == "prefix_optimum + expert_custom"
    )
