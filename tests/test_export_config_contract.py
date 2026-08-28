from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tether.export_config import (
    ExportConfigError,
    UnsupportedExportKindError,
    load_tether_config,
    normalize_legacy_tether_config,
    require_supported_export_kind,
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
