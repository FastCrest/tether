"""Small canonical export-config factory shared by runtime unit tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_test_onnx(path: Path, *, action_dim: int = 7) -> Path:
    """Write a tiny valid ONNX identity graph for artifact-inspection tests."""
    import onnx

    source = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, ["batch", action_dim]
    )
    target = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, ["batch", action_dim]
    )
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["output"])],
        path.stem,
        [source],
        [target],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 19)],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path


def write_test_export_config(
    root: Path,
    *,
    model_type: str = "smolvla",
    export_kind: str = "monolithic_onnx",
    action_dim: int = 32,
    num_denoising_steps: int = 10,
    artifacts: list[str] | None = None,
    **extras: Any,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if artifacts is None:
        artifacts = ["model.onnx"]
    for relative in artifacts:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"test artifact")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "model_id": f"test/{model_type}",
        "model_type": model_type,
        "action_dim": action_dim,
        "num_denoising_steps": num_denoising_steps,
        "opset": 19,
        "export_kind": export_kind,
        "artifacts": [{"path": path, "role": "model"} for path in artifacts],
        "io_contract": {
            "inputs": [
                {"name": "input", "dtype": "float32", "shape": [1, 50, action_dim]}
            ],
            "outputs": [
                {"name": "actions", "dtype": "float32", "shape": [1, 50, action_dim]}
            ],
        },
    }
    payload.update(extras)
    target = root / "tether_config.json"
    target.write_text(json.dumps(payload))
    return target
