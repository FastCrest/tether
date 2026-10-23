"""Actual ONNX conversion, parity, and transactional serving-export checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tether.export_config import build_producer_config, load_tether_config, write_tether_config
from tether.exporters.fp16_convert import (
    FP16ParityError,
    _output_parity,
    _size_with_external,
    convert_fp32_to_fp16,
    convert_monolithic_export_fp16,
    parity_gate,
)


def _export(root: Path, *, external=True, scale=0.123456789):
    root.mkdir()
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])
    y = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 64])
    # Distinct rows, signs and non-half-exact values exercise actual rounding.
    w = (np.arange(4096, dtype=np.float32).reshape(64, 64) % 17 - 8) * scale
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["actions"], name="project")],
        "fp16-test",
        [x],
        [y],
        [numpy_helper.from_array(w, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9
    onnx.save_model(
        model,
        str(root / "model.onnx"),
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location="arbitrary-weights.bin",
        size_threshold=0,
    )
    (root / "tokenizer").mkdir()
    (root / "tokenizer" / "tokenizer.json").write_text('{"fixture":true}')
    cfg = build_producer_config(
        root,
        producer="monolithic",
        model_id="fixture/fp32",
        model_type="smolvla",
        action_dim=64,
        num_denoising_steps=10,
        opset=19,
        optional_artifacts=[("tokenizer/tokenizer.json", "tokenizer")],
        metadata={
            "tokenizer_path": "tokenizer",
            "tokenizer_bundled": True,
            "chunk_size": 1,
            "max_state_dim": 64,
            "old_parity": {"passed": True},
            "precision": "fp32",
        },
    )
    write_tether_config(root, cfg)
    return root


def _cases():
    rng = np.random.default_rng(917)
    return [{"x": rng.normal(size=(1, 64)).astype(np.float32)} for _ in range(3)]


def _snapshot(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def _convert(src, dst, **kwargs):
    return convert_monolithic_export_fp16(
        src,
        dst,
        validation_cases=kwargs.pop("validation_cases", _cases()),
        max_abs_threshold=kwargs.pop("max_abs_threshold", 0.01),
        cos_threshold=kwargs.pop("cos_threshold", 0.999),
        **kwargs,
    )


def test_real_conversion_preserves_source_io_tokenizer_and_fresh_hashes(tmp_path):
    src = _export(tmp_path / "fp32")
    before = _snapshot(src)
    result = _convert(src, tmp_path / "fp16")
    assert result["parity_verdict"] == "PASS"
    assert result["model_bytes_after"] < result["model_bytes_before"] * 0.6
    assert len(result["cases"]) == 3
    assert result["cases"][0]["outputs"]["actions"]["max_abs_diff"] > 0
    assert _snapshot(src) == before
    converted = load_tether_config(tmp_path / "fp16")
    assert converted["io_contract"] == load_tether_config(src)["io_contract"]
    assert converted["precision"] == "mixed_float16"
    assert "old_parity" not in converted
    assert (tmp_path / "fp16/tokenizer/tokenizer.json").read_bytes() == (
        src / "tokenizer/tokenizer.json"
    ).read_bytes()
    assert all("sha256" in a for a in converted["artifacts"])
    receipt = json.loads((tmp_path / "fp16/precision-conversion.json").read_text())
    assert receipt == result
    assert "target memory fit not measured" in receipt["limits"]
    # Source's arbitrary external location is counted, not unrelated siblings.
    assert "arbitrary-weights.bin" in result["source_artifacts"]
    assert not list(tmp_path.glob(".tether-export-fp16-*"))


def test_failed_measured_parity_exposes_receipt_without_publishing(tmp_path):
    src = _export(tmp_path / "fp32")
    before = _snapshot(src)
    with pytest.raises(FP16ParityError) as caught:
        _convert(src, tmp_path / "failed", max_abs_threshold=0)
    assert caught.value.receipt["parity_verdict"] == "FAIL"
    assert not (tmp_path / "failed").exists()
    assert _snapshot(src) == before
    assert not list(tmp_path.glob(".tether-export-fp16-*"))


@pytest.mark.parametrize("destination", ["fp32", "fp32/nested", "occupied"])
def test_refuses_existing_or_nested_destination_without_changes(tmp_path, destination):
    src = _export(tmp_path / "fp32")
    if destination == "occupied":
        (tmp_path / "occupied").mkdir()
        (tmp_path / "occupied/sentinel").write_bytes(b"keep")
    before = _snapshot(tmp_path)
    with pytest.raises((ValueError, FileExistsError)):
        _convert(src, tmp_path / destination)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "cases",
    [
        [],
        [{"wrong": np.ones((1, 64), np.float32)}],
        [{"x": np.ones((1, 64), np.float64)}],
        [{"x": np.full((1, 64), np.nan, np.float32)}],
    ],
)
def test_refuses_absent_wrong_named_wrong_typed_nonfinite_inputs(tmp_path, cases):
    src = _export(tmp_path / "fp32")
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

    with pytest.raises((ValueError, InvalidArgument)):
        _convert(src, tmp_path / "fp16", validation_cases=cases)
    assert not (tmp_path / "fp16").exists()


@pytest.mark.parametrize(
    "maxabs,cos",
    [
        (float("inf"), 0.99),
        (float("nan"), 0.99),
        (-1, 0.99),
        (0.01, float("nan")),
        (0.01, 2),
        (True, 0.99),
    ],
)
def test_refuses_invalid_thresholds_before_conversion(tmp_path, maxabs, cos):
    with pytest.raises(ValueError):
        _convert(
            tmp_path / "missing", tmp_path / "fp16", max_abs_threshold=maxabs, cos_threshold=cos
        )
    assert not (tmp_path / "fp16").exists()


def test_low_level_keeps_io_does_not_run_heuristic_dtype_repair(tmp_path, monkeypatch):
    src = _export(tmp_path / "fp32")

    def forbidden(*args, **kwargs):
        pytest.fail("heuristic casts must not run")

    monkeypatch.setattr("tether.exporters.fp16_convert.fix_fp16_dtype_mismatches", forbidden)
    result = convert_fp32_to_fp16(src / "model.onnx", tmp_path / "out/model.onnx")
    assert result["keep_io_types"] is True
    assert result["numerical_parity"] == "not_measured"
    assert result["cast_nodes_inserted"] == 0
    assert (tmp_path / "out/model.bin").exists()


def test_external_size_counts_exact_references_not_prefixes(tmp_path):
    src = _export(tmp_path / "fp32")
    (src / "model-unrelated.bin").write_bytes(b"x" * 1024)
    assert (
        _size_with_external(src / "model.onnx")
        == (src / "model.onnx").stat().st_size + (src / "arbitrary-weights.bin").stat().st_size
    )


def test_loadability_error_does_not_publish_or_touch_other_files(tmp_path, monkeypatch):
    src = _export(tmp_path / "fp32")
    dst = tmp_path / "out"
    dst.mkdir()
    (dst / "other-model.bin").write_bytes(b"keep")

    def fail(*args):
        raise RuntimeError("session-init failure")

    monkeypatch.setattr("tether.exporters.fp16_convert._cpu_session", fail)
    with pytest.raises(RuntimeError, match="session-init"):
        convert_fp32_to_fp16(src / "model.onnx", dst / "model.onnx")
    assert _snapshot(dst) == {"other-model.bin": hashlib.sha256(b"keep").hexdigest()}


def test_refuses_external_weights_outside_model_directory(tmp_path):
    src = _export(tmp_path / "fp32")
    model = onnx.load(src / "model.onnx", load_external_data=False)
    location = next(v for v in model.graph.initializer[0].external_data if v.key == "location")
    location.value = "../outside.bin"
    onnx.save_model(model, src / "model.onnx")
    with pytest.raises(ValueError, match="outside"):
        _convert(src, tmp_path / "fp16")
    assert not (tmp_path / "fp16").exists()


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -1])
def test_parity_gate_cannot_pass_invalid_measurement(value):
    assert parity_gate(value, 1)["passed"] is False


def test_integer_outputs_compare_exactly_and_zero_cosine_is_defined():
    assert _output_parity(np.array([1]), np.array([2]), max_abs=1, cosine=0)["passed"] is False
    assert _output_parity(np.zeros(3), np.zeros(3), max_abs=0, cosine=1)["passed"] is True
    assert _output_parity(np.zeros(3), np.ones(3), max_abs=10, cosine=0.99)["passed"] is False


def test_blocklisted_nonlinear_boundary_is_actually_loadable_and_measured(tmp_path):
    src = _export(tmp_path / "fp32")
    model = onnx.load(src / "model.onnx")
    model.graph.node[0].output[0] = "projected"
    model.graph.node.extend(
        [
            helper.make_node("Softmax", ["projected"], ["actions"], name="softmax", axis=-1),
        ]
    )
    onnx.save_model(model, src / "model.onnx")
    # Rebuild the manifest because this test deliberately changed storage.
    cfg = load_tether_config(src, inspect_artifacts=False)
    cfg["artifacts"] = [
        {"path": "model.onnx", "role": "model"},
        {"path": "tokenizer/tokenizer.json", "role": "tokenizer"},
    ]
    write_tether_config(src, cfg)
    receipt = _convert(src, tmp_path / "fp16")
    assert receipt["parity_verdict"] == "PASS"
    converted = onnx.load(tmp_path / "fp16/model.onnx")
    softmax = next(n for n in converted.graph.node if n.op_type == "Softmax")
    assert any(
        n.op_type == "Cast"
        and n.output[0] == softmax.input[0]
        and next(a for a in n.attribute if a.name == "to").i == TensorProto.FLOAT
        for n in converted.graph.node
    )


def test_commit_failure_rolls_back_only_own_directory(tmp_path, monkeypatch):
    src = _export(tmp_path / "fp32")
    before = _snapshot(src)
    import os

    real_replace = os.replace

    def fail_commit(source, target):
        if Path(target) == tmp_path / "fp16":
            raise OSError("simulated commit failure")
        return real_replace(source, target)

    monkeypatch.setattr("tether.exporters.fp16_convert.os.replace", fail_commit)
    with pytest.raises(OSError, match="commit failure"):
        _convert(src, tmp_path / "fp16")
    assert not (tmp_path / "fp16").exists()
    assert _snapshot(src) == before


def test_late_destination_conflict_preserves_existing_contents(tmp_path, monkeypatch):
    src = _export(tmp_path / "fp32")
    from tether.exporters import fp16_convert as module

    original = module._output_parity

    def introduce_conflict(*args, **kwargs):
        dst = tmp_path / "fp16"
        dst.mkdir(exist_ok=True)
        (dst / "sentinel").write_bytes(b"preserve concurrent destination")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_output_parity", introduce_conflict)
    with pytest.raises(FileExistsError):
        _convert(src, tmp_path / "fp16")
    assert (tmp_path / "fp16/sentinel").read_bytes() == b"preserve concurrent destination"


def test_refuses_array_like_inputs_instead_of_implicitly_casting(tmp_path):
    src = _export(tmp_path / "fp32")
    with pytest.raises(TypeError, match="NumPy array"):
        _convert(src, tmp_path / "fp16", validation_cases=[{"x": [[1] * 64]}])


def test_source_changes_during_measurement_fail_without_publication(tmp_path, monkeypatch):
    src = _export(tmp_path / "fp32")
    from tether.exporters import fp16_convert as module

    original = module._output_parity

    def mutate_source(*args, **kwargs):
        (src / "tokenizer/tokenizer.json").write_text("changed")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_output_parity", mutate_source)
    with pytest.raises(ValueError, match="source changed"):
        _convert(src, tmp_path / "fp16")
    assert not (tmp_path / "fp16").exists()


@pytest.mark.parametrize("tokenizer_path,bundled", [("../outside", True), ("missing", True)])
def test_refuses_unshippable_tokenizer_contract(tmp_path, tokenizer_path, bundled):
    src = _export(tmp_path / "fp32")
    cfg = load_tether_config(src)
    cfg["tokenizer_path"], cfg["tokenizer_bundled"] = tokenizer_path, bundled
    write_tether_config(src, cfg)
    with pytest.raises(ValueError, match="[Tt]okenizer"):
        _convert(src, tmp_path / "fp16")
    assert not (tmp_path / "fp16").exists()


def test_receipt_handles_gpu_only_ort_distribution(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    from tether.exporters import fp16_convert as module

    def gpu_only(name):
        if name == "onnxruntime":
            raise PackageNotFoundError(name)
        return "1.30.0" if name == "onnxruntime-gpu" else "fixture"

    monkeypatch.setattr(module, "version", gpu_only)
    assert module._conversion_versions() == {
        "onnx": "fixture",
        "numpy": "fixture",
        "onnxruntime-gpu": "1.30.0",
    }


def test_shared_blocklisted_input_reuses_identical_converter_cast(tmp_path):
    src = _export(tmp_path / "fp32")
    model = onnx.load(src / "model.onnx")
    model.graph.node[0].output[0] = "shared"
    model.graph.node.extend(
        [
            helper.make_node("Softmax", ["shared"], ["a"], name="softmax_a", axis=-1),
            helper.make_node("Softmax", ["shared"], ["b"], name="softmax_b", axis=-1),
            helper.make_node("Add", ["a", "b"], ["actions"], name="combine"),
        ]
    )
    onnx.save_model(model, src / "model.onnx")
    cfg = load_tether_config(src, inspect_artifacts=False)
    cfg["artifacts"] = [
        {"path": "model.onnx", "role": "model"},
        {"path": "tokenizer/tokenizer.json", "role": "tokenizer"},
    ]
    write_tether_config(src, cfg)
    receipt = _convert(src, tmp_path / "fp16")
    assert receipt["parity_verdict"] == "PASS"
    assert receipt["duplicate_casts_removed"] >= 1
    converted = onnx.load(tmp_path / "fp16/model.onnx", load_external_data=False)
    outputs = [name for node in converted.graph.node for name in node.output]
    assert len(outputs) == len(set(outputs))
    onnx.checker.check_model(str(tmp_path / "fp16/model.onnx"))


@pytest.mark.parametrize("conflict", ["target", "input", "op", "declaration"])
def test_structural_cast_dedup_refuses_conflicting_semantics(conflict):
    from tether.exporters.fp16_convert import _deduplicate_conversion_casts

    a = helper.make_node("Cast", ["x"], ["shared"], to=TensorProto.FLOAT, name="a")
    b = helper.make_node("Cast", ["x"], ["shared"], to=TensorProto.FLOAT, name="b")
    if conflict == "target":
        b.attribute[0].i = TensorProto.FLOAT16
    elif conflict == "input":
        b.input[0] = "other"
    elif conflict == "op":
        b.op_type = "Identity"
    graph = helper.make_graph([a, b], "conflicting", [], [])
    if conflict == "declaration":
        graph.value_info.extend(
            [
                helper.make_tensor_value_info("shared", TensorProto.FLOAT, [1]),
                helper.make_tensor_value_info("shared", TensorProto.FLOAT16, [1]),
            ]
        )
    model = helper.make_model(graph)
    with pytest.raises(ValueError, match="Conflicting"):
        _deduplicate_conversion_casts(model)


def test_blocked_rms_norm_preserves_large_fp32_intermediates(tmp_path):
    src = _export(tmp_path / "fp32")
    model = onnx.load(src / "model.onnx")
    model.graph.node[0].output[0] = "projected"
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array(2, np.float32), "two"),
            numpy_helper.from_array(np.array(1e-6, np.float32), "eps"),
            numpy_helper.from_array(np.array([-1], np.int64), "axes"),
        ]
    )
    model.graph.node.extend(
        [
            helper.make_node("Pow", ["projected", "two"], ["squared"], name="square"),
            helper.make_node("ReduceMean", ["squared", "axes"], ["mean"], name="mean", keepdims=1),
            helper.make_node("Add", ["mean", "eps"], ["variance"], name="epsilon"),
            helper.make_node("Sqrt", ["variance"], ["sqrt"], name="sqrt"),
            helper.make_node("Reciprocal", ["sqrt"], ["scale"], name="reciprocal"),
            helper.make_node("Mul", ["projected", "scale"], ["actions"], name="normalize"),
        ]
    )
    onnx.save_model(model, src / "model.onnx")
    cfg = load_tether_config(src, inspect_artifacts=False)
    cfg["artifacts"] = [
        {"path": "model.onnx", "role": "model"},
        {"path": "tokenizer/tokenizer.json", "role": "tokenizer"},
    ]
    write_tether_config(src, cfg)
    inputs = [{"x": np.full((1, 64), 2000, np.float32)}]
    receipt = _convert(src, tmp_path / "fp16", validation_cases=inputs)
    assert receipt["parity_verdict"] == "PASS"
    assert receipt["fp32_island_edges_preserved"] >= 4
    assert len(receipt["fp32_island_edges"]) == receipt["fp32_island_edges_preserved"]
    assert {"Pow", "ReduceMean", "Add", "Sqrt"} <= {
        e["producer_op"] for e in receipt["fp32_island_edges"]
    }


def test_blocked_rotary_phase_keeps_fractional_large_angle_precision(tmp_path):
    src = _export(tmp_path / "fp32")
    model = onnx.load(src / "model.onnx")
    model.graph.node[0].output[0] = "projected"
    model.graph.initializer.append(numpy_helper.from_array(np.array(1.13, np.float32), "period"))
    model.graph.node.extend(
        [
            helper.make_node("Div", ["projected", "period"], ["phase"], name="phase"),
            helper.make_node("Sin", ["phase"], ["actions"], name="sine"),
        ]
    )
    onnx.save_model(model, src / "model.onnx")
    cfg = load_tether_config(src, inspect_artifacts=False)
    cfg["artifacts"] = [
        {"path": "model.onnx", "role": "model"},
        {"path": "tokenizer/tokenizer.json", "role": "tokenizer"},
    ]
    write_tether_config(src, cfg)
    # Diagonal exactly representable weights isolate phase rounding from matmul.
    model = onnx.load(src / "model.onnx")
    model.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.eye(64, dtype=np.float32), "w"))
    onnx.save_model(model, src / "model.onnx")
    receipt = _convert(
        src, tmp_path / "fp16", validation_cases=[{"x": np.full((1, 64), 208, np.float32)}]
    )
    assert receipt["parity_verdict"] == "PASS"
    assert receipt["fp32_island_edges_preserved"] >= 1


def _roundtrip_graph():
    nodes = [
        helper.make_node("Add", ["x", "x"], ["p"], name="source"),
        helper.make_node("Cast", ["p"], ["half"], to=TensorProto.FLOAT16, name="generated_half"),
        helper.make_node("Cast", ["half"], ["full"], to=TensorProto.FLOAT, name="generated_full"),
        helper.make_node("Identity", ["half"], ["parallel"], name="parallel"),
    ]
    graph = helper.make_graph(
        nodes,
        "roundtrip",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [
            helper.make_tensor_value_info("full", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("parallel", TensorProto.FLOAT16, [1]),
        ],
        value_info=[
            helper.make_tensor_value_info("p", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("half", TensorProto.FLOAT16, [1]),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])


def test_fp32_island_preserves_parallel_fp16_consumer():
    from tether.exporters.fp16_convert import _preserve_fp32_island_edges

    model = _roundtrip_graph()
    assert _preserve_fp32_island_edges(model, {"source", "parallel"}, {"Add"}) == 1
    assert model.graph.node[2].input[0] == "p"
    assert model.graph.node[3].input[0] == "half"
    onnx.checker.check_model(model)


@pytest.mark.parametrize("original_cast", ["generated_half", "generated_full"])
def test_fp32_island_never_bypasses_original_semantic_rounding(original_cast):
    from tether.exporters.fp16_convert import _preserve_fp32_island_edges

    model = _roundtrip_graph()
    assert _preserve_fp32_island_edges(model, {"source", original_cast}, {"Add"}) == 0
    assert model.graph.node[2].input[0] == "half"


def test_fp32_island_requires_complete_declared_types():
    from tether.exporters.fp16_convert import _preserve_fp32_island_edges

    model = _roundtrip_graph()
    del model.graph.value_info[:]
    assert _preserve_fp32_island_edges(model, {"source"}, {"Add"}) == 0
    assert model.graph.node[2].input[0] == "half"


def test_receipt_and_both_sessions_use_snapshot_before_caller_mutation(tmp_path, monkeypatch):
    from tether.exporters import fp16_convert as module

    src = _export(tmp_path / "fp32")
    caller_array = np.ones((1, 64), np.float32)
    identity = module._case_identity
    session_factory = module._cpu_session
    measured_inputs = []

    def identity_then_mutate_caller(case):
        result = identity(case)
        caller_array.fill(999)
        return result

    class ObservedSession:
        def __init__(self, session):
            self.session = session

        def get_inputs(self):
            return self.session.get_inputs()

        def get_outputs(self):
            return self.session.get_outputs()

        def run(self, outputs, case):
            measured_inputs.append(case["x"].copy())
            return self.session.run(outputs, case)

    monkeypatch.setattr(module, "_case_identity", identity_then_mutate_caller)
    monkeypatch.setattr(module, "_cpu_session", lambda path: ObservedSession(session_factory(path)))
    receipt = _convert(src, tmp_path / "fp16", validation_cases=[{"x": caller_array}])
    assert np.all(caller_array == 999)
    assert len(measured_inputs) == 2
    for measured in measured_inputs:
        assert np.all(measured == 1)
        assert (
            receipt["cases"][0]["inputs"]["x"]["sha256"]
            == hashlib.sha256(measured.tobytes()).hexdigest()
        )
