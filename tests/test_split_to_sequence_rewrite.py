"""Exported SplitToSequence+SequenceAt must become Split.

torch.onnx.export(optimize=False) lowers torch.split/chunk to a tensor sequence (370 times
in the pi0.5 expert). On ORT 1.25 + TensorRT EP the sequence crossed a CPU -> TensorRT
boundary and the TensorRT MemcpyFromHost kernel, which only copies tensors, failed the
first run with "Unsupported type proto value case:0".
"""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from tether.exporters.monolithic import _repair_exported_onnx  # noqa: E402
from tether.runtime.ort_providers import onnx_has_sequence_ops  # noqa: E402


def _const(name, value):
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(np.asarray(value)))


def _model(split, positions, *, axis=-1, length=2600, extra_consumer=False):
    nodes = [_const("split", np.asarray(split, np.int64)),
             helper.make_node("SplitToSequence", ["x", "split"], ["seq"], axis=axis, keepdims=1)]
    outputs = []
    for k, pos in enumerate(positions):
        nodes += [_const(f"p{k}", np.int64(pos)), helper.make_node("SequenceAt", ["seq", f"p{k}"], [f"y{k}"])]
        outputs.append(helper.make_tensor_value_info(f"y{k}", TensorProto.FLOAT, None))
    if extra_consumer:
        nodes.append(helper.make_node("SequenceLength", ["seq"], ["n"]))
        outputs.append(helper.make_tensor_value_info("n", TensorProto.INT64, []))
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, length])],
                              outputs)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)], ir_version=9)


def _run(path, x):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": x})


@pytest.mark.parametrize("split, positions", [
    (1024, [0, 1, 2]),          # pi0.5 expert form: scalar split, uneven last chunk
    (1024, [-1, 0]),            # negative position
    ([1000, 600, 1000], [2, 1, 0]),
])
def test_rewrite_matches_sequence_semantics_and_removes_sequences(tmp_path, split, positions):
    path = tmp_path / "m.onnx"
    onnx.save(_model(split, positions), path)
    x = np.random.default_rng(0).standard_normal((2, 3, 2600)).astype(np.float32)
    before = _run(path, x)
    assert onnx_has_sequence_ops(path)

    assert _repair_exported_onnx(path) == 1
    ops = {n.op_type for n in onnx.load(str(path)).graph.node}
    assert "SplitToSequence" not in ops and "SequenceAt" not in ops and "Split" in ops
    assert not onnx_has_sequence_ops(path)
    for a, b in zip(before, _run(path, x)):
        np.testing.assert_array_equal(a, b)


def test_rewrite_leaves_sequences_it_cannot_prove(tmp_path):
    path = tmp_path / "m.onnx"
    onnx.save(_model(1024, [0], extra_consumer=True), path)
    assert _repair_exported_onnx(path) == 0
    assert onnx_has_sequence_ops(path)


def test_provider_plan_refuses_trt_for_sequence_graphs(tmp_path, monkeypatch):
    """Old exports still carry sequences: TRT must be refused with the reason, not crash on run."""
    from tether.runtime import ort_providers

    monkeypatch.setattr(ort_providers, "gpu_is_blackwell", lambda: False)
    path = tmp_path / "m.onnx"
    onnx.save(_model(1024, [0, 1, 2]), path)
    available = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    plan = ort_providers.build_ort_provider_plan(tmp_path, available_providers=available, onnx_path=path)
    assert [ort_providers.provider_name(p) for p in plan.providers][0] == "CUDAExecutionProvider"
    assert "SplitToSequence" in plan.trt_disabled_reason

    _repair_exported_onnx(path)
    plan = ort_providers.build_ort_provider_plan(tmp_path, available_providers=available, onnx_path=path)
    assert ort_providers.provider_name(plan.providers[0]) == "TensorrtExecutionProvider"


@pytest.mark.parametrize("config, fp16", [
    ({}, True),                                    # FP32 export, unknown source: unchanged default
    ({"source_dtype": "float32"}, True),
    ({"source_dtype": "bfloat16"}, False),         # pi0.5: FP16 cos 0.079, BF16 0.988 on L4
    ({"precision": "mixed_float16"}, False),       # typed graph
])
def test_trt_fp16_default_respects_source_dtype(tmp_path, monkeypatch, config, fp16):
    import json

    from tether.runtime import ort_providers

    monkeypatch.setattr(ort_providers, "gpu_is_blackwell", lambda: False)
    (tmp_path / "tether_config.json").write_text(json.dumps(config))
    path = tmp_path / "m.onnx"
    onnx.save(_model(1024, [0]), path)
    _repair_exported_onnx(path)
    plan = ort_providers.build_ort_provider_plan(
        tmp_path, available_providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        onnx_path=path)
    assert plan.providers[0][0] == "TensorrtExecutionProvider"
    assert plan.providers[0][1]["trt_fp16_enable"] is fp16
