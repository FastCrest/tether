"""Exported ScatterND must not carry reduction="none": TensorRT 10.3-10.7 reject the attribute."""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper  # noqa: E402

from tether.exporters.monolithic import _repair_exported_onnx  # noqa: E402
from tether.runtime.ort_providers import onnx_has_scatternd_reduction  # noqa: E402


def scatter(name, data, out, reduction):
    kwargs = {} if reduction is None else {"reduction": reduction}
    return helper.make_node("ScatterND", [data, "idx", "upd"], [out], name=name, **kwargs)


def model(reductions, *, nested=False):
    nodes, prev = [], "data"
    for i, reduction in enumerate(reductions):
        nodes.append(scatter(f"s{i}", prev, f"o{i}", reduction))
        prev = f"o{i}"
    outputs = [helper.make_tensor_value_info(prev, TensorProto.FLOAT, [4, 3])]
    if nested:
        branch = lambda tag: helper.make_graph(  # noqa: E731
            [scatter(f"{tag}_s", prev, f"{tag}_out", "none")], tag, [],
            [helper.make_tensor_value_info(f"{tag}_out", TensorProto.FLOAT, [4, 3])])
        nodes.append(helper.make_node("If", ["cond"], ["final"], then_branch=branch("then"), else_branch=branch("else")))
        outputs = [helper.make_tensor_value_info("final", TensorProto.FLOAT, [4, 3])]
    inputs = [helper.make_tensor_value_info("data", TensorProto.FLOAT, [4, 3]),
              helper.make_tensor_value_info("idx", TensorProto.INT64, [2, 1]),
              helper.make_tensor_value_info("upd", TensorProto.FLOAT, [2, 3])]
    if nested:
        inputs.append(helper.make_tensor_value_info("cond", TensorProto.BOOL, []))
    m = helper.make_model(helper.make_graph(nodes, "g", inputs, outputs), opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 8
    return m


def attrs(m):
    found = []
    for node in m.graph.node:
        if node.op_type == "ScatterND":
            found.append({a.name: a.s for a in node.attribute})
        for a in node.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                found += [{x.name: x.s for x in n.attribute} for n in a.g.node]
    return found


def run(path, nested=False):
    feeds = {"data": np.arange(12, dtype=np.float32).reshape(4, 3), "idx": np.array([[2], [0]], dtype=np.int64),
             "upd": np.full((2, 3), 7.5, dtype=np.float32)}
    if nested:
        feeds["cond"] = np.array(True)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feeds)[0]


def test_default_reduction_is_stripped_with_identical_outputs(tmp_path):
    path = tmp_path / "m.onnx"
    onnx.save(model(["none", "none", None]), path)
    before = run(path)
    assert onnx_has_scatternd_reduction(path)
    assert _repair_exported_onnx(path) == 2
    after_model = onnx.load(path)
    onnx.checker.check_model(after_model)
    assert attrs(after_model) == [{}, {}, {}]
    assert not onnx_has_scatternd_reduction(path)
    np.testing.assert_array_equal(run(path), before)


def test_real_reductions_are_kept_and_still_keep_tensorrt_off(tmp_path):
    path = tmp_path / "m.onnx"
    onnx.save(model(["add", "none", "mul"]), path)
    before = run(path)
    assert _repair_exported_onnx(path) == 1
    assert attrs(onnx.load(path)) == [{"reduction": b"add"}, {}, {"reduction": b"mul"}]
    assert onnx_has_scatternd_reduction(path)
    np.testing.assert_array_equal(run(path), before)


def test_subgraph_scatternd_is_cleaned(tmp_path):
    path = tmp_path / "m.onnx"
    onnx.save(model(["none"], nested=True), path)
    before = run(path, nested=True)
    assert _repair_exported_onnx(path) == 3
    assert all(a == {} for a in attrs(onnx.load(path)))
    np.testing.assert_array_equal(run(path, nested=True), before)


def test_clean_graph_is_not_rewritten(tmp_path):
    path = tmp_path / "m.onnx"
    onnx.save(model([None]), path)
    stamp = path.stat().st_mtime_ns
    assert _repair_exported_onnx(path) == 0
    assert path.stat().st_mtime_ns == stamp
