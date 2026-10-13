"""weight_fusion must be algebraically exact: ONNX Runtime output before and
after fuse_weights agrees to float32 rounding.

Regression: a monolithic export unrolls the denoise loop, so one projection
weight is read by every step. Fusing each step's RMSNorm scale into that shared
weight compounded it (scale**num_steps), which moved a tiny SmolVLA export to
max_abs 1.9e-2 against PyTorch.
"""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from tether.exporters.weight_fusion import fuse_weights  # noqa: E402


def _graph(path, nodes, inits, out_shape=(2, 3)):
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
        [numpy_helper.from_array(v.astype(np.float32), name=k) for k, v in inits.items()],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), path)


def _run(path, x):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {"x": x})[0]


def _fuse_and_compare(path):
    x = np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32)
    before = _run(path, x)
    n_before = len(onnx.load(str(path)).graph.node)
    fuse_weights(path, num_steps=10)
    after = _run(path, x)
    np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6)
    return n_before - len(onnx.load(str(path)).graph.node)


def test_shared_weight_read_by_two_steps_is_not_compounded(tmp_path):
    rng = np.random.default_rng(1)
    inits = {"s": rng.uniform(0.1, 0.5, 4), "W": rng.standard_normal((4, 4))}
    nodes = [
        helper.make_node("Mul", ["x", "s"], ["a"]),
        helper.make_node("MatMul", ["a", "W"], ["h"]),
        helper.make_node("Mul", ["h", "s"], ["b"]),
        helper.make_node("MatMul", ["b", "W"], ["y"]),
    ]
    _graph(tmp_path / "m.onnx", nodes, inits, out_shape=(2, 4))
    assert _fuse_and_compare(tmp_path / "m.onnx") == 0


def test_single_reader_weight_fuses_exactly(tmp_path):
    rng = np.random.default_rng(2)
    inits = {"s": rng.uniform(0.1, 0.5, 4), "W": rng.standard_normal((4, 3))}
    nodes = [
        helper.make_node("Mul", ["x", "s"], ["a"]),
        helper.make_node("MatMul", ["a", "W"], ["y"]),
    ]
    _graph(tmp_path / "m.onnx", nodes, inits)
    assert _fuse_and_compare(tmp_path / "m.onnx") == 1


def test_gemm_trans_b_scales_the_input_axis(tmp_path):
    # Square weight: the old shape test picked rows, but transB=1 stores W^T.
    rng = np.random.default_rng(3)
    inits = {"s": rng.uniform(0.1, 0.5, 4), "W": rng.standard_normal((4, 4))}
    nodes = [
        helper.make_node("Mul", ["x", "s"], ["a"]),
        helper.make_node("Gemm", ["a", "W"], ["y"], transB=1),
    ]
    _graph(tmp_path / "m.onnx", nodes, inits, out_shape=(2, 4))
    assert _fuse_and_compare(tmp_path / "m.onnx") == 1


def test_dt_fold_skips_weight_with_other_readers(tmp_path):
    rng = np.random.default_rng(5)
    inits = {"W": rng.standard_normal((4, 3)), "dt": np.array(-0.1)}
    nodes = [
        helper.make_node("Mul", ["W", "dt"], ["Wdt"]),
        helper.make_node("MatMul", ["x", "Wdt"], ["p"]),
        helper.make_node("MatMul", ["x", "W"], ["q"]),
        helper.make_node("Add", ["p", "q"], ["y"]),
    ]
    _graph(tmp_path / "m.onnx", nodes, inits)
    _fuse_and_compare(tmp_path / "m.onnx")
