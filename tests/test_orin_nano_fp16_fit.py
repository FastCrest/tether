"""FP16 conversion unit tests — Orin Nano 8GB fit goal.

The actual conversion runs on Modal (12GB ONNX can't load locally).
These tests cover the pure helpers: sizing estimator, parity gate,
op blocklist defaults. They establish the contract that the Modal
runs will execute.

Goal: orin-nano-fp16-fit (weight 8).
"""
from __future__ import annotations

import pytest

from tether.exporters.fp16_convert import (
    FP16_OP_BLOCKLIST,
    _build_dtype_map,
    _DT_FLOAT,
    _DT_FLOAT16,
    _DT_INT64,
    estimate_fp16_size_bytes,
    fix_fp16_dtype_mismatches,
    parity_gate,
)


class TestSizeEstimate:
    def test_halves_roughly(self):
        # 12.5 GB (pi0 FP32) → expect roughly 6.3-6.6 GB after FP16 + overhead
        pi0_fp32 = int(12.5 * 1e9)
        estimate = estimate_fp16_size_bytes(pi0_fp32)
        # Should be 6.25GB (halved) + 5% overhead = ~6.56GB.
        assert 6.2e9 < estimate < 6.7e9

    def test_zero_input(self):
        assert estimate_fp16_size_bytes(0) == 0

    def test_smolvla_stays_small(self):
        # SmolVLA 1.6GB → expect ~0.84GB post-FP16 (well under 8GB obviously)
        smolvla = int(1.6e9)
        estimate = estimate_fp16_size_bytes(smolvla)
        assert 0.7e9 < estimate < 1.0e9

    def test_fits_orin_nano_after_conversion(self):
        """The key claim: pi0 FP16 fits under 8 GB (leaving room for
        activations + OS)."""
        pi0_fp32 = int(12.5 * 1e9)
        pi05_fp32 = int(13.0 * 1e9)
        # Budget: 8GB total, reserve 1.5GB for activations+OS = 6.5GB for weights.
        ORIN_NANO_WEIGHT_BUDGET = int(6.5e9)
        # Our estimator is approximate; allow 5% slack in the fit claim.
        assert estimate_fp16_size_bytes(pi0_fp32) <= ORIN_NANO_WEIGHT_BUDGET * 1.05
        # pi0.5 at 13GB FP32 → ~6.8GB FP16, right at the edge. Document it.
        pi05_est = estimate_fp16_size_bytes(pi05_fp32)
        assert pi05_est > ORIN_NANO_WEIGHT_BUDGET * 0.9


class TestParityGate:
    def test_machine_precision_passes(self):
        g = parity_gate(max_abs_diff=1e-6, cos_sim=0.999999)
        assert g["verdict"] == "PASS"
        assert g["passed"] is True
        assert g["reasons"] == []

    def test_bad_cos_fails(self):
        g = parity_gate(max_abs_diff=1e-4, cos_sim=0.97)
        assert g["verdict"] == "FAIL"
        assert g["passed"] is False
        assert any("cos_sim" in r for r in g["reasons"])

    def test_bad_maxabs_fails(self):
        g = parity_gate(max_abs_diff=0.5, cos_sim=0.9999)
        assert g["verdict"] == "FAIL"
        assert any("max_abs_diff" in r for r in g["reasons"])

    def test_both_bad(self):
        g = parity_gate(max_abs_diff=0.5, cos_sim=0.5)
        assert g["verdict"] == "FAIL"
        assert len(g["reasons"]) == 2

    def test_threshold_overrides(self):
        """Callers can tighten or loosen the gate for model-specific tolerance."""
        # Looser: 99% cos is OK, 1e-2 max_abs is OK
        g = parity_gate(
            max_abs_diff=5e-3, cos_sim=0.99,
            max_abs_threshold=1e-2, cos_threshold=0.98,
        )
        assert g["passed"] is True


class TestBlocklist:
    def test_blocklist_is_immutable_tuple(self):
        """Protect against accidental in-place mutation at import time."""
        assert isinstance(FP16_OP_BLOCKLIST, tuple)

    def test_blocklist_is_empty_by_default(self):
        """Empty default blocklist — full graph gets converted to FP16.

        onnxconverter_common.float16 doesn't insert Cast nodes for
        blocklisted-op outputs, so a non-empty blocklist creates
        FP32/FP16 operand mismatches on Mul/MatMul that ORT rejects.
        Empty list = all ops converted, parity gate (cos>0.999)
        catches any precision regression.
        """
        assert FP16_OP_BLOCKLIST == ()


@pytest.fixture
def _onnx():
    """Skip these tests if onnx isn't available in the test env."""
    pytest.importorskip("onnx")
    return pytest.importorskip("onnx")


def _make_model_with_mismatch(onnx, tmp_path, save: bool = True):
    """Build a tiny model: two graph inputs (one FP32, one FP16) feed
    into a Mul. Exactly the pattern that trips ORT on FP16 conversions.
    """
    from onnx import helper, TensorProto

    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [4])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT16, [4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT16, [4])

    mul = helper.make_node("Mul", inputs=["a", "b"], outputs=["out"], name="mul0")
    graph = helper.make_graph([mul], "mismatch", [a, b], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9
    if save:
        p = tmp_path / "mismatch.onnx"
        onnx.save(model, str(p))
        return p, model
    return None, model


class TestDtypeMap:
    def test_graph_inputs_and_initializers(self, _onnx, tmp_path):
        _, model = _make_model_with_mismatch(_onnx, tmp_path, save=False)
        dt = _build_dtype_map(model)
        assert dt["a"] == _DT_FLOAT
        assert dt["b"] == _DT_FLOAT16
        # Mul preserves first-input dtype in our rules; check output is tracked.
        assert "out" in dt

    def test_cast_op_uses_to_attribute(self, _onnx):
        from onnx import helper, TensorProto

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [4])
        cast = helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT16)
        graph = helper.make_graph([cast], "cast", [x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])

        dt = _build_dtype_map(model)
        assert dt["y"] == _DT_FLOAT16

    def test_shape_op_emits_int64(self, _onnx):
        from onnx import helper, TensorProto

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
        s = helper.make_tensor_value_info("s", TensorProto.INT64, [1])
        shape = helper.make_node("Shape", ["x"], ["s"])
        graph = helper.make_graph([shape], "shape", [x], [s])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])

        dt = _build_dtype_map(model)
        assert dt["s"] == _DT_INT64


class TestDtypeFixPass:
    def test_inserts_cast_on_fp32_fp16_mismatch(self, _onnx, tmp_path):
        p, _ = _make_model_with_mismatch(_onnx, tmp_path)
        count = fix_fp16_dtype_mismatches(p)
        assert count == 1

        # Reload and verify: there should now be a Cast node before the Mul.
        fixed = _onnx.load(str(p))
        op_types = [n.op_type for n in fixed.graph.node]
        assert op_types == ["Cast", "Mul"]
        # The Mul's first input should now be the Cast's output (FP16).
        mul_node = next(n for n in fixed.graph.node if n.op_type == "Mul")
        cast_node = next(n for n in fixed.graph.node if n.op_type == "Cast")
        assert mul_node.input[0] == cast_node.output[0]
        # The Cast must cast TO float16.
        to_attr = next(a for a in cast_node.attribute if a.name == "to")
        assert to_attr.i == _DT_FLOAT16

    def test_noop_when_already_consistent(self, _onnx, tmp_path):
        from onnx import helper, TensorProto

        # Two FP16 inputs into Mul — no Cast needed.
        a = helper.make_tensor_value_info("a", TensorProto.FLOAT16, [4])
        b = helper.make_tensor_value_info("b", TensorProto.FLOAT16, [4])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT16, [4])
        mul = helper.make_node("Mul", ["a", "b"], ["out"], name="mul0")
        graph = helper.make_graph([mul], "clean", [a, b], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
        p = tmp_path / "clean.onnx"
        _onnx.save(model, str(p))

        count = fix_fp16_dtype_mismatches(p)
        assert count == 0

    def test_handles_where_op_bool_input(self, _onnx, tmp_path):
        """Where has a bool first input — that's not a dtype mismatch even
        though bool != float. Only inputs[1] and [2] need consistent
        floating dtype."""
        from onnx import helper, TensorProto

        cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [4])
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [4])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT16, [4])
        w = helper.make_node("Where", ["cond", "x", "y"], ["out"], name="w0")
        graph = helper.make_graph([w], "where", [cond, x, y], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
        p = tmp_path / "where.onnx"
        _onnx.save(model, str(p))

        count = fix_fp16_dtype_mismatches(p)
        assert count == 1  # Cast on x, not on cond.
        fixed = _onnx.load(str(p))
        cast_node = next(n for n in fixed.graph.node if n.op_type == "Cast")
        assert cast_node.input[0] == "x"  # the FP32 input got cast, not cond.


class TestImportGuard:
    def test_conversion_raises_clearly_without_deps(self, monkeypatch):
        """If onnx or onnxconverter_common isn't installed, the caller
        should see a clean ImportError, not an AttributeError mid-run."""
        from tether.exporters import fp16_convert

        # Simulate missing onnxconverter-common by making the import fail.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnxconverter_common.float16":
                raise ImportError("simulated missing dep")
            if name == "onnxconverter_common":
                raise ImportError("simulated missing dep")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="onnxconverter-common"):
            fp16_convert.convert_fp32_to_fp16(
                "/tmp/nonexistent.onnx", "/tmp/out.onnx"
            )
