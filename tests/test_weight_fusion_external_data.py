from __future__ import annotations

from pathlib import Path

import pytest


def test_save_onnx_externalizes_attribute_tensors(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    from onnx import helper

    from reflex.exporters import weight_fusion

    captured = {}

    def _fake_save(model, path, **kwargs):
        captured.update(kwargs)
        out = Path(path)
        out.write_bytes(b"onnx-protobuf")
        (out.parent / kwargs["location"]).write_bytes(b"external-weights")

    monkeypatch.setattr(onnx, "save", _fake_save)

    # _save_onnx coerces inline tensors before save, so it needs a real model
    # with a `.graph`; an empty graph keeps coercion a no-op while still
    # exercising the external-data save kwargs this test asserts on.
    model = helper.make_model(helper.make_graph([], "empty", [], []))
    weight_fusion._save_onnx(model, tmp_path / "model.onnx")

    assert captured["save_as_external_data"] is True
    assert captured["convert_attribute"] is True
    assert captured["location"] == "model.onnx.data"


def test_inline_attribute_tensors_are_normalized_to_raw_data():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from reflex.exporters import weight_fusion

    tensor = helper.make_tensor(
        "constant_weights",
        TensorProto.FLOAT,
        [300],
        [float(i) for i in range(300)],
        raw=False,
    )
    node = helper.make_node("Constant", [], ["out"], value=tensor)
    graph = helper.make_graph(
        [node],
        "inline-tensor-test",
        [],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [300])],
    )
    model = helper.make_model(graph)
    attr_tensor = model.graph.node[0].attribute[0].t

    assert len(attr_tensor.float_data) == 300
    assert not attr_tensor.HasField("raw_data")

    converted, converted_bytes = weight_fusion._coerce_inline_tensor_data_to_raw(model)

    attr_tensor = model.graph.node[0].attribute[0].t
    assert converted == 1
    assert converted_bytes >= 1200
    assert attr_tensor.HasField("raw_data")
    assert len(attr_tensor.float_data) == 0
