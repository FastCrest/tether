from __future__ import annotations

from pathlib import Path

import pytest

from reflex.runtime.ort_providers import (
    build_ort_provider_plan,
    onnx_external_data_bytes,
    provider_name,
)


def test_cuda_prefers_trt_with_engine_and_timing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("reflex.runtime.ort_providers.gpu_is_blackwell", lambda: False)
    plan = build_ort_provider_plan(
        tmp_path,
        device="cuda",
        available_providers=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    assert provider_name(plan.providers[0]) == "TensorrtExecutionProvider"
    assert plan.used_trt is True
    trt_opts = plan.providers[0][1]
    assert trt_opts["trt_engine_cache_enable"] is True
    assert trt_opts["trt_timing_cache_enable"] is True
    assert Path(trt_opts["trt_engine_cache_path"]).is_dir()
    assert Path(trt_opts["trt_timing_cache_path"]).is_dir()
    assert [provider_name(p) for p in plan.providers[-2:]] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_cuda_disables_trt_when_batching_static_export(tmp_path, monkeypatch):
    monkeypatch.setattr("reflex.runtime.ort_providers.gpu_is_blackwell", lambda: False)
    plan = build_ort_provider_plan(
        tmp_path,
        device="cuda",
        max_batch=4,
        available_providers=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    assert plan.used_trt is False
    assert "max_batch=4" in plan.trt_disabled_reason
    assert provider_name(plan.providers[0]) == "CUDAExecutionProvider"


def test_explicit_providers_are_not_rewritten(tmp_path):
    requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    plan = build_ort_provider_plan(
        tmp_path,
        device="cuda",
        requested_providers=requested,
        available_providers=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    assert plan.providers == requested
    assert plan.trt_disabled_reason == "explicit providers supplied"


def test_env_can_disable_trt(tmp_path, monkeypatch):
    monkeypatch.setenv("REFLEX_TRT_EP", "0")
    monkeypatch.setattr("reflex.runtime.ort_providers.gpu_is_blackwell", lambda: False)
    plan = build_ort_provider_plan(
        tmp_path,
        device="cuda",
        available_providers=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    assert plan.used_trt is False
    assert "disabled" in plan.trt_disabled_reason.lower()
    assert provider_name(plan.providers[0]) == "CUDAExecutionProvider"


def test_scatternd_reduction_graph_disables_trt(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    onnx_path = tmp_path / "scatternd.onnx"
    graph = helper.make_graph(
        [
            helper.make_node(
                "ScatterND",
                ["data", "indices", "updates"],
                ["out"],
                reduction="add",
            )
        ],
        "scatternd_reduction",
        [
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 1]),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, [1, 2]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    onnx.save(model, onnx_path)

    monkeypatch.setattr("reflex.runtime.ort_providers.gpu_is_blackwell", lambda: False)
    plan = build_ort_provider_plan(
        tmp_path,
        device="cuda",
        available_providers=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        onnx_path=onnx_path,
    )

    assert plan.used_trt is False
    assert "ScatterND reduction" in plan.trt_disabled_reason
    assert provider_name(plan.providers[0]) == "CUDAExecutionProvider"


def test_session_options_default_to_error_logs(monkeypatch):
    pytest.importorskip("onnxruntime")
    from reflex.runtime.ort_providers import make_ort_session_options

    monkeypatch.delenv("REFLEX_ORT_LOG_SEVERITY", raising=False)
    opts = make_ort_session_options()

    assert opts.log_severity_level == 3


def test_large_external_data_disables_trt(tmp_path, monkeypatch):
    monkeypatch.setenv("REFLEX_LARGE_EXTERNAL_DATA_BYTES", "10")
    monkeypatch.setattr("reflex.runtime.ort_providers.gpu_is_blackwell", lambda: False)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    (tmp_path / "model.onnx.data").write_bytes(b"x" * 11)

    plan = build_ort_provider_plan(
        tmp_path,
        device="cuda",
        available_providers=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        onnx_path=onnx_path,
    )

    assert onnx_external_data_bytes(onnx_path) == 11
    assert plan.used_trt is False
    assert "large external-data ONNX" in plan.trt_disabled_reason
    assert provider_name(plan.providers[0]) == "CUDAExecutionProvider"


def test_large_external_data_disables_graph_optimizations(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    from reflex.runtime.ort_providers import make_ort_session_options

    monkeypatch.setenv("REFLEX_LARGE_EXTERNAL_DATA_BYTES", "10")
    monkeypatch.delenv("REFLEX_ORT_GRAPH_OPT_LEVEL", raising=False)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    (tmp_path / "model.onnx.data").write_bytes(b"x" * 11)

    opts = make_ort_session_options(onnx_path)

    assert opts.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
