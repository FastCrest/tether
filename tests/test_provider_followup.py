"""Provider selection facts without GPU execution or engine compilation."""

from unittest.mock import MagicMock
import sys

import pytest

from tether.runtime import ort_providers as providers
from tether.runtime.smolvla_onnx_server import SmolVLAOnnxServer
from tether.runtime.pi0_onnx_server import Pi0OnnxServer
from tether.runtime.pi05_onnx_server import Pi05OnnxServer

TRT, CUDA, CPU = "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"


def test_explicit_trt_receives_cache_precision_and_workspace_defaults(tmp_path):
    requested = [TRT, CUDA]
    plan = providers.build_ort_provider_plan(
        tmp_path, requested_providers=requested, available_providers=requested
    )
    assert [providers.provider_name(p) for p in plan.providers] == requested
    assert isinstance(plan.providers[0], tuple)
    options = plan.providers[0][1]
    assert options["trt_fp16_enable"] is True
    assert 0 < options["trt_max_workspace_size"] <= 4 * 1024**3
    assert options["trt_engine_cache_enable"] is True
    assert options["trt_timing_cache_enable"] is True
    assert requested == [TRT, CUDA], "caller list must remain unchanged"


def test_explicit_options_and_order_override_trt_defaults(tmp_path):
    options = {
        "trt_fp16_enable": False,
        "trt_max_workspace_size": "123456",
        "trt_engine_cache_enable": False,
        "trt_timing_cache_enable": False,
        "trt_engine_cache_path": "/caller/cache",
        "device_id": 1,
    }
    requested = [CUDA, (TRT, options), CPU]
    plan = providers.build_ort_provider_plan(
        tmp_path, requested_providers=requested, available_providers=[TRT, CUDA, CPU]
    )
    assert [providers.provider_name(p) for p in plan.providers] == [CUDA, TRT, CPU]
    assert plan.providers[1][1] == options
    assert requested[1][1] is options


def mocked_server(
    tmp_path, monkeypatch, *, active, strict=True, options=None, server_class=SmolVLAOnnxServer
):
    (tmp_path / "model.onnx").write_bytes(b"fake graph; ORT session is mocked")
    session = MagicMock()
    session.get_providers.return_value = active
    session.get_provider_options.return_value = options or {p: {} for p in active}
    session.get_inputs.return_value = []
    ort = MagicMock()
    ort.get_available_providers.return_value = [TRT, CUDA, CPU]
    ort.InferenceSession.return_value = session
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setattr(
        "tether.runtime.tokenizers.ensure_offline_tokenizer_bundle", lambda *a, **k: None
    )
    return server_class(
        tmp_path, providers=[(TRT, {"trt_fp16_enable": False}), CUDA], strict_providers=strict
    ), ort


@pytest.mark.parametrize("server_class", [SmolVLAOnnxServer, Pi0OnnxServer, Pi05OnnxServer])
def test_strict_explicit_trt_cannot_succeed_after_cuda_fallback(
    tmp_path, monkeypatch, server_class
):
    server, _ = mocked_server(tmp_path, monkeypatch, active=[CUDA, CPU], server_class=server_class)
    with pytest.raises(RuntimeError, match="TensorrtExecutionProvider"):
        server.load()


@pytest.mark.parametrize("server_class", [SmolVLAOnnxServer, Pi0OnnxServer, Pi05OnnxServer])
def test_trt_presence_does_not_claim_fp16_execution(tmp_path, monkeypatch, server_class):
    server, _ = mocked_server(
        tmp_path, monkeypatch, active=[TRT, CUDA, CPU], server_class=server_class
    )
    server.load()
    assert server._provider_mode == "onnx_trt"


@pytest.mark.parametrize(
    "memory,expected",
    [(None, 4 * 1024**3), (8 * 1024**3, 2 * 1024**3), (32 * 1024**3, 4 * 1024**3)],
)
def test_workspace_defaults_are_jetson_specific_and_bounded(
    tmp_path, monkeypatch, memory, expected
):
    monkeypatch.delenv("TETHER_TRT_WORKSPACE_BYTES", raising=False)
    monkeypatch.setattr(providers, "_jetson_memory_bytes", lambda: memory)
    plan = providers.build_ort_provider_plan(
        tmp_path, requested_providers=[TRT], available_providers=[TRT]
    )
    assert plan.providers[0][1]["trt_max_workspace_size"] == expected


@pytest.mark.parametrize("value", ["0", "-1", "garbage", "1.5", str(2**63)])
def test_invalid_workspace_configuration_is_actionable(tmp_path, monkeypatch, value):
    monkeypatch.setenv("TETHER_TRT_WORKSPACE_BYTES", value)
    with pytest.raises(ValueError, match="workspace|WORKSPACE"):
        providers.build_ort_provider_plan(
            tmp_path, requested_providers=[TRT], available_providers=[TRT]
        )


def test_caller_workspace_overrides_environment_without_reordering(tmp_path, monkeypatch):
    monkeypatch.setenv("TETHER_TRT_WORKSPACE_BYTES", "invalid ignored when caller supplies value")
    plan = providers.build_ort_provider_plan(
        tmp_path,
        requested_providers=[CPU, (TRT, {"trt_max_workspace_size": 1000000})],
        available_providers=[CPU, TRT],
    )
    assert providers.provider_name(plan.providers[0]) == CPU
    assert plan.providers[1][1]["trt_max_workspace_size"] == 1000000


def test_non_strict_fallback_reports_active_providers_and_options(tmp_path, monkeypatch, caplog):
    server, _ = mocked_server(tmp_path, monkeypatch, active=[CUDA, CPU], strict=False)
    server.load()
    facts = providers.provider_runtime_details(server._provider_plan, server._session)
    assert facts["requested_providers"][0] == (TRT, {"trt_fp16_enable": False})
    assert facts["configured_provider_options"][TRT]["trt_fp16_enable"] is False
    assert facts["active_providers"] == [CUDA, CPU]
    assert facts["precision"] == facts["node_placement"] == "not_measured"
    assert "Explicit primary TensorrtExecutionProvider" in caplog.text


def test_config_exposes_provider_facts_without_inference(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from tests.export_config_factory import write_test_export_config
    from tether.runtime.server import create_app

    server, _ = mocked_server(tmp_path, monkeypatch, active=[TRT, CUDA, CPU])
    server.load()
    write_test_export_config(
        tmp_path, model_type="smolvla", export_kind="monolithic_onnx", artifacts=["model.onnx"]
    )
    app = create_app(str(tmp_path), device="cpu", prewarm=False)
    original = app.state.tether_server
    # /config closes over the concrete server. Seed its observed loaded-session facts;
    # no lifespan/inference executes for this endpoint contract check.
    original._provider_plan, original._session = server._provider_plan, server._session
    response = TestClient(app).get("/config")
    assert response.status_code == 200
    facts = response.json()["execution_providers"]
    assert facts["configured_providers"] == [TRT, CUDA]
    assert facts["requested_providers"][0] == [TRT, {"trt_fp16_enable": False}]
    assert facts["active_provider_options"] == {TRT: {}, CUDA: {}, CPU: {}}
    assert facts["precision"] == "not_measured"


def test_rtc_horizon_is_a_plan_not_observed_execution():
    from tether.runtime.server import _action_execution_from_rtc_stats

    result = _action_execution_from_rtc_stats({"enabled": True, "execution_horizon": 8})
    assert result["planned_horizon"] == result["executed_horizon"] == 8
    assert result["evidence_scope"] == "runtime_schedule"
    assert result["physical_execution_observed"] is False


def test_valid_workspace_environment_is_forwarded_to_ort(tmp_path, monkeypatch):
    monkeypatch.setenv("TETHER_TRT_WORKSPACE_BYTES", str(512 * 1024**2))
    plan = providers.build_ort_provider_plan(
        tmp_path, requested_providers=[TRT], available_providers=[TRT]
    )
    assert plan.providers[0][1]["trt_max_workspace_size"] == 512 * 1024**2


def test_arm64_alone_does_not_select_the_jetson_memory_default(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(Path, "is_file", lambda path: False)
    monkeypatch.setattr(
        providers.os, "sysconf", lambda *a: pytest.fail("non-Jetson must not infer shared RAM")
    )
    assert providers._jetson_memory_bytes() is None


@pytest.mark.parametrize("active", [[TRT, CUDA], [CUDA, CPU]])
def test_legacy_onnx_server_enforces_explicit_primary_and_truthful_mode(
    tmp_path, monkeypatch, active
):
    from tether.runtime.server import TetherServer

    _, ort = mocked_server(tmp_path, monkeypatch, active=active)
    server = object.__new__(TetherServer)
    server.export_dir = tmp_path
    server._requested_device = "cuda"
    server._requested_providers = [(TRT, {"trt_fp16_enable": False}), CUDA]
    server._strict_providers, server._max_batch = True, 1
    if active[0] != TRT:
        with pytest.raises(RuntimeError, match="Explicit primary TensorrtExecutionProvider"):
            server._load_onnx(tmp_path / "model.onnx")
    else:
        server._load_onnx(tmp_path / "model.onnx")
        assert server._inference_mode == "onnx_trt"
        assert server._active_providers == active


def test_provider_mode_follows_registered_priority_not_trt_presence():
    assert providers.primary_provider_mode([CPU, TRT, CUDA]) == "onnx_cpu"
    assert providers.primary_provider_mode([CUDA, TRT, CPU]) == "onnx_gpu"
    assert providers.primary_provider_mode([TRT, CUDA, CPU]) == "onnx_trt"


def test_mixed_fp16_exports_keep_their_fp32_islands_under_tensorrt(tmp_path):
    # trt_fp16_enable on a mixed-FP16 SmolVLA export gave action cosine 0.32-0.56 on an A10G.
    import json

    for precision, expected in (
        ("mixed_float16", False),
        ("float16", False),
        ("float32", True),
        (None, True),
    ):
        export = tmp_path / str(precision)
        export.mkdir()
        if precision is not None:
            (export / "tether_config.json").write_text(json.dumps({"precision": precision}))
        plan = providers.build_ort_provider_plan(
            export,
            device="cuda",
            available_providers=[TRT, CUDA, CPU],
            requested_providers=[TRT, CUDA],
        )
        assert plan.providers[0][1]["trt_fp16_enable"] is expected, precision
    explicit = providers.build_ort_provider_plan(
        tmp_path / "mixed_float16",
        device="cuda",
        available_providers=[TRT, CUDA, CPU],
        requested_providers=[(TRT, {"trt_fp16_enable": True}), CUDA],
    )
    assert explicit.providers[0][1]["trt_fp16_enable"] is True
