"""Provider intent survives alternate dispatch and split runtime stages."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import sys

import pytest

from tests.export_config_factory import write_test_export_config
from tether.runtime.decomposed_server import Pi05DecomposedServer
from tether.runtime.smolvla_native import SmolVLANativeServer

TRT, CUDA, CPU = 'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'


def test_direct_native_runtime_refuses_ignored_ort_providers(tmp_path):
    with pytest.raises(ValueError, match='ORT providers.*native'):
        SmolVLANativeServer(tmp_path, providers=[TRT, CUDA])


def test_explicit_provider_intent_wins_over_ambient_native_override(tmp_path, monkeypatch):
    from tether.runtime.server import create_app
    from tether.runtime.smolvla_onnx_server import SmolVLAOnnxServer

    write_test_export_config(tmp_path)
    monkeypatch.setenv('TETHER_NATIVE', '1')
    app = create_app(tmp_path, device='cuda', providers=[TRT, CUDA], prewarm=False)
    assert isinstance(app.state.tether_server, SmolVLAOnnxServer)


def split_fixture(tmp_path, monkeypatch, active_prefix, active_expert):
    write_test_export_config(tmp_path, model_type='pi05_decomposed', export_kind='decomposed_onnx',
                             artifacts=['vlm_prefix.onnx', 'expert_denoise.onnx'],
                             decomposed={'vlm_prefix_onnx': 'vlm_prefix.onnx',
                                         'expert_denoise_onnx': 'expert_denoise.onnx'})
    sessions = []
    for active in (active_prefix, active_expert):
        session = MagicMock(spec=["get_providers", "get_provider_options", "get_inputs"])
        session.get_providers.return_value = active
        session.get_provider_options.return_value = {name: {} for name in active}
        session.get_inputs.return_value = []
        sessions.append(session)
    captured = {}

    def inference(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(_sess_prefix=sessions[0], _sess_expert=sessions[1])

    monkeypatch.setattr('tether.runtime.pi05_decomposed_server.Pi05DecomposedInference', inference)
    monkeypatch.setattr(Pi05DecomposedServer, '_load_normalizer_stats', lambda self: None)
    ort = MagicMock()
    ort.get_available_providers.return_value = [TRT, CUDA, CPU]
    monkeypatch.setitem(sys.modules, 'onnxruntime', ort)
    return captured


def test_split_refuses_one_stage_losing_explicit_trt(tmp_path, monkeypatch):
    split_fixture(tmp_path, monkeypatch, [TRT, CUDA, CPU], [CUDA, CPU])
    server = Pi05DecomposedServer(tmp_path, providers=[TRT, CUDA], device='cuda')
    with pytest.raises(RuntimeError, match='expert_denoise.*primary'):
        server.load()


def test_split_plans_and_reports_each_stage_without_precision_claim(tmp_path, monkeypatch):
    captured = split_fixture(tmp_path, monkeypatch, [TRT, CUDA, CPU], [CUDA, CPU])
    server = Pi05DecomposedServer(tmp_path, providers=[TRT, CUDA], device='cuda', strict_providers=False)
    server.load()
    stage_providers = captured['providers_by_stage']
    assert set(stage_providers) == {'vlm_prefix', 'expert_denoise'}
    first, second = (stage_providers[name][0][1] for name in ('vlm_prefix', 'expert_denoise'))
    assert first['trt_fp16_enable'] and second['trt_fp16_enable']
    assert first['trt_engine_cache_path'] != second['trt_engine_cache_path']
    report = server.config['execution_providers']
    assert report['active_provider_scope'] == 'union_of_stage_sessions'
    assert report['stages']['vlm_prefix']['active_providers'][0] == TRT
    assert report['stages']['expert_denoise']['active_providers'][0] == CUDA
    assert report['precision'] == report['node_placement'] == 'not_measured'
    assert server._inference_mode == 'onnx_mixed_decomposed'


@pytest.mark.parametrize('strict', [True, False])
def test_split_cuda_cpu_fallback_is_strict_or_truthfully_reported(tmp_path, monkeypatch, strict):
    split_fixture(tmp_path, monkeypatch, [CPU], [CPU])
    server = Pi05DecomposedServer(tmp_path, providers=[CUDA, CPU], device='cuda', strict_providers=strict)
    if strict:
        with pytest.raises(RuntimeError, match='vlm_prefix.*GPU provider requested'):
            server.load()
    else:
        server.load()
        assert server._inference_mode == 'onnx_cpu_decomposed'
        assert server.config['execution_providers']['active_providers'] == [CPU]


def test_split_primitive_forwards_distinct_plans_to_real_session_factory(tmp_path, monkeypatch):
    from tether.runtime.pi05_decomposed_server import Pi05DecomposedInference

    write_test_export_config(tmp_path, model_type='pi05_decomposed', export_kind='decomposed_onnx',
                             artifacts=['vlm_prefix.onnx', 'expert_denoise.onnx'],
                             decomposed={'vlm_prefix_onnx': 'vlm_prefix.onnx',
                                         'expert_denoise_onnx': 'expert_denoise.onnx',
                                         'past_kv_tensor_names': [], 'paligemma_layers': 0})
    calls = []

    def session(path, providers):
        calls.append((path, providers))
        return SimpleNamespace(get_providers=lambda: [CPU], get_inputs=lambda: [], get_outputs=lambda: [])

    ort = MagicMock()
    ort.InferenceSession.side_effect = session
    monkeypatch.setitem(sys.modules, 'onnxruntime', ort)
    plans = {'vlm_prefix': [(TRT, {'trt_engine_cache_path': '/prefix'})],
             'expert_denoise': [(TRT, {'trt_engine_cache_path': '/expert'})]}
    Pi05DecomposedInference(tmp_path, providers=[CPU], providers_by_stage=plans)
    assert [providers for _, providers in calls] == [plans['vlm_prefix'], plans['expert_denoise']]
    assert calls[0][0].endswith('vlm_prefix.onnx')
    assert calls[1][0].endswith('expert_denoise.onnx')
    with pytest.raises(ValueError, match='cannot be combined with cuda_graphs_enabled'):
        Pi05DecomposedInference(tmp_path, providers_by_stage=plans, cuda_graphs_enabled=True)


def test_split_config_endpoint_keeps_per_stage_registration(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from tether.runtime.server import create_app

    split_fixture(tmp_path, monkeypatch, [TRT, CUDA, CPU], [CUDA, CPU])
    server = Pi05DecomposedServer(tmp_path, providers=[TRT, CUDA], device='cuda', strict_providers=False)
    server.load()
    app = create_app(tmp_path, device='cpu', prewarm=False)
    # Endpoint closes over its dispatched server. The mocked session factory
    # keeps the real lifespan local; no model inference or downloads occur.
    app.state.tether_server.config['execution_providers'] = server.config['execution_providers']
    with TestClient(app) as client:
        facts = client.get('/config').json()['execution_providers']
    assert facts['stages']['vlm_prefix']['active_providers'][0] == TRT
    assert facts['stages']['expert_denoise']['active_providers'][0] == CUDA
