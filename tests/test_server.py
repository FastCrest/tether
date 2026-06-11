"""Tests for the VLA inference server."""

import json
import time
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from tether.runtime.server import TetherServer


@pytest.fixture
def mock_export_dir(tmp_path):
    """Create a mock export directory with config."""
    config = {
        "model_id": "lerobot/smolvla_base",
        "target": "desktop",
        "action_chunk_size": 50,
        "expert": {
            "expert_hidden": 720,
            "action_dim": 32,
            "num_layers": 16,
        },
    }
    config_path = tmp_path / "tether_config.json"
    config_path.write_text(json.dumps(config))
    return tmp_path


class TestTetherServer:
    def test_loads_config(self, mock_export_dir):
        server = TetherServer(mock_export_dir, device="cpu")
        assert server.config["model_id"] == "lerobot/smolvla_base"
        assert server.config["expert"]["action_dim"] == 32

    def test_not_ready_before_load(self, mock_export_dir):
        server = TetherServer(mock_export_dir, device="cpu")
        assert not server.ready

    def test_predict_before_load_returns_error(self, mock_export_dir):
        server = TetherServer(mock_export_dir, device="cpu")
        result = server.predict()
        assert "error" in result

    def test_loads_with_missing_onnx(self, mock_export_dir):
        server = TetherServer(mock_export_dir, device="cpu")
        server.load()
        assert not server.ready  # No ONNX file, so not ready


class TestTetherServerWithMockORT:
    def test_predict_async_offloads_non_batched_predict(self, mock_export_dir):
        import asyncio

        server = TetherServer(mock_export_dir, device="cpu", max_batch=1)

        def slow_predict(**kwargs):
            time.sleep(0.15)
            return {"ok": True, "kwargs": kwargs}

        server.predict = slow_predict

        async def run_check():
            start = time.perf_counter()
            task = asyncio.create_task(
                server.predict_async(instruction="pick", state=[0.0])
            )
            await asyncio.sleep(0.01)
            elapsed = time.perf_counter() - start
            result = await task
            return elapsed, result

        elapsed, result = asyncio.run(run_check())

        assert elapsed < 0.08
        assert result["ok"] is True
        assert result["kwargs"]["instruction"] == "pick"

    def test_batch_worker_offloads_sync_batch_predict(self, mock_export_dir):
        import asyncio

        server = TetherServer(
            mock_export_dir,
            device="cpu",
            max_batch=2,
            batch_timeout_ms=1,
        )

        def slow_batch(batch):
            time.sleep(0.15)
            return [{"ok": True, "batch_size": len(batch)} for _ in batch]

        server._predict_batch_sync = slow_batch

        async def run_check():
            await server.start_batch_worker()
            try:
                start = time.perf_counter()
                task = asyncio.create_task(
                    server.predict_async(instruction="pick", state=[0.0])
                )
                await asyncio.sleep(0.01)
                elapsed = time.perf_counter() - start
                result = await task
                return elapsed, result
            finally:
                await server.stop_batch_worker()

        elapsed, result = asyncio.run(run_check())

        assert elapsed < 0.08
        assert result["ok"] is True
        assert result["batch_size"] == 1

    def test_denoise_uses_iobinding_when_enabled(self, mock_export_dir):
        class _FakeOutput:
            name = "velocity"

        class _FakeOrtValue:
            def __init__(self, array):
                self._array = array

            def numpy(self):
                return self._array

        class _FakeBinding:
            def __init__(self, velocity):
                self.velocity = velocity
                self.bound_inputs = []
                self.bound_outputs = []
                self.clear_outputs_calls = 0

            def bind_cpu_input(self, name, array):
                self.bound_inputs.append((name, array.shape))

            def bind_output(self, name, *args):
                self.bound_outputs.append((name, args))

            def clear_binding_outputs(self):
                self.clear_outputs_calls += 1

            def get_outputs(self):
                return [_FakeOrtValue(self.velocity)]

        class _FakeSession:
            def __init__(self, velocity):
                self.binding = _FakeBinding(velocity)
                self.run_calls = 0
                self.run_with_iobinding_calls = 0

            def get_outputs(self):
                return [_FakeOutput()]

            def io_binding(self):
                return self.binding

            def run(self, *_args, **_kwargs):
                self.run_calls += 1
                raise AssertionError("session.run should not be used")

            def run_with_iobinding(self, _binding):
                self.run_with_iobinding_calls += 1

        server = TetherServer(mock_export_dir, device="cpu")
        server.num_denoising_steps = 1
        server._expert_input_names = []
        server._ort_iobinding_enabled = True

        noisy = np.zeros((1, 2, 3), dtype=np.float32)
        velocity = np.ones_like(noisy)
        fake_session = _FakeSession(velocity)
        server._ort_session = fake_session

        actions, steps = server._run_denoise(
            noisy_actions=noisy,
            position_ids=np.arange(2, dtype=np.int64)[None, :],
        )

        np.testing.assert_allclose(actions, -np.ones_like(noisy))
        assert steps == 1
        assert fake_session.run_calls == 0
        assert fake_session.run_with_iobinding_calls == 1
        bound_names = [name for name, _shape in fake_session.binding.bound_inputs]
        assert bound_names == ["position_ids", "noisy_actions", "timestep"]
        assert fake_session.binding.bound_outputs == [("velocity", ("cpu",))]

    def test_predict_returns_actions(self, mock_export_dir):
        server = TetherServer(mock_export_dir, device="cpu")
        server.action_dim = 32
        server.chunk_size = 50
        server.expert_hidden = 720
        server._inference_mode = "onnx"
        server._ready = True

        # Mock ORT session
        mock_session = MagicMock()
        mock_session.run.return_value = [np.random.randn(1, 50, 32).astype(np.float32)]
        server._ort_session = mock_session

        result = server.predict()

        assert "actions" in result
        assert result["num_actions"] == 50
        assert result["action_dim"] == 32
        assert result["latency_ms"] > 0
        assert result["hz"] > 0
        assert result["denoising_steps"] == 10
        assert mock_session.run.call_count == 10  # 10 denoising steps

    def test_predict_action_shape(self, mock_export_dir):
        server = TetherServer(mock_export_dir, device="cpu")
        server.action_dim = 6
        server.chunk_size = 20
        server.expert_hidden = 720
        server._inference_mode = "onnx"
        server._ready = True

        mock_session = MagicMock()
        mock_session.run.return_value = [np.random.randn(1, 20, 6).astype(np.float32)]
        server._ort_session = mock_session

        result = server.predict()
        assert len(result["actions"]) == 20
        assert len(result["actions"][0]) == 6


class TestCreateApp:
    def test_app_creates(self, mock_export_dir):
        try:
            from tether.runtime.server import create_app
            app = create_app(str(mock_export_dir), device="cpu")
            assert app is not None
            assert app.title == "Tether VLA Server"
        except ImportError:
            pytest.skip("fastapi not installed")


class TestStrictProviderMode:
    """Phase I.1: silent CPU fallback is now a hard error by default.

    The Apr 14 benchmark showed we had been publishing "GPU" numbers that
    were actually ORT CPU execution due to a CUDA-12-vs-13 library mismatch.
    These tests codify the new contract: asking for CUDA and not getting it
    raises, rather than silently degrading.
    """

    def test_strict_raises_when_cuda_requested_but_unavailable(
        self, mock_export_dir, tmp_path
    ):
        # Drop a dummy ONNX file so _load_onnx actually runs
        (tmp_path / "expert_stack.onnx").write_bytes(b"\x08\x07")  # ONNX magic stub

        server = TetherServer(
            mock_export_dir, device="cuda", strict_providers=True,
        )

        # Mock ORT to return a session whose active providers is CPU-only
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            with pytest.raises(RuntimeError, match="fell back to CPU"):
                server._load_onnx(tmp_path / "expert_stack.onnx")

    def test_non_strict_allows_fallback(self, mock_export_dir, tmp_path):
        (tmp_path / "expert_stack.onnx").write_bytes(b"\x08\x07")
        server = TetherServer(
            mock_export_dir, device="cuda", strict_providers=False,
        )
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            # Should NOT raise
            server._load_onnx(tmp_path / "expert_stack.onnx")
            assert server._inference_mode == "onnx_cpu"

    def test_strict_accepts_when_cuda_active(self, mock_export_dir, tmp_path):
        (tmp_path / "expert_stack.onnx").write_bytes(b"\x08\x07")
        server = TetherServer(
            mock_export_dir, device="cuda", strict_providers=True,
        )
        mock_session = MagicMock()
        mock_session.get_providers.return_value = [
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ]
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ]

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            server._load_onnx(tmp_path / "expert_stack.onnx")
            assert server._inference_mode == "onnx_gpu"

    def test_explicit_cpu_device_skips_strict_check(
        self, mock_export_dir, tmp_path
    ):
        (tmp_path / "expert_stack.onnx").write_bytes(b"\x08\x07")
        server = TetherServer(
            mock_export_dir, device="cpu", strict_providers=True,
        )
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            server._load_onnx(tmp_path / "expert_stack.onnx")
            assert server._inference_mode == "onnx_cpu"

    def test_explicit_providers_list_overrides_device(
        self, mock_export_dir, tmp_path
    ):
        (tmp_path / "expert_stack.onnx").write_bytes(b"\x08\x07")
        # device=cpu but explicit CUDAExecutionProvider in list
        server = TetherServer(
            mock_export_dir,
            device="cpu",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            strict_providers=True,
        )
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            # CUDAExecutionProvider is in providers list → strict should fire
            with pytest.raises(RuntimeError, match="fell back to CPU"):
                server._load_onnx(tmp_path / "expert_stack.onnx")
