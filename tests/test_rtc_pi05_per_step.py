"""pi0.5 per-step expert loop with RTC on CPU ONNX Runtime, vs LeRobot.

Builds a tiny per-step expert ONNX with the real contract
(x_t, t, past_kv..., prefix_pad_masks) -> v_t, runs
``Pi05DecomposedInference._run_expert`` on the CPU EP (the loop used to bind
CUDA OrtValues unconditionally), and checks the guided result against
LeRobot's ``euler_integrate`` + ``RTCProcessor`` on the same velocity field.
"""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")
pytest.importorskip("lerobot.policies.rtc")

from lerobot.configs.types import RTCAttentionSchedule  # noqa: E402
from lerobot.policies.common.flow_matching import euler_integrate  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.policies.rtc.modeling_rtc import RTCProcessor  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from tether.runtime.pi05_decomposed_server import Pi05DecomposedInference  # noqa: E402
from tether.runtime.rtc_guidance import RtcGuidance  # noqa: E402

B, T, A, H, L = 1, 12, 8, 16, 5
RNG = np.random.default_rng(3)
W1 = (RNG.standard_normal((A, H)) * 0.4).astype(np.float32)
W2 = (RNG.standard_normal((H, A)) * 0.4).astype(np.float32)
KV = (RNG.standard_normal((B, T, H)) * 0.2).astype(np.float32)
MASK = np.ones((B, L), dtype=np.bool_)


def _expert_onnx(path):
    """v_t = tanh(x_t @ W1 + past_k_0 + t + mean(mask)) @ W2."""
    nodes = [
        helper.make_node("MatMul", ["x_t", "W1"], ["xw"]),
        helper.make_node("Add", ["xw", "past_k_0"], ["h0"]),
        helper.make_node("Reshape", ["t", "t_shape"], ["t3"]),
        helper.make_node("Add", ["h0", "t3"], ["h1"]),
        helper.make_node("Cast", ["prefix_pad_masks"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("ReduceMean", ["mask_f"], ["mask_m"], keepdims=0),
        helper.make_node("Add", ["h1", "mask_m"], ["h2"]),
        helper.make_node("Tanh", ["h2"], ["h3"]),
        helper.make_node("MatMul", ["h3", "W2"], ["v_t"]),
    ]
    graph = helper.make_graph(
        nodes,
        "per_step_expert",
        [
            helper.make_tensor_value_info("x_t", TensorProto.FLOAT, [B, T, A]),
            helper.make_tensor_value_info("t", TensorProto.FLOAT, [B]),
            helper.make_tensor_value_info("past_k_0", TensorProto.FLOAT, [B, T, H]),
            helper.make_tensor_value_info("prefix_pad_masks", TensorProto.BOOL, [B, L]),
        ],
        [helper.make_tensor_value_info("v_t", TensorProto.FLOAT, [B, T, A])],
        initializer=[
            numpy_helper.from_array(W1, "W1"),
            numpy_helper.from_array(W2, "W2"),
            numpy_helper.from_array(np.array([B, 1, 1], np.int64), "t_shape"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9  # what ORT 1.2x loads
    onnx.save(model, path)
    return path


def _torch_field(x, t):
    tt = t.reshape(-1, 1, 1)
    h = torch.tanh(x @ torch.from_numpy(W1) + torch.from_numpy(KV) + tt + 1.0)
    return h @ torch.from_numpy(W2)


@pytest.fixture(scope="module")
def inference(tmp_path_factory):
    path = _expert_onnx(tmp_path_factory.mktemp("pi05") / "expert_denoise.onnx")
    inst = Pi05DecomposedInference.__new__(Pi05DecomposedInference)
    inst._sess_expert = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inst._per_step_expert = True
    return inst


def _feed():
    return {"past_k_0": KV, "prefix_pad_masks": MASK}


@pytest.mark.parametrize("num_steps", [2, 5, 10])
@pytest.mark.parametrize("schedule", ["ZEROS", "ONES", "LINEAR", "EXP"])
@pytest.mark.parametrize("delay", [0, 2, 5])
def test_cpu_per_step_rtc_matches_lerobot(inference, num_steps, schedule, delay):
    inference._num_steps = num_steps
    rng = np.random.default_rng(num_steps * 100 + delay)
    noise = rng.standard_normal((B, T, A)).astype(np.float32)
    prev = rng.standard_normal((T - 3, 7)).astype(np.float32)  # 7 < A: padded
    g = RtcGuidance(delay, 8, prev, schedule, 10.0)
    ours = inference._run_expert(_feed(), noise, rtc=g)
    ref = euler_integrate(
        _torch_field,
        torch.from_numpy(noise),
        num_steps,
        rtc_processor=RTCProcessor(
            RTCConfig(
                prefix_attention_schedule=RTCAttentionSchedule(schedule),
                max_guidance_weight=10.0,
                execution_horizon=8,
            )
        ),
        rtc_enabled=True,
        inference_delay=delay,
        prev_chunk_left_over=torch.from_numpy(prev),
        execution_horizon=8,
    ).numpy()
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-5)


def test_cpu_per_step_without_rtc_matches_plain_euler(inference):
    inference._num_steps = 5
    noise = np.random.default_rng(0).standard_normal((B, T, A)).astype(np.float32)
    ours = inference._run_expert(_feed(), noise)
    ref = euler_integrate(_torch_field, torch.from_numpy(noise), 5).numpy()
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-5)


def test_baked_loop_refuses_rtc(inference):
    baked = Pi05DecomposedInference.__new__(Pi05DecomposedInference)
    baked._per_step_expert = False
    with pytest.raises(RuntimeError, match="per-step expert"):
        baked._run_expert(_feed(), np.zeros((B, T, A), np.float32), rtc=RtcGuidance(1, 4, None))
