"""Modal diagnostic: probe every ONNX op type present in the retained pi0.5
monolithic export for CUDAExecutionProvider support with CPU fallback
disabled, one single-op session per type.

The export fails the no-fallback gate ("nodes assigned to the default CPU
EP"); this names the exact op types responsible instead of guessing from
log lines.

Usage:
    modal run scripts/modal_pi05_ep_op_probe.py
"""

import os

import modal

# Exact op census of /parity_out/pi05/export/model.onnx (from placement
# diag 2026-09-18): Mul 3581, Add 3372, MatMul 2448, Transpose 1607,
# Reshape 1511, Slice 790, Unsqueeze 787, Concat 759, Pow 683,
# ReduceMean 405, Sqrt 405, Reciprocal 405, Expand 398, Neg 395, Gemm 390,
# Split 370, Softmax 278, Tanh 278, LayerNormalization 165, Sigmoid 20,
# And 12, Where 11, Conv 3, Div 3, Cast 3, Gather 2, Sub 2, Cos 2, Sin 2,
# LessOrEqual 1, CumSum 1, Pad 1, ReduceSum 1.

app = modal.App("tether-pi05-ep-op-probe")

HF_CACHE_PATH = "/root/.cache/huggingface"
_hf_cache_volume = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("onnx>=1.16", "numpy")
    .pip_install(
        "onnxruntime-gpu>=1.20,<1.24",
        "nvidia-cuda-runtime-cu12>=12.0,<13.0",
        "nvidia-cublas-cu12>=12.0,<13.0",
        "nvidia-cudnn-cu12>=9.0,<10.0",
        "nvidia-cufft-cu12>=11.0,<12.0",
        "nvidia-curand-cu12>=10.0,<11.0",
        extra_options="--no-deps",
    )
    .pip_install("onnxruntime-gpu>=1.20,<1.24")
    .env({
        "LD_LIBRARY_PATH": (
            "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:"
            "/usr/local/cuda/lib64"
        ),
    })
)


def _probes():
    """Return {op_type: (node, feeds)} with minimal valid single-op graphs."""
    import numpy as np
    from onnx import TensorProto, helper

    F = TensorProto.FLOAT
    B = TensorProto.BOOL
    I64 = TensorProto.INT64

    def vi(name, dtype, shape):
        return helper.make_tensor_value_info(name, dtype, shape)

    def const_i64(name, values):
        arr = np.asarray(values, dtype=np.int64)
        return helper.make_tensor(name, TensorProto.INT64, arr.shape, arr.ravel())

    def const_f(name, values):
        arr = np.asarray(values, dtype=np.float32)
        return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.ravel())

    def binary(op, dtype=F):
        node = helper.make_node(op, ["a", "b"], ["c"])
        feeds = {
            "a": np.ones((2, 3), dtype=np.float32 if dtype == F else bool),
            "b": np.ones((2, 3), dtype=np.float32 if dtype == F else bool),
        }
        return node, [vi("a", dtype, [2, 3]), vi("b", dtype, [2, 3])], [vi("c", dtype, [2, 3])], feeds, []

    def unary(op, **attrs):
        node = helper.make_node(op, ["a"], ["c"], **attrs)
        feeds = {"a": (np.ones((2, 3), dtype=np.float32) * 2.0)}
        return node, [vi("a", F, [2, 3])], [vi("c", F, [2, 3])], feeds, []

    table = {}
    for op in ("Add", "Sub", "Mul", "Div", "Pow"):
        table[op] = binary(op)
    table["And"] = binary("And", dtype=B)
    for op in ("Neg", "Sqrt", "Reciprocal", "Tanh", "Sigmoid", "Cos", "Sin"):
        table[op] = unary(op)
    table["Cast"] = unary("Cast", to=1)
    table["Softmax"] = unary("Softmax", axis=-1)
    table["MatMul"] = (
        helper.make_node("MatMul", ["a", "b"], ["c"]),
        [vi("a", F, [2, 3]), vi("b", F, [3, 2])],
        [vi("c", F, [2, 2])],
        {"a": np.ones((2, 3), np.float32), "b": np.ones((3, 2), np.float32)},
        [],
    )
    table["Gemm"] = (
        helper.make_node("Gemm", ["a", "b", "d"], ["c"]),
        [vi("a", F, [2, 3]), vi("b", F, [3, 2]), vi("d", F, [2, 2])],
        [vi("c", F, [2, 2])],
        {
            "a": np.ones((2, 3), np.float32),
            "b": np.ones((3, 2), np.float32),
            "d": np.ones((2, 2), np.float32),
        },
        [],
    )
    table["Transpose"] = unary("Transpose", perm=[1, 0])
    table["Reshape"] = (
        helper.make_node("Reshape", ["a", "shape"], ["c"]),
        [vi("a", F, [2, 3])],
        [vi("c", F, [3, 2])],
        {"a": np.ones((2, 3), np.float32)},
        [const_i64("shape", [3, 2])],
    )
    table["Unsqueeze"] = (
        helper.make_node("Unsqueeze", ["a", "axes"], ["c"]),
        [vi("a", F, [2, 3])],
        [vi("c", F, [2, 1, 3])],
        {"a": np.ones((2, 3), np.float32)},
        [const_i64("axes", [1])],
    )
    table["Concat"] = (
        helper.make_node("Concat", ["a", "b"], ["c"], axis=0),
        [vi("a", F, [2, 3]), vi("b", F, [2, 3])],
        [vi("c", F, [4, 3])],
        {"a": np.ones((2, 3), np.float32), "b": np.ones((2, 3), np.float32)},
        [],
    )
    table["Slice"] = (
        helper.make_node("Slice", ["a", "starts", "ends", "axes", "steps"], ["c"]),
        [vi("a", F, [2, 3])],
        [vi("c", F, [1, 3])],
        {"a": np.ones((2, 3), np.float32)},
        [
            const_i64("starts", [0, 0]),
            const_i64("ends", [1, 3]),
            const_i64("axes", [0, 1]),
            const_i64("steps", [1, 1]),
        ],
    )
    table["Expand"] = (
        helper.make_node("Expand", ["a", "shape"], ["c"]),
        [vi("a", F, [1, 3])],
        [vi("c", F, [2, 3])],
        {"a": np.ones((1, 3), np.float32)},
        [const_i64("shape", [2, 3])],
    )
    table["Split"] = (
        helper.make_node("Split", ["a"], ["c1", "c2"], axis=0, num_outputs=2),
        [vi("a", F, [2, 3])],
        [vi("c1", F, [1, 3]), vi("c2", F, [1, 3])],
        {"a": np.ones((2, 3), np.float32)},
        [],
    )
    table["Gather"] = (
        helper.make_node("Gather", ["a", "idx"], ["c"], axis=0),
        [vi("a", F, [3, 2]), vi("idx", I64, [2])],
        [vi("c", F, [2, 2])],
        {"a": np.ones((3, 2), np.float32), "idx": np.asarray([0, 2], np.int64)},
        [],
    )
    table["ReduceMean"] = unary("ReduceMean", axes=[1], keepdims=1)
    table["ReduceSum"] = unary("ReduceSum", axes=[1], keepdims=1)
    table["LayerNormalization"] = (
        helper.make_node("LayerNormalization", ["a", "s", "b"], ["c"], axis=-1),
        [vi("a", F, [2, 3]), vi("s", F, [3]), vi("b", F, [3])],
        [vi("c", F, [2, 3])],
        {
            "a": np.ones((2, 3), np.float32),
            "s": np.ones(3, np.float32),
            "b": np.zeros(3, np.float32),
        },
        [],
    )
    table["Where"] = (
        helper.make_node("Where", ["cond", "a", "b"], ["c"]),
        [vi("cond", B, [2, 3]), vi("a", F, [2, 3]), vi("b", F, [2, 3])],
        [vi("c", F, [2, 3])],
        {
            "cond": np.ones((2, 3), bool),
            "a": np.ones((2, 3), np.float32),
            "b": np.zeros((2, 3), np.float32),
        },
        [],
    )
    table["LessOrEqual"] = (
        helper.make_node("LessOrEqual", ["a", "b"], ["c"]),
        [vi("a", F, [2, 3]), vi("b", F, [2, 3])],
        [vi("c", B, [2, 3])],
        {"a": np.ones((2, 3), np.float32), "b": np.ones((2, 3), np.float32)},
        [],
    )
    table["CumSum"] = (
        helper.make_node("CumSum", ["a", "axis"], ["c"]),
        [vi("a", F, [2, 3])],
        [vi("c", F, [2, 3])],
        {"a": np.ones((2, 3), np.float32)},
        [const_i64("axis", [1])],
    )
    table["Pad"] = (
        helper.make_node("Pad", ["a", "pads", "value"], ["c"]),
        [vi("a", F, [2, 3])],
        [vi("c", F, [2, 5])],
        {"a": np.ones((2, 3), np.float32)},
        [const_i64("pads", [0, 0, 0, 2]), const_f("value", [0.0])],
    )
    table["Conv"] = (
        helper.make_node("Conv", ["a", "w"], ["c"]),
        [vi("a", F, [1, 1, 4, 4]), vi("w", F, [1, 1, 2, 2])],
        [vi("c", F, [1, 1, 3, 3])],
        {"a": np.ones((1, 1, 4, 4), np.float32), "w": np.ones((1, 1, 2, 2), np.float32)},
        [],
    )
    return table


@app.function(
    image=image,
    gpu="A10G",
    timeout=1200,
    volumes={HF_CACHE_PATH: _hf_cache_volume},
    secrets=[_hf_secret()],
)
def run_probe(watchdog_seconds: float = 900) -> dict:
    import os as _os
    import signal as _signal
    import threading as _threading

    _watchdog = _threading.Timer(
        float(_os.environ.get("PARITY_WATCHDOG_SECONDS", str(watchdog_seconds))),
        lambda: (_os.kill(_os.getpid(), _signal.SIGKILL)),
    )
    _watchdog.daemon = True
    _watchdog.start()

    from onnx import helper
    import onnxruntime as ort

    print(f"[probe] ort {ort.__version__}", flush=True)
    results: dict = {"ort_version": ort.__version__, "ops": {}}
    for op, (node, inputs, outputs, feeds, initializers) in sorted(_probes().items()):
        graph = helper.make_graph([node], f"probe-{op}", inputs, outputs, initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
        model.ir_version = 10
        opts = ort.SessionOptions()
        opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        opts.log_severity_level = 3
        try:
            sess = ort.InferenceSession(
                model.SerializeToString(),
                sess_options=opts,
                providers=["CUDAExecutionProvider"],
            )
            sess.run(None, feeds)
            status = f"ok providers={sess.get_providers()}"
        except Exception as exc:  # noqa: BLE001 -- probe record
            status = f"FAIL {type(exc).__name__}: {str(exc)[:220]}"
        results["ops"][op] = status
        print(f"[probe] {op:22s} {status}", flush=True)

    _watchdog.cancel()
    print("[probe] done", flush=True)
    return results


@app.local_entrypoint()
def main() -> None:
    result = run_probe.remote()
    bad = {op: st for op, st in result["ops"].items() if st.startswith("FAIL")}
    print(f"\n=== {len(bad)} failing op types ===")
    for op, status in bad.items():
        print(f"  {op}: {status}")
