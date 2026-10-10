"""Modal diagnostic: single-op GatherND session on CUDAExecutionProvider with
CPU fallback disabled -- the exact gate the GR00T export fails.

Usage:
    modal run scripts/modal_gathernd_probe.py
"""

import os

import modal

app = modal.App("tether-gathernd-probe")
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


@app.function(
    image=image,
    gpu="A10G",
    timeout=1200,
    volumes={"/root/.cache/huggingface": _hf_cache_volume},
    secrets=[_hf_secret()],
)
def run_probe(watchdog_seconds: float = 600) -> dict:
    import os as _os
    import signal as _signal
    import threading as _threading

    _watchdog = _threading.Timer(
        float(_os.environ.get("PARITY_WATCHDOG_SECONDS", str(watchdog_seconds))),
        lambda: (_os.kill(_os.getpid(), _signal.SIGKILL)),
    )
    _watchdog.daemon = True
    _watchdog.start()

    import numpy as np
    from onnx import TensorProto, helper
    import onnxruntime as ort

    print(f"[probe] ort {ort.__version__}", flush=True)
    node = helper.make_node("GatherND", ["data", "indices"], ["out"], batch_dims=0)
    graph = helper.make_graph(
        [node],
        "probe-gathernd",
        [
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 2]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 10
    opts = ort.SessionOptions()
    opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    opts.log_severity_level = 3
    result: dict = {}
    try:
        sess = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=opts,
            providers=["CUDAExecutionProvider"],
        )
        out = sess.run(
            None,
            {
                "data": np.ones((2, 2), np.float32),
                "indices": np.asarray([[0, 1]], np.int64),
            },
        )
        result["status"] = f"ok providers={sess.get_providers()} out={out}"
    except Exception as exc:  # noqa: BLE001 -- probe record
        result["status"] = f"FAIL {type(exc).__name__}: {str(exc)[:300]}"
    print(f"[probe] GatherND: {result['status']}", flush=True)
    _watchdog.cancel()
    return result


@app.local_entrypoint()
def main() -> None:
    result = run_probe.remote()
    print("\n=== RESULT ===")
    print(f"  {result['status']}")
