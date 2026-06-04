"""FP16 vs FP32 monolithic ONNX parity on shared seeded inputs.

Runs the SAME input batch through the FP32 ONNX and the FP16 ONNX,
compares per-action cos sim + max_abs diff. Uses `parity_gate()` from
`tether.exporters.fp16_convert` — PASS requires cos > 0.999 AND
max_abs < 5e-3 (the defaults for flow-matching VLAs where the fp16
quantization noise is expected to stay in that range).

Usage:
    modal run scripts/modal_fp16_parity.py \\
        --fp32-subdir smolvla_libero_monolithic \\
        --fp16-subdir smolvla_libero_monolithic_fp16 \\
        --model-kind smolvla
"""
import os
import subprocess
import modal

app = modal.App("tether-fp16-parity")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()


hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
ONNX_OUTPUT_PATH = "/onnx_out"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "numpy",
        "onnx>=1.16",
        "onnxruntime-gpu>=1.20",
    )
    .run_commands(
        f'pip install "tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether-vla@{_HEAD}"',
            secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def parity_modal(
    fp32_subdir: str = "smolvla_libero_monolithic",
    fp16_subdir: str = "smolvla_libero_monolithic_fp16",
    model_kind: str = "smolvla",
    seed: int = 42,
):
    """Run shared seeded inputs through FP32 and FP16 ONNX; report delta."""
    import time
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort

    from tether.exporters.fp16_convert import parity_gate

    fp32_path = Path(ONNX_OUTPUT_PATH) / fp32_subdir / "model.onnx"
    fp16_path = Path(ONNX_OUTPUT_PATH) / fp16_subdir / "model.onnx"
    if not fp32_path.exists():
        return {"status": "fail", "reason": f"{fp32_path} not found"}
    if not fp16_path.exists():
        return {"status": "fail", "reason": f"{fp16_path} not found"}

    providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    print(f"[parity] Loading FP32 ({fp32_path.stat().st_size / 1e9:.2f}GB)...")
    t0 = time.time()
    fp32_sess = ort.InferenceSession(str(fp32_path), providers=providers)
    print(f"[parity]   loaded in {time.time()-t0:.1f}s")

    print(f"[parity] Loading FP16 ({fp16_path.stat().st_size / 1e9:.2f}GB)...")
    t0 = time.time()
    fp16_sess = ort.InferenceSession(str(fp16_path), providers=providers)
    print(f"[parity]   loaded in {time.time()-t0:.1f}s")

    # Construct a shared seeded input batch matching the model's signature.
    np.random.seed(seed)
    if model_kind == "smolvla":
        # SmolVLAMonolithicWrapper inputs
        B = 1
        feed = {
            "img_cam1": np.random.randn(B, 3, 512, 512).astype(np.float32),
            "img_cam2": np.random.randn(B, 3, 512, 512).astype(np.float32),
            "img_cam3": np.random.randn(B, 3, 512, 512).astype(np.float32),
            "mask_cam1": np.ones(B, dtype=bool),
            "mask_cam2": np.ones(B, dtype=bool),
            "mask_cam3": np.ones(B, dtype=bool),
            "lang_tokens": np.random.randint(0, 49152, size=(B, 16)).astype(np.int64),
            "lang_masks": np.ones((B, 16), dtype=bool),
            "state": np.random.randn(B, 32).astype(np.float32),
            "noise": np.random.randn(B, 50, 32).astype(np.float32),
        }
    else:
        return {"status": "fail", "reason": f"unknown model_kind={model_kind}"}

    print("[parity] Running FP32...")
    t0 = time.time()
    fp32_out = fp32_sess.run(["actions"], feed)[0]
    print(f"[parity]   FP32 shape={fp32_out.shape} in {time.time()-t0:.2f}s")

    # FP16 session: for big models, the FP16 ONNX was exported with
    # keep_io_types=False (end-to-end FP16). Detect the expected input
    # dtype from the session metadata and cast as needed.
    fp16_feed = {}
    fp16_inputs = {i.name: i for i in fp16_sess.get_inputs()}
    for k, v in feed.items():
        meta = fp16_inputs.get(k)
        if meta is None:
            fp16_feed[k] = v
            continue
        expected = meta.type  # "tensor(float16)", "tensor(int64)", etc.
        if "float16" in expected and v.dtype == np.float32:
            fp16_feed[k] = v.astype(np.float16)
        else:
            fp16_feed[k] = v

    print("[parity] Running FP16...")
    t0 = time.time()
    fp16_out = fp16_sess.run(["actions"], fp16_feed)[0]
    print(f"[parity]   FP16 shape={fp16_out.shape} dtype={fp16_out.dtype} "
          f"in {time.time()-t0:.2f}s")

    # FP16 session outputs may be float16 if keep_io_types=True failed;
    # upcast for fair comparison.
    fp32_a = fp32_out.astype(np.float64).flatten()
    fp16_a = fp16_out.astype(np.float64).flatten()

    max_abs = float(np.abs(fp32_a - fp16_a).max())
    mean_abs = float(np.abs(fp32_a - fp16_a).mean())
    cos = float(
        np.dot(fp32_a, fp16_a) / (np.linalg.norm(fp32_a) * np.linalg.norm(fp16_a) + 1e-12)
    )

    verdict = parity_gate(max_abs_diff=max_abs, cos_sim=cos)
    print(f"[parity] cos={cos:+.6f}  max_abs={max_abs:.4e}  mean_abs={mean_abs:.4e}")
    print(f"[parity] VERDICT: {verdict['verdict']}")
    if verdict["reasons"]:
        for r in verdict["reasons"]:
            print(f"[parity]   - {r}")

    return {
        "status": "ok",
        "verdict": verdict["verdict"],
        "cos_sim": cos,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "first_action_fp32": fp32_out[0, 0, :7].tolist() if fp32_out.ndim >= 3 else None,
        "first_action_fp16": fp16_out[0, 0, :7].tolist() if fp16_out.ndim >= 3 else None,
    }


@app.local_entrypoint()
def main(
    fp32_subdir: str = "smolvla_libero_monolithic",
    fp16_subdir: str = "smolvla_libero_monolithic_fp16",
    model_kind: str = "smolvla",
):
    r = parity_modal.remote(
        fp32_subdir=fp32_subdir,
        fp16_subdir=fp16_subdir,
        model_kind=model_kind,
    )
    print("\n=== RESULT ===")
    for k, v in r.items():
        if isinstance(v, (int, float, str, bool, list)):
            print(f"  {k}: {v}")
