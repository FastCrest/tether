"""Modal: convert SnapFlow student FP32 ONNX to FP16 + parity check.

Halves the 12.99 GB student export down toward Orin Nano's 8 GB RAM.
After conversion, runs both FP32 and FP16 ONNX on the same seeded
inputs and reports cos-sim + max_abs on the action output.

Usage:
    modal run scripts/modal_fp16_snapflow_student.py \\
      --fp32-dir /onnx_out/distill_v031_pi05_libero_r4/onnx_1nfe \\
      --fp16-dir /onnx_out/distill_v031_pi05_libero_r4/onnx_1nfe_fp16
"""
import os
import subprocess
import modal

app = modal.App("tether-fp16-snapflow-student")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


def _build_bust() -> str:
    import time
    return str(int(time.time()))


_HEAD = _repo_head_sha()
_BUST = _build_bust()

onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
ONNX_OUT = "/onnx_out"

# Lighter image — just ONNX tooling + tether for convert_fp32_to_fp16.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "onnx>=1.16",
        "onnxconverter-common>=1.14",
        "onnxruntime>=1.20",
        "numpy",
    )
    .run_commands(
        f'echo "build_bust={_BUST}"',
        f'pip install "tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    timeout=7200,
    volumes={ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def fp16_modal(fp32_dir: str, fp16_dir: str, seed: int = 7):
    """FP32 → FP16 conversion + parity check on seeded inputs."""
    import logging
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("fp16")

    fp32_path = Path(fp32_dir) / "model.onnx"
    if not fp32_path.exists():
        raise FileNotFoundError(f"FP32 ONNX missing: {fp32_path}")

    fp16_path = Path(fp16_dir) / "model.onnx"
    fp16_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: convert FP32 → FP16 via tether.exporters.fp16_convert ----
    from tether.exporters.fp16_convert import convert_fp32_to_fp16
    log.info("converting %s → %s", fp32_path, fp16_path)
    summary = convert_fp32_to_fp16(
        fp32_onnx_path=fp32_path,
        fp16_onnx_path=fp16_path,
    )
    log.info("size: %.2f GB → %.2f GB (%.1f%% reduction, %d casts inserted)",
             summary["src_bytes"] / 1e9,
             summary["dst_bytes"] / 1e9,
             summary["reduction_ratio"] * 100,
             summary["cast_nodes_inserted"])

    # ---- Step 2: build seeded inputs (same as FP32 parity run) ----
    np.random.seed(seed)
    B = 1
    chunk = 50
    action_dim = 32
    # NB: seed-based numpy gives us reproducibility ONLY within the
    # same run; to match the FP32 parity from the previous script we
    # would need the same torch.manual_seed path. Here we just check
    # FP32 ONNX output vs FP16 ONNX output on the same seeded inputs
    # — that's the quantization-impact question anyway.
    rng = np.random.default_rng(seed)
    inputs = {
        "img_base": rng.standard_normal((B, 3, 224, 224)).astype(np.float32),
        "img_wrist_l": rng.standard_normal((B, 3, 224, 224)).astype(np.float32),
        "img_wrist_r": rng.standard_normal((B, 3, 224, 224)).astype(np.float32),
        "mask_base": np.ones((B,), dtype=bool),
        "mask_wrist_l": np.ones((B,), dtype=bool),
        "mask_wrist_r": np.ones((B,), dtype=bool),
        "lang_tokens": rng.integers(0, 257152, (B, 16), dtype=np.int64),
        "lang_masks": np.ones((B, 16), dtype=bool),
        "noise": rng.standard_normal((B, chunk, action_dim)).astype(np.float32),
    }

    # ---- Step 3: run FP32 reference ----
    log.info("running FP32 reference ONNX ...")
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    actions_fp32 = sess_fp32.run(["actions"], inputs)[0]
    log.info("FP32 output shape: %s dtype: %s", actions_fp32.shape, actions_fp32.dtype)
    del sess_fp32

    # ---- Step 4: run FP16 ----
    # keep_io_types was flipped to False in convert_fp32_to_fp16 for oversized
    # models — inputs/outputs are FP16 on the converted graph. Cast our FP32
    # inputs to FP16 before passing to FP16 session, and cast output back.
    import onnx
    fp16_model = onnx.load(str(fp16_path), load_external_data=False)
    fp16_inputs_meta = {i.name: i.type.tensor_type.elem_type for i in fp16_model.graph.input}
    _FP16_ID = 10  # TensorProto.FLOAT16
    inputs_fp16 = {}
    for name, val in inputs.items():
        target_id = fp16_inputs_meta.get(name)
        if target_id == _FP16_ID and val.dtype == np.float32:
            inputs_fp16[name] = val.astype(np.float16)
        else:
            inputs_fp16[name] = val

    log.info("running FP16 ONNX ...")
    sess_fp16 = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])
    actions_fp16 = sess_fp16.run(["actions"], inputs_fp16)[0]
    log.info("FP16 output shape: %s dtype: %s", actions_fp16.shape, actions_fp16.dtype)

    # Cast FP16 output back to FP32 for comparison
    if actions_fp16.dtype != np.float32:
        actions_fp16_f32 = actions_fp16.astype(np.float32)
    else:
        actions_fp16_f32 = actions_fp16

    # ---- Step 5: parity ----
    from tether.exporters.fp16_convert import parity_gate
    assert actions_fp32.shape == actions_fp16_f32.shape
    diff = actions_fp32.astype(np.float64) - actions_fp16_f32.astype(np.float64)
    max_abs = float(np.abs(diff).max())
    mean_abs = float(np.abs(diff).mean())
    a = actions_fp32.reshape(-1).astype(np.float64)
    b = actions_fp16_f32.reshape(-1).astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    verdict = parity_gate(max_abs, cos)
    log.info("==== FP16 PARITY ====")
    log.info("  shape:     %s", actions_fp32.shape)
    log.info("  cos_sim:   %.10f", cos)
    log.info("  max_abs:   %.4e", max_abs)
    log.info("  mean_abs:  %.4e", mean_abs)
    log.info("  verdict:   %s", verdict["verdict"])
    if verdict["reasons"]:
        for r in verdict["reasons"]:
            log.info("  reason:    %s", r)
    log.info("  fp32 sample: %s", actions_fp32.flatten()[:5])
    log.info("  fp16 sample: %s", actions_fp16_f32.flatten()[:5])

    onnx_output.commit()

    return {
        "status": "ok",
        "fp32_size_gb": summary["src_bytes"] / 1e9,
        "fp16_size_gb": summary["dst_bytes"] / 1e9,
        "reduction_pct": summary["reduction_ratio"] * 100,
        "cast_nodes_inserted": summary["cast_nodes_inserted"],
        "cos_sim": cos,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "verdict": verdict["verdict"],
    }


@app.local_entrypoint()
def main(
    fp32_dir: str = "/onnx_out/distill_v031_pi05_libero_r4/onnx_1nfe",
    fp16_dir: str = "/onnx_out/distill_v031_pi05_libero_r4/onnx_1nfe_fp16",
    seed: int = 7,
):
    """
    --fp32-dir   Volume path to directory containing the FP32 model.onnx
                 + model.onnx.data (from export_snapflow_student_monolithic).
    --fp16-dir   Output volume path for the FP16 model.onnx + .bin.
    --seed       Seed for parity-check inputs. Default 7.
    """
    r = fp16_modal.remote(fp32_dir=fp32_dir, fp16_dir=fp16_dir, seed=seed)
    print("\n=== FP16 RESULT ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
