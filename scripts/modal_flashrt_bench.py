"""Modal: build + bench FlashRT public Python version on L40S.

Rewritten 2026-05-27 for FlashRT's current HEAD — major changes since
the 2026-04-30 v3 attempt (all v1-v3 failed on FA2 build issues):

  - Package renamed from ``flash_vla`` → ``flash_rt``
  - CMake overhaul: GPU_ARCH auto-detection, FA2 slimming flags, SM89
    build tested by community + documented in docs/deployment_rtx4090.md
  - Pi0.5 SM89 path confirmed working with FP8 via cuBLASLt
  - Python API unchanged: ``flash_rt.load_model()`` + ``model.predict()``

Uses CUDA 12.5 devel image + torch 2.5.1 (matches our prior Modal stack
for FlashRT; their docs reference CUDA 13 / torch 2.9 in NGC containers
but the cmake build supports CUDA 12.x fine).

Hardware: L40S (SM89, same compute capability as RTX 4090).
Cost: ~$5-10 (image build ~$3-5 first time, GPU run ~$2-3).

Usage:
    modal profile activate novarepmarketing
    modal run scripts/modal_flashrt_bench.py
"""
from __future__ import annotations

import os
import modal

app = modal.App("flashrt-bench-v4")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Bump this string to force a fresh git clone + rebuild of FlashRT HEAD.
_BUILD_BUST = "20260527-v3-paligemma-tokenizer"


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"
FLASHRT_DIR = "/opt/flashrt"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "wget", "ninja-build")
    .pip_install(
        "cmake>=3.24,<4",
        "torch==2.5.1",
        "safetensors>=0.4.0",
        "numpy<2.0",
        "pybind11>=2.12",
        "huggingface_hub>=0.20",
        "transformers>=4.40,<5.4",
        "sentencepiece",
        "Pillow",
        "ml_dtypes",
    )
    .run_commands("which cmake && cmake --version && gcc --version")
    .env({
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": f"{HF_CACHE}/transformers",
        "CUDACXX": "/usr/local/cuda/bin/nvcc",
        "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    })
    .run_commands(
        f'echo "build_bust={_BUILD_BUST}"',
        # Clone latest FlashRT HEAD + CUTLASS 4.4.2 (pinned per their docs).
        f"mkdir -p {FLASHRT_DIR} && cd {FLASHRT_DIR} && "
        "git clone --depth 1 https://github.com/LiangSu8899/FlashRT.git . && "
        "git clone --depth 1 --branch v4.4.2 "
        "  https://github.com/NVIDIA/cutlass.git third_party/cutlass && "
        "mkdir -p build && cd build && "
        # SM89 = L40S/4090 (Ada Lovelace). FA2 vendored in-SO, built for
        # sm_80 which runs natively on SM89. FA2_ARCH_NATIVE_ONLY skips
        # sm_120 PTX (requires CUDA 12.8+, we have 12.5).
        "cmake .. -GNinja "
        "  -DCMAKE_BUILD_TYPE=Release "
        "  -DCMAKE_CUDA_ARCHITECTURES=89 "
        "  -DGPU_ARCH=89 "
        "  -DENABLE_FA2=ON "
        "  -DFA2_ARCH_NATIVE_ONLY=ON "
        "  -DFA2_HDIMS='128;256' "
        "  -DFA2_DTYPES='bf16' && "
        "ninja -j$(nproc) && "
        # Skip `ninja install` — it has a path bug where it can't find
        # the .so that ninja already linked into flash_rt/. Instead,
        # manually copy from build/ per docs/deployment_rtx4090.md §4.
        f"cd {FLASHRT_DIR} && "
        "cp build/flash_rt_kernels.cpython-312-x86_64-linux-gnu.so "
        "   flash_rt/flash_rt_kernels.cpython-312-x86_64-linux-gnu.so 2>/dev/null || true && "
        "cp build/flash_rt_fa2.cpython-312-x86_64-linux-gnu.so "
        "   flash_rt/flash_rt_fa2.cpython-312-x86_64-linux-gnu.so 2>/dev/null || true && "
        "pip install -e '.[torch]' && "
        # FlashRT's Pi0.5 frontend needs the PaliGemma tokenizer .model
        # file which isn't bundled with HF checkpoints.
        "mkdir -p /root/.cache/flash_rt && "
        "curl -L https://storage.googleapis.com/big_vision/paligemma_tokenizer.model "
        "  -o /root/.cache/flash_rt/paligemma_tokenizer.model",
        gpu="L40S",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    volumes={HF_CACHE: hf_cache},
    secrets=[_hf_secret()],
    timeout=2400,
)
def bench(
    checkpoint_id: str = "lerobot/pi05_libero_finetuned_v044",
    benchmark_iters: int = 50,
    warmup_iters: int = 100,
    num_views: int = 2,
    prompt: str = "pick up the red block and place it in the tray",
):
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    print("=" * 60)
    print("FlashRT bench on L40S (SM89)")
    print(f"checkpoint: {checkpoint_id}")
    print(f"benchmark_iters: {benchmark_iters}, warmup: {warmup_iters}")
    print(f"num_views: {num_views}")
    print("=" * 60)

    # 1. Verify flash_rt is importable
    print("\n[1/4] verifying flash_rt import...", flush=True)
    try:
        import flash_rt
        print(f"  flash_rt version: {getattr(flash_rt, '__version__', '(none)')}")
        assert hasattr(flash_rt, "load_model"), "flash_rt.load_model missing"
        print("  flash_rt.load_model OK")
    except Exception as exc:
        return {"status": "fail", "stage": "import", "error": repr(exc)}

    # 2. Download checkpoint
    print(f"\n[2/4] downloading {checkpoint_id}...", flush=True)
    from huggingface_hub import snapshot_download
    try:
        ckpt_path = snapshot_download(repo_id=checkpoint_id, cache_dir=HF_CACHE)
        print(f"  checkpoint at: {ckpt_path}")
    except Exception as exc:
        return {"status": "fail", "stage": "checkpoint_download", "error": repr(exc)}

    # 3. Run quickstart benchmark
    print(f"\n[3/4] running quickstart.py --benchmark {benchmark_iters}...", flush=True)
    cmd = [
        sys.executable,
        f"{FLASHRT_DIR}/examples/quickstart.py",
        "--checkpoint", str(ckpt_path),
        "--framework", "torch",
        "--num_views", str(num_views),
        "--prompt", prompt,
        "--benchmark", str(benchmark_iters),
        "--warmup", str(warmup_iters),
        "--autotune", "3",
        "--config", "pi05",
        "--hardware", "rtx_sm89",
    ]
    print(f"  cmd: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env={**os.environ}, timeout=1800,
    )
    wall_s = time.perf_counter() - t0
    print(f"\n--- stdout ---")
    print(proc.stdout[-6000:] if len(proc.stdout) > 6000 else proc.stdout)
    if proc.returncode != 0:
        print(f"\n--- stderr ---")
        print(proc.stderr[-4000:])
        return {
            "status": "fail",
            "stage": "benchmark",
            "exit_code": proc.returncode,
            "stderr_tail": proc.stderr[-4000:],
            "stdout_tail": proc.stdout[-4000:],
            "wall_s": wall_s,
        }
    print(f"\n  wall: {wall_s:.1f}s, exit 0")

    # 4. Summary + comparison context
    print(f"\n[4/4] context")
    print(f"  Reflex Triton+Graph (A100, pi0.5): 51.0 ms (2.5x over PyTorch)")
    print(f"  Reflex cuda-graphs A/B (A100):     207.74 ms / chunk (1.30x)")
    print(f"  FlashRT published (4090):           community-tested SM89")
    print(f"  FlashRT published (5090):           17.58 ms Pi0.5 2-view")
    print(f"  FlashRT published (Thor):           44 ms / 39.78 ms NVFP4")
    print(f"  This run: L40S (SM89, datacenter Ada)")

    return {
        "status": "ok",
        "checkpoint": checkpoint_id,
        "wall_s": wall_s,
        "stdout_tail": proc.stdout[-6000:] if len(proc.stdout) > 6000 else proc.stdout,
    }


@app.local_entrypoint()
def main(
    checkpoint_id: str = "lerobot/pi05_libero_finetuned_v044",
    benchmark_iters: int = 50,
    warmup_iters: int = 100,
):
    print("FlashRT bench → L40S (SM89) → quickstart --benchmark")
    print(f"  checkpoint: {checkpoint_id}")
    print(f"  iters: {benchmark_iters} (warmup: {warmup_iters})")
    print()
    result = bench.remote(
        checkpoint_id=checkpoint_id,
        benchmark_iters=benchmark_iters,
        warmup_iters=warmup_iters,
    )
    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  status: {result.get('status')}")
    if result.get("status") == "ok":
        print(f"  wall:   {result.get('wall_s', '?'):.1f}s")
        print()
        print("Stdout tail:")
        print(result.get("stdout_tail", "(none)"))
    else:
        print(f"  stage:  {result.get('stage')}")
        print(f"  error:  {result.get('error', result.get('stderr_tail', '(none)'))}")
