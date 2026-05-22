"""Modal benchmark: ONNX (GPU) vs raw PyTorch vs torch.compile.

Answers two foundational questions:
1. Does our exported ONNX actually beat `torch.compile` eager PyTorch on the
   same forward pass? If no, reflex export has no reason to exist.
2. What's the actual memory footprint (weights + peak forward memory) for
   each model? Determines Jetson SKU feasibility (Orin Nano 8GB / Orin 32GB /
   Thor 128GB).

For each of the 4 flow-matching VLAs, measures per-denoising-step latency
at p50/p95/p99 over 100 trials after 10 warmups:
- PyTorch eager (FP32, CUDA)
- torch.compile(mode="reduce-overhead")
- ONNX Runtime CUDAExecutionProvider

Usage:
    modal run scripts/modal_bench_onnx_vs_torch.py
"""

import modal

app = modal.App("reflex-bench-onnx-vs-torch")

# onnxruntime-gpu ≥1.17 bundles its own CUDA libs, so we can pip install cleanly.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch", "safetensors", "huggingface_hub",
        "transformers>=4.51", "onnx", "onnxruntime-gpu",
        "onnxscript", "numpy", "Pillow",
        "typer", "rich", "pydantic>=2.0", "pyyaml",
    )
    .add_local_dir("src/reflex", "/root/reflex-vla/src/reflex", copy=True)
    .add_local_file("pyproject.toml", "/root/reflex-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/reflex-vla/README.md", copy=True)
    .run_commands("cd /root/reflex-vla && pip install -e .")
)


@app.function(image=image, gpu="A100-40GB", timeout=2400, scaledown_window=60)
def bench():
    import gc
    import os
    import subprocess
    import time
    import numpy as np
    import torch
    import onnxruntime as ort

    results = {}

    # ---- Models to benchmark ----
    models = [
        {
            "tag": "smolvla",
            "hf_id": "lerobot/smolvla_base",
            "builder_mod": "reflex.exporters.smolvla",
            "builder_fn": "build_expert_stack",
            "builder_kwargs": {"head_dim": 64},
        },
        {
            "tag": "pi0",
            "hf_id": "lerobot/pi0_base",
            "builder_mod": "reflex.exporters.pi0",
            "builder_fn": "build_pi0_expert_stack",
            "builder_kwargs": {"head_dim": 128},
        },
        {
            "tag": "pi05",
            "hf_id": "lerobot/pi05_base",
            "builder_mod": "reflex.exporters.pi0",
            "builder_fn": "build_pi05_expert_stack",
            "builder_kwargs": {"head_dim": 128},
        },
        {
            "tag": "gr00t",
            "hf_id": "nvidia/GR00T-N1.6-3B",
            "builder_mod": "reflex.exporters.gr00t",
            "builder_fn": "build_gr00t_full_stack",
            "builder_kwargs": {"embodiment_id": 0},
        },
    ]

    def bench_callable(fn, n_warmup=10, n_trials=100):
        """Time a callable, returning dict of stats in ms."""
        # Warmup
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()

        latencies = []
        for _ in range(n_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies.sort()
        n = len(latencies)
        return {
            "mean_ms": round(sum(latencies) / n, 3),
            "p50_ms": round(latencies[n // 2], 3),
            "p95_ms": round(latencies[min(n - 1, int(n * 0.95))], 3),
            "p99_ms": round(latencies[min(n - 1, int(n * 0.99))], 3),
            "min_ms": round(min(latencies), 3),
            "max_ms": round(max(latencies), 3),
        }

    for m in models:
        tag = m["tag"]
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {tag} ({m['hf_id']})", flush=True)
        print(f"{'='*60}", flush=True)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # ---- Load + build stack ----
        from reflex.checkpoint import load_checkpoint
        import importlib

        t0 = time.time()
        state_dict, _ = load_checkpoint(m["hf_id"])
        load_s = time.time() - t0

        builder_module = importlib.import_module(m["builder_mod"])
        builder = getattr(builder_module, m["builder_fn"])

        t0 = time.time()
        stack, meta = builder(state_dict, **m["builder_kwargs"])
        build_s = time.time() - t0

        # Move to GPU
        stack = stack.cuda()
        stack.eval()

        # Weights size on disk (approx — total param count × dtype)
        weights_bytes = sum(p.numel() for p in stack.parameters()) * 4  # FP32
        weights_gb = weights_bytes / 1e9

        # ---- Dummy inputs ----
        chunk_size = 50
        action_dim = meta.get("raw_action_dim", meta.get("action_dim", 32))
        dummy_actions = torch.randn(1, chunk_size, action_dim, device="cuda")
        dummy_time = torch.tensor([0.5], device="cuda")
        dummy_pos = torch.arange(chunk_size, device="cuda").unsqueeze(0)

        # ---- PyTorch eager (GPU) ----
        print(f"  [pytorch-eager]", flush=True)
        def _run_eager():
            with torch.no_grad():
                return stack(dummy_actions, dummy_time, dummy_pos)
        eager_stats = bench_callable(_run_eager)
        print(f"    {eager_stats}", flush=True)
        torch.cuda.synchronize()
        peak_mem_eager = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        # ---- torch.compile (reduce-overhead) ----
        print(f"  [torch-compile]", flush=True)
        try:
            compiled = torch.compile(stack, mode="reduce-overhead", fullgraph=False)
            def _run_compiled():
                with torch.no_grad():
                    return compiled(dummy_actions, dummy_time, dummy_pos)
            # Extra warmup for compile
            for _ in range(5):
                _run_compiled()
            compile_stats = bench_callable(_run_compiled)
            print(f"    {compile_stats}", flush=True)
        except Exception as e:
            compile_stats = {"error": str(e)[:200]}
            print(f"    FAILED: {e}", flush=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # ---- ONNX export + Runtime GPU ----
        print(f"  [onnxruntime-gpu]", flush=True)
        onnx_stats = None
        ort_mem_gb = None
        try:
            onnx_dir = f"/tmp/bench_{tag}"
            os.makedirs(onnx_dir, exist_ok=True)
            onnx_path = f"{onnx_dir}/expert.onnx"
            cpu_stack = stack.cpu()  # export needs CPU
            # Make CPU dummies for export
            cpu_actions = dummy_actions.cpu()
            cpu_time = dummy_time.cpu()
            cpu_pos = dummy_pos.cpu()
            with torch.no_grad():
                torch.onnx.export(
                    cpu_stack,
                    (cpu_actions, cpu_time, cpu_pos),
                    onnx_path,
                    input_names=["noisy_actions", "timestep", "position_ids"],
                    output_names=["velocity"],
                    dynamic_axes={"noisy_actions": {0: "batch"}},
                    opset_version=19,
                )
            # Move stack back to GPU for parity
            stack = stack.cuda()

            # Create ORT session on CUDA
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            session = ort.InferenceSession(onnx_path, providers=providers)
            active_providers = session.get_providers()
            print(f"    active providers: {active_providers}", flush=True)

            np_actions = dummy_actions.cpu().numpy()
            np_time = dummy_time.cpu().numpy()
            np_pos = dummy_pos.cpu().numpy().astype(np.int64)

            def _run_onnx():
                return session.run(None, {
                    "noisy_actions": np_actions,
                    "timestep": np_time,
                    "position_ids": np_pos,
                })[0]

            # Warmup
            for _ in range(10):
                _run_onnx()

            latencies = []
            for _ in range(100):
                t0 = time.perf_counter()
                _run_onnx()
                latencies.append((time.perf_counter() - t0) * 1000)
            latencies.sort()
            n = len(latencies)
            onnx_stats = {
                "mean_ms": round(sum(latencies) / n, 3),
                "p50_ms": round(latencies[n // 2], 3),
                "p95_ms": round(latencies[min(n - 1, int(n * 0.95))], 3),
                "p99_ms": round(latencies[min(n - 1, int(n * 0.99))], 3),
                "min_ms": round(min(latencies), 3),
                "max_ms": round(max(latencies), 3),
                "active_providers": active_providers,
            }
            print(f"    {onnx_stats}", flush=True)
        except Exception as e:
            import traceback
            onnx_stats = {"error": str(e)[:300], "trace": traceback.format_exc()[:500]}
            print(f"    FAILED: {e}", flush=True)

        results[tag] = {
            "hf_id": m["hf_id"],
            "load_s": round(load_s, 1),
            "build_s": round(build_s, 1),
            "chunk_size": chunk_size,
            "action_dim": action_dim,
            "params_m": round(meta.get("total_params_m", 0), 1),
            "full_stack_params_m": round(meta.get("full_stack_params_m", meta.get("total_params_m", 0)), 1),
            "weights_gb_fp32": round(weights_gb, 2),
            "weights_gb_fp16": round(weights_gb / 2, 2),
            "peak_gpu_gb_eager": round(peak_mem_eager, 2),
            "pytorch_eager": eager_stats,
            "torch_compile": compile_stats,
            "onnxruntime_gpu": onnx_stats,
        }

        # Free memory between models
        del stack, state_dict
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Summary table ----
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY: per-denoising-step latency (ms) on A100 / GPU", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Model':<8} {'Params':>8} {'FP32 GB':>8} {'Eager':>10} {'Compile':>10} {'ORT-GPU':>10}", flush=True)
    for tag, r in results.items():
        eager = r["pytorch_eager"].get("mean_ms", "—")
        compile_ = r["torch_compile"].get("mean_ms", "—") if isinstance(r["torch_compile"], dict) else "—"
        onnx = r["onnxruntime_gpu"].get("mean_ms", "—") if isinstance(r["onnxruntime_gpu"], dict) else "—"
        params = r.get("full_stack_params_m") or r.get("params_m")
        print(f"{tag:<8} {params:>7}M {r['weights_gb_fp32']:>7.2f} "
              f"{str(eager):>10} {str(compile_):>10} {str(onnx):>10}", flush=True)

    # ---- Jetson fit analysis ----
    print(f"\n{'='*80}", flush=True)
    print("JETSON FIT ANALYSIS (FP16 weights + 2x overhead)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Model':<8} {'FP16 GB':>8} {'+2x':>8} {'Nano 8GB':>10} {'Orin 32':>10} {'Thor 128':>10}", flush=True)
    for tag, r in results.items():
        fp16 = r["weights_gb_fp16"]
        with_overhead = fp16 * 2
        print(f"{tag:<8} {fp16:>7.2f} {with_overhead:>7.2f} "
              f"{'✓' if with_overhead < 8 else '✗':>10} "
              f"{'✓' if with_overhead < 32 else '✗':>10} "
              f"{'✓' if with_overhead < 128 else '✗':>10}", flush=True)

    return results


@app.local_entrypoint()
def main():
    print("Running ONNX-vs-PyTorch benchmark + memory analysis on Modal A100...")
    results = bench.remote()
    import json
    print("\n\n=== Full JSON ===")
    print(json.dumps(results, indent=2, default=str))
