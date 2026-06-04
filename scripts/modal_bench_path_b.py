"""Path B benchmark: can Tether actually beat torch.compile on GPU?

Tests four execution paths for each of 4 VLAs:
1. PyTorch eager (baseline)
2. torch.compile(mode="reduce-overhead") (the bar)
3. ORT-GPU (FIXED CUDA provider loading)
4. CUDA graph capture of full 10-step denoise loop (wedge 4 turbo)

Also measures FP32 weight memory and Jetson-fit implications.

Usage:
    modal run scripts/modal_bench_path_b.py
"""

import modal

app = modal.App("tether-bench-path-b")

# FIXED image: pin torch to a version using CUDA 12 (matches ORT 1.20),
# install matching cuDNN and cuBLAS.
# torch 2.5.1 ships with CUDA 12.4. onnxruntime-gpu 1.20.1 requires CUDA 12 + cuDNN 9.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        # Torch 2.5 uses CUDA 12.4 (NOT 13 like torch 2.11).
        "torch==2.5.1",
        "safetensors", "huggingface_hub",
        "transformers>=4.40,<5.0",
        "onnx", "onnxscript",
        # ORT 1.20.x uses cuDNN 9 + CUDA 12.x.
        "onnxruntime-gpu==1.20.1",
        "numpy<2.0", "Pillow",
        "typer", "rich", "pydantic>=2.0", "pyyaml",
    )
    .add_local_dir("src/tether", "/root/tether-vla/src/tether", copy=True)
    .add_local_file("pyproject.toml", "/root/tether-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/tether-vla/README.md", copy=True)
    .run_commands("cd /root/tether-vla && pip install -e . --no-deps")
)


@app.function(image=image, gpu="A100-40GB", timeout=2700, scaledown_window=60)
def bench():
    import gc
    import os
    import time
    import numpy as np
    import torch
    import onnxruntime as ort

    print(f"torch={torch.__version__}, cuda available={torch.cuda.is_available()}", flush=True)
    print(f"ORT={ort.__version__}, available providers={ort.get_available_providers()}", flush=True)

    results = {}

    models = [
        {
            "tag": "smolvla",
            "hf_id": "lerobot/smolvla_base",
            "builder_mod": "tether.exporters.smolvla_exporter",
            "builder_fn": "build_expert_stack",
            "builder_kwargs": {"head_dim": 64},
        },
        {
            "tag": "pi0",
            "hf_id": "lerobot/pi0_base",
            "builder_mod": "tether.exporters.pi0",
            "builder_fn": "build_pi0_expert_stack",
            "builder_kwargs": {"head_dim": 128},
        },
        {
            "tag": "pi05",
            "hf_id": "lerobot/pi05_base",
            "builder_mod": "tether.exporters.pi0",
            "builder_fn": "build_pi05_expert_stack",
            "builder_kwargs": {"head_dim": 128},
        },
        {
            "tag": "gr00t",
            "hf_id": "nvidia/GR00T-N1.6-3B",
            "builder_mod": "tether.exporters.gr00t",
            "builder_fn": "build_gr00t_full_stack",
            "builder_kwargs": {"embodiment_id": 0},
        },
    ]

    def bench_callable(fn, n_warmup=10, n_trials=50):
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
            "min_ms": round(min(latencies), 3),
            "max_ms": round(max(latencies), 3),
        }

    for m in models:
        tag = m["tag"]
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {tag}", flush=True)
        print(f"{'='*60}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

        # --- load + build ---
        from tether.checkpoint import load_checkpoint
        import importlib
        state_dict, _ = load_checkpoint(m["hf_id"])
        builder_module = importlib.import_module(m["builder_mod"])
        builder = getattr(builder_module, m["builder_fn"])
        stack, meta = builder(state_dict, **m["builder_kwargs"])
        stack = stack.cuda().eval()

        chunk_size = 50
        action_dim = meta.get("raw_action_dim", meta.get("action_dim", 32))
        num_steps = 10
        dt = -1.0 / num_steps

        noisy = torch.randn(1, chunk_size, action_dim, device="cuda")
        pos_ids = torch.arange(chunk_size, device="cuda").unsqueeze(0)

        # ============================================================
        # (a) torch.compile — single forward pass, the bar
        # ============================================================
        print("  [torch.compile single-pass]", flush=True)
        compile_single = torch.compile(stack, mode="reduce-overhead", fullgraph=False)

        def _single_compile():
            with torch.no_grad():
                t = torch.tensor([0.5], device="cuda")
                return compile_single(noisy, t, pos_ids)
        for _ in range(5):
            _single_compile()
        compile_single_stats = bench_callable(_single_compile)
        print(f"    {compile_single_stats}", flush=True)

        # ============================================================
        # (b) torch.compile — FULL LOOP in Python (the realistic baseline)
        # ============================================================
        print("  [torch.compile full 10-step loop, Python driver]", flush=True)
        def _full_loop_compile():
            with torch.no_grad():
                x = noisy.clone()
                for s in range(num_steps):
                    t = torch.tensor([1.0 + s * dt], device="cuda")
                    v = compile_single(x, t, pos_ids)
                    x = x + v * dt
                return x
        for _ in range(3):
            _full_loop_compile()
        compile_loop_stats = bench_callable(_full_loop_compile, n_warmup=5, n_trials=30)
        print(f"    {compile_loop_stats}", flush=True)

        # ============================================================
        # (c) ORT-GPU — verify CUDA provider actually loads now
        # ============================================================
        print("  [ONNX Runtime GPU]", flush=True)
        ort_stats = None
        try:
            import tempfile
            onnx_dir = tempfile.mkdtemp(prefix=f"bench_{tag}_")
            onnx_path = f"{onnx_dir}/expert.onnx"
            stack_cpu = stack.cpu()
            cpu_noisy = noisy.cpu()
            cpu_pos = pos_ids.cpu()
            cpu_t = torch.tensor([0.5])
            with torch.no_grad():
                torch.onnx.export(
                    stack_cpu,
                    (cpu_noisy, cpu_t, cpu_pos),
                    onnx_path,
                    input_names=["noisy_actions", "timestep", "position_ids"],
                    output_names=["velocity"],
                    dynamic_axes={"noisy_actions": {0: "batch"}},
                    opset_version=19,
                )
            stack = stack.cuda()

            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            session = ort.InferenceSession(onnx_path, providers=providers)
            active = session.get_providers()
            print(f"    active providers: {active}", flush=True)

            np_noisy = noisy.cpu().numpy()
            np_pos = pos_ids.cpu().numpy().astype(np.int64)
            np_t = np.array([0.5], dtype=np.float32)

            def _ort_single():
                return session.run(None, {
                    "noisy_actions": np_noisy,
                    "timestep": np_t,
                    "position_ids": np_pos,
                })[0]
            for _ in range(10):
                _ort_single()
            latencies = []
            for _ in range(50):
                t0 = time.perf_counter()
                _ort_single()
                latencies.append((time.perf_counter() - t0) * 1000)
            latencies.sort()
            n = len(latencies)
            ort_stats = {
                "mean_ms": round(sum(latencies) / n, 3),
                "p50_ms": round(latencies[n // 2], 3),
                "p95_ms": round(latencies[min(n - 1, int(n * 0.95))], 3),
                "active_providers": active,
            }
            print(f"    {ort_stats}", flush=True)
        except Exception as e:
            ort_stats = {"error": str(e)[:300]}
            print(f"    FAILED: {e}", flush=True)

        # ============================================================
        # (d) CUDA graph capture of full 10-step loop (Tether turbo)
        # ============================================================
        print("  [Tether turbo CUDA graph — full-loop capture]", flush=True)
        cuda_graph_stats = None
        try:
            from tether.kernels.turbo import TurboOptimizer, TurboConfig
            turbo = TurboOptimizer(TurboConfig(strategy="cuda_graph"))
            # First capture sets up the graph and returns initial timing
            initial = turbo.denoise_cuda_graph(stack, noisy, pos_ids, num_steps=num_steps)
            print(f"    capture latency: {initial.latency_ms:.3f}ms", flush=True)

            def _graph_replay():
                return turbo.replay_cuda_graph(noisy)
            for _ in range(10):
                _graph_replay()
            cuda_graph_stats = bench_callable(_graph_replay, n_warmup=10, n_trials=100)
            print(f"    {cuda_graph_stats}", flush=True)
        except Exception as e:
            import traceback
            cuda_graph_stats = {"error": str(e)[:400], "trace": traceback.format_exc()[:600]}
            print(f"    FAILED: {e}\n{traceback.format_exc()[:400]}", flush=True)

        weights_gb = sum(p.numel() for p in stack.parameters()) * 4 / 1e9

        results[tag] = {
            "hf_id": m["hf_id"],
            "action_dim": action_dim,
            "full_stack_params_m": round(meta.get("full_stack_params_m", meta.get("total_params_m", 0)), 1),
            "weights_gb_fp32": round(weights_gb, 2),
            "compile_single_step": compile_single_stats,
            "compile_full_loop_10_steps": compile_loop_stats,
            "ort_gpu_single_step": ort_stats,
            "tether_turbo_cuda_graph_10_steps": cuda_graph_stats,
        }

        # Free memory between models
        del stack, state_dict
        gc.collect()
        torch.cuda.empty_cache()

    # --- summary ---
    print(f"\n{'='*100}", flush=True)
    print("SUMMARY: per-CHUNK latency (ms) on A100 — 10 denoising steps, lower is better", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"{'Model':<10} {'Params':>10} "
          f"{'Compile×10':>14} {'ORT×10':>14} {'Tether CUDA-graph':>20}", flush=True)
    for tag, r in results.items():
        loop = r["compile_full_loop_10_steps"].get("mean_ms", "—")
        ort_single = r["ort_gpu_single_step"].get("mean_ms", "—") if isinstance(r["ort_gpu_single_step"], dict) and "mean_ms" in r["ort_gpu_single_step"] else "—"
        ort_loop = round(float(ort_single) * 10, 1) if isinstance(ort_single, (int, float)) else "—"
        cuda_g = r["tether_turbo_cuda_graph_10_steps"].get("mean_ms", "—") if isinstance(r["tether_turbo_cuda_graph_10_steps"], dict) and "mean_ms" in r["tether_turbo_cuda_graph_10_steps"] else "—"
        params_s = f"{r['full_stack_params_m']}M"
        print(f"{tag:<10} {params_s:>10} {str(loop):>14} {str(ort_loop):>14} {str(cuda_g):>20}", flush=True)

    return results


@app.local_entrypoint()
def main():
    print("Path B benchmark — CUDA graph vs torch.compile vs ORT-GPU")
    results = bench.remote()
    import json
    print("\n=== Full JSON ===")
    print(json.dumps(results, indent=2, default=str))
