"""E2E benchmark across all supported VLA models: SmolVLA + pi0 + pi0.5.

For each model:
1. tether export <hf_id>
2. tether serve <export_dir> (background)
3. POST /act 10 times
4. Record latency percentiles

Produces a publishable benchmark table.

Usage:
    modal run scripts/modal_e2e_all_models.py
"""

import modal

app = modal.App("tether-e2e-all-models")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch", "safetensors", "huggingface_hub",
        "transformers>=4.51", "onnx", "onnxruntime",
        "onnxscript", "numpy", "Pillow",
        "typer", "rich", "pydantic>=2.0", "pyyaml",
        "fastapi", "uvicorn", "httpx",
    )
    .add_local_dir("src/tether", "/root/tether-vla/src/tether", copy=True)
    .add_local_file("pyproject.toml", "/root/tether-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/tether-vla/README.md", copy=True)
    .run_commands("cd /root/tether-vla && pip install -e .")
)


@app.function(image=image, gpu="A100-40GB", timeout=2400, scaledown_window=60)
def benchmark_all():
    """Export and serve all 3 supported VLA models. Collect latency."""
    import json
    import os
    import subprocess
    import time
    import httpx

    models = [
        ("smolvla", "lerobot/smolvla_base"),
        ("pi0", "lerobot/pi0_base"),
        ("pi05", "lerobot/pi05_base"),
        ("gr00t", "nvidia/GR00T-N1.6-3B"),
    ]

    results = {}
    port_base = 8765

    for idx, (tag, hf_id) in enumerate(models):
        print(f"\n{'='*60}", flush=True)
        print(f"Model {idx+1}/3: {tag} ({hf_id})", flush=True)
        print(f"{'='*60}", flush=True)

        model_result = {"hf_id": hf_id, "steps": {}}
        export_dir = f"/tmp/reflex_{tag}_export"
        port = port_base + idx
        server_log = f"/tmp/reflex_{tag}_server.log"

        # --- Export ---
        start = time.time()
        r = subprocess.run([
            "tether", "export", hf_id,
            "--target", "desktop",
            "--output", export_dir,
        ], capture_output=True, text=True, timeout=600)
        export_s = time.time() - start

        if r.returncode != 0:
            model_result["steps"]["export"] = {"status": "fail", "detail": r.stderr[-300:]}
            results[tag] = model_result
            print(f"  FAIL export: {r.stderr[-300:]}", flush=True)
            continue

        # Parse validation from stdout
        max_diff = None
        for line in r.stdout.splitlines():
            if "max_diff" in line.lower():
                try:
                    # e.g. "Validation: PASS (max_diff=3.81e-06)"
                    max_diff = float(line.split("max_diff=")[1].rstrip(")"))
                except Exception:
                    pass

        onnx_path = os.path.join(export_dir, "expert_stack.onnx")
        onnx_mb = os.path.getsize(onnx_path) / 1e6 if os.path.exists(onnx_path) else 0
        data_path = onnx_path + ".data"
        data_mb = os.path.getsize(data_path) / 1e6 if os.path.exists(data_path) else 0

        model_result["steps"]["export"] = {
            "status": "pass",
            "seconds": round(export_s, 1),
            "onnx_mb": round(onnx_mb, 2),
            "weights_data_mb": round(data_mb, 1),
            "validation_max_diff": max_diff,
        }
        print(f"  PASS export: {export_s:.1f}s, onnx={onnx_mb:.1f}MB, data={data_mb:.1f}MB, max_diff={max_diff}", flush=True)

        # --- Parse config ---
        config_path = os.path.join(export_dir, "tether_config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        model_result["steps"]["config"] = {
            "model_type": cfg.get("model_type", "smolvla"),
            "action_dim": cfg.get("expert", {}).get("action_dim"),
            "num_layers": cfg.get("expert", {}).get("num_layers"),
            "total_params_m": cfg.get("expert", {}).get("total_params_m"),
        }

        # --- Serve in background ---
        log_fh = open(server_log, "wb")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        server_process = subprocess.Popen(
            ["tether", "serve", export_dir, "--port", str(port), "--host", "127.0.0.1", "--device", "cpu"],
            stdout=log_fh, stderr=subprocess.STDOUT,
            env=env,
        )

        # Wait for /health
        server_ready = False
        for i in range(60):
            time.sleep(1)
            if server_process.poll() is not None:
                break
            try:
                resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                if resp.status_code == 200 and resp.json().get("model_loaded"):
                    server_ready = True
                    print(f"  serve ready after {i+1}s", flush=True)
                    break
            except Exception:
                continue

        if not server_ready:
            log_fh.close()
            with open(server_log, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 500))
                tail = f.read().decode(errors="replace")
            model_result["steps"]["serve"] = {"status": "fail", "detail": tail[-300:]}
            server_process.kill()
            results[tag] = model_result
            print(f"  FAIL serve: {tail[-300:]}", flush=True)
            continue

        # --- Benchmark 10 /act requests ---
        latencies_ms = []
        for j in range(10):
            t0 = time.perf_counter()
            try:
                resp = httpx.post(
                    f"http://127.0.0.1:{port}/act",
                    json={"instruction": "pick up the cup", "state": [0.1]*6},
                    timeout=30.0,
                )
                dt = (time.perf_counter() - t0) * 1000
                if resp.status_code == 200:
                    latencies_ms.append(dt)
            except Exception as e:
                print(f"  request {j} failed: {e}", flush=True)

        if latencies_ms:
            latencies_ms.sort()
            n = len(latencies_ms)
            mean_ms = sum(latencies_ms) / n
            p50 = latencies_ms[n // 2]
            p95 = latencies_ms[min(n - 1, int(n * 0.95))]
            model_result["steps"]["serve"] = {
                "status": "pass",
                "requests": n,
                "mean_ms": round(mean_ms, 1),
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "hz": round(1000.0 / mean_ms, 1),
            }
            print(f"  PASS benchmark: {n} reqs, mean={mean_ms:.1f}ms p50={p50:.1f} p95={p95:.1f} ({1000.0/mean_ms:.1f}Hz)", flush=True)
        else:
            model_result["steps"]["serve"] = {"status": "fail", "detail": "no successful requests"}
            print(f"  FAIL benchmark: no successful requests", flush=True)

        # Cleanup
        server_process.kill()
        try:
            server_process.wait(timeout=5)
        except Exception:
            pass
        log_fh.close()

        # Free disk between models (checkpoints are big)
        subprocess.run(["rm", "-rf", export_dir], check=False)

        results[tag] = model_result

    # --- Summary ---
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Model':<10} {'Export':>8} {'max_diff':>12} {'Mean ms':>10} {'Hz':>8}", flush=True)
    for tag, r in results.items():
        exp = r["steps"].get("export", {})
        serve = r["steps"].get("serve", {})
        max_diff = exp.get("validation_max_diff")
        max_diff_s = f"{max_diff:.2e}" if max_diff is not None else "—"
        exp_s = f"{exp.get('seconds', '—')}s"
        mean = serve.get("mean_ms", "—")
        hz = serve.get("hz", "—")
        print(f"{tag:<10} {exp_s:>8} {max_diff_s:>12} {str(mean):>10} {str(hz):>8}", flush=True)

    return results


@app.local_entrypoint()
def main():
    print("Running E2E benchmark across all supported VLA models on Modal A100...\n")
    results = benchmark_all.remote()

    # Dump as JSON for external use
    import json
    print("\nFull results:")
    print(json.dumps(results, indent=2))
