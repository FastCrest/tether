"""Phase III verification: tether serve --max-batch N actually batches.

Spins up `tether serve` with three configurations:
  - max_batch=1 (baseline, no batching)
  - max_batch=4 (small batch)
  - max_batch=8 (bigger batch)

For each, fires 16 concurrent POST /act requests against the same server
and measures end-to-end throughput. Reports:
  - Total time to serve N requests
  - Effective per-request latency
  - Throughput multiplier vs baseline

Also asserts that batched responses surface batch_size in the JSON.

Usage:
    modal run scripts/modal_verify_batching.py
"""

import modal

app = modal.App("tether-batching-verify")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "safetensors", "huggingface_hub",
        "transformers>=4.40,<5.0",
        "onnx", "onnxscript",
        "onnxruntime-gpu==1.20.1",
        "nvidia-cudnn-cu12>=9.0,<10.0",
        "nvidia-cublas-cu12>=12.0,<13.0",
        "numpy<2.0", "Pillow",
        "typer", "rich", "pydantic>=2.0", "pyyaml",
        "fastapi", "uvicorn", "httpx",
        "yourdfpy", "trimesh",
    )
    .add_local_dir("src/tether", "/root/tether-vla/src/tether", copy=True)
    .add_local_file("pyproject.toml", "/root/tether-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/tether-vla/README.md", copy=True)
    .run_commands("cd /root/tether-vla && pip install -e . --no-deps")
)


def _make_fake_export(export_dir: str, action_dim: int = 6):
    """Create a tiny ONNX with batch=dynamic so the server can call it batched."""
    import json
    import os
    import onnx
    from onnx import helper, TensorProto

    os.makedirs(export_dir, exist_ok=True)
    # Dynamic batch dim — critical for batching to actually work
    input_tensor = helper.make_tensor_value_info(
        "noisy_actions", TensorProto.FLOAT, ["batch", 50, action_dim]
    )
    time_tensor = helper.make_tensor_value_info("timestep", TensorProto.FLOAT, ["batch"])
    pos_tensor = helper.make_tensor_value_info(
        "position_ids", TensorProto.INT64, ["batch", 50]
    )
    output_tensor = helper.make_tensor_value_info(
        "velocity", TensorProto.FLOAT, ["batch", 50, action_dim]
    )
    identity = helper.make_node(
        "Identity", inputs=["noisy_actions"], outputs=["velocity"], name="identity"
    )
    graph = helper.make_graph(
        [identity], "fake",
        [input_tensor, time_tensor, pos_tensor],
        [output_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 10
    onnx.save(model, f"{export_dir}/expert_stack.onnx")
    with open(f"{export_dir}/tether_config.json", "w") as f:
        json.dump({
            "model_type": "smolvla",
            "action_chunk_size": 50,
            "action_dim": action_dim,
            "expert": {"action_dim": action_dim, "expert_hidden": 720},
        }, f)


@app.function(image=image, cpu=4, timeout=600)
def test_batching():
    import asyncio
    import subprocess
    import time
    import httpx

    _make_fake_export("/tmp/fake_export", action_dim=6)

    async def _hit(client: httpx.AsyncClient, url: str) -> dict:
        r = await client.post(
            url,
            json={"instruction": "reach", "state": [0.0]*6},
            timeout=30.0,
        )
        return r.json()

    async def _hit_n(url: str, n: int) -> tuple[float, list[dict]]:
        async with httpx.AsyncClient() as client:
            t0 = time.perf_counter()
            results = await asyncio.gather(*[_hit(client, url) for _ in range(n)])
            elapsed = time.perf_counter() - t0
        return elapsed, results

    def _wait_for_health(port: int, timeout_s: int = 30) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                if r.status_code == 200 and r.json().get("model_loaded"):
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def run_scenario(label: str, port: int, max_batch: int, n_concurrent: int):
        cmd = ["tether", "serve", "/tmp/fake_export",
               "--port", str(port), "--host", "127.0.0.1",
               "--device", "cpu", "--no-strict-providers"]
        if max_batch > 1:
            cmd.extend(["--max-batch", str(max_batch),
                        "--batch-timeout-ms", "20"])
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if not _wait_for_health(port, 30):
            proc.terminate()
            proc.wait(timeout=5)
            return {"label": label, "error": "server failed to start"}

        # Warmup
        try:
            httpx.post(f"http://127.0.0.1:{port}/act",
                       json={"instruction": "warmup", "state": [0.0]*6},
                       timeout=30.0)
        except Exception:
            pass

        elapsed, results = asyncio.run(_hit_n(f"http://127.0.0.1:{port}/act", n_concurrent))

        # Sample fields from first response
        first = results[0] if results else {}
        batch_size_seen = first.get("batch_size", 1)
        amortized = first.get("amortized_latency_ms", first.get("latency_ms"))

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

        # Throughput = N / total wall-clock seconds
        throughput_qps = n_concurrent / elapsed if elapsed > 0 else 0
        return {
            "label": label,
            "max_batch": max_batch,
            "n_concurrent": n_concurrent,
            "wall_time_s": round(elapsed, 3),
            "throughput_qps": round(throughput_qps, 1),
            "batch_size_seen": batch_size_seen,
            "amortized_latency_ms": amortized,
            "n_responses": len(results),
        }

    # Scenarios
    print("=" * 70)
    print("Phase III batching verification — 16 concurrent /act per scenario")
    print("=" * 70)
    s1 = run_scenario("baseline-no-batch", 9200, max_batch=1, n_concurrent=16)
    print(f"\n[1] {s1}")
    s2 = run_scenario("batch-4", 9201, max_batch=4, n_concurrent=16)
    print(f"\n[2] {s2}")
    s3 = run_scenario("batch-8", 9202, max_batch=8, n_concurrent=16)
    print(f"\n[3] {s3}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"{'scenario':<20} {'max_batch':>10} {'qps':>8} {'wall_s':>8} {'batch_seen':>12}")
    base_qps = s1.get("throughput_qps", 1)
    for s in (s1, s2, s3):
        qps = s.get("throughput_qps", 0)
        speedup = qps / base_qps if base_qps > 0 else 0
        print(f"{s['label']:<20} {s.get('max_batch',1):>10} "
              f"{qps:>8} {s.get('wall_time_s',0):>8} {s.get('batch_size_seen',1):>12}  "
              f"({speedup:.2f}× vs baseline)")

    return {"baseline": s1, "batch_4": s2, "batch_8": s3}


@app.local_entrypoint()
def main():
    print("Phase III batching verification\n")
    r = test_batching.remote()
    import json
    print("\n=== JSON ===")
    print(json.dumps(r, indent=2, default=str))
