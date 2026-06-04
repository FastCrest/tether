"""Phase III.2: real-model batching benchmark.

Tests `tether serve --max-batch N` with an actual VLA (pi0, ~50ms per chunk
on GPU). With a real compute load, batching should give 2-3x throughput at
batch=4 — the fake-Identity-op test only measured queue overhead.

Strategy:
  1. Export pi0 to ONNX (with dynamic batch axis from the existing exporter).
  2. Boot tether serve in three configs (batch=1, 4, 8) on A10G with ORT-GPU.
  3. Fire 32 concurrent /act requests at each, measure throughput.

Usage:
    modal run scripts/modal_verify_batching_real.py
"""

import modal

app = modal.App("tether-batching-real")

image = (
    # Use NVIDIA's TRT container so cuDNN 9 (including libcudnn_adv) is
    # already on the system path. The pip-installed nvidia-cudnn-cu12
    # wheel is missing libcudnn_adv.so.9 which ORT 1.20+ requires.
    modal.Image.from_registry(
        "nvcr.io/nvidia/tensorrt:24.10-py3",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "safetensors", "huggingface_hub",
        "transformers>=4.40,<5.0",
        "onnx", "onnxscript",
        "onnxruntime-gpu>=1.20,<1.24",
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


@app.function(image=image, gpu="A10G", timeout=2400, scaledown_window=60)
def test_real_batching():
    import asyncio
    import os
    import subprocess
    import time
    import httpx

    # Step 1: Export pi0 (smallest-but-real flow-matching VLA we have)
    print("=== Exporting pi0 (~3.5GB checkpoint) ===", flush=True)
    export_dir = "/tmp/pi0_export"
    t0 = time.time()
    r = subprocess.run(
        ["tether", "export", "lerobot/pi0_base", "--target", "desktop",
         "--output", export_dir],
        capture_output=True, text=True, timeout=600,
    )
    print(f"  export exit={r.returncode}, took {time.time()-t0:.1f}s", flush=True)
    if r.returncode != 0:
        return {"error": "export failed", "stderr": r.stderr[-500:]}

    async def _hit(client: httpx.AsyncClient, url: str) -> dict:
        r = await client.post(
            url,
            json={"instruction": "reach", "state": [0.0]*6},
            timeout=60.0,
        )
        return r.json()

    async def _hit_n(url: str, n: int) -> tuple[float, list[dict]]:
        async with httpx.AsyncClient() as client:
            t0 = time.perf_counter()
            results = await asyncio.gather(*[_hit(client, url) for _ in range(n)])
            elapsed = time.perf_counter() - t0
        return elapsed, results

    def _wait_for_health(port: int, timeout_s: int = 60) -> bool:
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
        cmd = ["tether", "serve", export_dir,
               "--port", str(port), "--host", "127.0.0.1",
               "--device", "cuda"]
        if max_batch > 1:
            cmd.extend(["--max-batch", str(max_batch),
                        "--batch-timeout-ms", "20"])
        # Use a real file for stdout — subprocess.PIPE deadlocks if the
        # process logs more than one OS pipe buffer (~64KB) before we read.
        stdout_path = f"/tmp/tether_serve_{port}.log"
        stdout_fh = open(stdout_path, "wb")
        proc = subprocess.Popen(
            cmd, stdout=stdout_fh, stderr=subprocess.STDOUT,
        )
        # 180s — pi0 ONNX is ~1.7GB; first ORT-GPU session creation can be slow.
        ready = _wait_for_health(port, 180)
        if not ready:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            stdout_fh.close()
            with open(stdout_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 2000))
                tail = f.read().decode(errors="replace")
            print(f"  STDOUT TAIL [{label}]:\n{tail}", flush=True)
            return {
                "label": label,
                "error": "server failed to start (180s timeout)",
                "stdout_tail": tail[-1500:],
                "exit_code": proc.returncode,
            }

        # 5 sequential warmups so the GPU is hot
        for _ in range(5):
            try:
                httpx.post(f"http://127.0.0.1:{port}/act",
                           json={"instruction": "warmup", "state": [0.0]*6},
                           timeout=60.0)
            except Exception:
                pass

        elapsed, results = asyncio.run(_hit_n(f"http://127.0.0.1:{port}/act", n_concurrent))

        # Sample fields
        first = results[0] if results else {}
        batch_size_seen = first.get("batch_size", 1)
        amortized = first.get("amortized_latency_ms", first.get("latency_ms"))
        per_request_lats = [r.get("amortized_latency_ms", r.get("latency_ms", 0)) for r in results]

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        stdout_fh.close()

        throughput_qps = n_concurrent / elapsed if elapsed > 0 else 0
        return {
            "label": label,
            "max_batch": max_batch,
            "n_concurrent": n_concurrent,
            "wall_time_s": round(elapsed, 3),
            "throughput_qps": round(throughput_qps, 1),
            "batch_size_seen": batch_size_seen,
            "amortized_latency_ms": amortized,
            "per_request_latency_avg_ms": round(sum(per_request_lats)/len(per_request_lats), 1) if per_request_lats else 0,
            "n_responses": len(results),
        }

    print("\n" + "=" * 70)
    print("Phase III.2 — real-model (pi0) batching, 32 concurrent /act")
    print("=" * 70)

    s1 = run_scenario("baseline-no-batch", 9300, max_batch=1, n_concurrent=32)
    print(f"\n[1] {s1}", flush=True)
    s2 = run_scenario("batch-4", 9301, max_batch=4, n_concurrent=32)
    print(f"\n[2] {s2}", flush=True)
    s3 = run_scenario("batch-8", 9302, max_batch=8, n_concurrent=32)
    print(f"\n[3] {s3}", flush=True)
    s4 = run_scenario("batch-16", 9303, max_batch=16, n_concurrent=32)
    print(f"\n[4] {s4}", flush=True)

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT — pi0 on A10G, 32 concurrent requests")
    print("=" * 70)
    print(f"{'scenario':<20} {'max_batch':>10} {'qps':>8} {'wall_s':>8} {'batch_seen':>12} {'speedup':>10}")
    base_qps = s1.get("throughput_qps", 1)
    for s in (s1, s2, s3, s4):
        qps = s.get("throughput_qps", 0)
        speedup = qps / base_qps if base_qps > 0 else 0
        print(f"{s['label']:<20} {s.get('max_batch',1):>10} "
              f"{qps:>8} {s.get('wall_time_s',0):>8} {s.get('batch_size_seen',1):>12}  "
              f"{speedup:.2f}×")

    return {"baseline": s1, "batch_4": s2, "batch_8": s3, "batch_16": s4}


@app.local_entrypoint()
def main():
    print("Phase III.2 — real-model batching benchmark on A10G\n")
    r = test_real_batching.remote()
    import json
    print("\n=== JSON ===")
    print(json.dumps(r, indent=2, default=str))
