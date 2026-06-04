"""Verify the TRT FP16 + multi-batch interaction.

Known sharp edge: TRT engines compile against a specific input shape.
Our exporters bake static shapes (batch=1). When `tether serve --max-batch N`
fires a batched request, the TRT engine has to handle batch=N input.

ORT's TensorRT EP supports this in two ways:
  1. Falls back to CUDAExecutionProvider for shapes the engine doesn't have.
  2. Builds and caches a new engine per distinct shape (slow first hit).

This test runs `tether serve --max-batch 4` against pi0 and verifies:
  - The server starts cleanly (warmup builds engine for batch=1)
  - Concurrent batched requests (4 at a time) succeed
  - inference_mode reported in responses
  - Whether batched throughput improves over baseline

If batched mode silently falls back from TRT to CUDA EP, document it.
If batched mode fails entirely, that's a real bug — TRT auto-mode is
incompatible with --max-batch and we need to either:
  (a) Disable TRT EP when --max-batch > 1
  (b) Export with dynamic batch shape
  (c) Pre-build engines for common batch sizes

Usage:
    modal run scripts/modal_verify_trt_with_batch.py
"""

import modal

app = modal.App("tether-trt-batch-verify")

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/tensorrt:24.10-py3",
        add_python="3.12",
    )
    .apt_install("git")
)


@app.function(image=image, gpu="A10G", timeout=2400)
def test_trt_batch():
    import asyncio
    import json as _json
    import os
    import subprocess
    import time
    import urllib.request

    # Install tether from git
    print("=== pip install tether[serve,gpu] ===", flush=True)
    r = subprocess.run(
        ["pip", "install",
         "tether-vla[serve,gpu] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether-vla"],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        return {"error": "install failed", "stderr": r.stderr[-500:]}
    print("  install ok", flush=True)

    # Export pi0 — the model we benchmarked batching on earlier
    print("\n=== tether export lerobot/pi0_base ===", flush=True)
    t0 = time.time()
    r = subprocess.run(
        ["tether", "export", "lerobot/pi0_base", "--target", "desktop",
         "--output", "/tmp/pi0"],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        return {"error": "export failed", "stderr": r.stderr[-500:]}
    print(f"  export ok ({time.time()-t0:.1f}s)", flush=True)

    def _get_json(url, timeout=2.0):
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, _json.loads(resp.read().decode())

    def _post_json(url, payload, timeout=60.0):
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _json.loads(resp.read().decode())

    def wait_for_health(port, timeout_s=300):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                status, body = _get_json(f"http://127.0.0.1:{port}/health", timeout=2.0)
                if status == 200 and body.get("model_loaded"):
                    return body.get("inference_mode", "?")
            except Exception:
                pass
            time.sleep(0.5)
        return None

    async def hit_concurrent(url, n):
        loop = asyncio.get_event_loop()
        def _hit():
            try:
                status, body = _post_json(url, {"instruction": "reach", "state": [0.0]*6})
                return body
            except Exception as e:
                return {"error": str(e)[:200]}
        results = await asyncio.gather(
            *[loop.run_in_executor(None, _hit) for _ in range(n)]
        )
        return results

    def run_scenario(label, port, max_batch, n_concurrent):
        cmd = ["tether", "serve", "/tmp/pi0", "--port", str(port),
               "--host", "127.0.0.1", "--device", "cuda"]
        if max_batch > 1:
            cmd.extend(["--max-batch", str(max_batch),
                        "--batch-timeout-ms", "20"])
        log_path = f"/tmp/serve_{port}.log"
        log_fh = open(log_path, "wb")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)

        mode = wait_for_health(port, timeout_s=300)
        if mode is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            log_fh.close()
            with open(log_path) as f:
                tail = f.read()[-1500:]
            return {"label": label, "error": "server failed to start", "log_tail": tail}

        t0 = time.perf_counter()
        results = asyncio.run(
            hit_concurrent(f"http://127.0.0.1:{port}/act", n_concurrent),
        )
        elapsed = time.perf_counter() - t0
        n_ok = sum(1 for r in results if "actions" in r)
        sample = results[0] if results else {}

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log_fh.close()

        return {
            "label": label,
            "max_batch": max_batch,
            "n_concurrent": n_concurrent,
            "n_ok": n_ok,
            "wall_s": round(elapsed, 3),
            "qps": round(n_concurrent / elapsed, 1) if elapsed else 0,
            "startup_inference_mode": mode,
            "sample_response_mode": sample.get("inference_mode"),
            "sample_batch_size": sample.get("batch_size"),
            "sample_latency_ms": sample.get("latency_ms"),
            "errors": [r["error"] for r in results if "error" in r][:3],
        }

    print("\n=== Scenario 1: --max-batch 1 (baseline, TRT engine for batch=1) ===", flush=True)
    s1 = run_scenario("batch_1", 9400, max_batch=1, n_concurrent=8)
    print(f"  {s1}", flush=True)

    print("\n=== Scenario 2: --max-batch 4 (TRT now hits batch=4 input) ===", flush=True)
    s2 = run_scenario("batch_4", 9401, max_batch=4, n_concurrent=8)
    print(f"  {s2}", flush=True)

    print("\n=== Scenario 3: --max-batch 8 ===", flush=True)
    s3 = run_scenario("batch_8", 9402, max_batch=8, n_concurrent=8)
    print(f"  {s3}", flush=True)

    print(f"\n{'='*70}\nVERDICT — TRT FP16 + --max-batch interaction\n{'='*70}", flush=True)
    print(f"{'scenario':<12} {'max_batch':>10} {'n_ok':>6} {'qps':>8} {'mode':>22}", flush=True)
    for s in (s1, s2, s3):
        if "error" in s:
            print(f"{s['label']:<12} {s.get('max_batch','?'):>10} ERROR: {s['error']}", flush=True)
        else:
            mode = s.get("sample_response_mode") or s.get("startup_inference_mode") or "?"
            print(f"{s['label']:<12} {s['max_batch']:>10} "
                  f"{s['n_ok']}/{s['n_concurrent']:<5} {s['qps']:>8} {mode:>22}", flush=True)

    return {"baseline": s1, "batch_4": s2, "batch_8": s3}


@app.local_entrypoint()
def main():
    print("Verifying TRT FP16 + multi-batch interaction\n")
    r = test_trt_batch.remote()
    import json
    print("\n=== JSON ===")
    print(json.dumps(r, indent=2, default=str))
