"""End-to-end demo: tether export → tether serve → POST /act → actions.

This is the full user flow, running as one script on Modal A100.

Usage:
    modal run scripts/modal_e2e_demo.py
"""

import json
import os
import time

import modal

app = modal.App("tether-e2e-demo")

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


@app.function(image=image, gpu="A100-40GB", timeout=900, scaledown_window=60)
def run_e2e_demo():
    """Full flow: export SmolVLA, start server, send request, get actions."""
    import subprocess
    import httpx

    results = {"steps": []}

    def log(name, status, detail=""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else "FAIL"
        print(f"{tag}: {name} — {detail}", flush=True)

    export_dir = "/tmp/reflex_e2e_export"
    server_log = "/tmp/tether_server.log"

    def read_server_log(tail_bytes=2000):
        """Non-blocking read of server log file."""
        if os.path.exists(server_log):
            with open(server_log, "rb") as f:
                f.seek(0, 2)  # end
                size = f.tell()
                f.seek(max(0, size - tail_bytes))
                return f.read().decode(errors="replace")
        return ""

    # Step 1: Run tether export
    print("=== Step 1: tether export lerobot/smolvla_base ===", flush=True)
    start = time.time()
    r = subprocess.run([
        "tether", "export", "lerobot/smolvla_base",
        "--target", "desktop",
        "--output", export_dir,
    ], capture_output=True, text=True, timeout=300)
    elapsed = time.time() - start

    if r.returncode == 0:
        files = os.listdir(export_dir) if os.path.exists(export_dir) else []
        log("export", "pass", f"{elapsed:.1f}s, {len(files)} files: {files}")
    else:
        log("export", "fail", r.stderr[-300:])
        return results

    # Step 2: Verify export contents
    print("\n=== Step 2: Verify export ===", flush=True)
    onnx_path = os.path.join(export_dir, "expert_stack.onnx")
    config_path = os.path.join(export_dir, "tether_config.json")
    if os.path.exists(onnx_path) and os.path.exists(config_path):
        onnx_size = os.path.getsize(onnx_path) / 1e6
        log("verify_export", "pass", f"ONNX: {onnx_size:.1f}MB, config exists")
    else:
        log("verify_export", "fail", "Missing expected files")
        return results

    # Step 3: Start tether serve — redirect output to file to avoid pipe-buffer deadlock
    print("\n=== Step 3: Start tether serve ===", flush=True)
    log_fh = open(server_log, "wb")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    server_process = subprocess.Popen(
        ["tether", "serve", export_dir, "--port", "8765", "--host", "127.0.0.1", "--device", "cpu"],
        stdout=log_fh, stderr=subprocess.STDOUT,
        env=env,
    )

    # Wait for server to start
    server_ready = False
    last_log = ""
    for i in range(90):
        time.sleep(1)
        # Check if server process died
        if server_process.poll() is not None:
            log_fh.close()
            log("server_start", "fail", f"Process died (code={server_process.returncode}): {read_server_log()[-500:]}")
            return results

        # Try /health
        try:
            resp = httpx.get("http://127.0.0.1:8765/health", timeout=2.0)
            if resp.status_code == 200:
                health = resp.json()
                if health.get("model_loaded"):
                    server_ready = True
                    log("server_start", "pass", f"ready after {i+1}s, mode={health.get('inference_mode')}")
                    break
                else:
                    # Server up but model not loaded yet
                    if i % 5 == 0:
                        last_log = read_server_log(500)
                        print(f"  [{i+1}s] server responding but model not loaded. Log tail: {last_log[-200:]}", flush=True)
        except Exception:
            if i % 10 == 0:
                last_log = read_server_log(500)
                print(f"  [{i+1}s] waiting... log tail: {last_log[-200:]}", flush=True)

    if not server_ready:
        log_fh.close()
        log("server_start", "fail", f"Server not ready after 90s: {read_server_log()[-500:]}")
        server_process.kill()
        return results

    # Step 4: POST /act
    print("\n=== Step 4: POST /act ===", flush=True)
    try:
        resp = httpx.post(
            "http://127.0.0.1:8765/act",
            json={
                "instruction": "pick up the red cup",
                "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            },
            timeout=30.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            actions = data.get("actions", [])
            log("act", "pass",
                f"got {len(actions)} actions × {len(actions[0]) if actions else 0} dims, "
                f"latency={data.get('latency_ms')}ms ({data.get('hz')}Hz), "
                f"mode={data.get('inference_mode')}")
        else:
            log("act", "fail", f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        log("act", "fail", str(e)[:200])

    # Step 5: Benchmark 10 requests
    print("\n=== Step 5: Benchmark 10 /act requests ===", flush=True)
    try:
        latencies = []
        for i in range(10):
            start = time.perf_counter()
            resp = httpx.post(
                "http://127.0.0.1:8765/act",
                json={"instruction": "", "state": [0.0]*6},
                timeout=30.0,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            if resp.status_code == 200:
                latencies.append(elapsed_ms)

        if latencies:
            latencies.sort()
            avg = sum(latencies) / len(latencies)
            log("benchmark", "pass",
                f"mean={avg:.1f}ms, p50={latencies[len(latencies)//2]:.1f}ms, p95={latencies[-1]:.1f}ms, "
                f"{1000/avg:.1f}Hz end-to-end")
    except Exception as e:
        log("benchmark", "fail", str(e)[:200])

    # Step 6: Check /config endpoint
    print("\n=== Step 6: GET /config ===", flush=True)
    try:
        resp = httpx.get("http://127.0.0.1:8765/config", timeout=5.0)
        if resp.status_code == 200:
            config = resp.json()
            log("config", "pass", f"target={config.get('target')}, action_dim={config.get('expert',{}).get('action_dim')}")
    except Exception as e:
        log("config", "fail", str(e)[:200])

    # Cleanup
    server_process.kill()
    try:
        server_process.wait(timeout=5)
    except Exception:
        pass
    log_fh.close()

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    passed = sum(1 for s in results["steps"] if s["status"] == "pass")
    failed = sum(1 for s in results["steps"] if s["status"] == "fail")
    print(f"Passed: {passed}, Failed: {failed}", flush=True)
    results["summary"] = {"passed": passed, "failed": failed}
    return results


@app.local_entrypoint()
def main():
    print("Running end-to-end tether demo on Modal A100...")
    results = run_e2e_demo.remote()

    for step in results["steps"]:
        tag = "PASS" if step["status"] == "pass" else "FAIL"
        print(f"  {tag}: {step['step']} — {step['detail']}")
