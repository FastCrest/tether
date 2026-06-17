"""Test GR00T full-stack export (raw actions in, raw actions out).

Exercises the full stack: action_encoder → DiT → action_decoder, pinned
to embodiment 0. Validates that `tether export` produces an ONNX that
`tether serve` can drive with its standard flow-matching loop.

Usage:
    modal run scripts/modal_test_gr00t_full.py
"""

import modal

app = modal.App("tether-gr00t-full-test")

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


@app.function(image=image, gpu="A100-40GB", timeout=1800, scaledown_window=60)
def run_full():
    import os
    import subprocess
    import time
    import httpx

    results = {"steps": []}

    def log(name, status, detail=""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else "FAIL"
        print(f"{tag}: {name} — {detail}", flush=True)

    export_dir = "/tmp/tether_gr00t_full_export"
    server_log = "/tmp/tether_gr00t_full_server.log"

    # Step 1: Build full stack
    print("=== Step 1: Build GR00T full stack ===", flush=True)
    start = time.time()
    try:
        from tether.checkpoint import load_checkpoint
        from tether.exporters.gr00t import build_gr00t_full_stack
        state_dict, _ = load_checkpoint("nvidia/GR00T-N1.6-3B")
        full, meta = build_gr00t_full_stack(state_dict, embodiment_id=0)
        elapsed = time.time() - start
        log("build", "pass",
            f"{elapsed:.1f}s, raw_action_dim={meta['raw_action_dim']}, "
            f"params={meta['total_params_m']:.1f}M, "
            f"full_params={meta['full_stack_params_m']:.1f}M, "
            f"full_buffers={meta['full_stack_buffers_m']:.1f}M")
    except Exception as e:
        import traceback
        log("build", "fail", f"{str(e)[:300]}\n{traceback.format_exc()[:500]}")
        return results

    # Step 2: PyTorch forward
    print("\n=== Step 2: PyTorch forward (raw actions in/out) ===", flush=True)
    try:
        import torch
        dummy_actions = torch.randn(1, 50, meta["raw_action_dim"])
        dummy_time = torch.tensor([0.5])
        dummy_pos = torch.arange(50).unsqueeze(0)
        with torch.no_grad():
            out = full(dummy_actions, dummy_time, dummy_pos)
        log("forward", "pass",
            f"output shape={tuple(out.shape)} (expected (1, 50, {meta['raw_action_dim']})), "
            f"mean={out.mean().item():.4f}, std={out.std().item():.4f}")
    except Exception as e:
        import traceback
        log("forward", "fail", f"{str(e)[:300]}\n{traceback.format_exc()[:500]}")
        return results

    # Step 3: tether export (CLI, uses export_gr00t_full now)
    print("\n=== Step 3: tether export nvidia/GR00T-N1.6-3B ===", flush=True)
    del state_dict
    del full
    start = time.time()
    r = subprocess.run([
        "tether", "export", "nvidia/GR00T-N1.6-3B",
        "--target", "desktop",
        "--output", export_dir,
    ], capture_output=True, text=True, timeout=900)
    elapsed = time.time() - start
    if r.returncode == 0:
        files = os.listdir(export_dir)
        log("export", "pass", f"{elapsed:.1f}s, files: {files}")
        for line in r.stdout.splitlines():
            if "Validation" in line or "max_diff" in line:
                print(f"  {line}", flush=True)
    else:
        # Capture BOTH stderr + stdout — see modal_test_gr00t.py note
        # for the 2026-05-22 stdout-only error-path bug + fix.
        tail = (r.stderr or "")[-400:] + "\n---stdout tail---\n" + (r.stdout or "")[-600:]
        log("export", "fail", tail)
        return results

    # Step 4: tether serve
    print("\n=== Step 4: tether serve + POST /act ===", flush=True)
    log_fh = open(server_log, "wb")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    server_process = subprocess.Popen(
        ["tether", "serve", export_dir, "--port", "8799", "--host", "127.0.0.1", "--device", "cpu"],
        stdout=log_fh, stderr=subprocess.STDOUT,
        env=env,
    )

    server_ready = False
    for i in range(60):
        time.sleep(1)
        if server_process.poll() is not None:
            break
        try:
            resp = httpx.get("http://127.0.0.1:8799/health", timeout=2.0)
            if resp.status_code == 200 and resp.json().get("model_loaded"):
                server_ready = True
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
        log("serve_start", "fail", tail[-400:])
        server_process.kill()
        return results

    log("serve_start", "pass", f"ready after {i+1}s")

    try:
        resp = httpx.post(
            "http://127.0.0.1:8799/act",
            json={"instruction": "reach", "state": [0.0]*7},
            timeout=60.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            actions = data.get("actions", [])
            log("act", "pass",
                f"{len(actions)} actions × {len(actions[0]) if actions else 0} dims, "
                f"latency={data.get('latency_ms')}ms")
        else:
            log("act", "fail", f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        log("act", "fail", str(e)[:200])

    server_process.kill()
    try:
        server_process.wait(timeout=5)
    except Exception:
        pass
    log_fh.close()

    print("\n=== SUMMARY ===", flush=True)
    passed = sum(1 for s in results["steps"] if s["status"] == "pass")
    failed = sum(1 for s in results["steps"] if s["status"] == "fail")
    print(f"Passed: {passed}, Failed: {failed}", flush=True)
    results["summary"] = {"passed": passed, "failed": failed}
    return results


@app.local_entrypoint()
def main():
    print("Testing GR00T full-stack export + serve on Modal A100...")
    results = run_full.remote()

    for step in results["steps"]:
        tag = "PASS" if step["status"] == "pass" else "FAIL"
        print(f"  {tag}: {step['step']} — {step['detail']}")
