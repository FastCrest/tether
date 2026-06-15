"""Test that the README quickstart install path actually works on a fresh box.

Spins up a clean Linux container, runs the EXACT command from README:
  pip install 'fastcrest-tether[serve,gpu] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether'
  tether export lerobot/smolvla_base --target desktop --output ./sv
  tether serve ./sv --port 8000 --device cuda
  curl -X POST http://localhost:8000/act ...

Validates the entire user journey from `pip install` to first action.
If this fails, the launch fails — fix before posting publicly.

Usage:
    modal run scripts/modal_verify_install_path.py
"""

import modal

app = modal.App("tether-install-path-verify")

# CLEAN base image — no Tether preinstalled, just python + git.
# We're testing what a real user gets when they run the README pip command.
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/tensorrt:24.10-py3",
        add_python="3.12",
    )
    .apt_install("git", "curl")
    # NOTE: we do NOT pre-install tether. Test simulates a fresh box.
)


@app.function(image=image, gpu="A10G", timeout=900)
def test_fresh_install():
    import subprocess
    import time

    results = []

    def step(name, cmd, timeout=300, check=True):
        print(f"\n--- {name} ---", flush=True)
        print(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
        t0 = time.time()
        r = subprocess.run(
            cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        passed = (r.returncode == 0) if check else True
        results.append({
            "step": name, "exit_code": r.returncode, "elapsed_s": round(elapsed, 1),
            "passed": passed,
            "stdout_tail": r.stdout[-400:] if r.stdout else "",
            "stderr_tail": r.stderr[-400:] if r.stderr else "",
        })
        if not passed:
            print(f"  FAIL ({elapsed:.1f}s, exit {r.returncode})", flush=True)
            print(f"  stderr: {r.stderr[-500:]}", flush=True)
        else:
            print(f"  OK ({elapsed:.1f}s)", flush=True)
        return r

    # Step 1: pip install with the EXACT README command
    step(
        "1. pip install fastcrest-tether[serve,gpu] from git",
        "pip install 'fastcrest-tether[serve,gpu] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether'",
        timeout=600,
    )

    # Step 2: tether --help
    step("2. tether --help", ["tether", "--help"], timeout=30)

    # Step 3: tether models
    step("3. tether models", ["tether", "models"], timeout=30)

    # Step 4: tether export smolvla (smallest model for fastest test)
    step(
        "4. tether export lerobot/smolvla_base",
        ["tether", "export", "lerobot/smolvla_base", "--target", "desktop",
         "--output", "/tmp/sv"],
        timeout=600,
    )

    # Step 5: verify export contents
    import os
    files = os.listdir("/tmp/sv") if os.path.exists("/tmp/sv") else []
    results.append({
        "step": "5. export contents",
        "passed": "expert_stack.onnx" in files and "tether_config.json" in files,
        "elapsed_s": 0,
        "exit_code": 0,
        "stdout_tail": f"files: {files}",
        "stderr_tail": "",
    })
    print(f"\n--- 5. verify export contents ---", flush=True)
    print(f"  files: {files}", flush=True)

    # Step 6: tether serve in background, wait for /health, POST /act
    print(f"\n--- 6. tether serve + POST /act ---", flush=True)
    serve_log = open("/tmp/serve.log", "wb")
    serve = subprocess.Popen(
        ["tether", "serve", "/tmp/sv", "--port", "8765",
         "--host", "127.0.0.1", "--device", "cuda"],
        stdout=serve_log, stderr=subprocess.STDOUT,
    )

    # Use stdlib urllib so we don't depend on a pip-installed test client
    # (the whole point of this test is to verify what users get with the
    # advertised install path).
    import json as _json
    import urllib.request

    def _get_json(url, timeout=2.0):
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _json.loads(resp.read().decode())

    def _post_json(url, payload, timeout=30.0):
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _json.loads(resp.read().decode())

    # 240s — server now does a warmup pass during lifespan startup. With TRT
    # EP enabled, the first denoising loop also builds + caches a TRT engine,
    # which can take 30-90s for our smallest model and longer for pi0/gr00t.
    ready = False
    t0 = time.time()
    while time.time() - t0 < 240:
        if serve.poll() is not None:
            break
        try:
            status, body = _get_json("http://127.0.0.1:8765/health", timeout=2.0)
            if status == 200 and body.get("model_loaded"):
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    act_ok = False
    act_detail = ""
    if ready:
        try:
            _, data = _post_json(
                "http://127.0.0.1:8765/act",
                {"instruction": "reach", "state": [0.0]*6},
                timeout=30.0,
            )
            actions = data.get("actions", [])
            act_ok = len(actions) > 0
            act_detail = (
                f"{len(actions)} actions × {len(actions[0]) if actions else 0} dims, "
                f"latency={data.get('latency_ms')}ms, mode={data.get('inference_mode')}"
            )
        except Exception as e:
            act_detail = f"POST failed: {e}"

    serve.terminate()
    try:
        serve.wait(timeout=5)
    except subprocess.TimeoutExpired:
        serve.kill()
        serve.wait(timeout=5)
    serve_log.close()

    with open("/tmp/serve.log") as f:
        serve_log_tail = f.read()[-1500:]

    results.append({
        "step": "6. serve + POST /act",
        "passed": ready and act_ok,
        "elapsed_s": 0,
        "exit_code": serve.returncode,
        "stdout_tail": f"ready={ready}, act_ok={act_ok}, detail={act_detail}\n\n--- server log ---\n{serve_log_tail}"[-1500:],
        "stderr_tail": "",
    })
    print(f"  ready={ready}, act_ok={act_ok}", flush=True)
    print(f"  detail: {act_detail}", flush=True)

    # Verdict
    print(f"\n{'='*70}", flush=True)
    print("VERDICT — quickstart install path", flush=True)
    print(f"{'='*70}", flush=True)
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        elapsed = f"{r['elapsed_s']}s" if r["elapsed_s"] else ""
        print(f"  [{status}] {r['step']:<50} {elapsed:>8}", flush=True)
    print(f"\nOverall: {n_pass}/{n_total}", flush=True)
    return {"passed": n_pass, "total": n_total, "results": results}


@app.local_entrypoint()
def main():
    print("Verifying README quickstart install path on fresh box\n")
    r = test_fresh_install.remote()
    if r["passed"] < r["total"]:
        print("\nFAILED steps:")
        for s in r["results"]:
            if not s["passed"]:
                print(f"  - {s['step']}")
                print(f"    stderr: {s.get('stderr_tail','')[-300:]}")
                print(f"    stdout: {s.get('stdout_tail','')[-300:]}")
