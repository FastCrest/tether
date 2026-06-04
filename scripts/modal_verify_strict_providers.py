"""Phase I.1 verification: tether serve now fails loudly on silent CPU fallback.

Three scenarios tested:
  1. `onnxruntime-gpu` installed + CUDA 12 libs present + --device cuda → starts
  2. `onnxruntime-gpu` installed + CUDA 12 libs present + --device cpu → starts
  3. `onnxruntime` (CPU-only) installed + --device cuda (default) → exits with
     a clear error instead of silently running on CPU

Usage:
    modal run scripts/modal_verify_strict_providers.py
"""

import modal

app = modal.App("tether-strict-providers-verify")

# GPU-capable image with correctly pinned CUDA 12 / cuDNN 9 stack
image_gpu = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",  # torch 2.5.x uses CUDA 12.4
        "safetensors", "huggingface_hub",
        "transformers>=4.40,<5.0",
        "onnx", "onnxscript",
        "onnxruntime-gpu==1.20.1",
        "numpy<2.0", "Pillow",
        "typer", "rich", "pydantic>=2.0", "pyyaml",
        "fastapi", "uvicorn", "httpx",
    )
    .add_local_dir("src/tether", "/root/tether-vla/src/tether", copy=True)
    .add_local_file("pyproject.toml", "/root/tether-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/tether-vla/README.md", copy=True)
    .run_commands("cd /root/tether-vla && pip install -e . --no-deps")
)

# CPU-only image — installs `onnxruntime` (NOT the -gpu variant). Used to
# verify the CLI exits with a helpful error when a user pip-installed the
# wrong package.
image_cpu = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "safetensors", "huggingface_hub",
        "transformers>=4.40,<5.0",
        "onnx", "onnxscript",
        "onnxruntime==1.20.1",  # CPU-only!
        "numpy<2.0", "Pillow",
        "typer", "rich", "pydantic>=2.0", "pyyaml",
        "fastapi", "uvicorn", "httpx",
    )
    .add_local_dir("src/tether", "/root/tether-vla/src/tether", copy=True)
    .add_local_file("pyproject.toml", "/root/tether-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/tether-vla/README.md", copy=True)
    .run_commands("cd /root/tether-vla && pip install -e . --no-deps")
)


def _make_fake_export(export_dir: str):
    """Write a minimal tether_config.json + empty ONNX to let serve try to start."""
    import json
    import os

    os.makedirs(export_dir, exist_ok=True)
    # Minimal valid ONNX — just the magic bytes + a tiny trivial graph.
    # We don't need it to actually run; we just need `_load_onnx` to be reached.
    import onnx
    from onnx import helper, TensorProto
    input_tensor = helper.make_tensor_value_info(
        "noisy_actions", TensorProto.FLOAT, [1, 50, 32]
    )
    time_tensor = helper.make_tensor_value_info("timestep", TensorProto.FLOAT, [1])
    pos_tensor = helper.make_tensor_value_info("position_ids", TensorProto.INT64, [1, 50])
    output_tensor = helper.make_tensor_value_info(
        "velocity", TensorProto.FLOAT, [1, 50, 32]
    )
    identity = helper.make_node(
        "Identity", inputs=["noisy_actions"], outputs=["velocity"], name="identity"
    )
    graph = helper.make_graph(
        [identity],
        "fake",
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
            "action_dim": 32,
            "expert": {"action_dim": 32, "expert_hidden": 720},
        }, f)


def _wait_for_health(port: int, timeout_s: int = 60) -> tuple[bool, str]:
    """Poll /health until model_loaded OR timeout. Returns (ready, detail)."""
    import time
    import httpx

    t0 = time.time()
    last_err = ""
    while time.time() - t0 < timeout_s:
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("model_loaded"):
                    mode = data.get("inference_mode", "?")
                    return True, f"ready after {time.time()-t0:.1f}s (mode={mode})"
                last_err = f"not_ready: {data}"
        except Exception as e:
            last_err = str(e)[:100]
        time.sleep(0.5)
    return False, f"timeout after {timeout_s}s; last: {last_err}"


def _run_serve_with_health_check(port: int, extra_args: list[str], timeout_s: int = 60):
    """Launch tether serve, wait for /health, terminate cleanly. Returns dict."""
    import subprocess
    import time

    proc = subprocess.Popen(
        ["tether", "serve", "/tmp/fake_export", "--port", str(port),
         "--host", "127.0.0.1", *extra_args],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    # First make sure process is alive at all
    time.sleep(1)
    if proc.poll() is not None:
        # Already exited — capture output and return
        out, _ = proc.communicate(timeout=5)
        return {
            "exit_code": proc.returncode,
            "ready": False,
            "detail": "process exited before /health could be polled",
            "stdout_tail": out[-1500:],
        }

    ready, detail = _wait_for_health(port, timeout_s=timeout_s)

    # Clean shutdown (SIGTERM)
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate(timeout=5)

    return {
        "exit_code": proc.returncode,
        "ready": ready,
        "detail": detail,
        "stdout_tail": out[-1500:],
    }


@app.function(image=image_gpu, gpu="A10G", timeout=300)
def scenario_gpu_ok():
    """onnxruntime-gpu present, CUDA libs present → server should load on CUDA."""
    _make_fake_export("/tmp/fake_export")

    import onnxruntime as ort
    print(f"ORT version: {ort.__version__}", flush=True)
    print(f"Available providers: {ort.get_available_providers()}", flush=True)

    result = _run_serve_with_health_check(9001, ["--device", "cuda"], timeout_s=60)
    return {
        "scenario": "gpu_ok",
        **result,
        "expected": "server starts and /health returns ready on CUDA",
    }


@app.function(image=image_gpu, gpu="A10G", timeout=300)
def scenario_gpu_cpu_flag():
    """--device cpu should start cleanly even on a GPU box."""
    _make_fake_export("/tmp/fake_export")
    result = _run_serve_with_health_check(9002, ["--device", "cpu"], timeout_s=60)
    return {
        "scenario": "gpu_cpu_flag",
        **result,
        "expected": "server starts and /health returns ready on CPU",
    }


@app.function(image=image_cpu, gpu="A10G", timeout=300)
def scenario_cpu_only_silent_fallback_blocked():
    """CPU-only onnxruntime + --device cuda → should exit non-zero with helpful error."""
    import subprocess
    import time
    _make_fake_export("/tmp/fake_export")

    import onnxruntime as ort
    print(f"ORT version: {ort.__version__}", flush=True)
    print(f"Available providers: {ort.get_available_providers()}", flush=True)
    print(f"Expected: CUDAExecutionProvider NOT in the list.", flush=True)

    # Default device=cuda should now exit with code 1 before even starting
    proc = subprocess.run(
        ["tether", "serve", "/tmp/fake_export", "--port", "9003",
         "--host", "127.0.0.1"],  # default --device cuda
        capture_output=True, text=True, timeout=20,
    )
    return {
        "scenario": "cpu_only_silent_fallback_blocked",
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-1500:],
        "stderr_tail": proc.stderr[-1500:],
        "expected": "exit code 1 with message about CUDAExecutionProvider not available",
    }


@app.function(image=image_cpu, gpu="A10G", timeout=300)
def scenario_cpu_only_with_no_strict_flag():
    """--no-strict-providers on CPU-only box should allow fallback."""
    _make_fake_export("/tmp/fake_export")
    result = _run_serve_with_health_check(
        9004, ["--device", "cuda", "--no-strict-providers"], timeout_s=30,
    )
    return {
        "scenario": "cpu_only_with_no_strict_flag",
        **result,
        "expected": "server starts despite CUDA unavailable (best-effort fallback to CPU)",
    }


@app.local_entrypoint()
def main():
    print("Phase I.1 verification — tether serve strict provider mode\n")
    print("=" * 70)

    print("\n[1/4] GPU box + onnxruntime-gpu + --device cuda (should START)")
    r1 = scenario_gpu_ok.remote()
    print(f"  exit={r1['exit_code']}")
    print(f"  expected: {r1['expected']}")

    print("\n[2/4] GPU box + --device cpu (should START)")
    r2 = scenario_gpu_cpu_flag.remote()
    print(f"  exit={r2['exit_code']}")
    print(f"  expected: {r2['expected']}")

    print("\n[3/4] CPU-only ORT + --device cuda default (should EXIT 1)")
    r3 = scenario_cpu_only_silent_fallback_blocked.remote()
    print(f"  exit={r3['exit_code']}")
    print(f"  expected: {r3['expected']}")
    print(f"  stdout tail: {r3['stdout_tail'][-600:]}")

    print("\n[4/4] CPU-only ORT + --device cuda --no-strict-providers (should START)")
    r4 = scenario_cpu_only_with_no_strict_flag.remote()
    print(f"  exit={r4['exit_code']}")
    print(f"  expected: {r4['expected']}")

    # Verdict — scenarios 1, 2, 4 PASS if /health returned ready.
    # Scenario 3 PASS if exit code is 1 (refused to start).
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    passes = 0
    total = 4

    if r1.get("ready"):
        print(f"  1 PASS: server ready on GPU — {r1.get('detail')}")
        passes += 1
    else:
        print(f"  1 FAIL: ready={r1.get('ready')}, detail={r1.get('detail')}, "
              f"exit={r1.get('exit_code')}")
        print(f"  stdout tail:\n{r1.get('stdout_tail', '')[-2000:]}")

    if r2.get("ready"):
        print(f"  2 PASS: server ready on CPU — {r2.get('detail')}")
        passes += 1
    else:
        print(f"  2 FAIL: ready={r2.get('ready')}, detail={r2.get('detail')}, "
              f"exit={r2.get('exit_code')}")

    if r3["exit_code"] == 1:
        print(f"  3 PASS: refused to silently fall back to CPU (exit 1)")
        passes += 1
    else:
        print(f"  3 FAIL (expected exit 1, got {r3['exit_code']})")

    if r4.get("ready"):
        print(f"  4 PASS: --no-strict-providers allowed fallback — {r4.get('detail')}")
        passes += 1
    else:
        print(f"  4 FAIL: ready={r4.get('ready')}, detail={r4.get('detail')}, "
              f"exit={r4.get('exit_code')}")

    print(f"\nOverall: {passes}/{total}")
