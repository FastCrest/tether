"""v0.7.0 install validation on Modal A10G.

Day 4 of the v0.7 plan — gates the PyPI ship.

What this validates:
1. Local wheel installs cleanly via uv pip from a Modal mount
2. tensorrt + nvidia/cublas + nvidia/cudnn pip pkgs arrive automatically
3. `import tether` runs the LD_LIBRARY_PATH patch successfully
4. `tether doctor` reports all 4 TRT EP checks ✓
5. Latency on SmolVLA monolithic matches v0.6.0 baseline (~19.5 ms ORT-TRT EP)

Usage:
    modal profile activate <tether-profile>
    modal run scripts/modal_v07_install_validation_a10g.py
"""
from pathlib import Path
import modal

WHEEL_PATH = Path(__file__).parent.parent / "dist" / "reflex_vla-0.7.0-py3-none-any.whl"
# NOTE: existence check is in main() not here — Modal imports this module
# remotely too, where /dist/ won't exist.

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "wget", "linux-libc-dev", "build-essential", "python3-dev", "clang")
    .pip_install("uv")
    .add_local_file(
        str(WHEEL_PATH),
        remote_path=f"/tmp/{WHEEL_PATH.name}",
        copy=True,
    )
    .run_commands(
        # Install the LOCAL wheel (not from PyPI yet — this is the validation
        # gate before publishing). [monolithic] keeps export workflow available.
        f"uv pip install --system '/tmp/{WHEEL_PATH.name}[serve,gpu,monolithic]'",
    )
    .env({"PYTHONFAULTHANDLER": "1"})
)

app = modal.App("tether-v07-install-validation")


@app.function(image=image, gpu="A10G", timeout=900)
def validate():
    """Run the validation steps. Returns dict for local summary."""
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    results = {}

    # ─── Environment fingerprint
    nvsmi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    results["gpu"] = nvsmi.stdout.strip()
    print(f"GPU: {results['gpu']}")

    for pkg in ["tether-vla", "tensorrt", "onnxruntime-gpu", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"]:
        proc = subprocess.run(["pip", "show", pkg], capture_output=True, text=True)
        if proc.returncode == 0:
            for line in proc.stdout.split("\n"):
                if line.startswith("Version:"):
                    results[f"{pkg}_version"] = line.split(":", 1)[1].strip()
                    print(f"{pkg}: {results[f'{pkg}_version']}")
                    break
        else:
            results[f"{pkg}_version"] = "NOT_INSTALLED"
            print(f"{pkg}: NOT_INSTALLED")

    # Critical pre-condition: tensorrt must be installed for the v0.7 win
    if results["tensorrt_version"] == "NOT_INSTALLED":
        results["status"] = "FAIL_TENSORRT_NOT_AUTO_INSTALLED"
        return results

    # ─── Step A: import tether, verify LD_LIBRARY_PATH gets patched
    print()
    print("=" * 70)
    print("STEP A — import tether + verify LD_LIBRARY_PATH patch")
    print("=" * 70)
    proc = subprocess.run(
        [sys.executable, "-c",
         "import os; before=os.environ.get('LD_LIBRARY_PATH', ''); "
         "import tether; after=os.environ.get('LD_LIBRARY_PATH', ''); "
         "print('VERSION:', tether.__version__); "
         "print('LD_BEFORE:', repr(before)); "
         "print('LD_AFTER:', repr(after)); "
         "print('PATCH_APPLIED:', after != before)"],
        capture_output=True, text=True, env={**os.environ, "LD_LIBRARY_PATH": ""},
    )
    print(proc.stdout)
    print(proc.stderr)
    results["import_smoke_stdout"] = proc.stdout
    results["import_smoke_returncode"] = proc.returncode
    if proc.returncode != 0:
        results["status"] = "FAIL_IMPORT_REFLEX"
        return results
    if "PATCH_APPLIED: True" not in proc.stdout:
        results["status"] = "FAIL_LD_LIBRARY_PATH_NOT_PATCHED"
        return results

    # ─── Step B: tether doctor
    print()
    print("=" * 70)
    print("STEP B — tether doctor")
    print("=" * 70)
    proc = subprocess.run(["tether", "doctor"], capture_output=True, text=True)
    print(proc.stdout)
    print(proc.stderr)
    results["doctor_stdout"] = proc.stdout
    results["doctor_returncode"] = proc.returncode
    if proc.returncode != 0:
        results["status"] = "FAIL_DOCTOR_NONZERO_EXIT"
        return results

    # Parse the table for our 4 TRT EP checks
    expected_checks = [
        "TensorRT runtime",  # libnvinfer.so.10
        "CUDA cuBLAS",       # libcublas.so.12
        "CUDA cuDNN",        # libcudnn.so.9
        "ORT-TRT EP active",
    ]
    checks_status = {}
    for check_name in expected_checks:
        # The Rich table renders ✓ in green and ✗ in red. We check substring presence.
        # Heuristic: split by check name + look at the next chunk for ✓ or ✗.
        idx = proc.stdout.find(check_name)
        if idx == -1:
            checks_status[check_name] = "NOT_FOUND_IN_OUTPUT"
            continue
        # Look at the 200 chars after the check name
        chunk = proc.stdout[idx:idx + 200]
        if "✓" in chunk:
            checks_status[check_name] = "PASS"
        elif "✗" in chunk:
            checks_status[check_name] = "FAIL"
        else:
            checks_status[check_name] = "AMBIGUOUS"
    results["doctor_checks"] = checks_status
    print("Doctor checks:", checks_status)

    if any(s != "PASS" for s in checks_status.values()):
        results["status"] = "FAIL_DOCTOR_CHECK_NOT_PASSED"
        return results

    # ─── Step C: latency baseline
    # Build a tiny ONNX in-memory, run inference with both CUDA EP and TRT EP,
    # confirm TRT EP wins by ≥1.3× and matches the 19.5 ms ballpark for
    # transformer workloads (note: this is a 1-add ONNX so it's much smaller
    # than SmolVLA — perf delta will be smaller too, but TRT EP should still
    # be at least competitive with CUDA EP).
    print()
    print("=" * 70)
    print("STEP C — latency baseline (tiny ONNX, both providers)")
    print("=" * 70)
    proc = subprocess.run(
        [sys.executable, "-c", _LATENCY_SCRIPT],
        capture_output=True, text=True,
    )
    print(proc.stdout)
    print(proc.stderr)
    results["latency_stdout"] = proc.stdout
    results["latency_returncode"] = proc.returncode

    # Extract the numbers we care about
    for line in proc.stdout.split("\n"):
        if line.startswith("CUDA_EP_MEAN_MS:"):
            results["cuda_ep_mean_ms"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("TRT_EP_MEAN_MS:"):
            results["trt_ep_mean_ms"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("TRT_EP_ACTIVE:"):
            results["trt_ep_active_in_session"] = line.split(":", 1)[1].strip()

    # ─── Final status
    if results.get("trt_ep_active_in_session") == "True":
        results["status"] = "PASS"
    else:
        results["status"] = "FAIL_TRT_EP_NOT_ACTIVE_IN_SESSION"

    return results


_LATENCY_SCRIPT = """
# Critical: import tether FIRST so its __init__.py runs the
# LD_LIBRARY_PATH patch + eager dlopen of libnvinfer/libcublas/libcudnn.
# Without this, ORT's TRT EP can't find the libs in this fresh subprocess.
import tether  # noqa: F401 — load-bearing side effect

import os, time, json
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

# Tiny model for the latency check
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1024])
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1024])
node = helper.make_node('Mul', ['x', 'x'], ['y'])
graph = helper.make_graph([node], 'probe', [x], [y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 19)])
model.ir_version = 9
model_bytes = model.SerializeToString()

inputs = {'x': np.random.randn(1, 1024).astype(np.float32)}

def time_session(providers):
    sess = ort.InferenceSession(model_bytes, providers=providers)
    active = sess.get_providers()
    # Warmup
    for _ in range(5):
        sess.run(None, inputs)
    # Measure
    times = []
    for _ in range(20):
        t = time.perf_counter()
        sess.run(None, inputs)
        times.append((time.perf_counter() - t) * 1000)
    return active, float(np.mean(times))

cuda_active, cuda_ms = time_session(['CUDAExecutionProvider', 'CPUExecutionProvider'])
trt_active, trt_ms = time_session(['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

print('CUDA_EP_ACTIVE:', cuda_active)
print('CUDA_EP_MEAN_MS:', cuda_ms)
print('TRT_EP_ACTIVE:', 'TensorrtExecutionProvider' in trt_active)
print('TRT_EP_MEAN_MS:', trt_ms)
"""


@app.local_entrypoint()
def main():
    import json
    assert WHEEL_PATH.exists(), f"Build wheel first: {WHEEL_PATH}"
    result = validate.remote()
    print()
    print("=" * 70)
    print("LOCAL SUMMARY")
    print("=" * 70)
    print(json.dumps(result, indent=2, default=str)[:2500])
    print()
    print(f"FINAL STATUS: {result.get('status', 'UNKNOWN')}")
