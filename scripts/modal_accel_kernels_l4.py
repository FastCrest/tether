"""Modal: run the tether.accel GPU kernel tests on an L4 (sm_89, the RTX 4090 stand-in).

The L4 shares the 4090's compute capability, so it exercises the same Triton
and ``torch._scaled_mm`` FP8 code path. It is not a qualified accel target and
says nothing about 4090 latency.

Usage:
    MODAL_PROFILE=fastcrest modal run scripts/modal_accel_kernels_l4.py

Cost: L4 is ~$0.80/h; the 15-minute timeout caps a run well under $1.
"""

from __future__ import annotations

import os

import modal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = modal.App("tether-accel-kernels-l4")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0", "numpy", "pytest", "cryptography", "pydantic", "pyyaml", "rich", "typer"
    )
    .env({"PYTHONPATH": "/repo/src:/repo"})
    .add_local_dir(
        os.path.join(REPO_ROOT, "src", "tether"), "/repo/src/tether", ignore=["**/__pycache__"]
    )
    .add_local_file(os.path.join(REPO_ROOT, "tests", "__init__.py"), "/repo/tests/__init__.py")
    .add_local_file(
        os.path.join(REPO_ROOT, "tests", "test_accel_kernels_gpu.py"),
        "/repo/tests/test_accel_kernels_gpu.py",
    )
    .add_local_file(
        os.path.join(REPO_ROOT, "tests", "test_accel_graphs_gpu.py"),
        "/repo/tests/test_accel_graphs_gpu.py",
    )
)


@app.function(gpu="L4", image=image, timeout=900)
def run() -> str:
    import subprocess

    out = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-v",
            "-p",
            "no:cacheprovider",
            "--noconftest",
            "/repo/tests/test_accel_kernels_gpu.py",
            "/repo/tests/test_accel_graphs_gpu.py",
        ],
        cwd="/repo",
        capture_output=True,
        text=True,
    )
    return out.stdout[-12000:] + out.stderr[-4000:] + f"\nexit={out.returncode}"


@app.local_entrypoint()
def main():
    print(run.remote())
