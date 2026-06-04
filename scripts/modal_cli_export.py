"""Test the full tether CLI export pipeline on Modal A100.

Usage:
    modal run scripts/modal_cli_export.py
"""

import json
import os
import time

import modal

app = modal.App("tether-cli-export")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "safetensors",
        "huggingface_hub",
        "transformers>=4.51",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "numpy",
        "typer",
        "rich",
        "pydantic>=2.0",
        "pyyaml",
    )
    .add_local_dir("src/tether", "/root/tether-vla/src/tether", copy=True)
    .add_local_file("pyproject.toml", "/root/tether-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/tether-vla/README.md", copy=True)
    .run_commands("cd /root/tether-vla && pip install -e .")
)


@app.function(image=image, gpu="A100-40GB", timeout=600, scaledown_window=60)
def test_cli_export():
    """Run the tether CLI export command end-to-end."""
    import subprocess

    results = {"steps": []}

    def log(name, status, detail=""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else "FAIL"
        print(f"{tag}: {name} — {detail}")

    # Step 1: Verify CLI installed
    print("=== Step 1: Verify CLI ===")
    r = subprocess.run(["tether", "--version"], capture_output=True, text=True)
    if r.returncode == 0:
        log("cli_version", "pass", r.stdout.strip())
    else:
        log("cli_version", "fail", r.stderr[:200])
        return results

    # Step 2: Dry run
    print("\n=== Step 2: Dry run ===")
    r = subprocess.run([
        "tether", "export", "lerobot/smolvla_base",
        "--target", "orin-nano", "--dry-run", "--verbose",
    ], capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        log("dry_run", "pass", "Export check passed")
    else:
        log("dry_run", "fail", (r.stdout + r.stderr)[-300:])

    # Step 3: Full export
    print("\n=== Step 3: Full export ===")
    start = time.time()
    r = subprocess.run([
        "tether", "export", "lerobot/smolvla_base",
        "--target", "desktop",
        "--output", "/tmp/tether_cli_export",
        "--verbose",
    ], capture_output=True, text=True, timeout=300)
    elapsed = time.time() - start

    print(r.stdout[-1000:] if r.stdout else "")
    if r.stderr:
        print("STDERR:", r.stderr[-500:])

    if r.returncode == 0:
        log("full_export", "pass", f"{elapsed:.1f}s")
    else:
        log("full_export", "fail", (r.stdout + r.stderr)[-300:])

    # Step 4: Check output files
    print("\n=== Step 4: Check outputs ===")
    export_dir = "/tmp/tether_cli_export"
    if os.path.exists(export_dir):
        files = os.listdir(export_dir)
        total_size = sum(os.path.getsize(os.path.join(export_dir, f)) for f in files) / 1e6
        log("output_files", "pass", f"{len(files)} files, {total_size:.1f}MB total: {files}")

        # Check config
        config_path = os.path.join(export_dir, "tether_config.json")
        if os.path.exists(config_path):
            config = json.loads(open(config_path).read())
            log("config_valid", "pass", f"target={config.get('target')}, expert={config.get('expert', {}).get('num_layers')} layers")
        else:
            log("config_valid", "fail", "tether_config.json not found")
    else:
        log("output_files", "fail", f"{export_dir} does not exist")

    # Step 5: List targets
    print("\n=== Step 5: List targets ===")
    r = subprocess.run(["tether", "targets"], capture_output=True, text=True)
    if "orin-nano" in r.stdout and "Jetson Thor" in r.stdout:
        log("targets", "pass", "All hardware targets listed")
    else:
        log("targets", "fail", r.stdout[:200])

    # Summary
    print("\n=== SUMMARY ===")
    passed = sum(1 for s in results["steps"] if s["status"] == "pass")
    failed = sum(1 for s in results["steps"] if s["status"] == "fail")
    print(f"Passed: {passed}, Failed: {failed}")
    results["summary"] = {"passed": passed, "failed": failed}
    return results


@app.local_entrypoint()
def main():
    print("Testing tether CLI export on Modal A100...")
    results = test_cli_export.remote()

    for step in results["steps"]:
        tag = "PASS" if step["status"] == "pass" else "FAIL"
        print(f"  {tag}: {step['step']} — {step['detail']}")
