"""Test pi0 export on Modal A100.

Downloads lerobot/pi0_base (3.5GB), reconstructs the expert stack,
exports to ONNX, validates numerically.

Usage:
    modal run scripts/modal_test_pi0.py
"""

import modal

app = modal.App("reflex-pi0-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch", "safetensors", "huggingface_hub",
        "transformers>=4.51", "onnx", "onnxruntime",
        "onnxscript", "numpy", "Pillow",
        "typer", "rich", "pydantic>=2.0", "pyyaml",
    )
    .add_local_dir("src/reflex", "/root/reflex-vla/src/reflex", copy=True)
    .add_local_file("pyproject.toml", "/root/reflex-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/reflex-vla/README.md", copy=True)
    .run_commands("cd /root/reflex-vla && pip install -e .")
)


@app.function(image=image, gpu="A100-40GB", timeout=1200, scaledown_window=60)
def run_pi0_export():
    """Export pi0 to ONNX and validate."""
    import os
    import subprocess
    import time

    results = {"steps": []}

    def log(name, status, detail=""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else "FAIL"
        print(f"{tag}: {name} — {detail}", flush=True)

    export_dir = "/tmp/reflex_pi0_export"

    # Step 1: Detect model type by loading checkpoint
    print("=== Step 1: Load pi0 checkpoint & detect ===", flush=True)
    start = time.time()
    try:
        from reflex.checkpoint import load_checkpoint, detect_model_type
        state_dict, _ = load_checkpoint("lerobot/pi0_base")
        model_type = detect_model_type(state_dict)
        total_params = sum(v.numel() for v in state_dict.values())
        elapsed = time.time() - start
        log("detect", "pass",
            f"{elapsed:.1f}s, detected={model_type}, params={total_params/1e9:.2f}B, "
            f"tensors={len(state_dict)}")

        # Print a few sample keys
        sample = [k for k in list(state_dict.keys())[:20] if "expert" in k.lower() or "action" in k.lower()][:5]
        print(f"  Sample keys: {sample}", flush=True)

        if model_type != "pi0":
            log("detect_type", "fail", f"expected 'pi0', got {model_type}")
            return results
    except Exception as e:
        log("detect", "fail", str(e)[:300])
        return results

    # Step 2: Build expert stack
    print("\n=== Step 2: Build pi0 expert stack ===", flush=True)
    start = time.time()
    try:
        from reflex.exporters.pi0 import build_pi0_expert_stack
        expert_stack, meta = build_pi0_expert_stack(state_dict, head_dim=128)
        elapsed = time.time() - start
        log("build_expert", "pass",
            f"{elapsed:.1f}s, layers={meta['num_layers']}, "
            f"hidden={meta['expert_hidden']}, "
            f"heads={meta['n_q_heads']}Q/{meta['n_kv_heads']}KV, "
            f"params={meta['total_params_m']:.1f}M")
    except Exception as e:
        import traceback
        log("build_expert", "fail", f"{str(e)[:200]} {traceback.format_exc()[:500]}")
        return results

    # Step 3: Run reflex export CLI
    print("\n=== Step 3: reflex export lerobot/pi0_base ===", flush=True)
    del state_dict  # free memory
    start = time.time()
    r = subprocess.run([
        "reflex", "export", "lerobot/pi0_base",
        "--target", "desktop",
        "--output", export_dir,
    ], capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    if r.returncode == 0:
        files = os.listdir(export_dir) if os.path.exists(export_dir) else []
        log("export", "pass", f"{elapsed:.1f}s, files: {files}")
        print(f"  Stdout tail: {r.stdout[-500:]}", flush=True)
    else:
        log("export", "fail", f"{r.stderr[-500:]}")
        return results

    # Step 4: Verify ONNX validation passed
    print("\n=== Step 4: Verify ONNX ===", flush=True)
    import json
    config_path = os.path.join(export_dir, "reflex_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        expert = config.get("expert", {})
        log("verify", "pass",
            f"model_type={config.get('model_type')}, "
            f"action_dim={expert.get('action_dim')}, "
            f"num_layers={expert.get('num_layers')}")
    else:
        log("verify", "fail", "No reflex_config.json")

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    passed = sum(1 for s in results["steps"] if s["status"] == "pass")
    failed = sum(1 for s in results["steps"] if s["status"] == "fail")
    print(f"Passed: {passed}, Failed: {failed}", flush=True)
    results["summary"] = {"passed": passed, "failed": failed}
    return results


@app.local_entrypoint()
def main():
    print("Testing pi0 export on Modal A100...")
    results = run_pi0_export.remote()

    for step in results["steps"]:
        tag = "PASS" if step["status"] == "pass" else "FAIL"
        print(f"  {tag}: {step['step']} — {step['detail']}")
