"""Test pi0.5 export on Modal A100.

Downloads lerobot/pi05_base (3.62GB), builds AdaRMSNorm expert stack,
exports to ONNX, validates numerically.

Usage:
    modal run scripts/modal_test_pi05.py
"""

import modal

app = modal.App("tether-pi05-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch", "safetensors", "huggingface_hub",
        "transformers>=4.51", "onnx", "onnxruntime",
        "onnxscript", "numpy", "Pillow",
        "typer", "rich", "pydantic>=2.0", "pyyaml",
    )
    .add_local_dir("src/tether", "/root/tether-vla/src/tether", copy=True)
    .add_local_file("pyproject.toml", "/root/tether-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/tether-vla/README.md", copy=True)
    .run_commands("cd /root/tether-vla && pip install -e .")
)


@app.function(image=image, gpu="A100-40GB", timeout=1200, scaledown_window=60)
def run_pi05_export():
    """Export pi0.5 to ONNX and validate."""
    import os
    import subprocess
    import time

    results = {"steps": []}

    def log(name, status, detail=""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else "FAIL"
        print(f"{tag}: {name} — {detail}", flush=True)

    export_dir = "/tmp/reflex_pi05_export"

    # Step 1: Download and detect
    print("=== Step 1: Load pi0.5 checkpoint & detect ===", flush=True)
    start = time.time()
    try:
        from tether.checkpoint import load_checkpoint, detect_model_type
        state_dict, _ = load_checkpoint("lerobot/pi05_base")
        model_type = detect_model_type(state_dict)
        total_params = sum(v.numel() for v in state_dict.values())
        elapsed = time.time() - start

        # Confirm AdaRMSNorm markers
        adarms_keys = [k for k in state_dict.keys() if "input_layernorm.dense" in k]
        time_mlp_keys = [k for k in state_dict.keys() if k.startswith("time_mlp")]

        log("detect", "pass",
            f"{elapsed:.1f}s, detected={model_type}, params={total_params/1e9:.2f}B, "
            f"AdaRMSNorm keys={len(adarms_keys)}, time_mlp keys={len(time_mlp_keys)}")

        if model_type != "pi05":
            log("detect_type", "fail", f"expected 'pi05', got {model_type}")
            return results
    except Exception as e:
        import traceback
        log("detect", "fail", f"{str(e)[:300]}\n{traceback.format_exc()[:500]}")
        return results

    # Step 2: Build pi0.5 expert stack
    print("\n=== Step 2: Build pi0.5 expert stack (AdaRMSNorm) ===", flush=True)
    start = time.time()
    try:
        from tether.exporters.pi0 import build_pi05_expert_stack
        expert_stack, meta = build_pi05_expert_stack(state_dict, head_dim=128)
        elapsed = time.time() - start
        log("build_expert", "pass",
            f"{elapsed:.1f}s, layers={meta['num_layers']}, "
            f"hidden={meta['expert_hidden']}, "
            f"heads={meta['n_q_heads']}Q/{meta['n_kv_heads']}KV, "
            f"params={meta['total_params_m']:.1f}M, uses_adarms={meta.get('uses_adarms')}")
    except Exception as e:
        import traceback
        log("build_expert", "fail", f"{str(e)[:300]}\n{traceback.format_exc()[:500]}")
        return results

    # Step 3: Run a forward pass in PyTorch
    print("\n=== Step 3: PyTorch forward pass ===", flush=True)
    try:
        import torch
        dummy_actions = torch.randn(1, 50, meta["action_dim"])
        dummy_time = torch.tensor([0.5])
        dummy_pos = torch.arange(50).unsqueeze(0)
        with torch.no_grad():
            out = expert_stack(dummy_actions, dummy_time, dummy_pos)
        log("forward", "pass",
            f"output shape={tuple(out.shape)}, mean={out.mean().item():.4f}, "
            f"std={out.std().item():.4f}")
    except Exception as e:
        import traceback
        log("forward", "fail", f"{str(e)[:300]}\n{traceback.format_exc()[:500]}")
        return results

    # Step 4: Full tether export (including ONNX + validation)
    print("\n=== Step 4: tether export lerobot/pi05_base ===", flush=True)
    del state_dict  # free memory
    del expert_stack
    start = time.time()
    r = subprocess.run([
        "tether", "export", "lerobot/pi05_base",
        "--target", "desktop",
        "--output", export_dir,
    ], capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    if r.returncode == 0:
        files = os.listdir(export_dir) if os.path.exists(export_dir) else []
        log("export", "pass", f"{elapsed:.1f}s, files: {files}")
        # Show validation result from stdout
        for line in r.stdout.splitlines():
            if "Validation" in line or "max_diff" in line:
                print(f"  {line}", flush=True)
    else:
        log("export", "fail", f"{r.stderr[-500:]}")

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    passed = sum(1 for s in results["steps"] if s["status"] == "pass")
    failed = sum(1 for s in results["steps"] if s["status"] == "fail")
    print(f"Passed: {passed}, Failed: {failed}", flush=True)
    results["summary"] = {"passed": passed, "failed": failed}
    return results


@app.local_entrypoint()
def main():
    print("Testing pi0.5 export on Modal A100...")
    results = run_pi05_export.remote()

    for step in results["steps"]:
        tag = "PASS" if step["status"] == "pass" else "FAIL"
        print(f"  {tag}: {step['step']} — {step['detail']}")
