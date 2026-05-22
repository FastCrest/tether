"""Test GR00T N1.6 export on Modal A100.

Downloads nvidia/GR00T-N1.6-3B (6.6GB, 2 shards), builds DiT expert stack,
exports to ONNX, validates numerically.

Usage:
    modal run scripts/modal_test_gr00t.py
"""

import modal

app = modal.App("reflex-gr00t-test")

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


@app.function(image=image, gpu="A100-40GB", timeout=1800, scaledown_window=60)
def run_gr00t_export():
    import os
    import subprocess
    import time

    results = {"steps": []}

    def log(name, status, detail=""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else "FAIL"
        print(f"{tag}: {name} — {detail}", flush=True)

    export_dir = "/tmp/reflex_gr00t_export"

    # Step 1: Load + detect
    print("=== Step 1: Load GR00T checkpoint & detect ===", flush=True)
    start = time.time()
    try:
        from reflex.checkpoint import load_checkpoint, detect_model_type
        state_dict, _ = load_checkpoint("nvidia/GR00T-N1.6-3B")
        model_type = detect_model_type(state_dict)
        total_params = sum(v.numel() for v in state_dict.values())
        dit_keys = [k for k in state_dict.keys() if k.startswith("action_head.model.transformer_blocks.")]
        elapsed = time.time() - start
        log("detect", "pass",
            f"{elapsed:.1f}s, detected={model_type}, params={total_params/1e9:.2f}B, "
            f"dit_keys={len(dit_keys)}")
        if model_type != "gr00t":
            log("detect_type", "fail", f"expected 'gr00t', got {model_type}")
            return results
    except Exception as e:
        import traceback
        log("detect", "fail", f"{str(e)[:300]}\n{traceback.format_exc()[:500]}")
        return results

    # Step 2: Build DiT expert stack
    print("\n=== Step 2: Build GR00T DiT expert stack ===", flush=True)
    start = time.time()
    try:
        from reflex.exporters.gr00t import build_gr00t_expert_stack
        expert_stack, meta = build_gr00t_expert_stack(state_dict, embodiment_id=0)
        elapsed = time.time() - start
        log("build_expert", "pass",
            f"{elapsed:.1f}s, blocks={meta['num_layers']}, "
            f"hidden={meta['hidden']}, heads={meta['num_heads']}×hd{meta['head_dim']}, "
            f"ff_inner={meta['ff_inner']}, vlm_kv_dim={meta['vlm_kv_dim']}, "
            f"chunk={meta['chunk_size']}, out_dim={meta['output_dim']}, "
            f"params={meta['total_params_m']:.1f}M")
    except Exception as e:
        import traceback
        log("build_expert", "fail", f"{str(e)[:300]}\n{traceback.format_exc()[:500]}")
        return results

    # Step 3: PyTorch forward pass
    print("\n=== Step 3: PyTorch forward pass ===", flush=True)
    try:
        import torch
        chunk_size = meta["chunk_size"]
        hidden = meta["hidden"]
        dummy_tokens = torch.randn(1, chunk_size, hidden)
        dummy_time = torch.tensor([0.5])
        dummy_pos = torch.arange(chunk_size).unsqueeze(0)
        with torch.no_grad():
            out = expert_stack(dummy_tokens, dummy_time, dummy_pos)
        log("forward", "pass",
            f"output shape={tuple(out.shape)}, mean={out.mean().item():.4f}, "
            f"std={out.std().item():.4f}")
    except Exception as e:
        import traceback
        log("forward", "fail", f"{str(e)[:300]}\n{traceback.format_exc()[:500]}")
        return results

    # Step 4: Full reflex export (ONNX + validation)
    print("\n=== Step 4: reflex export nvidia/GR00T-N1.6-3B ===", flush=True)
    del state_dict
    del expert_stack
    start = time.time()
    r = subprocess.run([
        "reflex", "export", "nvidia/GR00T-N1.6-3B",
        "--target", "desktop",
        "--output", export_dir,
    ], capture_output=True, text=True, timeout=900)
    elapsed = time.time() - start

    if r.returncode == 0:
        files = os.listdir(export_dir) if os.path.exists(export_dir) else []
        log("export", "pass", f"{elapsed:.1f}s, files: {files}")
        for line in r.stdout.splitlines():
            if "Validation" in line or "max_diff" in line:
                print(f"  {line}", flush=True)
    else:
        # Capture BOTH stderr + stdout — some CLI error paths print to
        # stdout via rich (historical), so an empty stderr is misleading.
        # Fixed in 2026-05-22 follow-up; this diagnostic ensures we see
        # the cause even if a future regression re-introduces stdout-only
        # error paths.
        tail = (r.stderr or "")[-400:] + "\n---stdout tail---\n" + (r.stdout or "")[-600:]
        log("export", "fail", tail)

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    passed = sum(1 for s in results["steps"] if s["status"] == "pass")
    failed = sum(1 for s in results["steps"] if s["status"] == "fail")
    print(f"Passed: {passed}, Failed: {failed}", flush=True)
    results["summary"] = {"passed": passed, "failed": failed}
    return results


@app.local_entrypoint()
def main():
    print("Testing GR00T N1.6 export on Modal A100...")
    results = run_gr00t_export.remote()

    for step in results["steps"]:
        tag = "PASS" if step["status"] == "pass" else "FAIL"
        print(f"  {tag}: {step['step']} — {step['detail']}")
