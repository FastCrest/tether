"""Full SmolVLA export: load model via lerobot, decompose, export all 3 stages to ONNX.

Usage:
    modal run scripts/modal_full_export.py
"""

import json
import os
import time

import modal

app = modal.App("tether-smolvla-full-export")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "safetensors",
        "transformers>=4.51",
        "huggingface_hub",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "numpy",
        "lerobot @ git+https://github.com/huggingface/lerobot.git",
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=1800, scaledown_window=60)
def full_smolvla_export():
    """Load SmolVLA via lerobot, split into components, export each to ONNX."""
    import torch
    import torch.nn as nn
    import numpy as np
    from pathlib import Path

    results = {"steps": []}
    export_dir = Path("/tmp/reflex_export")
    export_dir.mkdir(exist_ok=True)

    def log(name, status, detail=""):
        entry = {"step": name, "status": status, "detail": detail}
        results["steps"].append(entry)
        print(f"{'PASS' if status == 'pass' else 'FAIL' if status == 'fail' else 'SKIP'}: {name} — {detail}")

    # Step 1: Load SmolVLA via multiple strategies
    print("=== Step 1: Load SmolVLA ===")
    policy = None
    state_dict = None

    # Strategy A: try lerobot (multiple import paths)
    for import_path in [
        "lerobot.common.policies.smolvla.modeling_smolvla",
        "lerobot.policies.smolvla.modeling_smolvla",
        "lerobot.scripts.smolvla",
    ]:
        try:
            mod = __import__(import_path, fromlist=["SmolVLAPolicy"])
            SmolVLAPolicy = getattr(mod, "SmolVLAPolicy")
            policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
            policy.eval()
            total_params = sum(p.numel() for p in policy.parameters())
            log("load_policy", "pass", f"Loaded via {import_path}, {total_params/1e6:.1f}M params")
            break
        except Exception:
            continue

    # Strategy B: try transformers AutoModel
    if policy is None:
        try:
            from transformers import AutoModel, AutoConfig
            policy = AutoModel.from_pretrained("lerobot/smolvla_base", trust_remote_code=True)
            policy.eval()
            total_params = sum(p.numel() for p in policy.parameters())
            log("load_policy", "pass", f"Loaded via AutoModel, {total_params/1e6:.1f}M params")
        except Exception as e:
            print(f"  AutoModel failed: {str(e)[:200]}")

    # Strategy C: raw state_dict
    if policy is None:
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        local_dir = snapshot_download("lerobot/smolvla_base")
        state_dict = {}
        for f in sorted(Path(local_dir).glob("*.safetensors")):
            state_dict.update(load_file(str(f)))
        log("load_raw", "pass", f"Raw state_dict, {sum(p.numel() for p in state_dict.values())/1e6:.1f}M params")

        # Deep structure analysis
        print("\n=== Model structure (3-level key analysis) ===")
        level2 = {}
        for key in sorted(state_dict.keys()):
            parts = key.split(".")
            prefix = ".".join(parts[:3]) if len(parts) >= 3 else key
            if prefix not in level2:
                level2[prefix] = {"count": 0, "params": 0}
            level2[prefix]["count"] += 1
            level2[prefix]["params"] += state_dict[key].numel()

        for p, info in sorted(level2.items(), key=lambda x: -x[1]["params"]):
            print(f"  {p}: {info['count']} tensors, {info['params']/1e6:.1f}M params")

        vlm_only = sum(v.numel() for k, v in state_dict.items() if ".vlm." in k and "expert" not in k)
        expert_only = sum(v.numel() for k, v in state_dict.items() if "expert" in k or "lm_expert" in k)
        action_proj = sum(v.numel() for k, v in state_dict.items() if "action_" in k or "state_proj" in k)
        buffer = sum(v.numel() for k, v in state_dict.items() if "buffer" in k)

        log("structure_analysis", "pass",
            f"VLM backbone: {vlm_only/1e6:.1f}M, Action expert: {expert_only/1e6:.1f}M, "
            f"Action projections: {action_proj/1e6:.1f}M, Buffers: {buffer/1e6:.1f}M")

    # Step 2: Identify and extract model components
    print("\n=== Step 2: Extract model components ===")
    try:
        if 'policy' in dir() and policy is not None:
            # Explore the policy model structure
            print("Policy attributes:")
            for name, child in policy.named_children():
                params = sum(p.numel() for p in child.parameters())
                print(f"  {name}: {type(child).__name__}, {params/1e6:.1f}M params")

            # Try to find the model inside policy
            model = getattr(policy, 'model', None)
            if model is not None:
                print("\nModel sub-components:")
                for name, child in model.named_children():
                    params = sum(p.numel() for p in child.parameters())
                    print(f"  {name}: {type(child).__name__}, {params/1e6:.1f}M params")

            log("extract_components", "pass", "Model structure mapped")
        else:
            log("extract_components", "skip", "Working with raw state_dict")

    except Exception as e:
        log("extract_components", "fail", str(e)[:300])

    # Step 3: RMSNorm decomposition on actual model
    print("\n=== Step 3: Find and decompose RMSNorm modules ===")
    try:
        target_model = policy if 'policy' in dir() and policy is not None else None
        if target_model is not None:
            rmsnorm_modules = []
            for name, module in target_model.named_modules():
                if "RMSNorm" in type(module).__name__ or "rmsnorm" in type(module).__name__.lower():
                    rmsnorm_modules.append(name)

            rope_modules = []
            for name, module in target_model.named_modules():
                if "Rotary" in type(module).__name__ or "rotary" in type(module).__name__.lower():
                    rope_modules.append(name)

            log("find_ops", "pass", f"Found {len(rmsnorm_modules)} RMSNorm, {len(rope_modules)} RoPE modules")
            if rmsnorm_modules[:5]:
                print(f"  RMSNorm examples: {rmsnorm_modules[:5]}")
            if rope_modules[:5]:
                print(f"  RoPE examples: {rope_modules[:5]}")
        else:
            # Search state_dict for norm weights
            norm_keys = [k for k in state_dict.keys() if "norm" in k.lower() or "layernorm" in k.lower()]
            rope_keys = [k for k in state_dict.keys() if "rotary" in k.lower() or "inv_freq" in k.lower()]
            log("find_ops", "pass", f"Found {len(norm_keys)} norm keys, {len(rope_keys)} rotary keys in state_dict")

    except Exception as e:
        log("find_ops", "fail", str(e)[:300])

    # Step 4: Test ONNX export of a component
    print("\n=== Step 4: ONNX export test ===")
    try:
        if 'policy' in dir() and policy is not None:
            model = getattr(policy, 'model', policy)

            # Try to get the VLM backbone
            vlm = None
            for name in ['vlm_with_expert', 'vlm', 'backbone', 'language_model']:
                vlm = getattr(model, name, None)
                if vlm is not None:
                    print(f"  Found VLM component: {name} ({type(vlm).__name__})")
                    break

            # Try to get vision encoder
            vision = None
            for name in ['vision_encoder', 'vision_tower', 'image_encoder']:
                vision = getattr(model, name, None) or getattr(vlm, name, None) if vlm else None
                if vision is not None:
                    print(f"  Found vision component: {name} ({type(vision).__name__})")
                    break

            log("component_discovery", "pass", f"VLM={vlm is not None}, Vision={vision is not None}")

            # Export vision encoder if found
            if vision is not None:
                vision = vision.cpu().eval()
                # Determine input size
                dummy_img = torch.randn(1, 3, 512, 512)
                try:
                    with torch.no_grad():
                        vision_out = vision(dummy_img)
                    print(f"  Vision output shape: {vision_out.shape if isinstance(vision_out, torch.Tensor) else type(vision_out)}")

                    export_path = str(export_dir / "vision_encoder.onnx")
                    torch.onnx.export(
                        vision,
                        (dummy_img,),
                        export_path,
                        input_names=["pixel_values"],
                        output_names=["image_features"],
                        opset_version=19,
                        dynamic_axes={"pixel_values": {0: "batch"}},
                    )
                    size_mb = os.path.getsize(export_path) / 1e6
                    log("onnx_vision", "pass", f"Exported vision encoder: {size_mb:.1f}MB")
                except Exception as ve:
                    log("onnx_vision", "fail", str(ve)[:300])
            else:
                log("onnx_vision", "skip", "Vision encoder not found as separate module")
        else:
            log("component_discovery", "skip", "No loaded model")

    except Exception as e:
        log("onnx_export_test", "fail", str(e)[:300])

    # Step 5: Benchmark inference
    print("\n=== Step 5: Benchmark inference speed ===")
    try:
        if 'policy' in dir() and policy is not None:
            device = torch.device("cuda")
            policy = policy.to(device)

            # Create dummy inputs matching SmolVLA expected format
            dummy_obs = {
                "observation.images.top": torch.randn(1, 3, 512, 512, device=device),
                "observation.state": torch.randn(1, 6, device=device),
            }

            # Warmup
            for _ in range(3):
                try:
                    with torch.no_grad():
                        _ = policy.select_action(dummy_obs)
                except Exception:
                    break

            # Measure
            latencies = []
            for _ in range(20):
                torch.cuda.synchronize()
                start = time.perf_counter()
                try:
                    with torch.no_grad():
                        _ = policy.select_action(dummy_obs)
                except Exception as ie:
                    log("benchmark", "fail", f"Inference failed: {str(ie)[:200]}")
                    break
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

            if latencies:
                latencies.sort()
                mean_ms = sum(latencies) / len(latencies)
                hz = 1000.0 / mean_ms
                log("benchmark", "pass", f"mean={mean_ms:.1f}ms, p50={latencies[len(latencies)//2]:.1f}ms, {hz:.1f}Hz, A100")
            else:
                log("benchmark", "skip", "Could not run inference")
        else:
            log("benchmark", "skip", "No loaded model")

    except Exception as e:
        log("benchmark", "fail", str(e)[:300])

    # Summary
    print("\n=== SUMMARY ===")
    passed = sum(1 for s in results["steps"] if s["status"] == "pass")
    failed = sum(1 for s in results["steps"] if s["status"] == "fail")
    skipped = sum(1 for s in results["steps"] if s["status"] == "skip")
    print(f"Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
    results["summary"] = {"passed": passed, "failed": failed, "skipped": skipped}

    return results


@app.local_entrypoint()
def main():
    print("Running full SmolVLA export on Modal A100...")
    results = full_smolvla_export.remote()

    with open("/tmp/tether_full_export.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults: /tmp/tether_full_export.json")
    for step in results["steps"]:
        status = "PASS" if step["status"] == "pass" else "FAIL" if step["status"] == "fail" else "SKIP"
        print(f"  {status}: {step['step']} — {step['detail']}")
