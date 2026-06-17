"""Export the SmolVLA VLM backbone (350M SigLIP + SmolLM2) on Modal A100.

The VLM runs ONCE per inference call to produce prefix embeddings + KV cache.
The action expert then runs 10 denoising steps using that cached KV.

Usage:
    modal run scripts/modal_vlm_export.py
"""

import json
import os
import time

import modal

app = modal.App("tether-vlm-export")

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
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=1200, scaledown_window=60)
def export_vlm():
    import torch
    import torch.nn as nn
    import numpy as np
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig

    results = {"steps": []}

    def log(name, status, detail=""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else "FAIL" if status == "fail" else "SKIP"
        print(f"{tag}: {name} — {detail}")

    export_dir = Path("/tmp/tether_vlm_export")
    export_dir.mkdir(exist_ok=True)

    # Step 1: Load the VLM directly from HuggingFace
    print("=== Step 1: Load SmolVLM2 ===")
    try:
        vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        vlm = AutoModelForImageTextToText.from_pretrained(vlm_model_name, torch_dtype=torch.float32)
        vlm.eval()
        total_params = sum(p.numel() for p in vlm.parameters())
        log("load_vlm", "pass", f"{total_params/1e6:.1f}M params, {type(vlm).__name__}")

        # Print structure
        print("\n  VLM top-level modules:")
        for name, child in vlm.named_children():
            params = sum(p.numel() for p in child.parameters())
            print(f"    {name}: {type(child).__name__}, {params/1e6:.1f}M")

    except Exception as e:
        import traceback
        traceback.print_exc()
        log("load_vlm", "fail", str(e)[:300])
        return results

    # Step 2: Load processor for creating dummy inputs
    print("\n=== Step 2: Load processor ===")
    try:
        processor = AutoProcessor.from_pretrained(vlm_model_name)
        log("load_processor", "pass", f"{type(processor).__name__}")
    except Exception as e:
        log("load_processor", "fail", str(e)[:300])

    # Step 3: Extract vision encoder
    print("\n=== Step 3: Extract vision encoder ===")
    try:
        vision_encoder = None
        for name in ["vision_model", "vision_encoder", "model.vision_model", "model.vision_tower"]:
            parts = name.split(".")
            obj = vlm
            for p in parts:
                obj = getattr(obj, p, None)
                if obj is None:
                    break
            if obj is not None:
                vision_encoder = obj
                print(f"  Found vision encoder at: {name}")
                break

        if vision_encoder is None:
            # Try deeper
            for name, mod in vlm.named_modules():
                if "vision" in name.lower() and "encoder" in type(mod).__name__.lower():
                    vision_encoder = mod
                    print(f"  Found vision encoder at: {name} ({type(mod).__name__})")
                    break

        if vision_encoder is not None:
            ve_params = sum(p.numel() for p in vision_encoder.parameters())
            log("extract_vision", "pass", f"{ve_params/1e6:.1f}M params, {type(vision_encoder).__name__}")
        else:
            log("extract_vision", "fail", "Could not find vision encoder module")
    except Exception as e:
        log("extract_vision", "fail", str(e)[:300])

    # Step 4: Test vision encoder forward
    print("\n=== Step 4: Test vision encoder forward ===")
    try:
        if vision_encoder is not None:
            # Try standard input
            dummy_pixel = torch.randn(1, 3, 384, 384)  # SigLIP default
            with torch.no_grad():
                try:
                    ve_out = vision_encoder(dummy_pixel)
                    if isinstance(ve_out, tuple):
                        ve_out = ve_out[0]
                    log("vision_forward", "pass", f"output shape={list(ve_out.shape)}")
                except Exception as e1:
                    # Try pixel_values kwarg
                    ve_out = vision_encoder(pixel_values=dummy_pixel)
                    if isinstance(ve_out, tuple):
                        ve_out = ve_out[0]
                    if hasattr(ve_out, 'last_hidden_state'):
                        ve_out = ve_out.last_hidden_state
                    log("vision_forward", "pass", f"output shape={list(ve_out.shape)} (via pixel_values)")
    except Exception as e:
        log("vision_forward", "fail", str(e)[:200])

    # Step 5: Test full VLM forward with dummy text
    print("\n=== Step 5: Test full VLM forward ===")
    try:
        device = torch.device("cuda")
        vlm = vlm.to(device)

        # Create simple text input
        dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        dummy_attention_mask = torch.ones_like(dummy_input_ids)

        with torch.no_grad():
            outputs = vlm(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)

        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        elif hasattr(outputs, 'logits'):
            hidden = outputs.logits
        else:
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs

        log("vlm_forward", "pass", f"output type={type(outputs).__name__}, shape={list(hidden.shape)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log("vlm_forward", "fail", str(e)[:300])

    # Step 6: Benchmark VLM forward (prefix encoding)
    print("\n=== Step 6: Benchmark VLM prefix encoding ===")
    try:
        # Simulate a realistic prefix: 5 tokens for text
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        attn_mask = torch.ones_like(input_ids)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = vlm(input_ids=input_ids, attention_mask=attn_mask)
        torch.cuda.synchronize()

        latencies = []
        for _ in range(50):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = vlm(input_ids=input_ids, attention_mask=attn_mask)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        mean_ms = sum(latencies) / len(latencies)
        log("benchmark_vlm", "pass",
            f"text-only: mean={mean_ms:.1f}ms, p50={latencies[25]:.1f}ms, p95={latencies[47]:.1f}ms")
    except Exception as e:
        log("benchmark_vlm", "fail", str(e)[:200])

    # Step 7: GPU memory for full VLM
    print("\n=== Step 7: Memory usage ===")
    mem_mb = torch.cuda.max_memory_allocated() / 1e6
    log("memory", "pass", f"VLM peak GPU: {mem_mb:.0f}MB ({mem_mb/1024:.2f}GB)")

    # Step 8: Truncate to 16 layers (as SmolVLA does)
    print("\n=== Step 8: Truncate VLM to 16 layers ===")
    try:
        text_model = None
        for name in ["model.text_model", "text_model", "model.language_model", "language_model"]:
            parts = name.split(".")
            obj = vlm
            for p in parts:
                obj = getattr(obj, p, None)
                if obj is None:
                    break
            if obj is not None:
                text_model = obj
                print(f"  Found text model at: {name}")
                break

        if text_model is not None:
            # Find layers
            layers_module = None
            for attr_name in ["layers", "model.layers", "decoder.layers"]:
                parts = attr_name.split(".")
                obj = text_model
                for p in parts:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        break
                if obj is not None:
                    layers_module = obj
                    print(f"  Found layers at: {attr_name}, count={len(layers_module)}")
                    break

            if layers_module is not None and len(layers_module) > 16:
                original_count = len(layers_module)
                # Truncate by replacing with first 16
                parent_name = "layers"  # Most common
                if hasattr(text_model, "layers"):
                    text_model.layers = nn.ModuleList(list(text_model.layers)[:16])
                elif hasattr(text_model, "model") and hasattr(text_model.model, "layers"):
                    text_model.model.layers = nn.ModuleList(list(text_model.model.layers)[:16])

                new_params = sum(p.numel() for p in vlm.parameters())
                log("truncate", "pass",
                    f"Truncated from {original_count} to 16 layers, {new_params/1e6:.1f}M params remaining")
            elif layers_module is not None:
                log("truncate", "pass", f"Already {len(layers_module)} layers, no truncation needed")
            else:
                log("truncate", "skip", "Could not find layers module")
        else:
            log("truncate", "skip", "Could not find text model")

    except Exception as e:
        import traceback
        traceback.print_exc()
        log("truncate", "fail", str(e)[:300])

    # Step 9: Extract hidden size and key dimensions
    print("\n=== Step 9: Model dimensions ===")
    try:
        config = AutoConfig.from_pretrained(vlm_model_name)
        text_config = config.text_config
        vision_config = config.vision_config if hasattr(config, "vision_config") else None

        dims = {
            "text_hidden_size": text_config.hidden_size,
            "text_num_heads": text_config.num_attention_heads,
            "text_num_kv_heads": text_config.num_key_value_heads,
            "text_num_layers": text_config.num_hidden_layers,
            "text_intermediate": text_config.intermediate_size,
            "text_head_dim": text_config.hidden_size // text_config.num_attention_heads,
            "vocab_size": text_config.vocab_size,
        }
        if vision_config:
            dims["vision_hidden_size"] = vision_config.hidden_size
            dims["vision_image_size"] = getattr(vision_config, "image_size", "unknown")
            dims["vision_patch_size"] = getattr(vision_config, "patch_size", "unknown")

        for k, v in dims.items():
            print(f"  {k}: {v}")
        log("dimensions", "pass", json.dumps(dims))

    except Exception as e:
        log("dimensions", "fail", str(e)[:200])

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
    print("Running VLM backbone export on Modal A100...")
    results = export_vlm.remote()

    with open("/tmp/tether_vlm_export.json", "w") as f:
        json.dump(results, f, indent=2)

    for step in results["steps"]:
        tag = "PASS" if step["status"] == "pass" else "FAIL" if step["status"] == "fail" else "SKIP"
        print(f"  {tag}: {step['step']} — {step['detail']}")
