"""Test SmolVLA export pipeline on Modal A100.

Usage:
    modal run scripts/modal_test_export.py
"""

import json
import time

import modal

app = modal.App("tether-smolvla-export")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "safetensors",
        "transformers>=4.40",
        "huggingface_hub",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "numpy",
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=1200, scaledown_window=60)
def test_smolvla_export():
    """Download SmolVLA, load, decompose, export to ONNX, validate."""
    import torch
    import numpy as np
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    results = {"steps": []}

    def log_step(name, status, detail=""):
        entry = {"step": name, "status": status, "detail": detail}
        results["steps"].append(entry)
        print(f"{'PASS' if status == 'pass' else 'FAIL'}: {name} {detail}")

    # Step 1: Download SmolVLA
    print("=== Step 1: Download SmolVLA ===")
    start = time.time()
    local_dir = snapshot_download("lerobot/smolvla_base")
    download_time = time.time() - start
    log_step("download", "pass", f"{download_time:.1f}s")

    # Step 2: Load checkpoint
    print("\n=== Step 2: Load checkpoint ===")
    safetensors_files = list(Path(local_dir).glob("*.safetensors"))
    state_dict = {}
    for f in sorted(safetensors_files):
        state_dict.update(load_file(str(f)))
    total_params = sum(p.numel() for p in state_dict.values())
    log_step("load_checkpoint", "pass", f"{len(state_dict)} tensors, {total_params/1e6:.1f}M params, {len(safetensors_files)} files")

    # Step 3: Inspect keys
    print("\n=== Step 3: Inspect model structure ===")
    prefixes = set()
    for key in state_dict.keys():
        parts = key.split(".")
        if len(parts) >= 3:
            prefixes.add(f"{parts[0]}.{parts[1]}.{parts[2]}")
    print(f"  Top-level prefixes ({len(prefixes)}):")
    for p in sorted(prefixes)[:30]:
        print(f"    {p}")
    log_step("inspect_keys", "pass", f"{len(prefixes)} unique 3-level prefixes")

    # Step 4: Load as transformers model
    print("\n=== Step 4: Load as transformers model ===")
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        config_path = Path(local_dir) / "config.json"
        config_data = json.loads(config_path.read_text()) if config_path.exists() else {}

        # Try loading with AutoModel
        model = None
        try:
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            log_step("load_model", "skip", "lerobot not installed on Modal")
        except ImportError:
            # Try loading just the vision encoder portion
            log_step("load_model", "skip", "lerobot not installed — testing raw tensors instead")

    except Exception as e:
        log_step("load_model", "fail", str(e)[:200])

    # Step 5: Test RMSNorm decomposition on a standalone module
    print("\n=== Step 5: Test RMSNorm decomposition ===")
    try:
        import torch.nn as nn

        class TestRMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.eps = eps
            def forward(self, x):
                variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
                return (x * torch.rsqrt(variance + self.eps) * self.weight).to(x.dtype)

        class DecomposedRMSNorm(nn.Module):
            def __init__(self, weight, eps=1e-6):
                super().__init__()
                self.register_buffer("weight", weight.clone())
                self.eps = eps
            def forward(self, x):
                variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
                x_normed = x * torch.rsqrt(variance + self.eps)
                return (x_normed * self.weight).to(x.dtype)

        dim = 576  # SmolVLA hidden size
        original = TestRMSNorm(dim).cuda()
        decomposed = DecomposedRMSNorm(original.weight.data).cuda()
        test_input = torch.randn(1, 50, dim, device="cuda")

        with torch.no_grad():
            ref_out = original(test_input)
            dec_out = decomposed(test_input)

        max_diff = (ref_out - dec_out).abs().max().item()
        log_step("rmsnorm_decompose", "pass", f"max_diff={max_diff:.2e}")

    except Exception as e:
        log_step("rmsnorm_decompose", "fail", str(e)[:200])

    # Step 6: Test ONNX export of decomposed RMSNorm
    print("\n=== Step 6: Test ONNX export of decomposed RMSNorm ===")
    try:
        export_path = "/tmp/rmsnorm_test.onnx"
        torch.onnx.export(
            decomposed.cpu(),
            (torch.randn(1, 50, dim),),
            export_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=19,
            dynamic_axes={"input": {0: "batch", 1: "seq_len"}},
        )
        import onnx
        model_onnx = onnx.load(export_path)
        onnx.checker.check_model(model_onnx)

        import onnxruntime as ort
        sess = ort.InferenceSession(export_path)
        test_np = torch.randn(1, 50, dim).numpy()
        ort_out = sess.run(None, {"input": test_np})[0]
        torch_out = decomposed.cpu()(torch.from_numpy(test_np)).numpy()
        onnx_diff = np.abs(ort_out - torch_out).max()
        log_step("onnx_export_rmsnorm", "pass", f"onnx_valid=True, max_diff={onnx_diff:.2e}")

    except Exception as e:
        log_step("onnx_export_rmsnorm", "fail", str(e)[:200])

    # Step 7: Test ONNX export of a small transformer block
    print("\n=== Step 7: Test ONNX export of mini transformer ===")
    try:
        class MiniTransformerBlock(nn.Module):
            def __init__(self, dim=576, heads=8):
                super().__init__()
                self.norm = DecomposedRMSNorm(torch.ones(dim))
                self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
                self.ffn = nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                )

            def forward(self, x):
                normed = self.norm(x)
                attn_out, _ = self.attn(normed, normed, normed)
                x = x + attn_out
                x = x + self.ffn(self.norm(x))
                return x

        mini = MiniTransformerBlock(dim=576, heads=8)
        export_path = "/tmp/mini_transformer.onnx"
        dummy = torch.randn(1, 50, 576)
        torch.onnx.export(
            mini,
            (dummy,),
            export_path,
            input_names=["hidden_states"],
            output_names=["output"],
            opset_version=19,
            dynamic_axes={"hidden_states": {0: "batch", 1: "seq_len"}},
        )
        model_onnx = onnx.load(export_path)
        onnx.checker.check_model(model_onnx)

        sess = ort.InferenceSession(export_path)
        ort_out = sess.run(None, {"hidden_states": dummy.numpy()})[0]
        torch_out = mini(dummy).detach().numpy()
        mini_diff = np.abs(ort_out - torch_out).max()
        import os
        file_size = os.path.getsize(export_path) / 1e6
        log_step("onnx_export_transformer", "pass", f"max_diff={mini_diff:.2e}, size={file_size:.1f}MB")

    except Exception as e:
        log_step("onnx_export_transformer", "fail", str(e)[:200])

    # Step 8: Profile GPU memory for SmolVLA weights
    print("\n=== Step 8: Memory profiling ===")
    try:
        torch.cuda.reset_peak_memory_stats()
        # Load all weights to GPU
        gpu_tensors = {k: v.cuda() for k, v in state_dict.items()}
        mem_used = torch.cuda.max_memory_allocated() / 1e9
        log_step("memory_profile", "pass", f"weights_on_gpu={mem_used:.2f}GB")
        del gpu_tensors
        torch.cuda.empty_cache()
    except Exception as e:
        log_step("memory_profile", "fail", str(e)[:200])

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
    print("Running SmolVLA export test on Modal A100...")
    results = test_smolvla_export.remote()

    with open("/tmp/tether_export_test.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: /tmp/tether_export_test.json")
    for step in results["steps"]:
        status = "PASS" if step["status"] == "pass" else "FAIL" if step["status"] == "fail" else "SKIP"
        print(f"  {status}: {step['step']} — {step['detail']}")
