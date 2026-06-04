"""Export real SmolVLA components to ONNX on Modal A100.

Usage:
    modal run scripts/modal_real_export.py
"""

import json
import os
import time

import modal

app = modal.App("tether-real-export")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "safetensors",
        "huggingface_hub",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "numpy",
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=600, scaledown_window=60)
def export_smolvla_components():
    """Load SmolVLA, extract suffix encoder + action projection, export to ONNX."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import math
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    results = {"steps": []}

    def log(name, status, detail=""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else "FAIL" if status == "fail" else "SKIP"
        print(f"{tag}: {name} — {detail}")

    export_dir = Path("/tmp/tether_smolvla_export")
    export_dir.mkdir(exist_ok=True)

    # Step 1: Download and load
    print("=== Step 1: Load SmolVLA ===")
    local_dir = snapshot_download("lerobot/smolvla_base")
    state_dict = {}
    for f in sorted(Path(local_dir).glob("*.safetensors")):
        state_dict.update(load_file(str(f)))
    total_params = sum(v.numel() for v in state_dict.values())
    log("load", "pass", f"{total_params/1e6:.1f}M params")

    # Step 2: Extract dimensions from weights
    print("\n=== Step 2: Extract dimensions ===")
    action_in_w = state_dict["model.action_in_proj.weight"]
    action_out_w = state_dict["model.action_out_proj.weight"]
    expert_hidden = action_in_w.shape[0]
    action_dim = action_in_w.shape[1]
    log("dimensions", "pass", f"expert_hidden={expert_hidden}, action_dim={action_dim}")

    # Step 3: Build suffix encoder
    print("\n=== Step 3: Build suffix encoder ===")

    def create_sinusoidal_pos_embedding(time_val, dimension, min_period=4e-3, max_period=4.0):
        d_half = dimension // 2
        exponent = torch.linspace(0, 1, d_half, device=time_val.device, dtype=time_val.dtype)
        freq = torch.exp(exponent * (math.log(min_period) - math.log(max_period))) / min_period
        angle = time_val.unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat([angle.cos(), angle.sin()], dim=-1)

    class SuffixEncoder(nn.Module):
        def __init__(self, act_dim, hidden):
            super().__init__()
            self.hidden = hidden
            self.action_in_proj = nn.Linear(act_dim, hidden)
            self.action_time_mlp_in = nn.Linear(hidden * 2, hidden)
            self.action_time_mlp_out = nn.Linear(hidden, hidden)

        def forward(self, noisy_actions, timestep):
            b, c, _ = noisy_actions.shape
            action_embs = self.action_in_proj(noisy_actions)
            time_emb = create_sinusoidal_pos_embedding(timestep, self.hidden)
            time_emb = time_emb.unsqueeze(1).expand(-1, c, -1)
            fused = torch.cat([action_embs, time_emb], dim=-1)
            return self.action_time_mlp_out(F.silu(self.action_time_mlp_in(fused)))

    suffix_enc = SuffixEncoder(action_dim, expert_hidden)
    suffix_enc.load_state_dict({
        "action_in_proj.weight": state_dict["model.action_in_proj.weight"],
        "action_in_proj.bias": state_dict["model.action_in_proj.bias"],
        "action_time_mlp_in.weight": state_dict["model.action_time_mlp_in.weight"],
        "action_time_mlp_in.bias": state_dict["model.action_time_mlp_in.bias"],
        "action_time_mlp_out.weight": state_dict["model.action_time_mlp_out.weight"],
        "action_time_mlp_out.bias": state_dict["model.action_time_mlp_out.bias"],
    })
    suffix_enc.eval()
    log("build_suffix", "pass", f"SuffixEncoder loaded, {sum(p.numel() for p in suffix_enc.parameters())/1e6:.2f}M params")

    # Step 4: Build action projection
    print("\n=== Step 4: Build action projection ===")

    class ActionProj(nn.Module):
        def __init__(self, hidden, act_dim):
            super().__init__()
            self.proj = nn.Linear(hidden, act_dim)

        def forward(self, x):
            return self.proj(x)

    action_proj = ActionProj(expert_hidden, action_dim)
    action_proj.load_state_dict({
        "proj.weight": state_dict["model.action_out_proj.weight"],
        "proj.bias": state_dict["model.action_out_proj.bias"],
    })
    action_proj.eval()
    log("build_action_proj", "pass", f"ActionProj loaded")

    # Step 5: Test forward pass
    print("\n=== Step 5: Test forward pass ===")
    device = torch.device("cuda")
    suffix_enc = suffix_enc.to(device)
    action_proj = action_proj.to(device)

    dummy_actions = torch.randn(1, 50, action_dim, device=device)
    dummy_time = torch.tensor([0.5], device=device)

    with torch.no_grad():
        suffix_out = suffix_enc(dummy_actions, dummy_time)
        velocity = action_proj(suffix_out)

    log("forward_pass", "pass",
        f"suffix_out={list(suffix_out.shape)}, velocity={list(velocity.shape)}")

    # Step 6: Export suffix encoder to ONNX
    print("\n=== Step 6: Export suffix encoder to ONNX ===")
    suffix_enc_cpu = suffix_enc.cpu()
    dummy_actions_cpu = torch.randn(1, 50, action_dim)
    dummy_time_cpu = torch.tensor([0.5])

    suffix_onnx_path = str(export_dir / "suffix_encoder.onnx")
    try:
        torch.onnx.export(
            suffix_enc_cpu,
            (dummy_actions_cpu, dummy_time_cpu),
            suffix_onnx_path,
            input_names=["noisy_actions", "timestep"],
            output_names=["suffix_embeddings"],
            opset_version=19,
            dynamic_axes={
                "noisy_actions": {0: "batch"},
                "timestep": {0: "batch"},
            },
        )
        import onnx
        model_onnx = onnx.load(suffix_onnx_path)
        onnx.checker.check_model(model_onnx)
        size_mb = os.path.getsize(suffix_onnx_path) / 1e6
        log("onnx_suffix", "pass", f"Exported: {size_mb:.2f}MB, valid ONNX")
    except Exception as e:
        log("onnx_suffix", "fail", str(e)[:300])

    # Step 7: Export action projection to ONNX
    print("\n=== Step 7: Export action projection to ONNX ===")
    action_proj_cpu = action_proj.cpu()
    dummy_hidden = torch.randn(1, 50, expert_hidden)

    proj_onnx_path = str(export_dir / "action_projection.onnx")
    try:
        torch.onnx.export(
            action_proj_cpu,
            (dummy_hidden,),
            proj_onnx_path,
            input_names=["expert_output"],
            output_names=["velocity"],
            opset_version=19,
            dynamic_axes={"expert_output": {0: "batch"}},
        )
        onnx.checker.check_model(onnx.load(proj_onnx_path))
        size_mb = os.path.getsize(proj_onnx_path) / 1e6
        log("onnx_action_proj", "pass", f"Exported: {size_mb:.2f}MB, valid ONNX")
    except Exception as e:
        log("onnx_action_proj", "fail", str(e)[:300])

    # Step 8: Validate ONNX vs PyTorch
    print("\n=== Step 8: Validate ONNX outputs ===")
    try:
        import onnxruntime as ort

        # Suffix encoder
        sess = ort.InferenceSession(suffix_onnx_path)
        ort_out = sess.run(None, {
            "noisy_actions": dummy_actions_cpu.numpy(),
            "timestep": dummy_time_cpu.numpy(),
        })[0]
        torch_out = suffix_enc_cpu(dummy_actions_cpu, dummy_time_cpu).detach().numpy()
        suffix_diff = float(np.abs(ort_out - torch_out).max())

        # Action projection
        sess2 = ort.InferenceSession(proj_onnx_path)
        ort_out2 = sess2.run(None, {"expert_output": dummy_hidden.numpy()})[0]
        torch_out2 = action_proj_cpu(dummy_hidden).detach().numpy()
        proj_diff = float(np.abs(ort_out2 - torch_out2).max())

        log("validate_onnx", "pass",
            f"suffix max_diff={suffix_diff:.2e}, proj max_diff={proj_diff:.2e}")
    except Exception as e:
        log("validate_onnx", "fail", str(e)[:300])

    # Step 9: Benchmark denoising loop
    print("\n=== Step 9: Benchmark denoising loop ===")
    suffix_enc = suffix_enc.to(device)
    action_proj = action_proj.to(device)

    def denoise_loop():
        actions = torch.randn(1, 50, action_dim, device=device)
        dt = -1.0 / 10
        for step in range(10):
            t = 1.0 + step * dt
            timestep = torch.tensor([t], device=device)
            with torch.no_grad():
                suffix_emb = suffix_enc(actions, timestep)
                # In real pipeline, suffix_emb goes through expert transformer
                # Here we just project directly (measures the projection overhead)
                velocity = action_proj(suffix_emb)
            actions = actions + velocity * dt
        return actions

    # Warmup
    for _ in range(10):
        denoise_loop()
    torch.cuda.synchronize()

    # Measure
    latencies = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        denoise_loop()
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    mean_ms = sum(latencies) / len(latencies)
    log("benchmark_denoise", "pass",
        f"mean={mean_ms:.2f}ms, p50={latencies[50]:.2f}ms, p95={latencies[95]:.2f}ms, "
        f"{1000/mean_ms:.0f}Hz (suffix+proj only, no expert transformer)")

    # Summary
    print("\n=== SUMMARY ===")
    passed = sum(1 for s in results["steps"] if s["status"] == "pass")
    failed = sum(1 for s in results["steps"] if s["status"] == "fail")
    results["summary"] = {"passed": passed, "failed": failed}
    print(f"Passed: {passed}, Failed: {failed}")

    return results


@app.local_entrypoint()
def main():
    print("Running real SmolVLA component export on Modal A100...")
    results = export_smolvla_components.remote()

    with open("/tmp/reflex_real_export.json", "w") as f:
        json.dump(results, f, indent=2)

    for step in results["steps"]:
        tag = "PASS" if step["status"] == "pass" else "FAIL"
        print(f"  {tag}: {step['step']} — {step['detail']}")
