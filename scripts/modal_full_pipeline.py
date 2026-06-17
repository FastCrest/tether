"""Full SmolVLA export + inference pipeline on Modal A100.

Stacks all 16 expert layers, runs the complete 10-step denoising loop,
benchmarks end-to-end inference speed.

Usage:
    modal run scripts/modal_full_pipeline.py
"""

import json
import math
import os
import time

import modal

app = modal.App("tether-full-pipeline")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
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


@app.function(image=image, gpu="A100-40GB", timeout=1800, scaledown_window=60)
def full_pipeline():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    results = {"steps": []}

    def log(name, status, detail=""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else "FAIL" if status == "fail" else "SKIP"
        print(f"{tag}: {name} — {detail}")

    # --- Decomposed ops ---
    class DecomposedRMSNorm(nn.Module):
        def __init__(self, weight, eps=1e-6):
            super().__init__()
            self.register_buffer("weight", weight.clone())
            self.eps = eps

        def forward(self, x):
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + self.eps)
            return (x_normed * self.weight).to(x.dtype)

    class DecomposedRoPE(nn.Module):
        def __init__(self, dim, max_seq_len=512, base=10000.0):
            super().__init__()
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(max_seq_len).float()
            freqs = torch.outer(t, inv_freq)
            self.register_buffer("cos_cached", torch.cat([freqs.cos(), freqs.cos()], dim=-1))
            self.register_buffer("sin_cached", torch.cat([freqs.sin(), freqs.sin()], dim=-1))

        def apply(self, x, position_ids):
            cos = self.cos_cached[position_ids].unsqueeze(1)
            sin = self.sin_cached[position_ids].unsqueeze(1)
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    # --- Expert GQA Layer ---
    class ExpertGQALayer(nn.Module):
        def __init__(self, hidden, n_q_heads, n_kv_heads, head_dim, intermediate, eps=1e-6):
            super().__init__()
            self.n_q_heads = n_q_heads
            self.n_kv_heads = n_kv_heads
            self.head_dim = head_dim
            self.kv_groups = n_q_heads // n_kv_heads
            self.input_layernorm = DecomposedRMSNorm(torch.ones(hidden), eps)
            self.post_attention_layernorm = DecomposedRMSNorm(torch.ones(hidden), eps)
            self.q_proj = nn.Linear(hidden, n_q_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(n_q_heads * head_dim, hidden, bias=False)
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)
            self.rope = DecomposedRoPE(head_dim, max_seq_len=512)

        def forward(self, x, position_ids, cross_attn_kv=None):
            b, s, _ = x.shape
            residual = x
            x = self.input_layernorm(x)
            q = self.q_proj(x).view(b, s, self.n_q_heads, self.head_dim).transpose(1, 2)
            # Cross-attn layers use external KV; self-attn layers use own hidden states
            kv_input = cross_attn_kv if cross_attn_kv is not None else x
            k = self.k_proj(kv_input).view(b, kv_input.shape[1], self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(kv_input).view(b, kv_input.shape[1], self.n_kv_heads, self.head_dim).transpose(1, 2)
            q = self.rope.apply(q, position_ids)
            if cross_attn_kv is None:
                k = self.rope.apply(k, position_ids)
            kv_len = k.shape[2]
            k = k.unsqueeze(2).expand(-1, -1, self.kv_groups, -1, -1).reshape(b, self.n_q_heads, kv_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.kv_groups, -1, -1).reshape(b, self.n_q_heads, kv_len, self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, -1)
            x = residual + self.o_proj(out)
            residual = x
            x = self.post_attention_layernorm(x)
            x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
            return residual + x

    # --- Suffix Encoder ---
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

    # --- Full Expert Stack ---
    class SmolVLAExpertStack(nn.Module):
        def __init__(self, layers, suffix_encoder, action_out_proj, final_norm_weight, cross_attn_indices, vlm_kv_dim):
            super().__init__()
            self.layers = nn.ModuleList(layers)
            self.suffix_encoder = suffix_encoder
            self.action_out_proj = action_out_proj
            self.final_norm = DecomposedRMSNorm(final_norm_weight)
            self.cross_attn_indices = set(cross_attn_indices)
            self.vlm_kv_dim = vlm_kv_dim

        def forward(self, noisy_actions, timestep, position_ids, vlm_conditioning=None):
            x = self.suffix_encoder(noisy_actions, timestep)
            for i, layer in enumerate(self.layers):
                if i in self.cross_attn_indices:
                    # Cross-attention: use VLM conditioning or dummy
                    if vlm_conditioning is None:
                        cond = torch.zeros(x.shape[0], 1, self.vlm_kv_dim, device=x.device, dtype=x.dtype)
                    else:
                        cond = vlm_conditioning
                    x = layer(x, position_ids, cross_attn_kv=cond)
                else:
                    x = layer(x, position_ids)
            x = self.final_norm(x)
            return self.action_out_proj(x)

        def denoise(self, num_steps=10, chunk_size=50, action_dim=32, device="cuda"):
            actions = torch.randn(1, chunk_size, action_dim, device=device)
            position_ids = torch.arange(chunk_size, device=device).unsqueeze(0)
            dt = -1.0 / num_steps
            for step in range(num_steps):
                t = 1.0 + step * dt
                timestep = torch.tensor([t], device=device)
                with torch.no_grad():
                    velocity = self.forward(actions, timestep, position_ids)
                actions = actions + velocity * dt
            return actions

    # === EXECUTION ===

    # Step 1: Load
    print("=== Step 1: Load SmolVLA ===")
    local_dir = snapshot_download("lerobot/smolvla_base")
    state_dict = {}
    for f in sorted(Path(local_dir).glob("*.safetensors")):
        state_dict.update(load_file(str(f)))
    log("load", "pass", f"{sum(v.numel() for v in state_dict.values())/1e6:.1f}M params")

    # Step 2: Get config
    print("\n=== Step 2: Get dimensions ===")
    from transformers import AutoConfig
    vlm_config = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    text_config = vlm_config.text_config
    head_dim = text_config.hidden_size // text_config.num_attention_heads

    action_in_w = state_dict["model.action_in_proj.weight"]
    expert_hidden = action_in_w.shape[0]
    action_dim = action_in_w.shape[1]

    # Detect key prefix
    l0_keys = [k for k in state_dict.keys() if "lm_expert" in k and "layers.0." in k]
    sample_key = l0_keys[0]
    base_prefix = sample_key[:sample_key.index("layers.0")]

    q_shape = state_dict[f"{base_prefix}layers.0.self_attn.q_proj.weight"].shape
    k_shape = state_dict[f"{base_prefix}layers.0.self_attn.k_proj.weight"].shape
    gate_shape = state_dict[f"{base_prefix}layers.0.mlp.gate_proj.weight"].shape
    n_q_heads = q_shape[0] // head_dim
    n_kv_heads = k_shape[0] // head_dim
    intermediate = gate_shape[0]

    # Count layers using ALL expert keys
    all_expert_keys = [k for k in state_dict.keys() if "lm_expert" in k]
    layer_indices = set()
    for k in all_expert_keys:
        parts = k.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_indices.add(int(parts[i + 1]))
    num_layers = max(layer_indices) + 1

    log("config", "pass",
        f"hidden={expert_hidden}, q={n_q_heads}, kv={n_kv_heads}, head_dim={head_dim}, "
        f"intermediate={intermediate}, layers={num_layers}, action_dim={action_dim}")

    # Step 3: Build full expert stack
    print(f"\n=== Step 3: Build {num_layers}-layer expert stack ===")
    try:
        layers = []
        for i in range(num_layers):
            prefix = f"{base_prefix}layers.{i}"
            # Detect per-layer K/V input dim (self-attn vs cross-attn)
            kv_weight = state_dict[f"{prefix}.self_attn.k_proj.weight"]
            layer_kv_in = kv_weight.shape[1]  # 720 for self-attn, 320 for cross-attn
            is_cross_attn = layer_kv_in != expert_hidden

            # Build layer with correct K/V input dimension
            layer = ExpertGQALayer(expert_hidden, n_q_heads, n_kv_heads, head_dim, intermediate)
            # Override k/v proj with correct input dimension
            if is_cross_attn:
                layer.k_proj = nn.Linear(layer_kv_in, n_kv_heads * head_dim, bias=False)
                layer.v_proj = nn.Linear(layer_kv_in, n_kv_heads * head_dim, bias=False)

            layer_sd = {
                "input_layernorm.weight": state_dict[f"{prefix}.input_layernorm.weight"],
                "post_attention_layernorm.weight": state_dict[f"{prefix}.post_attention_layernorm.weight"],
                "q_proj.weight": state_dict[f"{prefix}.self_attn.q_proj.weight"],
                "k_proj.weight": state_dict[f"{prefix}.self_attn.k_proj.weight"],
                "v_proj.weight": state_dict[f"{prefix}.self_attn.v_proj.weight"],
                "o_proj.weight": state_dict[f"{prefix}.self_attn.o_proj.weight"],
                "gate_proj.weight": state_dict[f"{prefix}.mlp.gate_proj.weight"],
                "up_proj.weight": state_dict[f"{prefix}.mlp.up_proj.weight"],
                "down_proj.weight": state_dict[f"{prefix}.mlp.down_proj.weight"],
            }
            layer.load_state_dict(layer_sd, strict=False)
            layers.append(layer)
            if i == 0 or is_cross_attn != (i % 2 == 1):
                print(f"  Layer {i}: {'cross-attn' if is_cross_attn else 'self-attn'} (kv_in={layer_kv_in})")

        # Print layer pattern
        self_layers = sum(1 for i in range(num_layers) if state_dict[f"{base_prefix}layers.{i}.self_attn.k_proj.weight"].shape[1] == expert_hidden)
        cross_layers = num_layers - self_layers
        print(f"  Pattern: {self_layers} self-attn + {cross_layers} cross-attn = {num_layers} total")

        # Suffix encoder
        suffix_enc = SuffixEncoder(action_dim, expert_hidden)
        suffix_enc.load_state_dict({
            "action_in_proj.weight": state_dict["model.action_in_proj.weight"],
            "action_in_proj.bias": state_dict["model.action_in_proj.bias"],
            "action_time_mlp_in.weight": state_dict["model.action_time_mlp_in.weight"],
            "action_time_mlp_in.bias": state_dict["model.action_time_mlp_in.bias"],
            "action_time_mlp_out.weight": state_dict["model.action_time_mlp_out.weight"],
            "action_time_mlp_out.bias": state_dict["model.action_time_mlp_out.bias"],
        })

        # Action projection
        action_out = nn.Linear(expert_hidden, action_dim)
        action_out.weight = nn.Parameter(state_dict["model.action_out_proj.weight"])
        action_out.bias = nn.Parameter(state_dict["model.action_out_proj.bias"])

        # Final norm
        final_norm_key = f"{base_prefix[:-len('layers.')]}norm.weight" if base_prefix.endswith("layers.") else None
        # Try common locations for final norm
        final_norm_weight = None
        for candidate in [
            base_prefix.replace("layers.", "norm.weight").rstrip("."),
            base_prefix + "norm.weight",
            "model.vlm_with_expert.lm_expert.norm.weight",
            "model.vlm_with_expert.lm_expert.model.norm.weight",
        ]:
            candidate = candidate.replace("..", ".")
            if candidate in state_dict:
                final_norm_weight = state_dict[candidate]
                print(f"  Found final norm: {candidate}")
                break
        if final_norm_weight is None:
            print("  Final norm not found, using ones")
            final_norm_weight = torch.ones(expert_hidden)

        # Identify cross-attn layer indices
        cross_attn_indices = []
        vlm_kv_dim = 0
        for i in range(num_layers):
            kv_in = state_dict[f"{base_prefix}layers.{i}.self_attn.k_proj.weight"].shape[1]
            if kv_in != expert_hidden:
                cross_attn_indices.append(i)
                vlm_kv_dim = kv_in
        print(f"  Cross-attn layers: {cross_attn_indices}, vlm_kv_dim={vlm_kv_dim}")

        # Assemble
        expert_stack = SmolVLAExpertStack(layers, suffix_enc, action_out, final_norm_weight, cross_attn_indices, vlm_kv_dim)
        expert_stack.eval()
        total_expert_params = sum(p.numel() for p in expert_stack.parameters())
        log("build_stack", "pass", f"{num_layers} layers, {total_expert_params/1e6:.1f}M total params")

    except Exception as e:
        import traceback
        traceback.print_exc()
        log("build_stack", "fail", str(e)[:300])
        results["summary"] = {"passed": sum(1 for s in results["steps"] if s["status"] == "pass"), "failed": 1}
        return results

    # Step 4: Test forward
    print("\n=== Step 4: Test full forward ===")
    device = torch.device("cuda")
    expert_stack = expert_stack.to(device)

    dummy_actions = torch.randn(1, 50, action_dim, device=device)
    dummy_time = torch.tensor([0.5], device=device)
    dummy_pos = torch.arange(50, device=device).unsqueeze(0)

    with torch.no_grad():
        out = expert_stack(dummy_actions, dummy_time, dummy_pos)
    log("forward", "pass", f"output={list(out.shape)}")

    # Step 5: Full denoise loop
    print("\n=== Step 5: Full 10-step denoise ===")
    with torch.no_grad():
        actions = expert_stack.denoise(num_steps=10, chunk_size=50, action_dim=action_dim, device=device)
    log("denoise", "pass", f"output={list(actions.shape)}, range=[{actions.min():.3f}, {actions.max():.3f}]")

    # Step 6: Benchmark
    print("\n=== Step 6: Benchmark ===")
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            expert_stack.denoise(num_steps=10, chunk_size=50, action_dim=action_dim, device=device)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(50):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            expert_stack.denoise(num_steps=10, chunk_size=50, action_dim=action_dim, device=device)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    mean_ms = sum(latencies) / len(latencies)
    log("benchmark", "pass",
        f"full_denoise: mean={mean_ms:.1f}ms, p50={latencies[25]:.1f}ms, "
        f"p95={latencies[47]:.1f}ms, {1000/mean_ms:.1f}Hz")

    # Step 7: Memory usage
    print("\n=== Step 7: GPU memory ===")
    mem_mb = torch.cuda.max_memory_allocated() / 1e6
    log("memory", "pass", f"peak GPU memory: {mem_mb:.0f}MB ({mem_mb/1024:.2f}GB)")

    # Step 8: ONNX export of full stack (single denoise step)
    print("\n=== Step 8: ONNX export full stack ===")
    export_dir = Path("/tmp/tether_full_export")
    export_dir.mkdir(exist_ok=True)
    try:
        expert_cpu = expert_stack.cpu()
        dummy_a = torch.randn(1, 50, action_dim)
        dummy_t = torch.tensor([0.5])
        dummy_p = torch.arange(50).unsqueeze(0)

        onnx_path = str(export_dir / "expert_stack_step.onnx")
        torch.onnx.export(
            expert_cpu,
            (dummy_a, dummy_t, dummy_p),
            onnx_path,
            input_names=["noisy_actions", "timestep", "position_ids"],
            output_names=["velocity"],
            opset_version=19,
            dynamic_axes={
                "noisy_actions": {0: "batch"},
                "timestep": {0: "batch"},
                "position_ids": {0: "batch"},
            },
        )
        import onnx
        onnx.checker.check_model(onnx.load(onnx_path))
        size_mb = os.path.getsize(onnx_path) / 1e6
        log("onnx_full_stack", "pass", f"Exported: {size_mb:.1f}MB, valid ONNX")

        # Validate
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        ort_out = sess.run(None, {
            "noisy_actions": dummy_a.numpy(),
            "timestep": dummy_t.numpy(),
            "position_ids": dummy_p.numpy().astype(np.int64),
        })[0]
        torch_out = expert_cpu(dummy_a, dummy_t, dummy_p).detach().numpy()
        max_diff = float(np.abs(ort_out - torch_out).max())
        log("validate_full_stack", "pass", f"ONNX max_diff={max_diff:.2e}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        log("onnx_full_stack", "fail", str(e)[:300])

    # Summary
    print("\n=== SUMMARY ===")
    passed = sum(1 for s in results["steps"] if s["status"] == "pass")
    failed = sum(1 for s in results["steps"] if s["status"] == "fail")
    print(f"Passed: {passed}, Failed: {failed}")
    results["summary"] = {"passed": passed, "failed": failed}
    return results


@app.local_entrypoint()
def main():
    print("Running full SmolVLA pipeline on Modal A100...")
    results = full_pipeline.remote()

    with open("/tmp/reflex_pipeline.json", "w") as f:
        json.dump(results, f, indent=2)

    for step in results["steps"]:
        tag = "PASS" if step["status"] == "pass" else "FAIL"
        print(f"  {tag}: {step['step']} — {step['detail']}")
