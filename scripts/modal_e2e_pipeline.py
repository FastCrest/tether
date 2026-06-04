"""End-to-end SmolVLA inference: image in → actions out.

Loads the full model (VLM + expert), runs prefix encoding + 10-step denoise,
benchmarks the complete pipeline.

Usage:
    modal run scripts/modal_e2e_pipeline.py
"""

import json
import math
import os
import time

import modal

app = modal.App("tether-e2e")

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
        "Pillow",
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=1800, scaledown_window=60)
def e2e_pipeline():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoConfig

    results = {"steps": []}
    device = torch.device("cuda")

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
            return (x * torch.rsqrt(variance + self.eps) * self.weight).to(x.dtype)

    class DecomposedRoPE(nn.Module):
        def __init__(self, dim, max_seq_len=512, base=10000.0):
            super().__init__()
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
            self.register_buffer("cos_cached", torch.cat([freqs.cos(), freqs.cos()], dim=-1))
            self.register_buffer("sin_cached", torch.cat([freqs.sin(), freqs.sin()], dim=-1))
        def apply(self, x, position_ids):
            cos = self.cos_cached[position_ids].unsqueeze(1)
            sin = self.sin_cached[position_ids].unsqueeze(1)
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    def create_sinusoidal_pos_embedding(t, dim, min_p=4e-3, max_p=4.0):
        exp = torch.linspace(0, 1, dim//2, device=t.device, dtype=t.dtype)
        freq = torch.exp(exp * (math.log(min_p) - math.log(max_p))) / min_p
        angle = t.unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat([angle.cos(), angle.sin()], dim=-1)

    # --- Expert Layer ---
    class ExpertGQALayer(nn.Module):
        def __init__(self, hidden, nq, nkv, hd, inter, kv_in=None):
            super().__init__()
            self.nq, self.nkv, self.hd = nq, nkv, hd
            self.kv_groups = nq // nkv
            self.is_cross = kv_in is not None and kv_in != hidden
            self.input_layernorm = DecomposedRMSNorm(torch.ones(hidden))
            self.post_attention_layernorm = DecomposedRMSNorm(torch.ones(hidden))
            self.q_proj = nn.Linear(hidden, nq * hd, bias=False)
            self.k_proj = nn.Linear(kv_in or hidden, nkv * hd, bias=False)
            self.v_proj = nn.Linear(kv_in or hidden, nkv * hd, bias=False)
            self.o_proj = nn.Linear(nq * hd, hidden, bias=False)
            self.gate_proj = nn.Linear(hidden, inter, bias=False)
            self.up_proj = nn.Linear(hidden, inter, bias=False)
            self.down_proj = nn.Linear(inter, hidden, bias=False)
            self.rope = DecomposedRoPE(hd)

        def forward(self, x, pos_ids, cross_kv=None):
            b, s, _ = x.shape
            res = x
            x = self.input_layernorm(x)
            q = self.q_proj(x).view(b, s, self.nq, self.hd).transpose(1, 2)
            kv_src = cross_kv if cross_kv is not None else x
            kv_len = kv_src.shape[1]
            k = self.k_proj(kv_src).view(b, kv_len, self.nkv, self.hd).transpose(1, 2)
            v = self.v_proj(kv_src).view(b, kv_len, self.nkv, self.hd).transpose(1, 2)
            q = self.rope.apply(q, pos_ids)
            if cross_kv is None:
                k = self.rope.apply(k, pos_ids)
            k = k.unsqueeze(2).expand(-1,-1,self.kv_groups,-1,-1).reshape(b, self.nq, kv_len, self.hd)
            v = v.unsqueeze(2).expand(-1,-1,self.kv_groups,-1,-1).reshape(b, self.nq, kv_len, self.hd)
            attn = F.softmax(torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.hd), dim=-1)
            x = res + self.o_proj(torch.matmul(attn, v).transpose(1,2).contiguous().view(b, s, -1))
            res = x
            x = self.post_attention_layernorm(x)
            return res + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    # --- Full SmolVLA ---
    class SmolVLAFull(nn.Module):
        def __init__(self, vlm, expert_layers, suffix_enc_weights, action_proj_weights,
                     state_proj_weights, final_norm_weight, cross_indices, vlm_kv_dim,
                     expert_hidden, action_dim):
            super().__init__()
            self.vlm = vlm
            self.expert_layers = nn.ModuleList(expert_layers)
            self.cross_indices = set(cross_indices)
            self.vlm_kv_dim = vlm_kv_dim
            self.expert_hidden = expert_hidden
            self.action_dim = action_dim

            # Suffix encoder
            self.action_in_proj = nn.Linear(action_dim, expert_hidden)
            self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
            self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)
            self.action_in_proj.weight = nn.Parameter(suffix_enc_weights["in_w"])
            self.action_in_proj.bias = nn.Parameter(suffix_enc_weights["in_b"])
            self.action_time_mlp_in.weight = nn.Parameter(suffix_enc_weights["t_in_w"])
            self.action_time_mlp_in.bias = nn.Parameter(suffix_enc_weights["t_in_b"])
            self.action_time_mlp_out.weight = nn.Parameter(suffix_enc_weights["t_out_w"])
            self.action_time_mlp_out.bias = nn.Parameter(suffix_enc_weights["t_out_b"])

            # Action output
            self.action_out_proj = nn.Linear(expert_hidden, action_dim)
            self.action_out_proj.weight = nn.Parameter(action_proj_weights["w"])
            self.action_out_proj.bias = nn.Parameter(action_proj_weights["b"])

            # State projection
            self.state_proj = nn.Linear(action_dim, vlm.config.text_config.hidden_size)
            self.state_proj.weight = nn.Parameter(state_proj_weights["w"])
            self.state_proj.bias = nn.Parameter(state_proj_weights["b"])

            self.final_norm = DecomposedRMSNorm(final_norm_weight)
            # Project VLM hidden (960) down to cross-attn KV dim (320)
            vlm_hidden_size = vlm.config.text_config.hidden_size
            if vlm_kv_dim > 0 and vlm_kv_dim != vlm_hidden_size:
                self.vlm_to_kv_proj = nn.Linear(vlm_hidden_size, vlm_kv_dim, bias=False)
            else:
                self.vlm_to_kv_proj = None

        def encode_suffix(self, noisy_actions, timestep):
            b, c, _ = noisy_actions.shape
            act = self.action_in_proj(noisy_actions)
            t_emb = create_sinusoidal_pos_embedding(timestep, self.expert_hidden)
            t_emb = t_emb.unsqueeze(1).expand(-1, c, -1)
            return self.action_time_mlp_out(F.silu(self.action_time_mlp_in(torch.cat([act, t_emb], dim=-1))))

        def denoise_step(self, x, pos_ids, vlm_hidden=None):
            for i, layer in enumerate(self.expert_layers):
                if i in self.cross_indices and vlm_hidden is not None:
                    x = layer(x, pos_ids, cross_kv=vlm_hidden)
                else:
                    x = layer(x, pos_ids)
            x = self.final_norm(x)
            return self.action_out_proj(x)

        @torch.no_grad()
        def inference(self, pixel_values, input_ids, attention_mask, state,
                      num_steps=10, chunk_size=50):
            """Full end-to-end: image + text + state → actions."""
            dev = pixel_values.device
            b = pixel_values.shape[0]

            # 1. VLM prefix encoding (runs once)
            vlm_out = self.vlm(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Get last hidden state from the VLM and project to cross-attn KV dim
            vlm_hidden = vlm_out.hidden_states[-1] if hasattr(vlm_out, 'hidden_states') and vlm_out.hidden_states else None
            if vlm_hidden is not None and self.vlm_to_kv_proj is not None:
                vlm_hidden = self.vlm_to_kv_proj(vlm_hidden)

            # 2. Euler denoise (runs 10 steps)
            actions = torch.randn(b, chunk_size, self.action_dim, device=dev)
            pos_ids = torch.arange(chunk_size, device=dev).unsqueeze(0).expand(b, -1)
            dt = -1.0 / num_steps

            for step in range(num_steps):
                t = 1.0 + step * dt
                timestep = torch.tensor([t], device=dev).expand(b)
                suffix = self.encode_suffix(actions, timestep)
                velocity = self.denoise_step(suffix, pos_ids, vlm_hidden)
                actions = actions + velocity * dt

            return actions

    # === EXECUTION ===

    # Step 1: Load VLM
    print("=== Step 1: Load VLM ===")
    vlm_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    vlm = AutoModelForImageTextToText.from_pretrained(vlm_name, torch_dtype=torch.float32)
    # Truncate to 16 layers
    if hasattr(vlm.model, "text_model") and hasattr(vlm.model.text_model, "layers"):
        vlm.model.text_model.layers = nn.ModuleList(list(vlm.model.text_model.layers)[:16])
    elif hasattr(vlm, "language_model") and hasattr(vlm.language_model, "model"):
        vlm.language_model.model.layers = nn.ModuleList(list(vlm.language_model.model.layers)[:16])
    vlm.eval()
    vlm_params = sum(p.numel() for p in vlm.parameters())
    log("load_vlm", "pass", f"{vlm_params/1e6:.1f}M params (truncated to 16 layers)")

    # Step 2: Load SmolVLA weights
    print("\n=== Step 2: Load SmolVLA weights ===")
    local_dir = snapshot_download("lerobot/smolvla_base")
    sd = {}
    for f in sorted(Path(local_dir).glob("*.safetensors")):
        sd.update(load_file(str(f)))
    log("load_weights", "pass", f"{sum(v.numel() for v in sd.values())/1e6:.1f}M params")

    # Step 3: Build expert layers
    print("\n=== Step 3: Build expert + assemble full model ===")
    try:
        config = AutoConfig.from_pretrained(vlm_name)
        head_dim = config.text_config.hidden_size // config.text_config.num_attention_heads
        expert_hidden = sd["model.action_in_proj.weight"].shape[0]
        action_dim = sd["model.action_in_proj.weight"].shape[1]

        # Find expert keys and config
        all_expert_keys = [k for k in sd.keys() if "lm_expert" in k]
        base_prefix = all_expert_keys[0][:all_expert_keys[0].index("layers.")]
        layer_indices = set()
        for k in all_expert_keys:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i+1 < len(parts) and parts[i+1].isdigit():
                    layer_indices.add(int(parts[i+1]))
        num_layers = max(layer_indices) + 1

        q_shape = sd[f"{base_prefix}layers.0.self_attn.q_proj.weight"].shape
        k_shape = sd[f"{base_prefix}layers.0.self_attn.k_proj.weight"].shape
        gate_shape = sd[f"{base_prefix}layers.0.mlp.gate_proj.weight"].shape
        nq = q_shape[0] // head_dim
        nkv = k_shape[0] // head_dim
        inter = gate_shape[0]

        # Build layers
        layers = []
        cross_indices = []
        vlm_kv_dim = 0
        for i in range(num_layers):
            prefix = f"{base_prefix}layers.{i}"
            kv_in = sd[f"{prefix}.self_attn.k_proj.weight"].shape[1]
            is_cross = kv_in != expert_hidden
            if is_cross:
                cross_indices.append(i)
                vlm_kv_dim = kv_in

            layer = ExpertGQALayer(expert_hidden, nq, nkv, head_dim, inter,
                                   kv_in=kv_in if is_cross else None)
            layer_sd = {
                "input_layernorm.weight": sd[f"{prefix}.input_layernorm.weight"],
                "post_attention_layernorm.weight": sd[f"{prefix}.post_attention_layernorm.weight"],
                "q_proj.weight": sd[f"{prefix}.self_attn.q_proj.weight"],
                "k_proj.weight": sd[f"{prefix}.self_attn.k_proj.weight"],
                "v_proj.weight": sd[f"{prefix}.self_attn.v_proj.weight"],
                "o_proj.weight": sd[f"{prefix}.self_attn.o_proj.weight"],
                "gate_proj.weight": sd[f"{prefix}.mlp.gate_proj.weight"],
                "up_proj.weight": sd[f"{prefix}.mlp.up_proj.weight"],
                "down_proj.weight": sd[f"{prefix}.mlp.down_proj.weight"],
            }
            layer.load_state_dict(layer_sd, strict=False)
            layers.append(layer)

        # Final norm
        final_norm_w = torch.ones(expert_hidden)
        for candidate in [f"{base_prefix}norm.weight", "model.vlm_with_expert.lm_expert.norm.weight"]:
            if candidate in sd:
                final_norm_w = sd[candidate]
                break

        # Assemble
        model = SmolVLAFull(
            vlm=vlm,
            expert_layers=layers,
            suffix_enc_weights={
                "in_w": sd["model.action_in_proj.weight"], "in_b": sd["model.action_in_proj.bias"],
                "t_in_w": sd["model.action_time_mlp_in.weight"], "t_in_b": sd["model.action_time_mlp_in.bias"],
                "t_out_w": sd["model.action_time_mlp_out.weight"], "t_out_b": sd["model.action_time_mlp_out.bias"],
            },
            action_proj_weights={"w": sd["model.action_out_proj.weight"], "b": sd["model.action_out_proj.bias"]},
            state_proj_weights={"w": sd["model.state_proj.weight"], "b": sd["model.state_proj.bias"]},
            final_norm_weight=final_norm_w,
            cross_indices=cross_indices,
            vlm_kv_dim=vlm_kv_dim,
            expert_hidden=expert_hidden,
            action_dim=action_dim,
        )
        model = model.to(device).eval()
        total = sum(p.numel() for p in model.parameters())
        log("assemble", "pass",
            f"Full model: {total/1e6:.1f}M, {num_layers} expert layers, "
            f"cross_attn={cross_indices}, vlm_kv_dim={vlm_kv_dim}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        log("assemble", "fail", str(e)[:300])
        results["summary"] = {"passed": sum(1 for s in results["steps"] if s["status"]=="pass"), "failed": 1}
        return results

    # Step 4: End-to-end inference
    print("\n=== Step 4: End-to-end inference ===")
    try:
        # Dummy inputs matching SmolVLA format
        dummy_pixels = torch.randn(1, 1, 3, 384, 384, device=device)  # [batch, num_images, C, H, W]
        dummy_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        dummy_mask = torch.ones(1, 5, device=device, dtype=torch.long)
        dummy_state = torch.randn(1, 6, device=device)

        actions = model.inference(
            pixel_values=dummy_pixels,  # Keep 5D for SmolVLM
            input_ids=dummy_ids,
            attention_mask=dummy_mask,
            state=dummy_state,
        )
        log("e2e_inference", "pass",
            f"actions shape={list(actions.shape)}, range=[{actions.min():.3f}, {actions.max():.3f}]")

    except Exception as e:
        import traceback
        traceback.print_exc()
        log("e2e_inference", "fail", str(e)[:300])

    # Step 5: Benchmark end-to-end
    print("\n=== Step 5: Benchmark end-to-end ===")
    try:
        # Warmup
        for _ in range(3):
            model.inference(dummy_pixels, dummy_ids, dummy_mask, dummy_state)
        torch.cuda.synchronize()

        latencies = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.inference(dummy_pixels, dummy_ids, dummy_mask, dummy_state)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        mean_ms = sum(latencies) / len(latencies)
        log("benchmark_e2e", "pass",
            f"mean={mean_ms:.1f}ms, p50={latencies[10]:.1f}ms, p95={latencies[19]:.1f}ms, "
            f"{1000/mean_ms:.1f}Hz")

    except Exception as e:
        log("benchmark_e2e", "fail", str(e)[:200])

    # Step 6: Memory
    mem_mb = torch.cuda.max_memory_allocated() / 1e6
    log("memory", "pass", f"peak GPU: {mem_mb:.0f}MB ({mem_mb/1024:.2f}GB)")

    # Summary
    print("\n=== SUMMARY ===")
    passed = sum(1 for s in results["steps"] if s["status"] == "pass")
    failed = sum(1 for s in results["steps"] if s["status"] == "fail")
    print(f"Passed: {passed}, Failed: {failed}")
    results["summary"] = {"passed": passed, "failed": failed}
    return results


@app.local_entrypoint()
def main():
    print("Running end-to-end SmolVLA pipeline on Modal A100...")
    results = e2e_pipeline.remote()

    with open("/tmp/reflex_e2e.json", "w") as f:
        json.dump(results, f, indent=2)

    for step in results["steps"]:
        tag = "PASS" if step["status"] == "pass" else "FAIL"
        print(f"  {tag}: {step['step']} — {step['detail']}")
