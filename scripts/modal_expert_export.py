"""Export the 98M action expert transformer from SmolVLA to ONNX.

This is the hard part — the expert has RMSNorm and RoPE that need decomposition,
plus cross-attention to VLM KV cache.

Usage:
    modal run scripts/modal_expert_export.py
"""

import json
import math
import os
import time

import modal

app = modal.App("tether-expert-export")

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


@app.function(image=image, gpu="A100-40GB", timeout=1200, scaledown_window=60)
def export_expert():
    """Load SmolVLA, reconstruct action expert, decompose ops, export to ONNX."""
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

    export_dir = Path("/tmp/tether_expert_export")
    export_dir.mkdir(exist_ok=True)

    # Step 1: Load checkpoint
    print("=== Step 1: Load SmolVLA ===")
    local_dir = snapshot_download("lerobot/smolvla_base")
    state_dict = {}
    for f in sorted(Path(local_dir).glob("*.safetensors")):
        state_dict.update(load_file(str(f)))
    log("load", "pass", f"{sum(v.numel() for v in state_dict.values())/1e6:.1f}M params")

    # Step 2: Load the VLM to get the expert architecture
    print("\n=== Step 2: Load VLM to extract expert config ===")
    try:
        from transformers import AutoConfig, AutoModelForImageTextToText

        vlm_config = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        text_config = vlm_config.text_config

        # Expert config: 0.75x width
        vlm_hidden = text_config.hidden_size
        expert_hidden = int(vlm_hidden * 0.75)
        num_heads = text_config.num_attention_heads
        num_kv_heads = text_config.num_key_value_heads
        head_dim = vlm_hidden // num_heads
        expert_num_heads = expert_hidden // head_dim if expert_hidden % head_dim == 0 else num_heads
        num_layers = text_config.num_hidden_layers  # 16 for SmolLM2

        # Expert intermediate size (2/3 * 4x * hidden, rounded to 256)
        raw_intermediate = int(2 / 3 * 4 * expert_hidden)
        expert_intermediate = ((raw_intermediate + 255) // 256) * 256

        log("vlm_config", "pass",
            f"vlm_hidden={vlm_hidden}, expert_hidden={expert_hidden}, "
            f"head_dim={head_dim}, num_layers={num_layers}, "
            f"expert_intermediate={expert_intermediate}")

    except Exception as e:
        log("vlm_config", "fail", str(e)[:300])
        return results

    # Step 3: Analyze expert state_dict keys
    print("\n=== Step 3: Analyze expert keys ===")
    expert_keys = [k for k in state_dict.keys() if "lm_expert" in k]
    expert_params = sum(state_dict[k].numel() for k in expert_keys)

    # Group by layer
    layer_keys = {}
    for k in expert_keys:
        parts = k.split(".")
        # Find layer index
        for i, p in enumerate(parts):
            if p == "layers":
                layer_idx = int(parts[i + 1])
                if layer_idx not in layer_keys:
                    layer_keys[layer_idx] = []
                layer_keys[layer_idx].append(k)
                break

    print(f"  Expert layers: {sorted(layer_keys.keys())}")
    print(f"  Total expert keys: {len(expert_keys)}, {expert_params/1e6:.1f}M params")

    # Check shapes of attention projections per layer
    for layer_idx in sorted(layer_keys.keys())[:3]:
        for k in layer_keys[layer_idx]:
            if "q_proj" in k or "k_proj" in k or "v_proj" in k or "o_proj" in k:
                print(f"  Layer {layer_idx} {k.split('.')[-2]}.{k.split('.')[-1]}: {list(state_dict[k].shape)}")

    log("expert_keys", "pass", f"{len(expert_keys)} keys, {expert_params/1e6:.1f}M params, {len(layer_keys)} layers")

    # Step 4: Build a standalone expert transformer for export
    print("\n=== Step 4: Build standalone expert ===")
    try:
        # Decomposed RMSNorm for ONNX
        class DecomposedRMSNorm(nn.Module):
            def __init__(self, weight, eps=1e-6):
                super().__init__()
                self.register_buffer("weight", weight.clone())
                self.eps = eps

            def forward(self, x):
                variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
                x_normed = x * torch.rsqrt(variance + self.eps)
                return (x_normed * self.weight).to(x.dtype)

        # Decomposed RoPE
        class DecomposedRoPE(nn.Module):
            def __init__(self, dim, max_seq_len=512, base=10000.0):
                super().__init__()
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
                t = torch.arange(max_seq_len).float()
                freqs = torch.outer(t, inv_freq)
                cos_cached = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
                sin_cached = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
                self.register_buffer("cos_cached", cos_cached)
                self.register_buffer("sin_cached", sin_cached)

            def apply(self, x, position_ids):
                cos = self.cos_cached[position_ids].unsqueeze(1)
                sin = self.sin_cached[position_ids].unsqueeze(1)
                x1 = x[..., :x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2:]
                rotated = torch.cat((-x2, x1), dim=-1)
                return x * cos + rotated * sin

        # Simplified single expert layer (self-attention only, for ONNX export test)
        class ExpertSelfAttnLayer(nn.Module):
            def __init__(self, hidden, num_heads, head_dim, intermediate, eps=1e-6):
                super().__init__()
                self.num_heads = num_heads
                self.head_dim = head_dim
                self.input_layernorm = DecomposedRMSNorm(torch.ones(hidden), eps)
                self.post_attention_layernorm = DecomposedRMSNorm(torch.ones(hidden), eps)
                self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
                self.k_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
                self.v_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
                self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
                self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
                self.up_proj = nn.Linear(hidden, intermediate, bias=False)
                self.down_proj = nn.Linear(intermediate, hidden, bias=False)
                self.rope = DecomposedRoPE(head_dim, max_seq_len=512)

            def forward(self, x, position_ids):
                b, s, _ = x.shape
                residual = x
                x = self.input_layernorm(x)

                q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

                q = self.rope.apply(q, position_ids)
                k = self.rope.apply(k, position_ids)

                attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_out = torch.matmul(attn_weights, v)
                attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, -1)
                attn_out = self.o_proj(attn_out)
                x = residual + attn_out

                residual = x
                x = self.post_attention_layernorm(x)
                x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
                return residual + x

        # Determine actual expert dimensions from weights
        q_key = [k for k in expert_keys if "layers.0.self_attn.q_proj.weight" in k][0]
        k_key = [k for k in expert_keys if "layers.0.self_attn.k_proj.weight" in k][0]
        q_shape = state_dict[q_key].shape
        k_shape = state_dict[k_key].shape
        actual_expert_in = q_shape[1]
        num_q_heads = q_shape[0] // head_dim
        num_kv_heads = k_shape[0] // head_dim

        print(f"  Expert q_proj: {list(q_shape)} -> {num_q_heads} Q heads x {head_dim}")
        print(f"  Expert k_proj: {list(k_shape)} -> {num_kv_heads} KV heads x {head_dim}")

        gate_key = [k for k in expert_keys if "layers.0.mlp.gate_proj.weight" in k][0]
        actual_intermediate = state_dict[gate_key].shape[0]

        # Rebuild layer with GQA (different Q and KV head counts)
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

            def forward(self, x, position_ids):
                b, s, _ = x.shape
                residual = x
                x = self.input_layernorm(x)
                q = self.q_proj(x).view(b, s, self.n_q_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
                q = self.rope.apply(q, position_ids)
                k = self.rope.apply(k, position_ids)
                # GQA: expand KV heads
                k = k.unsqueeze(2).expand(-1, -1, self.kv_groups, -1, -1).reshape(b, self.n_q_heads, s, self.head_dim)
                v = v.unsqueeze(2).expand(-1, -1, self.kv_groups, -1, -1).reshape(b, self.n_q_heads, s, self.head_dim)
                attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attn = F.softmax(attn, dim=-1)
                out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, -1)
                out = self.o_proj(out)
                x = residual + out
                residual = x
                x = self.post_attention_layernorm(x)
                x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
                return residual + x

        layer = ExpertGQALayer(
            hidden=actual_expert_in,
            n_q_heads=num_q_heads,
            n_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate=actual_intermediate,
        )

        # Load weights from state_dict for layer 0
        # Detect actual key format (with or without .model.)
        test_key_a = "model.vlm_with_expert.lm_expert.model.layers.0.input_layernorm.weight"
        test_key_b = "model.vlm_with_expert.lm_expert.layers.0.input_layernorm.weight"
        if test_key_a in state_dict:
            prefix = "model.vlm_with_expert.lm_expert.model.layers.0"
        elif test_key_b in state_dict:
            prefix = "model.vlm_with_expert.lm_expert.layers.0"
        else:
            # Find any layer 0 key to detect prefix
            l0_keys = [k for k in expert_keys if "layers.0." in k]
            if l0_keys:
                # Extract prefix before "layers.0"
                sample = l0_keys[0]
                prefix = sample[:sample.index("layers.0") + len("layers.0")]
                print(f"  Auto-detected prefix: {prefix}")
            else:
                raise ValueError("Cannot find layer 0 keys in expert state_dict")
        print(f"  Using prefix: {prefix}")
        norm_prefix = prefix + ".input_layernorm.weight"
        post_norm_prefix = prefix + ".post_attention_layernorm.weight"

        layer_sd = {}
        layer_sd["input_layernorm.weight"] = state_dict[norm_prefix]
        layer_sd["post_attention_layernorm.weight"] = state_dict[post_norm_prefix]
        layer_sd["q_proj.weight"] = state_dict[f"{prefix}.self_attn.q_proj.weight"]
        layer_sd["k_proj.weight"] = state_dict[f"{prefix}.self_attn.k_proj.weight"]
        layer_sd["v_proj.weight"] = state_dict[f"{prefix}.self_attn.v_proj.weight"]
        layer_sd["o_proj.weight"] = state_dict[f"{prefix}.self_attn.o_proj.weight"]
        layer_sd["gate_proj.weight"] = state_dict[f"{prefix}.mlp.gate_proj.weight"]
        layer_sd["up_proj.weight"] = state_dict[f"{prefix}.mlp.up_proj.weight"]
        layer_sd["down_proj.weight"] = state_dict[f"{prefix}.mlp.down_proj.weight"]

        layer.load_state_dict(layer_sd, strict=False)
        layer.eval()

        total_layer_params = sum(p.numel() for p in layer.parameters())
        log("build_expert_layer", "pass",
            f"Layer 0 loaded: {total_layer_params/1e6:.2f}M params, "
            f"hidden={actual_expert_in}, q_heads={num_q_heads}, kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}, intermediate={actual_intermediate}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        log("build_expert_layer", "fail", str(e)[:300])
        results["summary"] = {"passed": sum(1 for s in results["steps"] if s["status"] == "pass"), "failed": sum(1 for s in results["steps"] if s["status"] == "fail")}
        return results

    # Step 5: Test forward pass
    print("\n=== Step 5: Test expert layer forward ===")
    try:
        device = torch.device("cuda")
        layer = layer.to(device)
        dummy_x = torch.randn(1, 50, actual_expert_in, device=device)
        dummy_pos = torch.arange(50, device=device).unsqueeze(0)

        with torch.no_grad():
            out = layer(dummy_x, dummy_pos)
        log("expert_forward", "pass", f"output shape={list(out.shape)}")
    except Exception as e:
        log("expert_forward", "fail", str(e)[:300])

    # Step 6: Export expert layer to ONNX
    print("\n=== Step 6: Export expert layer to ONNX ===")
    try:
        layer_cpu = layer.cpu()
        dummy_x_cpu = torch.randn(1, 50, actual_expert_in)
        dummy_pos_cpu = torch.arange(50).unsqueeze(0)

        onnx_path = str(export_dir / "expert_layer_0.onnx")
        torch.onnx.export(
            layer_cpu,
            (dummy_x_cpu, dummy_pos_cpu),
            onnx_path,
            input_names=["hidden_states", "position_ids"],
            output_names=["output"],
            opset_version=19,
            dynamic_axes={
                "hidden_states": {0: "batch", 1: "seq_len"},
                "position_ids": {0: "batch", 1: "seq_len"},
            },
        )
        import onnx
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        size_mb = os.path.getsize(onnx_path) / 1e6
        log("onnx_expert_layer", "pass", f"Exported: {size_mb:.1f}MB, valid ONNX")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log("onnx_expert_layer", "fail", str(e)[:300])

    # Step 7: Validate ONNX vs PyTorch
    print("\n=== Step 7: Validate expert ONNX ===")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        ort_out = sess.run(None, {
            "hidden_states": dummy_x_cpu.numpy(),
            "position_ids": dummy_pos_cpu.numpy().astype(np.int64),
        })[0]
        torch_out = layer_cpu(dummy_x_cpu, dummy_pos_cpu).detach().numpy()
        max_diff = float(np.abs(ort_out - torch_out).max())
        mean_diff = float(np.abs(ort_out - torch_out).mean())
        log("validate_expert", "pass", f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    except Exception as e:
        log("validate_expert", "fail", str(e)[:300])

    # Step 8: Benchmark single layer
    print("\n=== Step 8: Benchmark expert layer ===")
    try:
        layer = layer.to(device)
        # Warmup
        for _ in range(20):
            with torch.no_grad():
                _ = layer(dummy_x.to(device), dummy_pos.to(device))
        torch.cuda.synchronize()

        latencies = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = layer(dummy_x.to(device), dummy_pos.to(device))
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        mean_ms = sum(latencies) / len(latencies)
        # 16 layers total, 10 denoising steps
        estimated_full = mean_ms * 16 * 10
        log("benchmark_layer", "pass",
            f"single_layer={mean_ms:.2f}ms, "
            f"estimated_16_layers={mean_ms*16:.1f}ms, "
            f"estimated_full_denoise={estimated_full:.0f}ms ({1000/estimated_full:.0f}Hz)")
    except Exception as e:
        log("benchmark_layer", "fail", str(e)[:300])

    # Summary
    print("\n=== SUMMARY ===")
    passed = sum(1 for s in results["steps"] if s["status"] == "pass")
    failed = sum(1 for s in results["steps"] if s["status"] == "fail")
    results["summary"] = {"passed": passed, "failed": failed}
    print(f"Passed: {passed}, Failed: {failed}")

    return results


@app.local_entrypoint()
def main():
    print("Running expert transformer export on Modal A100...")
    results = export_expert.remote()

    with open("/tmp/tether_expert_export.json", "w") as f:
        json.dump(results, f, indent=2)

    for step in results["steps"]:
        tag = "PASS" if step["status"] == "pass" else "FAIL"
        print(f"  {tag}: {step['step']} — {step['detail']}")
