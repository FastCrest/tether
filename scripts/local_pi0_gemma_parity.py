"""Real-scale pi0 Gemma backbone parity: PyTorch vs Optimum ONNX.

Extracts pi0's Gemma backbone from the cached pi0_base safetensors, saves as
a standard HF GemmaForCausalLM dir, exports via Optimum ONNX, and compares
PyTorch vs ONNX on the same seeded input.

Result on 2026-04-17 first run:
  logits cos_sim       = +0.99999994
  logits max_abs_diff  = 3.86e-05
  present.17.key diff  = 3.02e-05 (deepest layer)
  VERDICT: PASS

This is load-bearing real-scale evidence that pi0-onnx-parity is viable
via the Optimum Gemma path. See
reflex_context/03_research/pi0_empirical_derisk_findings.md

Prerequisites:
  pip install 'fastcrest-tether[native]' 'optimum[onnxruntime]>=1.22' onnx onnxruntime
  # Also: pi0_base cached in ~/.cache/huggingface (14GB), see scripts/local_full_diff.py
"""
import sys
import types

# Python 3.13 + lerobot 0.5.1 compat shim (same as local_full_diff.py)
for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
    _stub = types.ModuleType(_mod)
    _stub.GrootPolicy = None
    _stub.GR00TN15 = None
    sys.modules[_mod] = _stub

import json
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import GemmaConfig, GemmaForCausalLM


SAVE_DIR = Path("/tmp/pi0_gemma_backbone")
ONNX_DIR = Path("/tmp/pi0_gemma_onnx")


def extract_and_save():
    """Extract pi0's Gemma backbone from cached pi0_base and save as HF dir."""
    repo_dir = Path(snapshot_download("lerobot/pi0_base"))
    sf_files = list(repo_dir.glob("*.safetensors"))
    if not sf_files:
        raise FileNotFoundError(f"no safetensors in {repo_dir}")

    backbone_prefix = "paligemma_with_expert.paligemma.model.language_model."

    # Build a standalone Gemma-2b via transformers (pi0 uses exactly this arch)
    cfg = GemmaConfig(
        vocab_size=256000,
        hidden_size=2048,
        intermediate_size=16384,
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    model = GemmaForCausalLM(cfg).eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"target model: {total_params / 1e9:.2f}B params")

    state_dict = {}
    with safe_open(sf_files[0], framework="pt") as sf:
        for k in sf.keys():
            if k.startswith(backbone_prefix):
                state_dict[k[len(backbone_prefix):]] = sf.get_tensor(k)
    print(f"loaded {len(state_dict)} backbone keys from pi0_base")

    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    print(f"load: missing={len(missing)} unexpected={len(unexpected)}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    (SAVE_DIR / "tokenizer_config.json").write_text(
        json.dumps({
            "tokenizer_class": "GemmaTokenizer",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
        })
    )
    print(f"saved to {SAVE_DIR}")


def export_onnx():
    """Run optimum-cli export with text-generation-with-past task."""
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        ".venv/bin/optimum-cli", "export", "onnx",
        "--model", str(SAVE_DIR),
        "--task", "text-generation-with-past",
        "--framework", "pt",
        str(ONNX_DIR),
    ]
    print("running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"exported to {ONNX_DIR}")


def verify_parity():
    """Run PyTorch and ORT on the same seeded input, compute cos + max_abs."""
    pt_model = GemmaForCausalLM.from_pretrained(SAVE_DIR).eval()

    torch.manual_seed(42)
    input_ids = torch.randint(0, 256000, (1, 8), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        pt_out = pt_model(input_ids, attention_mask=attention_mask, use_cache=True)
    pt_logits = pt_out.logits.numpy()

    sess = ort.InferenceSession(str(ONNX_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    }
    for i in range(pt_model.config.num_hidden_layers):
        ort_inputs[f"past_key_values.{i}.key"] = np.zeros(
            (1, pt_model.config.num_key_value_heads, 0, pt_model.config.head_dim),
            dtype=np.float32,
        )
        ort_inputs[f"past_key_values.{i}.value"] = np.zeros(
            (1, pt_model.config.num_key_value_heads, 0, pt_model.config.head_dim),
            dtype=np.float32,
        )

    ort_outs = sess.run(None, ort_inputs)
    ort_logits = ort_outs[0]

    abs_diff = np.abs(pt_logits - ort_logits)
    cos = float(
        np.dot(pt_logits.flatten(), ort_logits.flatten())
        / (np.linalg.norm(pt_logits) * np.linalg.norm(ort_logits) + 1e-8)
    )

    print(f"logits max_abs_diff:  {abs_diff.max():.2e}")
    print(f"logits mean_abs_diff: {abs_diff.mean():.2e}")
    print(f"logits cos_sim:       {cos:+.8f}")

    pt_kv = pt_out.past_key_values
    if hasattr(pt_kv, "layers"):
        for layer_idx in [0, pt_model.config.num_hidden_layers - 1]:
            pt_k = pt_kv.layers[layer_idx].keys.numpy()
            ort_k = ort_outs[1 + layer_idx * 2]
            k_diff = np.abs(pt_k - ort_k).max()
            print(f"present.{layer_idx}.key diff:  {k_diff:.2e}")

    passed = abs_diff.max() < 1e-3 and cos > 0.9999
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    if not SAVE_DIR.exists():
        print("Step 1: extracting pi0 Gemma backbone...")
        extract_and_save()
    else:
        print(f"Step 1: using cached {SAVE_DIR}")

    if not (ONNX_DIR / "model.onnx").exists():
        print("Step 2: exporting to ONNX via Optimum...")
        export_onnx()
    else:
        print(f"Step 2: using cached {ONNX_DIR}")

    print("\nStep 3: verifying parity (PyTorch vs ORT)...")
    return 0 if verify_parity() else 1


if __name__ == "__main__":
    sys.exit(main())
