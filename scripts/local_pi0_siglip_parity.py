"""pi0 SigLIP vision tower parity: PyTorch vs Optimum ONNX.

Extracts pi0's SigLIP vision tower from cached pi0_base safetensors,
saves as standard HF SiglipVisionModel dir, exports via Optimum ONNX,
and compares PyTorch vs ONNX on the same seeded image input.

Result on 2026-04-17 first run:
  last_hidden_state cos_sim       = +0.99999994
  last_hidden_state max_abs_diff  = 3.59e-04
  VERDICT: PASS

Companion to scripts/local_pi0_gemma_parity.py. Together they validate
both halves of PaliGemma (vision + language) independently via Optimum.

Prerequisites:
  pip install 'fastcrest-tether[native]' 'optimum[onnxruntime]>=1.22' onnx onnxruntime
"""
import sys
import types

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
from transformers import SiglipVisionConfig, SiglipVisionModel


SAVE_DIR = Path("/tmp/pi0_siglip_vision")
ONNX_DIR = Path("/tmp/pi0_siglip_onnx")


def extract_and_save():
    repo_dir = Path(snapshot_download("lerobot/pi0_base"))
    sf_files = list(repo_dir.glob("*.safetensors"))
    prefix = "paligemma_with_expert.paligemma.model.vision_tower."

    # SigLIP so400m-patch14-224 (what PaliGemma-3B uses)
    cfg = SiglipVisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
    )
    model = SiglipVisionModel(cfg).eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"target model: {total_params / 1e6:.1f}M params")

    state_dict = {}
    with safe_open(sf_files[0], framework="pt") as sf:
        for k in sf.keys():
            if k.startswith(prefix):
                state_dict[k[len(prefix):]] = sf.get_tensor(k)
    print(f"loaded {len(state_dict)} vision tower keys from pi0_base")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"load: missing={len(missing)} unexpected={len(unexpected)}")
    # Missing keys are the attention-pool head (vision_model.head.*) — pi0 doesn't use it

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    (SAVE_DIR / "preprocessor_config.json").write_text(
        json.dumps({
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "rescale_factor": 1 / 255.0,
            "resample": 3,
            "size": {"height": 224, "width": 224},
            "image_processor_type": "SiglipImageProcessor",
        })
    )
    print(f"saved to {SAVE_DIR}")


def export_onnx():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        ".venv/bin/optimum-cli", "export", "onnx",
        "--model", str(SAVE_DIR),
        "--task", "feature-extraction",
        "--framework", "pt",
        str(ONNX_DIR),
    ]
    print("running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"exported to {ONNX_DIR}")


def verify_parity():
    pt_model = SiglipVisionModel.from_pretrained(SAVE_DIR).eval()

    torch.manual_seed(42)
    pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        pt_out = pt_model(pixel_values, output_hidden_states=False)
    pt_hidden = pt_out.last_hidden_state.numpy()

    sess = ort.InferenceSession(str(ONNX_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {"pixel_values": pixel_values.numpy()})
    ort_hidden = ort_outs[0]

    abs_diff = np.abs(pt_hidden - ort_hidden)
    cos = float(
        np.dot(pt_hidden.flatten(), ort_hidden.flatten())
        / (np.linalg.norm(pt_hidden) * np.linalg.norm(ort_hidden) + 1e-8)
    )

    print(f"last_hidden_state max_abs_diff:  {abs_diff.max():.2e}")
    print(f"last_hidden_state mean_abs_diff: {abs_diff.mean():.2e}")
    print(f"last_hidden_state cos_sim:       {cos:+.8f}")

    passed = abs_diff.max() < 1e-3 and cos > 0.9999
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    if not SAVE_DIR.exists():
        print("Step 1: extracting pi0 SigLIP...")
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
