"""Local stage-by-stage diff. No Modal. Runs in ~30 sec per iteration.

Usage:
    python scripts/local_stage_diff.py [--export-dir PATH]

Expects /tmp/tether_libero_export/ to already have our ONNX files. If not,
run `tether export lerobot/smolvla_libero --target desktop --output /tmp/tether_libero_export/` first.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def diff(name, torch_arr, onnx_arr):
    t = np.asarray(torch_arr, dtype=np.float64).flatten()
    o = np.asarray(onnx_arr, dtype=np.float64).flatten()
    if t.shape != o.shape:
        print(f"  {name}: SHAPE MISMATCH torch={t.shape} onnx={o.shape}")
        return
    max_abs = float(np.abs(t - o).max())
    l2 = float(np.linalg.norm(t - o))
    cos = float(np.dot(t, o) / (np.linalg.norm(t) * np.linalg.norm(o) + 1e-8))
    print(f"  {name}: max_abs={max_abs:.4e}  L2={l2:.4e}  cos={cos:+.4f}  "
          f"torch||={np.linalg.norm(t):.3e}  onnx||={np.linalg.norm(o):.3e}")


def main(export_dir: str):
    print("=" * 60)
    print("LOCAL stage diff: PyTorch vs ONNX")
    print("=" * 60)

    # ── Load policy ─────────────────────────────────────────────────
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")
    policy.eval()
    policy.to(dtype=torch.float32)
    device = "cpu"
    policy.to(device)

    vlm = policy.model.vlm_with_expert.get_vlm_model()
    vision = vlm.vision_tower if hasattr(vlm, "vision_tower") else vlm.vision_model
    connector = vlm.connector
    text_model = vlm.text_model

    # ── Build input ─────────────────────────────────────────────────
    rng = np.random.RandomState(42)
    H, W = 512, 512
    img_uint8 = rng.randint(0, 255, (H, W, 3), dtype=np.uint8)
    img_f = img_uint8.astype(np.float32) / 255.0
    img_f = img_f * 2.0 - 1.0
    pix_np = np.transpose(img_f, (2, 0, 1))[None, ...].astype(np.float32)

    # ── STAGE 1: vision ─────────────────────────────────────────────
    print("\nSTAGE 1: Vision encoder")
    pix_t = torch.from_numpy(pix_np)
    with torch.no_grad():
        vout = vision(pix_t)
        vfeats = vout.last_hidden_state if hasattr(vout, "last_hidden_state") else vout
        if connector is not None:
            vfeats = connector(vfeats)
    torch_vision = vfeats.numpy()

    import onnxruntime as ort
    sess = ort.InferenceSession(f"{export_dir}/vision_encoder.onnx",
                                providers=["CPUExecutionProvider"])
    onnx_vision = sess.run(None, {"pixel_values": pix_np})[0]
    diff("vision_embeds", torch_vision, onnx_vision)

    # ── STAGE 2: scale image embeds by sqrt(hidden) per real embed_prefix ───
    print("\nSTAGE 2: scaled vision (real embed_prefix scales by sqrt(hidden))")
    hidden = torch_vision.shape[-1]
    scale = float(hidden) ** 0.5
    torch_scaled = torch_vision * scale
    onnx_scaled = onnx_vision * scale   # should match torch side exactly
    diff("scaled_vision", torch_scaled, onnx_scaled)
    print(f"  scale factor: {scale:.3f}")
    print(f"  torch raw magnitude: {np.linalg.norm(torch_vision):.3f}")
    print(f"  torch scaled magnitude: {np.linalg.norm(torch_scaled):.3f}")

    # ── STAGE 3: pass thru decoder with scaled vs unscaled embeds ───
    print("\nSTAGE 3: decoder output sensitivity to scaling")

    # With unscaled input
    with torch.no_grad():
        mask = torch.ones(1, torch_vision.shape[1], dtype=torch.long)
        out_unscaled = text_model(
            inputs_embeds=torch.from_numpy(torch_vision),
            attention_mask=mask,
            output_hidden_states=False,
        ).last_hidden_state.numpy()
        out_scaled = text_model(
            inputs_embeds=torch.from_numpy(torch_scaled),
            attention_mask=mask,
            output_hidden_states=False,
        ).last_hidden_state.numpy()
    print(f"  unscaled decoder output norm: {np.linalg.norm(out_unscaled):.3f}")
    print(f"  scaled decoder output norm: {np.linalg.norm(out_scaled):.3f}")
    diff("unscaled_vs_scaled", out_unscaled, out_scaled)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default="/tmp/tether_libero_export")
    args = parser.parse_args()
    if not Path(args.export_dir).exists():
        print(f"ERROR: {args.export_dir} doesn't exist. Run `tether export "
              f"lerobot/smolvla_libero --target desktop --output {args.export_dir}` first.")
        sys.exit(1)
    main(args.export_dir)
