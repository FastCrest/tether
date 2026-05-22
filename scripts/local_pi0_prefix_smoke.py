"""Smoke test for pi0_prefix exporter — verify the fast (non-Optimum) paths.

Tests the two components pi0_prefix.py exports directly:
  1. multi_modal_projector (Linear 1152 -> 2048)
  2. text_embedder (Embedding 257152 x 2048, tied with lm_head)

Both should be bit-exact or near-bit-exact (cos ~= 1.0).

Does NOT re-run the Optimum exports for SigLIP and Gemma (those are
validated separately by scripts/local_pi0_siglip_parity.py and
scripts/local_pi0_gemma_parity.py). Runs in seconds.

Expected:
  projector PARITY: max_abs=<1e-5 cos=+1.0
  embed_tokens PARITY: max_abs=0 cos=+1.0
"""
import sys
import types

for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
    _stub = types.ModuleType(_mod)
    _stub.GrootPolicy = None
    _stub.GR00TN15 = None
    sys.modules[_mod] = _stub

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from reflex.exporters.pi0_prefix import (
    build_embed_tokens,
    build_projector,
    export_embed_tokens_onnx,
    export_projector_onnx,
    load_pi0_state_dict,
)


def main():
    out = Path("/tmp/pi0_prefix_export_smoke")
    out.mkdir(exist_ok=True)

    print("Loading pi0 state dict (from HF cache)...")
    state = load_pi0_state_dict("lerobot/pi0_base")

    # 1. Projector
    print("\n1. Multi-modal projector:")
    proj = build_projector(state)
    proj_onnx = export_projector_onnx(proj, out / "multi_modal_projector.onnx")

    dummy_vision = torch.randn(1, 256, 1152, dtype=torch.float32,
                               generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        pt = proj(dummy_vision).numpy()
    sess = ort.InferenceSession(str(proj_onnx), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"vision_features": dummy_vision.numpy()})[0]
    max_abs = float(np.abs(pt - ort_out).max())
    cos = float(
        np.dot(pt.flatten(), ort_out.flatten())
        / (np.linalg.norm(pt) * np.linalg.norm(ort_out) + 1e-8)
    )
    print(f"  projector PARITY: max_abs={max_abs:.2e} cos={cos:+.8f}")
    ok1 = max_abs < 1e-4 and cos > 0.9999

    # 2. Embed tokens
    print("\n2. Text embedder (tied lm_head):")
    emb = build_embed_tokens(state)
    emb_onnx = export_embed_tokens_onnx(emb, out / "text_embedder.onnx")

    dummy_tokens = torch.randint(0, emb.embed.num_embeddings, (1, 16), dtype=torch.long,
                                 generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        pt = emb(dummy_tokens).numpy()
    sess = ort.InferenceSession(str(emb_onnx), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input_ids": dummy_tokens.numpy()})[0]
    max_abs = float(np.abs(pt - ort_out).max())
    cos = float(
        np.dot(pt.flatten(), ort_out.flatten())
        / (np.linalg.norm(pt) * np.linalg.norm(ort_out) + 1e-8)
    )
    print(f"  embed_tokens PARITY: max_abs={max_abs:.2e} cos={cos:+.8f}")
    ok2 = max_abs < 1e-4 and cos > 0.9999

    print(f"\nSMOKE TEST: {'PASS' if ok1 and ok2 else 'FAIL'}")
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
