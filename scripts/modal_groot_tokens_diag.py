"""Modal CPU diagnostic (M4 #53): localize the GR00T full-chunk max_abs miss.

The A100 receipt run at 4cf9854 passed placement and cosine but failed
full_max_abs (1.2579e-03 vs < 1e-3; first-action 2.99e-04 passes). This
cheap CPU job reuses the retained export + cached snapshot and reports:

1. per-chunk-position max-abs profile (50 values) of reference vs export,
2. the argmax element (position, dim, ref/export values),
3. pre-decoder (DiT velocity-token) parity via a dit-only ONNX export,
   isolating decoder gain from DiT divergence.

No GPU, no gate touched. Read-only against the retained export.

Usage:
    modal run scripts/modal_groot_tokens_diag.py
"""

import modal

app = modal.App("tether-groot-tokens-diag")
_hf_cache_volume = modal.Volume.from_name("gr00t-hf-cache", create_if_missing=True)
_parity_out_volume = modal.Volume.from_name(
    "parity-gpu-receipts", create_if_missing=True
)

HF_CACHE_PATH = "/root/.cache/huggingface"
PARITY_OUT_PATH = "/parity_out"

TETHER_REVISION = "4cf9854a72ca9c97ec6a7a398f4246650b001d2a"
TETHER_REPO = "https://github.com/FastCrest/tether.git"
GROOT_REVISION = "d0814e7ecb19202e7c8468b46098b0b7ef3a6d61"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers==5.3.0",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "lerobot==0.5.1",
        "num2words",
        "onnx-diagnostic>=0.9",
        "optree",
        "scipy",
        "accelerate",
        "draccus",
        "numpy",
        "Pillow",
        "typer",
        "rich",
        "pydantic>=2.0",
        "pyyaml",
    )
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)


@app.function(
    image=image,
    timeout=3600,
    volumes={
        HF_CACHE_PATH: _hf_cache_volume,
        PARITY_OUT_PATH: _parity_out_volume,
    },
)
def run_diag() -> dict:
    import subprocess
    import sys
    from pathlib import Path

    import numpy as _np
    import torch as _torch

    pin_dir = Path("/root/tether-pin")
    if pin_dir.exists():
        subprocess.run(["rm", "-rf", str(pin_dir)], check=True)
    subprocess.run(["git", "clone", "--quiet", TETHER_REPO, str(pin_dir)], check=True)
    subprocess.run(
        ["git", "-C", str(pin_dir), "checkout", "--quiet", TETHER_REVISION],
        check=True,
    )
    sys.path.insert(0, str(pin_dir / "src"))

    from huggingface_hub import snapshot_download

    from tether.checkpoint import load_checkpoint
    from tether.exporters.gr00t import build_gr00t_full_stack

    snap = snapshot_download("nvidia/GR00T-N1.6-3B", revision=GROOT_REVISION)
    state_dict, _ = load_checkpoint(snap)
    full, metadata = build_gr00t_full_stack(state_dict, embodiment_id=0)
    full.eval()
    raw_action_dim = int(metadata["raw_action_dim"])
    print(f"[diag] raw_action_dim={raw_action_dim}", flush=True)

    input_rng = _np.random.RandomState(42)
    noise_rng = _np.random.RandomState(99)
    shared = {
        "noisy_actions": noise_rng.randn(1, 50, raw_action_dim).astype(_np.float32),
        "timestep": _np.array([input_rng.uniform(0.05, 0.95)], dtype=_np.float32),
        "position_ids": _np.arange(50, dtype=_np.int64)[None, :],
    }
    noisy_actions = _torch.from_numpy(shared["noisy_actions"])
    timestep = _torch.from_numpy(shared["timestep"])
    position_ids = _torch.from_numpy(shared["position_ids"])

    with _torch.no_grad():
        reference = full(noisy_actions, timestep, position_ids)
    reference_np = reference.numpy().astype(_np.float32)

    import onnxruntime as _ort

    export_path = Path(PARITY_OUT_PATH) / "groot" / "export" / "model.onnx"
    sess = _ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
    export_np = _np.asarray(sess.run(None, shared)[0], dtype=_np.float32)

    abs_err = _np.abs(reference_np - export_np)
    per_pos = abs_err[0].max(axis=1)
    flat = int(_np.argmax(abs_err))
    pos, dim = divmod(flat, raw_action_dim)
    print("[diag] per-position max_abs:", flush=True)
    for i, v in enumerate(per_pos):
        print(f"[diag]   pos={i:2d} max_abs={v:.4e}", flush=True)
    print(
        f"[diag] argmax pos={pos} dim={dim} ref={float(reference_np[0,pos,dim]):.6f} "
        f"ort={float(export_np[0,pos,dim]):.6f} err={float(abs_err[0,pos,dim]):.4e}",
        flush=True,
    )
    print(
        f"[diag] ref scale: max={float(_np.abs(reference_np).max()):.4f} "
        f"mean_abs={float(_np.abs(reference_np).mean()):.4e}",
        flush=True,
    )

    # Pre-decoder isolation: dit-only ONNX vs torch dit on identical inputs.
    # Recompute the forward's internals exactly as GR00TFullStack.forward does:
    from tether.exporters.gr00t import _sinusoidal_timestep

    with _torch.no_grad():
        t_sin_v = _sinusoidal_timestep(timestep, full.dit.sinusoidal_dim)
        time_emb_v = full.dit.timestep_linear_2(
            _torch.nn.functional.silu(full.dit.timestep_linear_1(t_sin_v))
        )
        action_tokens = full.action_encoder(noisy_actions, time_emb_v)
        # add_pos_embed defaults True: pass raw action_tokens exactly as
        # GR00TFullStack.forward does on the state=None path.
        ref_tokens = full.dit(action_tokens, timestep, position_ids, vlm_kv=None)
        ref_out = full.action_decoder(ref_tokens)
    print(
        f"[diag] torch dit->decoder recompute matches full: "
        f"{bool(_np.allclose(ref_out.numpy(), reference_np, atol=0))}",
        flush=True,
    )

    import torch.nn as _nn

    class _DitWrap(_nn.Module):
        def __init__(self, dit):
            super().__init__()
            self.dit = dit

        def forward(self, tokens, t, p):
            return self.dit(tokens, t, p, vlm_kv=None)

    wrap = _DitWrap(full.dit).eval()
    dit_path = Path("/tmp/dit_only.onnx")
    with _torch.no_grad():
        _torch.onnx.export(
            wrap,
            (action_tokens, timestep, position_ids),
            str(dit_path),
            input_names=["tokens", "timestep", "position_ids"],
            output_names=["velocity_tokens"],
            opset_version=19,
        )
    dit_sess = _ort.InferenceSession(str(dit_path), providers=["CPUExecutionProvider"])
    ort_tokens = _np.asarray(
        dit_sess.run(
            None,
            {
                "tokens": action_tokens.numpy(),
                "timestep": shared["timestep"],
                "position_ids": shared["position_ids"],
            },
        )[0],
        dtype=_np.float32,
    )
    ref_tokens_np = ref_tokens.numpy().astype(_np.float32)
    tok_err = _np.abs(ref_tokens_np - ort_tokens)
    print(
        f"[diag] PRE-DECODER tokens: max_abs={float(tok_err.max()):.4e} "
        f"cos={float(_np.dot(ref_tokens_np.reshape(-1).astype(_np.float64), ort_tokens.reshape(-1).astype(_np.float64)) / (_np.linalg.norm(ref_tokens_np.reshape(-1).astype(_np.float64)) * _np.linalg.norm(ort_tokens.reshape(-1).astype(_np.float64)))):.10f}",
        flush=True,
    )
    # Decoder gain: push the ORT token error through the torch decoder.
    with _torch.no_grad():
        gain_probe = full.action_decoder(_torch.from_numpy(ort_tokens)).numpy()
    print(
        f"[diag] decoder gain probe (torch decoder on ORT tokens): "
        f"max_abs={float(_np.abs(gain_probe - reference_np).max()):.4e}",
        flush=True,
    )
    return {
        "full_max_abs": float(abs_err.max()),
        "per_pos_max": [float(v) for v in per_pos],
        "argmax": {"pos": pos, "dim": dim},
        "pre_decoder_max_abs": float(tok_err.max()),
    }


@app.local_entrypoint()
def main() -> None:
    result = run_diag.remote()
    print("\n=== RESULT ===")
    print(f"  full_max_abs: {result['full_max_abs']}")
    print(f"  pre_decoder_max_abs: {result['pre_decoder_max_abs']}")
