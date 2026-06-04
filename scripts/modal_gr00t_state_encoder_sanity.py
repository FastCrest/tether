"""Step 3 sanity — extended GR00TFullStack loads + runs + responds to state.

Verifies the 2026-04-19 Step 2 port:
  1. build_gr00t_full_stack loads state_encoder from
     action_head.state_encoder.* keys (should detect them in N1.6)
  2. GR00TFullStack.forward accepts (noisy_actions, timestep, position_ids,
     state, vlm_kv) without errors, produces [B, chunk, action_dim].
  3. Changing the `state` input produces DIFFERENT velocity output than
     state=None. If output is identical, state conditioning is not actually
     wired in (dead code path).
  4. Changing `vlm_kv` from zero → random produces DIFFERENT output,
     confirming VLM cross-attn plumbing is live.

This is NOT a full parity test against lerobot's GR00TN15 — that needs
Eagle export first. This is a code-correctness sanity run.

Usage:
    modal run scripts/modal_gr00t_state_encoder_sanity.py
"""
import os
import subprocess
import modal

app = modal.App("tether-gr00t-state-sanity")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "clang")
    .pip_install(
        "torch",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy",
        "Pillow",
        "pydantic>=2.0",
        "pyyaml",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "typer",
        "rich",
    )
    .run_commands(
        f'pip install "tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether-vla@{_HEAD}"',
            secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1200,
    secrets=[_hf_secret()],
)
def run_sanity(model_id: str = "nvidia/GR00T-N1.6-3B"):
    import time
    import numpy as np
    import torch

    from tether.checkpoint import load_checkpoint
    from tether.exporters.gr00t import build_gr00t_full_stack

    print(f"[sanity] Loading {model_id}...")
    t0 = time.time()
    state_dict, _ = load_checkpoint(model_id)
    print(f"[sanity] Checkpoint loaded in {time.time()-t0:.1f}s, "
          f"{len(state_dict)} tensors")

    # Build the extended full stack
    t0 = time.time()
    full, meta = build_gr00t_full_stack(state_dict, embodiment_id=0)
    full = full.float().eval().to("cuda")
    print(f"[sanity] Stack built in {time.time()-t0:.1f}s")
    print(f"[sanity] meta:")
    for k, v in meta.items():
        print(f"         {k}: {v}")

    assert meta.get("has_state_encoder"), (
        "state_encoder was NOT loaded — keys missing from state_dict "
        "or soft-load path didn't trigger"
    )
    print(f"[sanity] ✓ state_encoder loaded ({meta['raw_state_dim']} → "
          f"{meta['state_hidden_out']})")

    # ─── Dummy inputs ───────────────────────────────────────────────
    torch.manual_seed(42)
    B = 1
    chunk = 50
    raw_action_dim = meta["raw_action_dim"]
    raw_state_dim = meta["raw_state_dim"]

    noisy_actions = torch.randn(B, chunk, raw_action_dim, device="cuda")
    timestep = torch.tensor([0.5], device="cuda")
    # Position ids: chunk+1 to accommodate prepended state token
    position_ids = torch.arange(chunk + 1, device="cuda").unsqueeze(0)
    state_a = torch.randn(B, raw_state_dim, device="cuda")  # state variant A
    state_b = torch.randn(B, raw_state_dim, device="cuda")  # state variant B (different)
    vlm_kv_zeros = torch.zeros(B, 256, 2048, device="cuda")
    vlm_kv_rand = torch.randn(B, 256, 2048, device="cuda")

    # ─── Test 1: forward runs (no errors, right output shape) ─────
    print(f"\n[sanity] Test 1: basic forward")
    with torch.no_grad():
        out = full(noisy_actions, timestep, position_ids,
                    state=state_a, vlm_kv=vlm_kv_zeros)
    print(f"         output shape: {tuple(out.shape)}")
    assert out.shape == (B, chunk, raw_action_dim), \
        f"Expected {(B, chunk, raw_action_dim)}, got {tuple(out.shape)}"
    print(f"         ✓ shape correct: [B={B}, chunk={chunk}, action_dim={raw_action_dim}]")
    out_a_zero = out.clone()

    # ─── Test 2: different state → different output? ──────────────
    print(f"\n[sanity] Test 2: state conditioning is LIVE (state_a vs state_b)")
    with torch.no_grad():
        out_b_zero = full(noisy_actions, timestep, position_ids,
                          state=state_b, vlm_kv=vlm_kv_zeros)
    diff_state = (out_a_zero - out_b_zero).abs().max().item()
    mean_a = out_a_zero.abs().mean().item()
    print(f"         max|a - b| = {diff_state:.4e}")
    print(f"         mean|a|   = {mean_a:.4e}")
    if diff_state < 1e-6:
        print(f"         ✗ STATE CONDITIONING DEAD — state input ignored!")
    else:
        ratio = diff_state / (mean_a + 1e-12)
        print(f"         ✓ state affects output (ratio {ratio:.2f})")

    # ─── Test 3: state=None → same as state input when VLM is also same? ──
    print(f"\n[sanity] Test 3: state=None path still works (back-compat)")
    with torch.no_grad():
        out_no_state = full(noisy_actions, timestep,
                            torch.arange(chunk, device="cuda").unsqueeze(0),
                            state=None, vlm_kv=vlm_kv_zeros)
    print(f"         output shape with state=None: {tuple(out_no_state.shape)}")
    assert out_no_state.shape == (B, chunk, raw_action_dim), \
        f"state=None path broke the output shape"
    print(f"         ✓ back-compat path works")

    # ─── Test 4: different vlm_kv → different output? ─────────────
    print(f"\n[sanity] Test 4: vlm_kv conditioning is LIVE (zeros vs random)")
    with torch.no_grad():
        out_a_rand = full(noisy_actions, timestep, position_ids,
                          state=state_a, vlm_kv=vlm_kv_rand)
    diff_vlm = (out_a_zero - out_a_rand).abs().max().item()
    print(f"         max|zero-vlm - rand-vlm| = {diff_vlm:.4e}")
    if diff_vlm < 1e-6:
        print(f"         ✗ VLM CONDITIONING DEAD — vlm_kv ignored!")
    else:
        ratio = diff_vlm / (mean_a + 1e-12)
        print(f"         ✓ vlm_kv affects output (ratio {ratio:.2f})")

    # ─── Summary ────────────────────────────────────────────────────
    print(f"\n====== SANITY SUMMARY ======")
    print(f"  has_state_encoder:   {meta.get('has_state_encoder')}")
    print(f"  state conditioning:  "
          f"{'LIVE' if diff_state > 1e-6 else 'DEAD'} (diff={diff_state:.4e})")
    print(f"  vlm_kv conditioning: "
          f"{'LIVE' if diff_vlm > 1e-6 else 'DEAD'} (diff={diff_vlm:.4e})")
    print(f"  back-compat path:    WORKS")
    if diff_state > 1e-6 and diff_vlm > 1e-6:
        print(f"  VERDICT: PASS — ready for Modal export + parity test")
    else:
        print(f"  VERDICT: FAIL — a conditioning path is dead")

    return {
        "status": "ok",
        "has_state_encoder": bool(meta.get("has_state_encoder")),
        "state_cond_live": bool(diff_state > 1e-6),
        "vlm_cond_live": bool(diff_vlm > 1e-6),
        "diff_state": float(diff_state),
        "diff_vlm": float(diff_vlm),
        "mean_abs_output": float(mean_a),
    }


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("GR00T state-encoder + vlm_kv plumbing sanity check")
    print("=" * 60)
    r = run_sanity.remote()
    print(f"\n=== RESULT ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
