"""Step 3 parity — compare our GR00TFullStack against a hand-rolled
reference that uses the SAME weights via the primitive encoder/decoder
classes. No lerobot dependency (0.5.1 can't load N1.6 checkpoint).

Goal: verify that our `GR00TFullStack.forward(noisy, t, pos, state, vlm_kv)`
produces bit-exact output vs a minimal re-implementation that:
  1. Calls action_encoder + state_encoder (same classes we use)
  2. Adds pos_embed to action tokens only (BEFORE concat)
  3. Concats state + actions (no future_tokens per N1.6 state dict)
  4. Feeds through the same DiT
  5. Slices velocity from position action_start:
  6. Decodes via action_decoder

If Path A (GR00TFullStack) == Path B (hand-rolled reference), our
wrapper is semantically correct. Any delta is a bug in the wrapper.

Usage:
    modal run scripts/modal_gr00t_vlm_parity.py
"""
import os
import subprocess
import modal

app = modal.App("tether-gr00t-vlm-parity")


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
        f'pip install "tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
            secrets=[modal.Secret.from_name("github-token")],
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1200,
    secrets=[_hf_secret()],
)
def run_parity(model_id: str = "nvidia/GR00T-N1.6-3B"):
    import time
    import torch

    from tether.checkpoint import load_checkpoint
    from tether.exporters.gr00t import build_gr00t_full_stack

    print(f"[parity] Loading {model_id}...")
    t0 = time.time()
    state_dict, _ = load_checkpoint(model_id)
    ours, meta = build_gr00t_full_stack(state_dict, embodiment_id=0)
    ours = ours.float().eval().to("cuda")
    print(f"[parity] Ours built in {time.time()-t0:.1f}s")

    # ─── Seeded synthetic inputs ─────────────────────────────────
    torch.manual_seed(42)
    B = 1
    chunk = 50
    raw_action_dim = meta["raw_action_dim"]      # 128
    raw_state_dim = meta["raw_state_dim"]        # 128
    vlm_kv_dim = 2048
    vlm_seq_len = 256

    noisy_actions = torch.randn(B, chunk, raw_action_dim, device="cuda")
    timestep = torch.tensor([0.5], device="cuda")
    state = torch.randn(B, raw_state_dim, device="cuda")
    vlm_kv = torch.randn(B, vlm_seq_len, vlm_kv_dim, device="cuda")
    position_ids = torch.arange(chunk + 1, device="cuda").unsqueeze(0)

    # ─── Path A: our GR00TFullStack ─────────────────────────────
    print(f"\n[parity] Path A (our GR00TFullStack)")
    with torch.no_grad():
        actions_a = ours(noisy_actions, timestep, position_ids,
                           state=state, vlm_kv=vlm_kv)
    print(f"         shape={tuple(actions_a.shape)}, "
          f"mean={actions_a.mean().item():+.4f} std={actions_a.std().item():.4f}")
    print(f"         first_action[:5] = {actions_a[0, 0, :5].cpu().numpy().tolist()}")

    # ─── Path B: hand-rolled reference ───────────────────────────
    # Uses the SAME primitives (ours.action_encoder, ours.state_encoder,
    # ours.dit, ours.action_decoder) but wires them directly, without
    # going through GR00TFullStack.forward. Any delta = wrapper bug.
    import torch.nn.functional as F
    from tether.exporters.gr00t import _sinusoidal_timestep

    print(f"\n[parity] Path B (hand-rolled reference, same primitives)")
    with torch.no_grad():
        # 1. time embedding
        t_sin = _sinusoidal_timestep(timestep, ours.dit.sinusoidal_dim)
        time_emb = ours.dit.timestep_linear_2(
            F.silu(ours.dit.timestep_linear_1(t_sin))
        )

        # 2. action_encoder → [B, chunk, 1536]
        action_tokens = ours.action_encoder(noisy_actions, time_emb)

        # 3. Add pos_embed to action tokens ONLY (not state)
        action_pos_ids = torch.arange(chunk, device="cuda")
        action_pos = ours.dit.pos_embed[action_pos_ids].unsqueeze(0)
        action_tokens_with_pos = action_tokens + action_pos

        # 4. state_encoder → [B, 1, 1536]
        state_token = ours.state_encoder(state)

        # 5. concat state + actions (no future_tokens in N1.6)
        sa_embs = torch.cat([state_token, action_tokens_with_pos], dim=1)
        print(f"         sa_embs shape: {tuple(sa_embs.shape)}")

        # 6. DiT forward — MUST skip internal pos_embed since we added
        #    it ourselves to action tokens.
        velocity_tokens_b = ours.dit(
            sa_embs, timestep, position_ids,
            vlm_kv=vlm_kv,
            add_pos_embed=False,
        )
        print(f"         velocity_tokens shape: {tuple(velocity_tokens_b.shape)}")

        # 7. Slice off state prefix
        velocity_tokens_b = velocity_tokens_b[:, 1:, :]

        # 8. action_decoder
        actions_b = ours.action_decoder(velocity_tokens_b)
    print(f"         shape={tuple(actions_b.shape)}, "
          f"mean={actions_b.mean().item():+.4f} std={actions_b.std().item():.4f}")
    print(f"         first_action[:5] = {actions_b[0, 0, :5].cpu().numpy().tolist()}")

    # ─── Compare ─────────────────────────────────────────────────
    a = actions_a.reshape(-1)
    b = actions_b.reshape(-1)
    max_abs = float((a - b).abs().max())
    cos = float(
        (a * b).sum() / (a.norm() * b.norm() + 1e-12)
    )
    print(f"\n====== PARITY RESULT (our wrapper vs hand-rolled ref) ======")
    print(f"  cos:        {cos:+.6f}")
    print(f"  max_abs:    {max_abs:.4e}")
    print(f"  mean_abs:   {float(a.abs().mean()):.4e}")
    if cos >= 0.99999 and max_abs < 1e-5:
        verdict = "PASS (bit-exact) — wrapper is semantically correct"
    elif cos >= 0.99:
        verdict = "CLOSE but not bit-exact — investigate small delta"
    else:
        verdict = f"FAIL — wrapper has a real bug (cos={cos:+.4f})"
    print(f"  VERDICT: {verdict}")

    return {
        "status": "ok",
        "cos": cos,
        "max_abs": max_abs,
        "path_a_first": actions_a[0, 0, :5].cpu().numpy().tolist(),
        "path_b_first": actions_b[0, 0, :5].cpu().numpy().tolist(),
        "verdict": verdict,
    }


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("GR00T extended GR00TFullStack vs hand-rolled reference parity")
    print("=" * 60)
    r = run_parity.remote()
    print(f"\n=== RESULT ===")
    for k, v in r.items():
        if isinstance(v, list):
            print(f"  {k}: {[round(x, 4) for x in v]}")
        else:
            print(f"  {k}: {v}")
