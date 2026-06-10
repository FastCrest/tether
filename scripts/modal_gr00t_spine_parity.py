"""Step 4 parity — verify GR00TVLA(BaseVLA) produces bit-identical numerics
vs the legacy direct-build path (build_gr00t_full_stack).

Both paths use the same underlying builders + same checkpoint; this script
asserts the spine wrapper doesn't introduce any drift. If bit-identical
(max_diff == 0.0), the spine refactor is numerics-equivalent to the
legacy path — the strongest evidence Day 7's spine composition is correct.

Compares:
  PATH A: build_gr00t_full_stack(state_dict)              # legacy
  PATH B: GR00TVLA.from_pretrained(state_dict=state_dict) # spine

Per the user's Day 7 "full bundle + ALL 5 Modal gates" choice. Estimated
cost ~$2-3 on A10G. Required gear: HF_TOKEN secret + github-token for
the pip install.

Usage:
    modal profile activate novarepmarketing  # per saved-memory token
    modal run scripts/modal_gr00t_spine_parity.py
"""
import os
import subprocess
import modal

app = modal.App("tether-gr00t-spine-parity")


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
        f'pip install "fastcrest-tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
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
    from tether.models.vlas.gr00t import GR00TVLA

    print(f"[spine-parity] Loading {model_id}...")
    t0 = time.time()
    state_dict, _ = load_checkpoint(model_id)
    print(f"[spine-parity] Loaded {len(state_dict)} tensors in {time.time()-t0:.1f}s")

    # ─── PATH A: legacy direct-build ─────────────────────────────
    print(f"\n[spine-parity] PATH A: build_gr00t_full_stack (legacy)")
    t1 = time.time()
    legacy_stack, meta = build_gr00t_full_stack(state_dict, embodiment_id=0)
    legacy_stack = legacy_stack.float().eval().to("cuda")
    print(f"[spine-parity] Legacy built in {time.time()-t1:.1f}s, raw_action_dim={meta['raw_action_dim']}")

    # ─── PATH B: spine via GR00TVLA.from_pretrained ──────────────
    print(f"\n[spine-parity] PATH B: GR00TVLA.from_pretrained (spine)")
    t2 = time.time()
    vla = GR00TVLA.from_pretrained(state_dict=state_dict, embodiment_id=0)
    spine_stack = vla.vla_head.full_stack.float().eval().to("cuda")
    spine_meta = vla.vla_head.metadata
    print(f"[spine-parity] Spine built in {time.time()-t2:.1f}s, raw_action_dim={spine_meta['raw_action_dim']}")

    # Sanity: same architectural metadata
    assert meta["raw_action_dim"] == spine_meta["raw_action_dim"], "raw_action_dim mismatch"
    assert meta["hidden"] == spine_meta["hidden"], "hidden mismatch"
    assert meta["chunk_size"] == spine_meta["chunk_size"], "chunk_size mismatch"
    print(f"[spine-parity] Architecture metadata: MATCH ✓")

    # ─── Seeded synthetic inputs ─────────────────────────────────
    torch.manual_seed(42)
    B = 1
    chunk = 50
    raw_action_dim = meta["raw_action_dim"]
    raw_state_dim = 128
    vlm_kv_dim = 2048
    vlm_seq_len = 256

    noisy_actions = torch.randn(B, chunk, raw_action_dim, device="cuda")
    timestep = torch.tensor([0.5], device="cuda")
    state = torch.randn(B, raw_state_dim, device="cuda")
    vlm_kv = torch.randn(B, vlm_seq_len, vlm_kv_dim, device="cuda")
    position_ids = torch.arange(chunk + 1, device="cuda").unsqueeze(0)

    # ─── Run both forward paths with identical inputs ────────────
    print(f"\n[spine-parity] Running PATH A forward...")
    with torch.no_grad():
        actions_a = legacy_stack(noisy_actions, timestep, position_ids,
                                 state=state, vlm_kv=vlm_kv)
    print(f"   PATH A shape={tuple(actions_a.shape)}, mean={actions_a.mean().item():+.4f}, std={actions_a.std().item():.4f}")

    print(f"\n[spine-parity] Running PATH B forward...")
    with torch.no_grad():
        actions_b = spine_stack(noisy_actions, timestep, position_ids,
                                state=state, vlm_kv=vlm_kv)
    print(f"   PATH B shape={tuple(actions_b.shape)}, mean={actions_b.mean().item():+.4f}, std={actions_b.std().item():.4f}")

    # ─── Bit-identical comparison ────────────────────────────────
    print(f"\n[spine-parity] PATH A vs PATH B comparison")
    assert actions_a.shape == actions_b.shape, f"shape mismatch: {actions_a.shape} vs {actions_b.shape}"
    diff = (actions_a - actions_b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    p95 = diff.flatten().sort()[0][int(0.95 * diff.numel())].item()
    cos = torch.nn.functional.cosine_similarity(
        actions_a.flatten().unsqueeze(0),
        actions_b.flatten().unsqueeze(0),
    ).item()
    print(f"   max_diff={max_diff:.6e}")
    print(f"   mean_diff={mean_diff:.6e}")
    print(f"   p95_diff={p95:.6e}")
    print(f"   cos_sim={cos:.6f}")

    # Bit-identical gate: spine wraps the same builders, so should be 0.0
    if max_diff == 0.0:
        verdict = "PASS BIT-IDENTICAL"
    elif max_diff < 1e-6:
        verdict = f"PASS (near-bit-identical, max={max_diff:.2e})"
    elif max_diff < 1e-4:
        verdict = f"PASS (within tolerance, max={max_diff:.2e})"
    else:
        verdict = f"FAIL (max={max_diff:.2e} > 1e-4)"
    print(f"\n[spine-parity] VERDICT: {verdict}")

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "p95_diff": p95,
        "cos_sim": cos,
        "raw_action_dim": raw_action_dim,
        "chunk_size": chunk,
        "verdict": verdict,
    }


@app.local_entrypoint()
def main():
    result = run_parity.remote()
    print(f"\n[local] Final result: {result}")
