"""Diagnostic: dump state-dict keys of nvidia/GR00T-N1.6-3B.

Answers the Step-1 land-mine question from the Eagle VLM export plan:
does N1.6 have `backbone.eagle_linear.weight` (the 2048→1536 projection
that N1.5 has) or does it skip straight from Qwen2's 2048-dim hidden
into DiT's `vlln = LayerNorm(2048)`?

Also enumerates:
- All `backbone.eagle_model.*` keys grouped by prefix (vision / text / mlp1)
- Presence/absence of `eagle_linear` and related projection heads
- Total eagle-side parameter count

Usage:
    modal run scripts/modal_gr00t_keys_dump.py
"""
import os
import modal

app = modal.App("tether-gr00t-keys-dump")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_dict({})


# Lightweight image — just torch + huggingface-hub + safetensors for
# loading the state dict. No lerobot / MuJoCo / robosuite needed.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "numpy",
    )
)


@app.function(
    image=image,
    timeout=900,
    secrets=[_hf_secret()],
)
def dump_keys(model_id: str = "nvidia/GR00T-N1.6-3B"):
    """Load the model's state dict and report the full key schema."""
    import time
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from pathlib import Path

    print(f"[dump] Downloading {model_id} (weights only)...")
    t0 = time.time()
    # Only fetch the index + safetensors files; skip configs etc. for speed
    repo_dir = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
    )
    print(f"[dump] Downloaded to {repo_dir} in {time.time()-t0:.1f}s")

    all_keys: list[tuple[str, tuple, str]] = []  # (key, shape, dtype)
    for st_file in sorted(Path(repo_dir).glob("*.safetensors")):
        with safe_open(str(st_file), framework="pt") as f:
            for key in f.keys():
                t = f.get_slice(key)
                all_keys.append((key, tuple(t.get_shape()), str(t.get_dtype())))

    print(f"\n[dump] Total keys: {len(all_keys)}")

    # Group by top-level prefix
    from collections import defaultdict
    prefix_counts: dict[str, int] = defaultdict(int)
    prefix_params: dict[str, int] = defaultdict(int)
    for key, shape, _ in all_keys:
        parts = key.split(".")
        top2 = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        prefix_counts[top2] += 1
        n = 1
        for d in shape:
            n *= d
        prefix_params[top2] += n

    print(f"\n[dump] Top-level prefix counts (key_count | total_params):")
    for prefix in sorted(prefix_counts.keys()):
        print(f"  {prefix:50s}  {prefix_counts[prefix]:5d}  {prefix_params[prefix]/1e6:8.1f}M")

    # ─── THE QUESTION: does eagle_linear exist? ────────────────────
    eagle_linear_keys = [k for k, _, _ in all_keys if "eagle_linear" in k]
    print(f"\n[dump] === EAGLE_LINEAR CHECK ===")
    if eagle_linear_keys:
        print(f"  FOUND {len(eagle_linear_keys)} eagle_linear key(s):")
        for k, shape, dtype in all_keys:
            if "eagle_linear" in k:
                print(f"    {k}: shape={shape} dtype={dtype}")
        print(f"  → N1.6 USES the 2048→1536 projection. Port must include it.")
    else:
        print(f"  ABSENT — no eagle_linear keys in state dict.")
        print(f"  → N1.6 SKIPS the projection. DiT's vlln=LayerNorm(2048) "
              "consumes Qwen2 hidden directly. Port must skip eagle_linear.")

    # Eagle substructure (actual prefix is backbone.model.*, not backbone.eagle.*)
    print(f"\n[dump] === BACKBONE SUB-PREFIX COUNTS (level-3) ===")
    backbone_sub: dict[str, tuple] = defaultdict(lambda: (0, 0))
    for k, shape, _ in all_keys:
        if k.startswith("backbone."):
            parts = k.split(".")
            sub = ".".join(parts[:3]) if len(parts) >= 3 else ".".join(parts)
            n = 1
            for d in shape:
                n *= d
            cnt, params = backbone_sub[sub]
            backbone_sub[sub] = (cnt + 1, params + n)
    for sub in sorted(backbone_sub.keys()):
        cnt, params = backbone_sub[sub]
        print(f"  {sub:60s}  {cnt:5d}  {params/1e6:8.1f}M")

    # State encoder full dump (found in N1.6, absent from our current export)
    print(f"\n[dump] === STATE ENCODER KEYS (new in N1.6?) ===")
    for k, shape, dtype in all_keys:
        if k.startswith("action_head.state_encoder"):
            print(f"  {k}: shape={shape} dtype={dtype}")

    # Look for vision-model-likely keys
    print(f"\n[dump] === POSSIBLE VISION TOWER KEYS (first 5 matches) ===")
    seen_v = 0
    for k, shape, dtype in all_keys:
        if any(m in k.lower() for m in ("vision", "siglip", "vit", "patch_embed", "mlp1")):
            print(f"  {k}: shape={shape} dtype={dtype}")
            seen_v += 1
            if seen_v >= 5:
                break

    # mlp1 connector (Eagle's 2-layer MLP mapping vision → text hidden)
    print(f"\n[dump] === MLP1 CONNECTOR KEYS ===")
    for k, shape, dtype in all_keys:
        if "mlp1" in k.lower():
            print(f"  {k}: shape={shape} dtype={dtype}")

    # Future tokens — learnable prefix tokens in the DiT sequence (per lerobot
    # flow_matching_action_head.py L324-325: sa_embs = cat(state, future, action))
    print(f"\n[dump] === FUTURE TOKENS / DIT SEQUENCE PREFIX KEYS ===")
    for k, shape, dtype in all_keys:
        if any(m in k.lower() for m in ("future_tokens", "future_token",
                                          "prefix_token", "readout")):
            print(f"  {k}: shape={shape} dtype={dtype}")

    # action_head.model.* level-3 breakdown (inside DiT)
    print(f"\n[dump] === ACTION_HEAD.MODEL.* SUB-PREFIXES (level-3) ===")
    ah_sub: dict[str, int] = defaultdict(int)
    for k, _, _ in all_keys:
        if k.startswith("action_head.model."):
            parts = k.split(".")
            sub = ".".join(parts[:3]) if len(parts) >= 3 else ".".join(parts)
            ah_sub[sub] += 1
    for sub in sorted(ah_sub.keys()):
        print(f"  {sub:60s}  {ah_sub[sub]:5d}")

    # Sample one key per action_head.model.* sub to see shapes
    print(f"\n[dump] === ACTION_HEAD.MODEL.* FIRST-KEY PER SUB (shape) ===")
    seen_sub: set[str] = set()
    for k, shape, dtype in all_keys:
        if k.startswith("action_head.model."):
            parts = k.split(".")
            sub = ".".join(parts[:3]) if len(parts) >= 3 else ".".join(parts)
            if sub not in seen_sub:
                seen_sub.add(sub)
                print(f"  [{sub}] {k}: {shape} {dtype}")

    # Sample key names to sanity-check structure
    print(f"\n[dump] === SAMPLE KEYS (first 5 per top-2 prefix) ===")
    seen: dict[str, int] = defaultdict(int)
    for k, shape, dtype in all_keys:
        parts = k.split(".")
        top2 = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        if seen[top2] < 5:
            print(f"  [{top2}] {k}: {shape}")
            seen[top2] += 1

    return {
        "model_id": model_id,
        "total_keys": len(all_keys),
        "has_eagle_linear": bool(eagle_linear_keys),
        "eagle_linear_keys": eagle_linear_keys,
        "prefix_counts": dict(prefix_counts),
        "prefix_params_m": {k: round(v / 1e6, 1) for k, v in prefix_params.items()},
    }


@app.local_entrypoint()
def main(model_id: str = "nvidia/GR00T-N1.6-3B"):
    print("=" * 60)
    print(f"GR00T state-dict key dump: {model_id}")
    print("=" * 60)
    r = dump_keys.remote(model_id=model_id)
    print(f"\n=== RESULT ===")
    print(f"  total_keys: {r['total_keys']}")
    print(f"  has_eagle_linear: {r['has_eagle_linear']}")
    print(f"  eagle_linear_keys: {r['eagle_linear_keys']}")
