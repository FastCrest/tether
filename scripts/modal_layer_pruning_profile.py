"""Modal: cosine-similarity profiling of pi0.5 PaliGemma layers.

Day-1 spike for the language-layer-pruning-adaptive feature
(features/01_serve/subfeatures/_perf_compound/language-layer-pruning-adaptive).
Decision the run informs:

    Are there pairs of consecutive PaliGemma layers in pi0.5 with
    cosine similarity > 0.95 between their output hidden states?

If yes -> EfficientVLA's pruning approach (arxiv 2506.10100) likely
transfers to pi0.5 flow-matching (validated on CogACT diffusion).
Continue with full Phase 1.5 build (1-2 weeks + ~$20-40 Modal validation).

If no  -> kill the feature; pi0.5's layer-redundancy profile is
fundamentally different from CogACT's. Document the negative finding
+ move to next perf-compound candidate.

Minimum-viable: N=10 synthetic-input forward passes through the prefix.
17 layer-pair sims per sample, mean over batch + tokens. A10G, ~5 min
wall, target $1-2 cost.

Usage:
    modal run scripts/modal_layer_pruning_profile.py
"""
import os
import subprocess
import modal

app = modal.App("reflex-layer-pruning-profile")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


def _build_bust() -> str:
    import time
    return str(int(time.time()))


_HEAD = _repo_head_sha()
_BUST = _build_bust()

hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "lerobot==0.5.1",
        "transformers==5.3.0",
        "num2words",
        "safetensors>=0.4.0",
        "scipy",
        "numpy",
        "accelerate",
        "draccus",
    )
    .run_commands(
        f'echo "build_bust={_BUST}"',
        f'pip install "reflex-vla[monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": f"{HF_CACHE}/transformers",
    })
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={HF_CACHE: hf_cache},
    secrets=[_hf_secret()],
)
def profile_modal(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    n_samples: int = 10,
    sim_threshold: float = 0.95,
):
    """Profile pi0.5 PaliGemma layer-output cosine similarity."""
    import logging
    import time
    import numpy as np
    import torch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("layer-prune")

    logger.info("[profile] loading %s", model_id)
    t0 = time.time()
    # Direct PI05Policy.from_pretrained bypasses lerobot's make_policy +
    # draccus config parser, which trips on the `type` field saved in
    # canonical lerobot configs (caught by 2026-04-25 distill smoke v3).
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained(model_id).eval().to("cuda")
    logger.info("[profile] policy loaded in %.1fs", time.time() - t0)

    paligemma = policy.model.paligemma_with_expert.paligemma
    language_model = paligemma.model.language_model
    layers = language_model.layers
    n_layers = len(layers)
    logger.info("[profile] PaliGemma language_model layers: %d", n_layers)

    captured: list[torch.Tensor] = []

    def make_hook(idx: int):
        def hook(_module, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            captured.append(t.detach().to(torch.float32).cpu())
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]

    pair_sims_all: list[list[float]] = []  # per-sample list of (n_layers-1) sims

    try:
        for s in range(n_samples):
            captured.clear()
            seed = 1000 + s
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Synthetic prefix forward. Use the same image+lang shapes the
            # decomposed exporter uses (PI05_PREFIX_SEQ_LEN ~= 968 for the
            # default 3-cam + 200-token lang config). We don't need real
            # data because we're measuring intrinsic layer-output redundancy,
            # not task semantics.
            B = 1
            from reflex.exporters.decomposed import (
                PI05_PALIGEMMA_LAYERS,
            )
            assert n_layers == PI05_PALIGEMMA_LAYERS, (
                f"layer count mismatch: hooks={n_layers}, "
                f"PI05_PALIGEMMA_LAYERS={PI05_PALIGEMMA_LAYERS}"
            )

            # Build dummy inputs matching the prefix wrapper's signature.
            img = torch.rand(B, 3, 224, 224, device="cuda", dtype=torch.float32)
            mask = torch.ones(B, dtype=torch.bool, device="cuda")
            lang_tokens = torch.randint(0, 100, (B, 16), device="cuda", dtype=torch.long)
            lang_masks = torch.ones(B, 16, dtype=torch.bool, device="cuda")

            with torch.no_grad():
                images = [img, img, img]
                img_masks = [mask, mask, mask]
                from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
                prefix_embs, prefix_pad_masks, prefix_att_masks = (
                    policy.model.embed_prefix(
                        images, img_masks, lang_tokens, lang_masks,
                    )
                )
                prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                prefix_att_2d_4d = (
                    policy.model._prepare_attention_masks_4d(prefix_att_2d)
                )
                language_model.config._attn_implementation = "eager"
                _ = policy.model.paligemma_with_expert.forward(
                    attention_mask=prefix_att_2d_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                )

            assert len(captured) == n_layers, (
                f"sample {s}: expected {n_layers} captured layers, "
                f"got {len(captured)}"
            )
            # Cosine sim per consecutive pair, mean over (B, seq, hidden)
            # along the hidden axis -> single scalar per token; then mean
            # over batch + tokens for a single sim per layer-pair.
            sims = []
            for i in range(n_layers - 1):
                a = captured[i].reshape(-1, captured[i].shape[-1])
                b = captured[i + 1].reshape(-1, captured[i + 1].shape[-1])
                cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
                sims.append(float(cos.mean().item()))
            pair_sims_all.append(sims)
            logger.info(
                "[profile] sample %d/%d: max_sim=%.4f min_sim=%.4f mean_sim=%.4f",
                s + 1, n_samples,
                max(sims), min(sims), sum(sims) / len(sims),
            )
    finally:
        for h in handles:
            h.remove()

    arr = np.array(pair_sims_all)  # (n_samples, n_layers-1)
    mean_per_pair = arr.mean(axis=0)
    p95_per_pair = np.percentile(arr, 95, axis=0)
    above_threshold = mean_per_pair > sim_threshold

    print("\n" + "=" * 72)
    print(f"PI0.5 PALIGEMMA LAYER COSINE-SIMILARITY PROFILE  (N={n_samples})")
    print("=" * 72)
    print(f"  layers      : {n_layers}")
    print(f"  layer pairs : {n_layers - 1}")
    print(f"  threshold   : {sim_threshold}")
    print()
    print(f"  {'pair':<8} {'mean_sim':<12} {'p95_sim':<12} {'>thresh':<10}")
    print(f"  {'-'*8:<8} {'-'*12:<12} {'-'*12:<12} {'-'*10:<10}")
    for i in range(n_layers - 1):
        marker = " *" if above_threshold[i] else ""
        print(
            f"  {i:>2}->{i+1:<3}    "
            f"{mean_per_pair[i]:>9.4f}    "
            f"{p95_per_pair[i]:>9.4f}    "
            f"{'YES' if above_threshold[i] else 'no':<10}{marker}"
        )
    print()
    n_above = int(above_threshold.sum())
    print(f"  Pairs with mean sim > {sim_threshold}: {n_above} of {n_layers - 1}")
    print(
        f"  Max mean sim: {mean_per_pair.max():.4f} "
        f"(pair {int(mean_per_pair.argmax())} -> {int(mean_per_pair.argmax()) + 1})"
    )
    print(
        f"  Min mean sim: {mean_per_pair.min():.4f} "
        f"(pair {int(mean_per_pair.argmin())} -> {int(mean_per_pair.argmin()) + 1})"
    )
    print()
    if n_above == 0:
        print("  VERDICT: NO LAYER PAIRS ABOVE THRESHOLD")
        print("  -> EfficientVLA's pi0.5 transfer FAILS at the standard 0.95 threshold")
        print("  -> Try lower threshold (0.90, 0.85) before killing the feature")
    else:
        print(f"  VERDICT: {n_above} PAIR(S) ABOVE THRESHOLD")
        print("  -> EfficientVLA approach LIKELY transfers; proceed with Phase 1.5")
        print("  -> Next: per-task LIBERO N=50 validation gating ($5-10 Modal)")
    print("=" * 72)

    return {
        "model_id": model_id,
        "n_samples": n_samples,
        "n_layers": n_layers,
        "threshold": sim_threshold,
        "mean_per_pair": mean_per_pair.tolist(),
        "p95_per_pair": p95_per_pair.tolist(),
        "n_pairs_above_threshold": n_above,
        "max_mean_sim": float(mean_per_pair.max()),
        "min_mean_sim": float(mean_per_pair.min()),
    }


@app.local_entrypoint()
def main(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    n_samples: int = 10,
    sim_threshold: float = 0.95,
):
    r = profile_modal.remote(
        model_id=model_id,
        n_samples=n_samples,
        sim_threshold=sim_threshold,
    )
    print("\n=== RESULT (return value) ===")
    print(f"  n_pairs_above_threshold: {r['n_pairs_above_threshold']}")
    print(f"  max_mean_sim: {r['max_mean_sim']:.4f}")
    print(f"  min_mean_sim: {r['min_mean_sim']:.4f}")
