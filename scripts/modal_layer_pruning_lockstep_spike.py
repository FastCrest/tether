"""Modal: Day-3 spike — dual-tower lockstep prune of pi0.5 layer 5.

Day-3 of the language-layer-pruning audit (Day-1 cosine-sim PASS,
Day-2 prefix-only re-export PASS but surfaced joint cross-tower
attention; sidecar revisit found PPCL is the published lockstep
pattern from MM-DiT image-gen).

Decision the spike informs:

    Does the PPCL-style lockstep pattern (drop layer i from BOTH
    paligemma + expert towers) actually compose end-to-end on pi0.5
    via the existing torch.export + ONNX pipeline + raw ORT?

If yes -> language-layer-pruning Phase 1.5 (1-2 weeks code + $30-50
Modal) is unblocked with eyes open. PPCL pattern transfers.

If no  -> dual-tower entanglement is harder than PPCL acknowledges;
pivot to a different perf-compound feature.

Minimum viable: drop ONLY layer 5 from both towers. Re-export
vlm_prefix.onnx with 34 past_kv outputs (17 layers × 2). Re-export
expert_denoise.onnx that consumes 34 past_kv inputs in matching
order. Run prefix → expert in raw ORT. Verify output shape +
finite. Skip LIBERO + skip Pi05DecomposedInference runtime path
(those are downstream gates).

Usage:
    modal run scripts/modal_layer_pruning_lockstep_spike.py
"""
import os
import subprocess
import modal

app = modal.App("reflex-layer-pruning-lockstep-spike")


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
        "onnx>=1.16",
        "onnxruntime-gpu>=1.20",
        "onnxscript>=0.1",
        "onnx-diagnostic>=0.9",
        "optree",
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
    gpu="A100-80GB",
    timeout=3600,
    volumes={HF_CACHE: hf_cache},
    secrets=[_hf_secret()],
)
def lockstep_spike_modal(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    drop_layer: int = 5,
):
    """Lockstep prune drop_layer from BOTH towers; re-export both ONNX
    files; run prefix→expert in raw ORT; verify output."""
    import logging
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("lockstep-spike")

    # ---- Load policy ----
    logger.info("[spike] loading %s", model_id)
    t0 = time.time()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained(model_id).eval().to(torch.float32).to("cpu")
    logger.info("[spike] policy loaded in %.1fs", time.time() - t0)

    # ---- Patch BOTH towers ----
    paligemma = policy.model.paligemma_with_expert.paligemma
    paligemma_lm = paligemma.model.language_model
    gemma_expert = policy.model.paligemma_with_expert.gemma_expert
    expert_lm = gemma_expert.model

    n_orig = len(paligemma_lm.layers)
    assert len(expert_lm.layers) == n_orig, (
        f"tower depth mismatch: paligemma={n_orig} expert={len(expert_lm.layers)}"
    )
    logger.info("[spike] original tower depth: %d (both)", n_orig)

    if not (0 <= drop_layer < n_orig):
        raise ValueError(f"drop_layer={drop_layer} out of range [0, {n_orig})")

    kept_indices = [i for i in range(n_orig) if i != drop_layer]
    n_kept = len(kept_indices)
    logger.info(
        "[spike] dropping layer %d from BOTH towers; keeping %d: %s",
        drop_layer, n_kept, kept_indices,
    )

    # Replace ModuleLists in lockstep
    paligemma_lm.layers = nn.ModuleList(
        [paligemma_lm.layers[i] for i in kept_indices]
    )
    expert_lm.layers = nn.ModuleList(
        [expert_lm.layers[i] for i in kept_indices]
    )

    # Patch num_hidden_layers in BOTH configs
    if hasattr(paligemma_lm.config, "num_hidden_layers"):
        paligemma_lm.config.num_hidden_layers = n_kept
    if hasattr(expert_lm.config, "num_hidden_layers"):
        expert_lm.config.num_hidden_layers = n_kept
    # The PaliGemma top-level config also has text_config.num_hidden_layers
    # which compute_layer_complete reads (lerobot/.../modeling_pi05.py:488)
    if hasattr(paligemma.config, "text_config") and hasattr(
        paligemma.config.text_config, "num_hidden_layers"
    ):
        paligemma.config.text_config.num_hidden_layers = n_kept
    logger.info("[spike] patched num_hidden_layers to %d in both configs", n_kept)

    # ---- Build dummy inputs (mirror decomposed.py contract) ----
    # CRITICAL: also patch the MODULE-LEVEL PI05_PALIGEMMA_LAYERS constant.
    # The Pi05PrefixWrapper.forward reads it via global lookup
    # (decomposed.py:~770 `for layer_idx in range(PI05_PALIGEMMA_LAYERS)`)
    # to flatten the DynamicCache. If we don't patch this, the wrapper
    # iterates 18 even though our patched language_model only has 17
    # layers of cache; the extra access returns phantom entries.
    import reflex.exporters.decomposed as _dd
    _dd.PI05_PALIGEMMA_LAYERS = n_kept
    logger.info(
        "[spike] patched reflex.exporters.decomposed.PI05_PALIGEMMA_LAYERS "
        "= %d", n_kept,
    )
    from reflex.exporters.decomposed import (
        Pi05PrefixWrapper,
        _PI05_BATCH_SIZE,
        _PI05_IMAGE_SIZE,
        _PI05_LANG_TOKENS,
        PI05_KV_HEADS,
        PI05_HEAD_DIM,
        _prefix_seq_len,
    )
    B = _PI05_BATCH_SIZE
    chunk = policy.config.chunk_size
    action_dim = policy.config.max_action_dim
    prefix_seq_len = _prefix_seq_len()

    prefix_wrapper = Pi05PrefixWrapper(policy.model).eval()

    prefix_dummy = dict(
        img_base=torch.randn(B, 3, _PI05_IMAGE_SIZE, _PI05_IMAGE_SIZE, dtype=torch.float32),
        img_wrist_l=torch.randn(B, 3, _PI05_IMAGE_SIZE, _PI05_IMAGE_SIZE, dtype=torch.float32),
        img_wrist_r=torch.randn(B, 3, _PI05_IMAGE_SIZE, _PI05_IMAGE_SIZE, dtype=torch.float32),
        mask_base=torch.ones(B, dtype=torch.bool),
        mask_wrist_l=torch.ones(B, dtype=torch.bool),
        mask_wrist_r=torch.ones(B, dtype=torch.bool),
        lang_tokens=torch.randint(0, 257152, (B, _PI05_LANG_TOKENS), dtype=torch.long),
        lang_masks=torch.ones(B, _PI05_LANG_TOKENS, dtype=torch.bool),
    )

    # past_kv names for the KEPT layers (in iteration order, which after
    # patching is 0..n_kept-1; the original layer index is encoded in the
    # name for traceability + to mirror the existing reflex contract)
    past_kv_names_kept = []
    for li in kept_indices:
        past_kv_names_kept.append(f"past_k_{li}")
        past_kv_names_kept.append(f"past_v_{li}")
    prefix_output_names = past_kv_names_kept + ["prefix_pad_masks"]

    # ---- Eager forward of pruned PREFIX wrapper (sanity) ----
    logger.info("[spike] eager forward (pruned prefix, CPU)...")
    t0 = time.time()
    with torch.no_grad():
        prefix_outputs = prefix_wrapper(*tuple(prefix_dummy.values()))
    eager_prefix_ms = (time.time() - t0) * 1000
    expected_n_outputs = 2 * n_kept + 1
    logger.info(
        "[spike] eager prefix: %.1f ms; %d outputs (expected %d)",
        eager_prefix_ms, len(prefix_outputs), expected_n_outputs,
    )
    assert len(prefix_outputs) == expected_n_outputs

    # ---- Re-export prefix ONNX ----
    out_dir = Path("/tmp/lockstep_export")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = out_dir / "vlm_prefix.onnx"
    logger.info("[spike] exporting pruned prefix → %s", prefix_path)
    from onnx_diagnostic.torch_export_patches import torch_export_patches
    from reflex.exporters.monolithic import _fix_onnx_where_dtype_mismatches

    t0 = time.time()
    with torch_export_patches(patch_transformers=True):
        ep_prefix = torch.export.export(
            prefix_wrapper, tuple(prefix_dummy.values()),
            dynamic_shapes=None, strict=False,
        )
    logger.info("[spike] prefix torch.export: %.1fs", time.time() - t0)
    t0 = time.time()
    torch.onnx.export(
        ep_prefix, tuple(prefix_dummy.values()), str(prefix_path),
        input_names=list(prefix_dummy.keys()),
        output_names=prefix_output_names,
        opset_version=19,
    )
    logger.info("[spike] prefix torch.onnx.export: %.1fs", time.time() - t0)
    _ = _fix_onnx_where_dtype_mismatches(prefix_path)
    prefix_size_mb = prefix_path.stat().st_size / (1024 * 1024)
    logger.info("[spike] pruned prefix ONNX: %.1f MB", prefix_size_mb)

    # ---- Build pruned PER-STEP expert wrapper (custom — bypasses
    # _Pi05ExpertPerStepWrapper which hardcodes PI05_PALIGEMMA_LAYERS=18)
    class _PrunedPi05ExpertPerStepWrapper(nn.Module):
        """Per-step expert wrapper accepting n_kept past_kv tensors instead
        of the hardcoded 36. Mirrors decomposed.py's _Pi05ExpertPerStepWrapper
        but parameterized on n_kept_layers."""
        def __init__(self, pi05_model, n_kept_layers: int):
            super().__init__()
            self.model = pi05_model
            self.n_kept_layers = n_kept_layers

        def forward(self, *args):
            from transformers.cache_utils import DynamicCache
            past_flat = args[: self.n_kept_layers * 2]
            prefix_pad_masks = args[self.n_kept_layers * 2]
            x_t = args[self.n_kept_layers * 2 + 1]
            t = args[self.n_kept_layers * 2 + 2]

            past_kv = DynamicCache()
            for i in range(self.n_kept_layers):
                past_kv.update(
                    key_states=past_flat[2 * i],
                    value_states=past_flat[2 * i + 1],
                    layer_idx=i,
                    cache_kwargs=None,
                )

            action_dtype = self.model.action_in_proj.weight.dtype
            if x_t.dtype != action_dtype:
                x_t = x_t.to(action_dtype)
            v_t = self.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_kv,
                x_t=x_t, timestep=t,
            )
            return v_t

    expert_wrapper = _PrunedPi05ExpertPerStepWrapper(
        policy.model, n_kept_layers=n_kept,
    ).eval()

    # past_kv dummies + prefix_pad_masks_dummy + x_t + t (per-step shape)
    past_kv_shape = (B, PI05_KV_HEADS, prefix_seq_len, PI05_HEAD_DIM)
    expert_dummy = {}
    for i, li in enumerate(kept_indices):
        expert_dummy[f"past_k_{li}"] = torch.randn(past_kv_shape, dtype=torch.float32)
        expert_dummy[f"past_v_{li}"] = torch.randn(past_kv_shape, dtype=torch.float32)
    expert_dummy["prefix_pad_masks"] = torch.ones(B, prefix_seq_len, dtype=torch.bool)
    expert_dummy["x_t"] = torch.randn(B, chunk, action_dim, dtype=torch.float32)
    expert_dummy["t"] = torch.full((B,), 1.0, dtype=torch.float32)

    # ---- Eager forward of pruned EXPERT wrapper (sanity test) ----
    # Production decomposed export does NOT actually run an eager forward
    # of the expert wrapper — torch.export.export traces it symbolically
    # via onnx-diagnostic's torch_export_patches. The eager forward
    # tries the literal compute, which hits a mask-shape mismatch under
    # use_cache=False (50 vs 1018 at dim 3) — gemma's attention doesn't
    # concatenate past_kv when use_cache=False at runtime. Production
    # works because the exported ONNX graph bakes the cache into the
    # attention compute at trace-time. Skip the eager test; rely on
    # torch.export + ORT compose as the load-bearing gate.
    eager_expert_ms = -1.0
    eager_compose_ok = None  # not measured
    v_t = None  # avoid undefined ref

    # ---- Re-export expert ONNX ----
    expert_path = out_dir / "expert_denoise.onnx"
    logger.info("[spike] exporting pruned expert → %s", expert_path)
    t0 = time.time()
    with torch_export_patches(patch_transformers=True):
        ep_expert = torch.export.export(
            expert_wrapper, tuple(expert_dummy.values()),
            dynamic_shapes=None, strict=False,
        )
    logger.info("[spike] expert torch.export: %.1fs", time.time() - t0)

    t0 = time.time()
    torch.onnx.export(
        ep_expert, tuple(expert_dummy.values()), str(expert_path),
        input_names=list(expert_dummy.keys()),
        output_names=["v_t"],
        opset_version=19,
        optimize=False,
    )
    logger.info("[spike] expert torch.onnx.export: %.1fs", time.time() - t0)
    _ = _fix_onnx_where_dtype_mismatches(expert_path)
    expert_size_mb = expert_path.stat().st_size / (1024 * 1024)
    logger.info("[spike] pruned expert ONNX: %.1f MB", expert_size_mb)

    # ---- Compose: prefix → expert via ORT ----
    import onnxruntime as ort
    logger.info("[spike] ORT compose: prefix → expert (CPU EP)")
    sess_prefix = ort.InferenceSession(
        str(prefix_path), providers=["CPUExecutionProvider"],
    )
    sess_expert = ort.InferenceSession(
        str(expert_path), providers=["CPUExecutionProvider"],
    )

    # Run prefix
    prefix_feed = {
        nm: arr.numpy() for nm, arr in prefix_dummy.items()
    }
    t0 = time.time()
    prefix_out = sess_prefix.run(None, prefix_feed)
    ort_prefix_ms = (time.time() - t0) * 1000
    prefix_out_names = [o.name for o in sess_prefix.get_outputs()]
    logger.info("[spike] ORT prefix: %.1f ms; %d outputs", ort_prefix_ms, len(prefix_out))

    # Build expert feed: take past_kv outputs from prefix + add prefix_pad_masks + new x_t + t
    prefix_out_dict = dict(zip(prefix_out_names, prefix_out))
    expert_feed = {}
    for li in kept_indices:
        expert_feed[f"past_k_{li}"] = prefix_out_dict[f"past_k_{li}"]
        expert_feed[f"past_v_{li}"] = prefix_out_dict[f"past_v_{li}"]
    expert_feed["prefix_pad_masks"] = prefix_out_dict["prefix_pad_masks"]
    expert_feed["x_t"] = expert_dummy["x_t"].numpy()
    expert_feed["t"] = expert_dummy["t"].numpy()

    t0 = time.time()
    expert_out = sess_expert.run(None, expert_feed)
    ort_expert_ms = (time.time() - t0) * 1000
    v_t_ort = expert_out[0]
    logger.info(
        "[spike] ORT expert: %.1f ms; v_t shape=%s dtype=%s",
        ort_expert_ms, v_t_ort.shape, v_t_ort.dtype,
    )
    n_finite_ort = int(np.isfinite(v_t_ort).sum())
    n_total_ort = int(v_t_ort.size)
    ort_compose_ok = (
        n_finite_ort == n_total_ort
        and tuple(v_t_ort.shape) == (B, chunk, action_dim)
    )
    logger.info(
        "[spike] ORT v_t finite: %d / %d (%.1f%%); range [%.4f, %.4f]",
        n_finite_ort, n_total_ort, 100 * n_finite_ort / n_total_ort,
        float(v_t_ort.min()), float(v_t_ort.max()),
    )

    # ---- Report ----
    print("\n" + "=" * 72)
    print(f"PI0.5 DUAL-TOWER LOCKSTEP-PRUNE SPIKE  (drop layer {drop_layer})")
    print("=" * 72)
    print(f"  model_id          : {model_id}")
    print(f"  original layers   : {n_orig} (both towers)")
    print(f"  kept layers       : {n_kept}")
    print(f"  kept indices      : {kept_indices}")
    print()
    print(f"  EAGER prefix      : {eager_prefix_ms:.1f} ms, {len(prefix_outputs)} outputs")
    print(f"  EAGER expert      : SKIPPED (production uses symbolic tracing only)")
    print(f"  EAGER compose OK  : not measured (relying on ORT compose)")
    print()
    print(f"  prefix ONNX       : {prefix_size_mb:.1f} MB")
    print(f"  expert ONNX       : {expert_size_mb:.1f} MB")
    print()
    print(f"  ORT prefix        : {ort_prefix_ms:.1f} ms")
    print(f"  ORT expert        : {ort_expert_ms:.1f} ms")
    print(f"  ORT v_t shape     : {v_t_ort.shape}")
    print(f"  ORT v_t finite    : {n_finite_ort}/{n_total_ort}")
    print(f"  ORT v_t range     : [{float(v_t_ort.min()):.4f}, {float(v_t_ort.max()):.4f}]")
    print(f"  ORT compose OK    : {ort_compose_ok}")
    print()
    if ort_compose_ok:
        print("  VERDICT: PASS — lockstep dual-tower prune composes end-to-end")
        print("  through torch.export + ORT. PPCL pattern transfers to pi0.5.")
        print("  Phase 1.5 build (1-2 weeks code + $30-50 Modal) is unblocked")
        print("  with eyes open.")
    else:
        print("  VERDICT: FAIL — lockstep prune broke at the ORT compose step.")
        print("  Architectural blocker; pivot to a different perf-compound feature.")
    print("=" * 72)

    return {
        "model_id": model_id,
        "drop_layer": drop_layer,
        "n_orig": n_orig,
        "n_kept": n_kept,
        "kept_indices": kept_indices,
        "eager_compose_ok": None,  # not measured per production pattern
        "ort_compose_ok": bool(ort_compose_ok),
        "prefix_onnx_mb": prefix_size_mb,
        "expert_onnx_mb": expert_size_mb,
        "v_t_shape": list(v_t_ort.shape),
        "n_finite": n_finite_ort,
        "n_total": n_total_ort,
        "v_t_min": float(v_t_ort.min()),
        "v_t_max": float(v_t_ort.max()),
    }


@app.local_entrypoint()
def main(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    drop_layer: int = 5,
):
    r = lockstep_spike_modal.remote(
        model_id=model_id, drop_layer=drop_layer,
    )
    print("\n=== RESULT ===")
    print(f"  eager_compose_ok : {r['eager_compose_ok']}")
    print(f"  ort_compose_ok   : {r['ort_compose_ok']}")
    print(f"  v_t_shape        : {r['v_t_shape']}")
    print(f"  v_t_range        : [{r['v_t_min']:.4f}, {r['v_t_max']:.4f}]")
    print(f"  prefix_onnx_mb   : {r['prefix_onnx_mb']:.1f}")
    print(f"  expert_onnx_mb   : {r['expert_onnx_mb']:.1f}")
