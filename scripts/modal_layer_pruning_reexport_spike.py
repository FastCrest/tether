"""Modal: re-export feasibility spike for language-layer-pruning-adaptive.

Day-2 spike for the language-layer-pruning-adaptive feature (Day-1
cosine-sim profiling already PASS at 15/17 pairs > 0.95 — see
03_experiments/2026-05-06-language-layer-pruning-cosine-sim-spike.md).

Decision the run informs:

    Can torch.onnx.export trace a Pi05PrefixWrapper whose internal
    paligemma.language_model.layers is a SHORTENED ModuleList? Does
    the resulting ONNX file have fewer past_kv outputs + smaller file
    size?

If yes -> static-shape constraint does NOT block adaptive depth via
re-export (Option A from the research sidecar Lens 3 is viable).

If no  -> Option A is blocked; the entire feature requires re-thinking
the static-shape strategy (likely deferred to Phase 2 dynamic-shape
re-export work).

Test prune set: drop layers 1-7 (most-redundant per Day-1 spike at
mean sim 0.9947-0.9974). Keep layers 0 (input projection), 8-17 (final
specialization). Expected: ~40% layer reduction, ~40% prefix latency
reduction, prefix ONNX file size drops proportionally.

Spike does NOT validate downstream composition with expert_denoise.onnx
(separate concern: expert reads 36 past_kv tensors, pruned prefix
outputs 22). That gate is the next spike.

Usage:
    modal run scripts/modal_layer_pruning_reexport_spike.py
"""
import os
import subprocess
import modal

app = modal.App("reflex-layer-pruning-reexport-spike")


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
        "onnxruntime>=1.20",
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
    gpu="A10G",
    timeout=3600,
    volumes={HF_CACHE: hf_cache},
    secrets=[_hf_secret()],
)
def reexport_spike_modal(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    prune_indices: str = "1,2,3,4,5,6,7",  # comma-separated; default drop layers 1-7
):
    """Re-export pi0.5 vlm_prefix with a subset of language-model layers."""
    import logging
    import time
    from pathlib import Path
    import torch
    import torch.nn as nn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("reexport-spike")

    skip_set = {int(x) for x in prune_indices.split(",")}
    logger.info("[spike] prune indices: %s", sorted(skip_set))

    # ---- Load policy ----
    logger.info("[spike] loading %s", model_id)
    t0 = time.time()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained(model_id).eval().to(torch.float32).to("cpu")
    logger.info("[spike] policy loaded in %.1fs", time.time() - t0)

    # ---- Inspect baseline ----
    paligemma = policy.model.paligemma_with_expert.paligemma
    language_model = paligemma.model.language_model
    original_layers = list(language_model.layers)
    n_orig = len(original_layers)
    logger.info("[spike] original PaliGemma language_model layers: %d", n_orig)

    if any(i >= n_orig for i in skip_set):
        raise ValueError(
            f"prune_indices contains values >= n_layers ({n_orig}): {sorted(skip_set)}"
        )

    kept_indices = [i for i in range(n_orig) if i not in skip_set]
    logger.info(
        "[spike] keeping %d of %d layers: %s",
        len(kept_indices), n_orig, kept_indices,
    )

    # ---- Patch language_model.layers ----
    # Replace the 18-layer ModuleList with a shortened ModuleList containing
    # only the kept layers, in order. The PaliGemma forward iterates
    # `for layer in self.layers:` so this should produce a structurally
    # smaller compute graph at trace time.
    pruned_layers = nn.ModuleList([original_layers[i] for i in kept_indices])
    language_model.layers = pruned_layers
    # PaliGemma may also read config.num_hidden_layers — patch it too.
    n_kept = len(kept_indices)
    if hasattr(language_model.config, "num_hidden_layers"):
        old = language_model.config.num_hidden_layers
        language_model.config.num_hidden_layers = n_kept
        logger.info(
            "[spike] patched language_model.config.num_hidden_layers: %d -> %d",
            old, n_kept,
        )

    logger.info("[spike] pruned to %d layers", len(language_model.layers))

    # ---- Build wrapper + dummy inputs (mirror decomposed.py's contract) ----
    from reflex.exporters.decomposed import (
        Pi05PrefixWrapper,
        _PI05_BATCH_SIZE,
        _PI05_IMAGE_SIZE,
        _PI05_LANG_TOKENS,
    )

    B = _PI05_BATCH_SIZE
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

    # past_kv_names for KEPT layers (each kept layer outputs K + V)
    past_kv_names_pruned = []
    for li in kept_indices:
        past_kv_names_pruned.append(f"past_k_{li}")
        past_kv_names_pruned.append(f"past_v_{li}")
    prefix_output_names = past_kv_names_pruned + ["prefix_pad_masks"]

    # ---- Eager forward to verify the pruned model runs ----
    logger.info("[spike] running pruned forward (eager)...")
    t0 = time.time()
    with torch.no_grad():
        eager_outputs = prefix_wrapper(*tuple(prefix_dummy.values()))
    eager_ms = (time.time() - t0) * 1000
    logger.info("[spike] eager forward: %.1f ms; %d outputs", eager_ms, len(eager_outputs))
    expected_n_outputs = 2 * n_kept + 1  # 2 * kept layers (K+V) + prefix_pad_masks
    if len(eager_outputs) != expected_n_outputs:
        logger.warning(
            "[spike] output count mismatch: got %d, expected %d (= 2*%d kept + 1 prefix_pad)",
            len(eager_outputs), expected_n_outputs, n_kept,
        )

    # ---- Re-export pruned prefix ONNX ----
    out_dir = Path("/tmp/pruned_export")
    out_dir.mkdir(parents=True, exist_ok=True)
    pruned_path = out_dir / f"vlm_prefix_skip_{'_'.join(str(i) for i in sorted(skip_set))}.onnx"

    logger.info("[spike] exporting pruned prefix → %s", pruned_path)
    from onnx_diagnostic.torch_export_patches import torch_export_patches
    from reflex.exporters.monolithic import _fix_onnx_where_dtype_mismatches

    t0 = time.time()
    with torch_export_patches(patch_transformers=True):
        ep = torch.export.export(
            prefix_wrapper, tuple(prefix_dummy.values()),
            dynamic_shapes=None, strict=False,
        )
    logger.info("[spike] torch.export: %.1fs", time.time() - t0)

    t0 = time.time()
    torch.onnx.export(
        ep, tuple(prefix_dummy.values()), str(pruned_path),
        input_names=list(prefix_dummy.keys()),
        output_names=prefix_output_names,
        opset_version=19,
    )
    logger.info("[spike] torch.onnx.export: %.1fs", time.time() - t0)

    cast_fixes = _fix_onnx_where_dtype_mismatches(pruned_path)
    logger.info("[spike] post-export Cast fixes: %d", cast_fixes)

    pruned_size_mb = pruned_path.stat().st_size / (1024 * 1024)
    logger.info("[spike] pruned prefix ONNX size: %.1f MB", pruned_size_mb)

    # ---- Sanity-check ORT load (no execution; just session creation) ----
    import onnxruntime as ort
    logger.info("[spike] ORT session creation...")
    t0 = time.time()
    sess = ort.InferenceSession(
        str(pruned_path), providers=["CPUExecutionProvider"],
    )
    sess_ms = (time.time() - t0) * 1000
    n_inputs = len(sess.get_inputs())
    n_outputs = len(sess.get_outputs())
    logger.info(
        "[spike] ORT session OK in %.1f ms: %d inputs, %d outputs",
        sess_ms, n_inputs, n_outputs,
    )
    output_names = [o.name for o in sess.get_outputs()]

    # ---- Report ----
    print("\n" + "=" * 72)
    print(f"PI0.5 PRUNED-PREFIX RE-EXPORT SPIKE")
    print("=" * 72)
    print(f"  model_id        : {model_id}")
    print(f"  original layers : {n_orig}")
    print(f"  kept layers     : {n_kept} (of {n_orig})")
    print(f"  pruned indices  : {sorted(skip_set)}")
    print(f"  kept indices    : {kept_indices}")
    print()
    print(f"  eager forward   : {eager_ms:.1f} ms")
    print(f"  ONNX file size  : {pruned_size_mb:.1f} MB")
    print(f"  ORT session OK  : {sess_ms:.1f} ms")
    print(f"  ONNX inputs     : {n_inputs}")
    print(
        f"  ONNX outputs    : {n_outputs} "
        f"(expected 2*{n_kept}+1={2*n_kept + 1})"
    )
    print()
    print("  Output names (first 6):")
    for nm in output_names[:6]:
        print(f"    {nm}")
    if len(output_names) > 6:
        print(f"    ... +{len(output_names) - 6} more")
    print()
    n_outputs_ok = (n_outputs == 2 * n_kept + 1)
    if n_outputs_ok:
        print("  VERDICT: PASS — pruned re-export produces a structurally smaller")
        print("  ONNX with the expected past_kv output count. Static-shape constraint")
        print("  does NOT block adaptive depth via Option A (per-strategy re-export).")
        print()
        print("  Next gate: re-export expert_denoise.onnx with matching past_kv input")
        print("  count + parity-check the composed pruned-prefix + pruned-expert pair.")
    else:
        print("  VERDICT: PARTIAL — export succeeded but output count diverges from")
        print("  expectation. Surfaces a contract issue worth investigating before")
        print("  proceeding to expert re-export.")
    print("=" * 72)

    return {
        "model_id": model_id,
        "original_layers": n_orig,
        "kept_layers": n_kept,
        "pruned_indices": sorted(skip_set),
        "kept_indices": kept_indices,
        "eager_forward_ms": eager_ms,
        "pruned_onnx_mb": pruned_size_mb,
        "ort_session_ok": True,
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
        "expected_n_outputs": 2 * n_kept + 1,
        "verdict": "PASS" if n_outputs_ok else "PARTIAL",
    }


@app.local_entrypoint()
def main(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    prune_indices: str = "1,2,3,4,5,6,7",
):
    r = reexport_spike_modal.remote(
        model_id=model_id, prune_indices=prune_indices,
    )
    print("\n=== RESULT ===")
    print(f"  verdict        : {r['verdict']}")
    print(f"  kept_layers    : {r['kept_layers']} of {r['original_layers']}")
    print(f"  pruned_onnx_mb : {r['pruned_onnx_mb']:.1f}")
    print(f"  n_outputs      : {r['n_outputs']} (expected {r['expected_n_outputs']})")
