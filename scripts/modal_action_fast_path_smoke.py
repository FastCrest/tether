"""Modal: production-config validation of action-similarity-fast-path
on real pi0.5 decomposed export.

Single A100 fire, ~$3-5. Validates the only feature shipped today that
touches the /act inference path under realistic conditions (unit + mock
integration tests already covered the wire-up; this validates against a
real ORT session + real pi0.5 expert output distribution).

Sweep design:
  1. Load Pi05DecomposedInference from a pre-existing decomposed export
     on the pi0-onnx-outputs Modal volume (pi05_decomposed_smoke_local_auto).
  2. Fire N=20 predict_action_chunk calls with IDENTICAL image + lang +
     noise. ORT determinism → identical expert outputs → L2 distance
     between consecutive chunks = 0 → ALWAYS below threshold=0.05.
  3. Compare:
       Run A: action_similarity_threshold=0.0 (disabled, baseline)
       Run B: action_similarity_threshold=0.05 (enabled)
  4. Verify:
       - Both runs return finite (B, chunk_size, action_dim) actions
       - Run A: expert called 20 times, 0 skips
       - Run B: expert called ~10 times (every-other pattern per
                consume_skip semantics), ~10 skips
       - Run B's actions are bit-exact to Run A's at the same call index
         WHEN the call uses the cached chunk (skip path)

Decision the run informs: does the action-similarity-fast-path wire-up
actually fire on real pi0.5 expert outputs? If yes → today's ship is
production-validated. If no → debug + fix.

Min-viable N: 20 calls × 2 modes = 40 inferences total. Pi0.5 decomposed
~270 ms/call on A100 → ~11s of compute + ~2 min model load + ~1 min image
build = ~5 min wall clock. Cost ~$2-3.

Customer-trace-archive + uncertainty scoring NOT validated here — both
are non-/act-path and have full unit + smoke test coverage already.

Usage:
    modal run scripts/modal_action_fast_path_smoke.py
"""
import os
import subprocess
import modal

app = modal.App("reflex-action-fast-path-smoke")


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
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"
ONNX_OUT = "/onnx_out"

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
    timeout=1200,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def smoke_modal(
    export_subdir: str = "pi05_decomposed_smoke_local_auto",
    n_calls: int = 20,
):
    """Fire N=n_calls predict_action_chunk calls in two modes (fast-path
    OFF then ON), report skip-count + bit-exactness."""
    import logging
    import time
    from pathlib import Path
    import numpy as np
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("fp-smoke")

    # Trigger reflex's eager dlopen for ORT-CUDA EP (caught yesterday in
    # 2026-05-06 ort-streams-spike — image had CUDA-13 nvidia libs while
    # ORT 1.20 wants CUDA-12; reflex's __init__ fixes the loadchain).
    import reflex
    logger.info("[smoke] reflex imported (eager-dlopen-nvidia-libs)")

    export_dir = Path(ONNX_OUT) / export_subdir
    if not export_dir.exists():
        raise FileNotFoundError(
            f"export not found: {export_dir}. List the volume:\n"
            f"  modal volume ls pi0-onnx-outputs {export_subdir}"
        )
    logger.info("[smoke] export_dir: %s", export_dir)

    from reflex.runtime.pi05_decomposed_server import Pi05DecomposedInference

    # ---- BASELINE: ORT-CUDA EP loaded check ----
    # Disabled-fast-path run — also serves as the "did CUDA EP actually
    # come up" sanity gate (per feedback_validate_baseline_and_check_modal_midflight).
    logger.info("[smoke] Run A: action_similarity_threshold=0.0 (disabled)")
    t0 = time.time()
    inf_off = Pi05DecomposedInference(
        export_dir=str(export_dir),
        enable_cache=False,         # disable prefix cache to isolate fast-path
        cache_level="none",
        action_similarity_threshold=0.0,
        max_similar_skips=3,
    )
    logger.info("[smoke] Run A loaded in %.1fs", time.time() - t0)

    # Get expert session providers — fail loud if not CUDA
    raw_expert = getattr(inf_off._sess_expert, "session", inf_off._sess_expert)
    active_providers = raw_expert.get_providers()
    logger.info("[smoke] active providers: %s", active_providers)
    if "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError(
            f"BASELINE FAIL — CUDA EP not active. Got {active_providers}. "
            f"Spike measurements would be CPU-tier; not representative of "
            f"production. Fix CUDA loadchain before re-running."
        )

    # Build deterministic dummy inputs — same per call so expert outputs
    # are identical (ORT determinism) → L2 distance = 0 → fast path
    # ALWAYS triggers when threshold > 0. Cleanest signal for the
    # wire-up validation.
    cfg = inf_off.config
    chunk_size = cfg.get("chunk_size", 50)
    action_dim = cfg.get("action_dim", 7)
    image_size = cfg.get("decomposed", {}).get("image_size", 224)
    # Probe lang_tokens shape from the prefix session — Pi05DecomposedInference
    # doesn't expose lang_seq_len directly (Pi05DecomposedServer does).
    raw_prefix = getattr(inf_off._sess_prefix, "session", inf_off._sess_prefix)
    lang_seq_len = 16
    for inp in raw_prefix.get_inputs():
        if inp.name == "lang_tokens":
            shape = inp.shape
            if len(shape) >= 2 and isinstance(shape[1], int):
                lang_seq_len = int(shape[1])
            break
    logger.info("[smoke] probed lang_seq_len=%d", lang_seq_len)

    rng = np.random.RandomState(0)
    img = rng.rand(1, 3, image_size, image_size).astype(np.float32)
    mask = np.ones(1, dtype=bool)
    lang_tokens = rng.randint(0, 257152, size=(1, lang_seq_len)).astype(np.int64)
    lang_masks = np.ones((1, lang_seq_len), dtype=bool)
    noise = rng.randn(1, chunk_size, action_dim).astype(np.float32)

    common_args = dict(
        img_base=img, img_wrist_l=img, img_wrist_r=img,
        mask_base=mask, mask_wrist_l=mask, mask_wrist_r=mask,
        lang_tokens=lang_tokens, lang_masks=lang_masks,
        noise=noise,
    )

    # ---- Run A: threshold=0.0 (disabled) ----
    actions_off = []
    t0 = time.time()
    for i in range(n_calls):
        a = inf_off.predict_action_chunk(**common_args)
        actions_off.append(a.copy())
    elapsed_off = time.time() - t0
    stats_off = inf_off._fast_path.stats
    logger.info(
        "[smoke] Run A: %d calls in %.1fs; expert_calls=%d skip_count=%d",
        n_calls, elapsed_off, stats_off.expert_calls, stats_off.skip_count,
    )

    # ---- Run B: threshold=0.05 (enabled) ----
    logger.info("[smoke] Run B: action_similarity_threshold=0.05 (enabled)")
    t0 = time.time()
    inf_on = Pi05DecomposedInference(
        export_dir=str(export_dir),
        enable_cache=False,
        cache_level="none",
        action_similarity_threshold=0.05,
        max_similar_skips=3,
    )
    logger.info("[smoke] Run B loaded in %.1fs", time.time() - t0)

    actions_on = []
    t0 = time.time()
    for i in range(n_calls):
        a = inf_on.predict_action_chunk(**common_args)
        actions_on.append(a.copy())
    elapsed_on = time.time() - t0
    stats_on = inf_on._fast_path.stats
    logger.info(
        "[smoke] Run B: %d calls in %.1fs; expert_calls=%d skip_count=%d",
        n_calls, elapsed_on, stats_on.expert_calls, stats_on.skip_count,
    )

    # ---- Verify ----
    # 1. Run A: 0 skips, expert called n_calls times
    # 2. Run B: skips > 0 (fast path actually fired)
    # 3. Run B: bit-exact actions (since deterministic inputs + cache reuse
    #    AND expert reruns on same input produce identical outputs)
    # 4. Latency: Run B should be FASTER (some calls skip the expert)

    n_total = len(actions_off)
    bit_exact_count = 0
    for a, b in zip(actions_off, actions_on):
        if np.array_equal(a, b):
            bit_exact_count += 1
        else:
            max_diff = float(np.abs(a - b).max())
            mean_diff = float(np.abs(a - b).mean())
            logger.info(
                "[smoke] mismatch at call %d: max_diff=%.6e mean_diff=%.6e",
                actions_off.index(a), max_diff, mean_diff,
            )

    speedup = elapsed_off / elapsed_on if elapsed_on > 0 else 0

    # ---- Report ----
    print("\n" + "=" * 72)
    print("ACTION-SIMILARITY-FAST-PATH PRODUCTION SMOKE")
    print("=" * 72)
    print(f"  GPU              : A100-80GB")
    print(f"  export_dir       : {export_dir}")
    print(f"  N calls per mode : {n_calls}")
    print()
    print(f"  Run A (disabled, threshold=0.0):")
    print(f"    elapsed       : {elapsed_off:.1f}s")
    print(f"    expert_calls  : {stats_off.expert_calls}")
    print(f"    skip_count    : {stats_off.skip_count}")
    print()
    print(f"  Run B (enabled, threshold=0.05):")
    print(f"    elapsed       : {elapsed_on:.1f}s")
    print(f"    expert_calls  : {stats_on.expert_calls}")
    print(f"    skip_count    : {stats_on.skip_count}")
    print()
    print(f"  Bit-exact actions: {bit_exact_count} / {n_total}")
    print(f"  Speedup          : {speedup:.2f}x  (Run A elapsed / Run B elapsed)")
    print()

    # Verdict
    expected_skip_ratio = 0.4  # ~50% but consume_skip pattern means just under
    skip_ratio = stats_on.skip_count / n_calls if n_calls > 0 else 0

    issues = []
    if stats_off.skip_count != 0:
        issues.append(f"Run A had skips when threshold=0 (got {stats_off.skip_count})")
    if stats_on.skip_count == 0:
        issues.append(f"Run B had no skips — fast path didn't fire on identical inputs")
    if skip_ratio < expected_skip_ratio:
        issues.append(
            f"Skip ratio {skip_ratio:.1%} below expected ~{expected_skip_ratio:.0%}; wire-up may be broken"
        )
    if bit_exact_count != n_total:
        issues.append(
            f"Bit-exact count {bit_exact_count} != {n_total}; deterministic inputs should produce identical outputs across modes"
        )
    if speedup < 1.0:
        issues.append(f"Run B was SLOWER than Run A (speedup {speedup:.2f}x); fast path adds overhead instead of saving")

    if not issues:
        print("  VERDICT: PASS — fast path fires correctly on production pi0.5,")
        print("           bit-exact outputs, measurable speedup.")
    else:
        print("  VERDICT: FAIL")
        for x in issues:
            print(f"    - {x}")
    print("=" * 72)

    return {
        "n_calls": n_calls,
        "run_a": {
            "elapsed_s": elapsed_off,
            "expert_calls": stats_off.expert_calls,
            "skip_count": stats_off.skip_count,
        },
        "run_b": {
            "elapsed_s": elapsed_on,
            "expert_calls": stats_on.expert_calls,
            "skip_count": stats_on.skip_count,
        },
        "bit_exact_count": bit_exact_count,
        "speedup": speedup,
        "skip_ratio": skip_ratio,
        "issues": issues,
        "verdict": "PASS" if not issues else "FAIL",
    }


@app.local_entrypoint()
def main(
    export_subdir: str = "pi05_decomposed_smoke_local_auto",
    n_calls: int = 20,
):
    r = smoke_modal.remote(export_subdir=export_subdir, n_calls=n_calls)
    print("\n=== RESULT ===")
    print(f"  verdict      : {r['verdict']}")
    print(f"  speedup      : {r['speedup']:.2f}x")
    print(f"  Run A skips  : {r['run_a']['skip_count']}")
    print(f"  Run B skips  : {r['run_b']['skip_count']}")
    print(f"  bit-exact    : {r['bit_exact_count']} / {r['n_calls']}")
    if r["issues"]:
        print("  issues:")
        for x in r["issues"]:
            print(f"    - {x}")
