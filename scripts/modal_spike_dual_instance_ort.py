"""Modal: Day-0 dual-instance ORT spike for policy-versioning.

Per ADR 2026-04-25-policy-versioning-architecture + plan Day 0:
"Instantiate two Pi05DecomposedInference instances with the same export
in one process. Run alternating predict_action_chunk calls and compare
against single-instance baseline. PASS = bitwise-identical outputs
within numerical noise (<1e-5 absolute) + zero crashes + no ORT log
interleaving that corrupts state. FAIL = output divergence, ORT
allocator pool pollution, CUDA context deadlock."

Two layers tested in one run, cheap to expensive:

1. Raw ORT layer — 2 sessions on vlm_prefix.onnx + 2 sessions on
   expert_denoise.onnx, alternating runs. Tests the load-bearing concern
   (CUDA context / stream / allocator isolation between sessions on the
   same model). Cheap, fast, focused.

2. Full Pi05DecomposedInference layer — 2 instances of the production
   class with cache disabled, alternating predict_action_chunk runs.
   Tests that the wrapper class's per-instance state (action_guard,
   episode_cache, rtc_adapter, _call_index) doesn't leak between
   instances when both are alive in the same process.

Both layers must PASS for the policy-versioning architecture to be
unblocked. Either FAIL → refactor to subprocess IPC (heavier, but
correctness-safe).

Usage:
    modal profile activate novarepmarketing
    modal run scripts/modal_spike_dual_instance_ort.py

Reference:
- ADR: 01_decisions/2026-04-25-policy-versioning-architecture.md
- Plan: features/01_serve/subfeatures/_ecosystem/policy-versioning/policy-versioning_plan.md (Day 0)
- Research: same folder, _research.md (Lens 3 riskiest-assumption)

Cost: ~$1-2 on A10G, ~3-5 min wall-clock + image build.
"""
from __future__ import annotations

import os
import modal

app = modal.App("tether-policy-versioning-spike")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_bust() -> str:
    import time
    return str(int(time.time()))


_BUST = _build_bust()

onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=False)
ONNX_OUT = "/onnx_out"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "onnxruntime-gpu==1.20.1",
        "numpy<2.0",
        "onnx>=1.16",
        "prometheus-client>=0.19",
        "fastapi>=0.110",
        "torch==2.5.1",
        "transformers>=4.40",
    )
    .env({
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu",
    })
    .add_local_dir(
        os.path.join(REPO_ROOT, "src"),
        remote_path="/root/tether-vla/src",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_file(
        os.path.join(REPO_ROOT, "pyproject.toml"),
        remote_path="/root/tether-vla/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        os.path.join(REPO_ROOT, "README.md"),
        remote_path="/root/tether-vla/README.md",
        copy=True,
    )
    .add_local_file(
        os.path.join(REPO_ROOT, "LICENSE"),
        remote_path="/root/tether-vla/LICENSE",
        copy=True,
    )
    .run_commands(
        f'echo "build_bust={_BUST}"',
        'pip install -e "/root/tether-vla" --no-deps',
    )
)


# ---------------------------------------------------------------------------
# Layer 1: raw ORT dual-session test
# ---------------------------------------------------------------------------


def _layer1_raw_ort(export_dir, n_iters):
    """Test that two ort.InferenceSession on the same ONNX produce
    bitwise-identical outputs to a single session."""
    import json
    import time
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort

    cfg_path = export_dir / "tether_config.json"
    cfg = json.loads(cfg_path.read_text())
    prefix_path = export_dir / cfg["decomposed"]["vlm_prefix_onnx"]
    expert_path = export_dir / cfg["decomposed"]["expert_denoise_onnx"]

    _ORT_TO_NP = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
    }

    def _make_feed(session, seed: int):
        rng = np.random.default_rng(seed)
        feed = {}
        for inp in session.get_inputs():
            shape = [1 if (isinstance(d, str) or d is None) else int(d) for d in inp.shape]
            dtype = _ORT_TO_NP.get(inp.type, np.float32)
            if np.issubdtype(dtype, np.floating):
                feed[inp.name] = rng.standard_normal(shape).astype(dtype)
            elif dtype == np.bool_:
                feed[inp.name] = rng.integers(0, 2, size=shape) > 0
            else:
                feed[inp.name] = rng.integers(0, 100, size=shape, dtype=dtype)
        return feed

    providers = [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]

    layer1_results = {}
    for layer_name, model_path in (
        ("vlm_prefix", prefix_path),
        ("expert_denoise", expert_path),
    ):
        print(f"\n--- Layer 1 raw ORT: {layer_name} ({model_path.name}) ---", flush=True)

        # Build single baseline first to capture reference output.
        baseline = ort.InferenceSession(str(model_path), providers=providers)
        feed = _make_feed(baseline, seed=42)
        baseline_out = baseline.run(None, feed)
        print(f"  baseline single-session: produced {len(baseline_out)} outputs", flush=True)

        # Now build TWO sessions (baseline still alive — three sessions resident).
        sess_a = ort.InferenceSession(str(model_path), providers=providers)
        sess_b = ort.InferenceSession(str(model_path), providers=providers)
        print(f"  built sessions A + B alongside baseline", flush=True)

        # Compare first calls of A + B against baseline
        out_a = sess_a.run(None, feed)
        out_b = sess_b.run(None, feed)

        for i, (ref, a, b) in enumerate(zip(baseline_out, out_a, out_b)):
            np.testing.assert_allclose(ref, a, atol=1e-5, err_msg=f"{layer_name} sess_A output[{i}] diverged from baseline")
            np.testing.assert_allclose(ref, b, atol=1e-5, err_msg=f"{layer_name} sess_B output[{i}] diverged from baseline")

        # Alternating-run loop — A, B, A, B, ... each compared to baseline.
        max_abs_drift = 0.0
        last_log_t = time.perf_counter()
        for i in range(n_iters):
            sess = sess_a if i % 2 == 0 else sess_b
            label = "A" if i % 2 == 0 else "B"
            outs = sess.run(None, feed)
            for j, (ref, got) in enumerate(zip(baseline_out, outs)):
                drift = float(np.max(np.abs(ref.astype(np.float64) - got.astype(np.float64))))
                max_abs_drift = max(max_abs_drift, drift)
                if drift > 1e-5:
                    raise AssertionError(
                        f"{layer_name} iter {i} sess_{label} output[{j}] drift={drift} exceeded 1e-5"
                    )
            now = time.perf_counter()
            if (now - last_log_t) >= 30.0 or i == n_iters - 1:
                print(f"  alternating progress: {i+1}/{n_iters} max_abs_drift={max_abs_drift:.2e}", flush=True)
                last_log_t = now

        layer1_results[layer_name] = {
            "status": "ok",
            "n_iters": n_iters,
            "max_abs_drift": max_abs_drift,
        }
        print(f"  PASS: {layer_name} dual-session bitwise-identical (max drift {max_abs_drift:.2e})", flush=True)

    return layer1_results


# ---------------------------------------------------------------------------
# Layer 2: full Pi05DecomposedInference dual-instance test
# ---------------------------------------------------------------------------


def _layer2_pi05_decomposed(export_dir, n_iters):
    """Test that two Pi05DecomposedInference (cache disabled) on the same
    export produce bitwise-identical outputs to a single instance.

    cache disabled so each call is deterministic: same input → same output."""
    import json
    import time

    import numpy as np

    from tether.runtime.pi05_decomposed_server import Pi05DecomposedInference

    cfg = json.loads((export_dir / "tether_config.json").read_text())
    chunk_size = int(cfg.get("chunk_size", 50))
    action_dim = int(cfg.get("action_dim", 7))
    max_state_dim = int(cfg.get("max_state_dim", 32))
    expert_takes_state = bool(cfg.get("decomposed", {}).get("expert_takes_state", True))

    print(f"\n--- Layer 2 Pi05DecomposedInference dual-instance ---", flush=True)
    print(f"  chunk_size={chunk_size} action_dim={action_dim} max_state_dim={max_state_dim} state={expert_takes_state}", flush=True)

    # Build baseline first (single instance), capture reference output.
    baseline = Pi05DecomposedInference(export_dir, enable_cache=False)
    print(f"  baseline single-instance built", flush=True)

    rng = np.random.default_rng(2026)
    img_base = rng.standard_normal((1, 3, 224, 224), dtype=np.float32)
    img_wrist_l = rng.standard_normal((1, 3, 224, 224), dtype=np.float32)
    img_wrist_r = rng.standard_normal((1, 3, 224, 224), dtype=np.float32)
    mask_base = np.ones((1,), dtype=bool)
    mask_wrist_l = np.ones((1,), dtype=bool)
    mask_wrist_r = np.ones((1,), dtype=bool)
    lang_tokens = rng.integers(0, 100, size=(1, 200), dtype=np.int64)
    lang_masks = np.ones((1, 200), dtype=bool)
    noise = rng.standard_normal((1, chunk_size, action_dim), dtype=np.float32)
    state = rng.standard_normal((1, max_state_dim), dtype=np.float32) if expert_takes_state else None

    def _kwargs():
        return dict(
            img_base=img_base,
            img_wrist_l=img_wrist_l,
            img_wrist_r=img_wrist_r,
            mask_base=mask_base,
            mask_wrist_l=mask_wrist_l,
            mask_wrist_r=mask_wrist_r,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            noise=noise,
            state=state,
        )

    baseline_out = baseline.predict_action_chunk(**_kwargs())
    print(f"  baseline output shape: {baseline_out.shape} dtype: {baseline_out.dtype}", flush=True)

    # Build two more instances. Total 3 instances = 6 ORT sessions resident.
    inst_a = Pi05DecomposedInference(export_dir, enable_cache=False)
    inst_b = Pi05DecomposedInference(export_dir, enable_cache=False)
    print(f"  built instances A + B alongside baseline (3 Pi05DecomposedInference total = 6 ORT sessions)", flush=True)

    # Initial sanity check: A + B match baseline on the same input.
    out_a = inst_a.predict_action_chunk(**_kwargs())
    out_b = inst_b.predict_action_chunk(**_kwargs())
    np.testing.assert_allclose(baseline_out, out_a, atol=1e-5, err_msg="instance A diverged from baseline")
    np.testing.assert_allclose(baseline_out, out_b, atol=1e-5, err_msg="instance B diverged from baseline")
    print(f"  initial parity check: A and B both match baseline", flush=True)

    # Alternating loop.
    max_abs_drift = 0.0
    last_log_t = time.perf_counter()
    for i in range(n_iters):
        inst = inst_a if i % 2 == 0 else inst_b
        label = "A" if i % 2 == 0 else "B"
        out = inst.predict_action_chunk(**_kwargs())
        drift = float(np.max(np.abs(baseline_out.astype(np.float64) - out.astype(np.float64))))
        max_abs_drift = max(max_abs_drift, drift)
        if drift > 1e-5:
            raise AssertionError(
                f"layer 2 iter {i} inst_{label} drift={drift} exceeded 1e-5"
            )
        now = time.perf_counter()
        if (now - last_log_t) >= 30.0 or i == n_iters - 1:
            print(f"  alternating progress: {i+1}/{n_iters} max_abs_drift={max_abs_drift:.2e}", flush=True)
            last_log_t = now

    print(f"  PASS: dual-instance Pi05DecomposedInference bitwise-identical (max drift {max_abs_drift:.2e})", flush=True)
    return {
        "status": "ok",
        "n_iters": n_iters,
        "max_abs_drift": max_abs_drift,
    }


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="A10G",
    volumes={ONNX_OUT: onnx_output},
    timeout=1800,
)
def spike_a10g(export_subdir: str, n_iters: int):
    return _spike_body(export_subdir, n_iters, "a10g")


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={ONNX_OUT: onnx_output},
    timeout=1800,
)
def spike_a100(export_subdir: str, n_iters: int):
    return _spike_body(export_subdir, n_iters, "a100")


def _spike_body(export_subdir: str, n_iters: int, hw_label: str):
    import json
    from pathlib import Path

    import onnxruntime as ort

    print("=" * 60, flush=True)
    print(f"policy-versioning Day-0 dual-instance ORT spike (hw={hw_label})", flush=True)
    print(f"export={export_subdir} ORT={ort.__version__} n_iters={n_iters}", flush=True)
    print(f"providers={ort.get_available_providers()}", flush=True)
    print("=" * 60, flush=True)

    export_dir = Path(ONNX_OUT) / export_subdir
    if not export_dir.exists():
        return {"status": "fail", "reason": f"export_dir_missing: {export_dir}"}

    layer1 = None
    layer2 = None
    layer1_err = None
    layer2_err = None

    try:
        layer1 = _layer1_raw_ort(export_dir, n_iters=n_iters)
    except Exception as exc:
        layer1_err = repr(exc)
        print(f"layer 1 FAIL: {exc!r}", flush=True)

    try:
        layer2 = _layer2_pi05_decomposed(export_dir, n_iters=n_iters)
    except Exception as exc:
        layer2_err = repr(exc)
        print(f"layer 2 FAIL: {exc!r}", flush=True)

    return {
        "status": "ok" if (layer1 and layer2) else "fail",
        "layer1": layer1, "layer1_err": layer1_err,
        "layer2": layer2, "layer2_err": layer2_err,
    }


@app.local_entrypoint()
def main(
    hw: str = "a10g",
    export_subdir: str = "pi05_decomposed_smoke_local_auto",
    n_iters: int = 100,
):
    if hw == "a10g":
        result = spike_a10g.remote(export_subdir=export_subdir, n_iters=n_iters)
    elif hw == "a100":
        result = spike_a100.remote(export_subdir=export_subdir, n_iters=n_iters)
    else:
        raise ValueError(f"hw must be 'a10g' or 'a100', got {hw!r}")
    print()
    print("=" * 60)
    print("DUAL-INSTANCE SPIKE RESULT")
    print("=" * 60)
    if result.get("status") == "ok":
        print(f"OVERALL: PASS")
        for layer, key in (("Layer 1 raw ORT", "layer1"), ("Layer 2 Pi05DecomposedInference", "layer2")):
            r = result[key]
            print(f"  {layer}: max drift = {max((s.get('max_abs_drift', 0) for s in (r.values() if isinstance(r, dict) and 'status' not in r else [r])), default=0):.2e}")
        print()
        print("policy-versioning Day-0 gate UNBLOCKED — proceed with Day 1-2 plan.")
    else:
        print(f"OVERALL: FAIL")
        if result.get("layer1_err"):
            print(f"  Layer 1 error: {result['layer1_err']}")
        if result.get("layer2_err"):
            print(f"  Layer 2 error: {result['layer2_err']}")
        print()
        print("policy-versioning Day-0 gate FAILED — investigate before Day 1-2.")
        print("Contingency: refactor 2-policy to subprocess IPC (heavier, correctness-safe).")
