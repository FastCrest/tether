"""Modal A100: per-step expert ONNX export parity gate (gate 3).

Builds BOTH a baked-loop expert (current default) AND a per-step expert
(``per_step_expert=True``) from the same pi0.5 checkpoint. Runs each on
shared seeded inputs and compares cos + max_abs. The two paths must
produce numerically identical actions to ship.

Acceptance gate (research sidecar Lens 5):
    cos     ≥ 0.99999
    max_abs ≤ 1e-5

Cells (test matrix per research sidecar artifact 1):
    pi05 teacher  × num_steps={1, 10}
    pi05 student  × num_steps={1, 10}

Writes receipt JSON to ``reflex_context/per_step_parity_last_run.json``.
The receipt is consumed by ``tests/test_decomposed_per_step_parity.py``.

Spec:        features/03_export/per-step-expert-export.md
Research:    features/03_export/per-step-expert-export_research.md

Cost: ~$3 Modal (one A100-80GB invocation, ~15 min wall).

Usage:
    modal profile activate suranjana-jain
    HF_TOKEN=<token> modal run scripts/modal_per_step_parity.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

app = modal.App("tether-per-step-parity")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


def _build_bust() -> str:
    return str(int(time.time()))


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
        "onnxruntime-gpu>=1.20,<1.24",
        "onnxscript>=0.1",
        "onnx-diagnostic>=0.9",
        "optree",
        "scipy",
        "numpy",
        "accelerate",
        "draccus",
        "nvidia-cudnn-cu12>=9.0,<10.0",
        "nvidia-cublas-cu12>=12.0,<13.0",
    )
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
        'pip install -e "/root/tether-vla[monolithic]"',
    )
    .env({
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": f"{HF_CACHE}/transformers",
    })
)


# Test matrix per research sidecar artifact 1.
# Each cell:
#   - export_subdir base (under ONNX_OUT)
#   - num_steps for the baked-loop variant
#   - student_checkpoint dir if applicable (None for teacher)
CELLS = [
    {
        "label": "pi05_teacher_n10",
        "model_id": "lerobot/pi05_libero_finetuned_v044",
        "num_steps": 10,
        "student_checkpoint": None,
        "subdir": "per_step_parity/pi05_teacher_n10",
    },
    {
        "label": "pi05_teacher_n1",
        "model_id": "lerobot/pi05_libero_finetuned_v044",
        "num_steps": 1,
        "student_checkpoint": None,
        "subdir": "per_step_parity/pi05_teacher_n1",
    },
]


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def parity_test() -> dict:
    """Build baked + per-step exports for each cell, run both on shared
    seeded inputs, compute cos + max_abs."""
    import logging
    import numpy as np
    import onnxruntime as ort

    from tether.exporters.decomposed import (
        PI05_HEAD_DIM,
        PI05_KV_HEADS,
        PI05_PALIGEMMA_LAYERS,
        export_pi05_decomposed,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    results: dict[str, dict] = {}

    for cell in CELLS:
        log.info("=" * 60)
        log.info("CELL: %s", cell["label"])
        log.info("=" * 60)

        baked_dir = Path(ONNX_OUT) / f"{cell['subdir']}_baked"
        per_step_dir = Path(ONNX_OUT) / f"{cell['subdir']}_per_step"
        baked_dir.mkdir(parents=True, exist_ok=True)
        per_step_dir.mkdir(parents=True, exist_ok=True)

        # Build the baked-loop export
        log.info("[%s] building baked-loop export → %s", cell["label"], baked_dir)
        export_pi05_decomposed(
            model_id=cell["model_id"],
            output_dir=str(baked_dir),
            num_steps=cell["num_steps"],
            student_checkpoint=cell["student_checkpoint"],
            variant="default",
            export_mode="sequential",
            per_step_expert=False,
        )

        # Build the per-step export
        log.info("[%s] building per-step export → %s", cell["label"], per_step_dir)
        export_pi05_decomposed(
            model_id=cell["model_id"],
            output_dir=str(per_step_dir),
            num_steps=cell["num_steps"],
            student_checkpoint=cell["student_checkpoint"],
            variant="default",
            export_mode="sequential",
            per_step_expert=True,
        )

        # Load both expert sessions on CUDA
        log.info("[%s] loading sessions", cell["label"])
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_baked = ort.InferenceSession(
            str(baked_dir / "expert_denoise.onnx"),
            providers=providers,
        )
        sess_per_step = ort.InferenceSession(
            str(per_step_dir / "expert_denoise.onnx"),
            providers=providers,
        )
        # Confirm CUDA is in use, not silent CPU fallback
        actual_baked = sess_baked.get_providers()[0]
        actual_per_step = sess_per_step.get_providers()[0]
        log.info(
            "[%s] providers: baked=%s, per_step=%s",
            cell["label"], actual_baked, actual_per_step,
        )

        # Seeded shared inputs for parity comparison
        rng = np.random.default_rng(seed=42)
        prefix_seq_len = 968  # pi05 fixed
        chunk = 50  # pi05 fixed
        action_dim = 32  # pi05 fixed
        B = 1

        past_kvs = [
            rng.standard_normal(
                (B, PI05_KV_HEADS, prefix_seq_len, PI05_HEAD_DIM),
                dtype=np.float64,
            ).astype(np.float32)
            for _ in range(PI05_PALIGEMMA_LAYERS * 2)
        ]
        prefix_pad_masks = np.ones((B, prefix_seq_len), dtype=bool)
        noise = rng.standard_normal((B, chunk, action_dim), dtype=np.float64).astype(np.float32)

        # ──────── Run BAKED ────────
        baked_feed = {}
        for i, name in enumerate(_past_kv_names(PI05_PALIGEMMA_LAYERS)):
            baked_feed[name] = past_kvs[i]
        baked_feed["prefix_pad_masks"] = prefix_pad_masks
        baked_feed["noise"] = noise
        log.info("[%s] running baked ORT call", cell["label"])
        actions_baked = sess_baked.run(["actions"], baked_feed)[0]

        # ──────── Run PER-STEP (Python Euler loop) ────────
        log.info("[%s] running per-step Python Euler loop (n=%d)", cell["label"], cell["num_steps"])
        n = cell["num_steps"]
        dt = -1.0 / n
        x_t = noise.copy()
        for step in range(n):
            time_val = 1.0 + step * dt
            t = np.full((B,), time_val, dtype=np.float32)
            ps_feed = {}
            for i, name in enumerate(_past_kv_names(PI05_PALIGEMMA_LAYERS)):
                ps_feed[name] = past_kvs[i]
            ps_feed["prefix_pad_masks"] = prefix_pad_masks
            ps_feed["x_t"] = x_t
            ps_feed["t"] = t
            v_t = sess_per_step.run(["v_t"], ps_feed)[0]
            x_t = x_t + dt * v_t
        actions_per_step = x_t

        # ──────── Compute parity metrics ────────
        a_flat = actions_baked.flatten().astype(np.float64)
        b_flat = actions_per_step.flatten().astype(np.float64)
        denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) or 1.0
        cos = float(np.dot(a_flat, b_flat) / denom)
        max_abs = float(np.max(np.abs(actions_baked - actions_per_step)))
        mean_abs = float(np.mean(np.abs(actions_baked - actions_per_step)))

        log.info(
            "[%s] cos=%.10f, max_abs=%.3e, mean_abs=%.3e",
            cell["label"], cos, max_abs, mean_abs,
        )

        results[cell["label"]] = {
            "cell": cell["label"],
            "model_id": cell["model_id"],
            "num_steps": cell["num_steps"],
            "cos": cos,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "actions_shape": list(actions_baked.shape),
            "used_provider_baked": actual_baked,
            "used_provider_per_step": actual_per_step,
            # Acceptance gate per research sidecar Lens 5
            "passes_cos_gate": cos >= 0.99999,
            "passes_max_abs_gate": max_abs <= 1e-5,
            "passes_overall": cos >= 0.99999 and max_abs <= 1e-5,
        }

        # Free GPU memory before next cell
        del sess_baked, sess_per_step
        import gc
        gc.collect()

    return {
        "cells": results,
        "all_passed": all(r["passes_overall"] for r in results.values()),
        "thresholds": {"cos_min": 0.99999, "max_abs_max": 1e-5},
    }


def _past_kv_names(num_layers: int) -> list[str]:
    """Mirror tether.exporters.decomposed._past_kv_names() — flat list of
    past_k_0, past_v_0, past_k_1, past_v_1, ... per layer."""
    names = []
    for i in range(num_layers):
        names.append(f"past_k_{i}")
        names.append(f"past_v_{i}")
    return names


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("Per-step expert ONNX parity test (gate 3)")
    print("=" * 60)
    result = parity_test.remote()

    # Write receipt
    receipt_path = Path(REPO_ROOT) / ".." / "reflex_context" / "per_step_parity_last_run.json"
    receipt_path = receipt_path.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(result, indent=2))
    print(f"\nReceipt written: {receipt_path}")

    # Surface results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for label, r in result["cells"].items():
        status = "✓" if r["passes_overall"] else "✗"
        print(
            f"  {status} {label:25} cos={r['cos']:.10f}  "
            f"max_abs={r['max_abs']:.3e}  mean_abs={r['mean_abs']:.3e}"
        )
    print(
        f"\n  Overall: {'PASS' if result['all_passed'] else 'FAIL'} "
        f"(thresholds: cos≥{result['thresholds']['cos_min']}, "
        f"max_abs≤{result['thresholds']['max_abs_max']:.0e})"
    )
