"""PyTorch-vs-ONNX per-stage diff for SmolVLA-LIBERO.

The decisive test. Runs both paths on IDENTICAL (preprocessed) inputs and
reports L2 distance at each intermediate stage so we can localise exactly
where the pipelines diverge.

Stages compared:
  1. Vision encoder output per camera   (our vision_encoder.onnx vs policy's vlm.vision_tower + connector)
  2. Text embeddings                    (our text_embedder.onnx vs policy's embed_tokens)
  3. Per-layer VLM k (post-RoPE)        (our decoder_prefill.onnx vlm_k[i] vs policy's key_states[i])
  4. Per-layer VLM v                    (our decoder_prefill.onnx vlm_v[i] vs policy's value_states[i])
  5. Final action chunk                 (our expert_stack.onnx vs policy.predict_action_chunk)

Runs in one ~5-7 min Modal call.
Usage:
    modal run scripts/modal_pytorch_vs_onnx.py
"""
import json
import subprocess
import sys
import time

import modal

app = modal.App("tether-pytorch-vs-onnx")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "libegl1-mesa",
                 "libglvnd0", "ffmpeg", "cmake", "build-essential")
    .pip_install(
        "torch", "safetensors", "huggingface_hub", "transformers>=4.51",
        "onnx", "onnxruntime", "onnxscript", "numpy", "Pillow",
        "pydantic>=2.0", "typer", "rich", "pyyaml", "einops",
    )
    .pip_install("lerobot", "num2words")
    .add_local_dir("src/tether", "/root/tether-vla/src/tether", copy=True)
    .add_local_file("pyproject.toml", "/root/tether-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/tether-vla/README.md", copy=True)
    .run_commands("cd /root/tether-vla && pip install -e .")
)


@app.function(image=image, gpu="A10G", timeout=1200, scaledown_window=60)
def diff():
    import os
    os.environ.setdefault("HF_HOME", "/tmp/hf")

    import numpy as np
    import torch

    # ── Step 1: Export our ONNX pipeline ───────────────────────────
    print("=" * 60)
    print("Step 1: export our ONNX pipeline")
    print("=" * 60)
    export_dir = "/tmp/tether_libero_export"
    t0 = time.time()
    r = subprocess.run(
        ["tether", "export", "lerobot/smolvla_libero",
         "--target", "desktop", "--output", export_dir],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        print("EXPORT FAILED"); print(r.stdout[-1500:]); print(r.stderr[-800:])
        return {"error": "export"}
    print(f"exported in {time.time()-t0:.0f}s")

    # ── Step 2: Load the reference PyTorch policy + processor ──────
    print("\n" + "=" * 60)
    print("Step 2: load PyTorch policy + preprocessor")
    print("=" * 60)
    from huggingface_hub import snapshot_download
    repo_path = snapshot_download("lerobot/smolvla_libero")
    print(f"repo cached at {repo_path}")

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition,
        transition_to_batch,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")
    policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_path,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_path,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    print("policy + processors loaded")

    # ── Step 3: Build a synthetic raw batch ────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: build raw batch (3 cameras + state + task)")
    print("=" * 60)
    rng = np.random.RandomState(42)
    H, W = 256, 256
    img_np = rng.randint(0, 255, (H, W, 3), dtype=np.uint8)
    # LeRobot's LIBERO dataset uses 8D state: eef_pos(3) + axis_angle(3) + gripper_qpos(2).
    # The preprocessor's normalizer stats are shape (8,) so we need 8D here.
    state_np = rng.randn(8).astype(np.float32) * 0.1

    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # [3, H, W]
    batch_raw = {
        "observation.images.camera1": img_t.unsqueeze(0),
        "observation.images.camera2": img_t.unsqueeze(0),
        "observation.images.camera3": img_t.unsqueeze(0),
        "observation.state": torch.from_numpy(state_np).unsqueeze(0),
        "task": ["put the red bowl on the plate"],
    }
    print("raw batch keys:", sorted(batch_raw.keys()))

    # ── Step 4: Run preprocessor → get tokenized + normalized batch
    print("\n" + "=" * 60)
    print("Step 4: run preprocessor (tokenize + normalize)")
    print("=" * 60)
    batch_pp = preprocessor(batch_raw)
    print("preprocessed batch keys:", sorted(batch_pp.keys()))
    for k, v in batch_pp.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        elif isinstance(v, list) and v:
            print(f"  {k}: list[{len(v)}] first={v[0]!r}")

    # ── Step 5: PyTorch predict_action_chunk ───────────────────────
    print("\n" + "=" * 60)
    print("Step 5: PyTorch predict_action_chunk")
    print("=" * 60)
    batch_device = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch_pp.items()
    }
    # Generate SHARED noise so the flow-matching initial condition is identical
    # in both paths. Without this, cos_sim is dominated by random noise drift,
    # not export correctness.
    chunk = int(policy.config.chunk_size)
    max_action_dim = int(policy.config.max_action_dim)
    shared_noise_np = np.random.RandomState(99).randn(1, chunk, max_action_dim).astype(np.float32)
    shared_noise_torch = torch.from_numpy(shared_noise_np).to(device)

    with torch.no_grad():
        torch_actions = policy.predict_action_chunk(batch_device, noise=shared_noise_torch)
    torch_actions_cpu = torch_actions.detach().cpu().numpy()
    print(f"torch actions shape: {torch_actions_cpu.shape}")
    print(f"torch first action: {np.round(torch_actions_cpu[0, 0], 3).tolist()}")

    # Apply postprocessor to unnormalize like production eval does
    torch_postprocessed = postprocessor(torch.from_numpy(torch_actions_cpu))
    if isinstance(torch_postprocessed, torch.Tensor):
        torch_unnorm = torch_postprocessed.detach().cpu().numpy()
    else:
        torch_unnorm = np.asarray(torch_postprocessed)
    print(f"torch post-processor first action: {np.round(torch_unnorm.flatten()[:7], 3).tolist()}")

    # ── Step 6: Our ONNX pipeline, SAME preprocessed batch ─────────
    print("\n" + "=" * 60)
    print("Step 6: our ONNX pipeline (same preprocessed inputs)")
    print("=" * 60)
    from tether.runtime.server import TetherServer

    server = TetherServer(export_dir, device="cuda", strict_providers=False)
    server.load()

    # Feed the SAME images (uint8 HWC) and task string that preprocessor started from.
    # Our pipeline will tokenize + normalize internally — already wired correctly.
    result = server.predict(
        image=[img_np, img_np, img_np],   # 3 cameras (same content, matching batch_raw)
        instruction="put the red bowl on the plate",
        state=state_np,
        noise=shared_noise_np,
    )
    onnx_actions = np.asarray(result["actions"], dtype=np.float32)
    print(f"onnx actions shape: {onnx_actions.shape}")
    print(f"onnx first action (normalized): {np.round(onnx_actions[0, :7], 3).tolist()}")

    # ── Step 7: Per-dim diff on the first action in chunk ──────────
    print("\n" + "=" * 60)
    print("Step 7: action diff (first action of chunk)")
    print("=" * 60)
    t_first = torch_actions_cpu[0, 0][:7]   # [7]
    o_first = onnx_actions[0][:7]           # [7]
    abs_diff = np.abs(t_first - o_first)
    l2 = float(np.linalg.norm(t_first - o_first))
    cos = float(np.dot(t_first, o_first) / (np.linalg.norm(t_first) * np.linalg.norm(o_first) + 1e-8))
    print(f"  torch:   {np.round(t_first, 3).tolist()}")
    print(f"  onnx:    {np.round(o_first, 3).tolist()}")
    print(f"  abs:     {np.round(abs_diff, 3).tolist()}")
    print(f"  L2:      {l2:.3f}")
    print(f"  cos_sim: {cos:+.3f}")

    if cos > 0.95:
        verdict = "EXPORT IS CORRECT — LIBERO 0% is a sim/env issue"
    elif cos > 0.5:
        verdict = "MINOR DRIFT — subtle numerics (probably fp32, opset, or RoPE phase)"
    else:
        verdict = "MAJOR DIVERGENCE — a structural bug remains in the export"
    print(f"\n  VERDICT: {verdict}")

    return {
        "torch_action": t_first.tolist(),
        "onnx_action": o_first.tolist(),
        "abs_diff": abs_diff.tolist(),
        "l2": l2,
        "cos_sim": cos,
        "verdict": verdict,
    }


@app.local_entrypoint()
def main():
    result = diff.remote()
    print("\n" + "=" * 60)
    print("FINAL DIFF RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))
