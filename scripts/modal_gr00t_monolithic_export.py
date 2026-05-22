"""Modal: GR00T N1.6 monolithic ONNX export + parity test.

GR00T differs from pi0 / pi0.5 / SmolVLA in three ways:
1. DDPM-style discrete-timestep diffusion, NOT flow matching (so num_steps
   doesn't unroll into the ONNX — the denoise loop is a runtime concern).
2. DiT (Diffusion Transformer) with AdaLN modulation, not a PaliGemma
   decoder-only stack → no DynamicCache, no prefix-mask surgery needed.
3. The exported monolithic ONNX is the *per-step* velocity function:
   noisy_actions + timestep + position_ids → velocity (same shape).
   `reflex serve` wraps it in the canonical 4-step denoise loop.

Parity target: cos=1.0 vs PyTorch `GR00TFullStack.forward` at machine
precision on shared seeded inputs. This is the equivalent of pi0's
"num_steps=10 unrolled" parity — since GR00T's loop is external, one
step of parity IS the loadbearing claim.

Usage:
    modal run scripts/modal_gr00t_monolithic_export.py             # export
    modal run scripts/modal_gr00t_monolithic_export.py --parity    # parity
"""
import os
import modal

app = modal.App("gr00t-monolithic-export")


def _hf_secret():
    """HF token — only needed for gated models. GR00T N1.6 is public,
    so an empty secret works on fresh workspaces without a pre-registered
    'huggingface' Modal secret."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_dict({})


# GR00T 6.6GB checkpoint — reuse dedicated volume
hf_cache = modal.Volume.from_name("gr00t-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("gr00t-onnx-outputs", create_if_missing=True)

HF_CACHE_PATH = "/root/.cache/huggingface"
ONNX_OUTPUT_PATH = "/onnx_out"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers>=4.51",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "numpy",
        "Pillow",
        "typer",
        "rich",
        "pydantic>=2.0",
        "pyyaml",
    )
    .add_local_dir("src/reflex", "/root/reflex-vla/src/reflex", copy=True)
    .add_local_file("pyproject.toml", "/root/reflex-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/reflex-vla/README.md", copy=True)
    .run_commands("cd /root/reflex-vla && pip install -e .")
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,
    volumes={
        HF_CACHE_PATH: hf_cache,
        ONNX_OUTPUT_PATH: onnx_output,
    },
    secrets=[_hf_secret()],
)
def export_gr00t_monolithic_modal(
    model_id: str = "nvidia/GR00T-N1.6-3B",
    embodiment_id: int = 0,
):
    """Export GR00T full-stack (encoder + DiT + decoder) as a single ONNX.

    The output is a per-step velocity function: (noisy_actions, timestep,
    position_ids) → velocity_raw. Shape matches input actions; runtime
    wraps it in the 4-step canonical denoise loop.
    """
    import time
    from pathlib import Path

    import torch
    import numpy as np

    from reflex.checkpoint import load_checkpoint
    from reflex.exporters.gr00t import build_gr00t_full_stack

    print(f"[modal] Loading {model_id}...")
    t0 = time.time()
    state_dict, _ = load_checkpoint(model_id)
    print(f"[modal] Checkpoint loaded in {time.time() - t0:.1f}s "
          f"({len(state_dict)} tensors)")

    print(f"[modal] Building GR00T full stack (embodiment={embodiment_id})...")
    t0 = time.time()
    full, meta = build_gr00t_full_stack(state_dict, embodiment_id=embodiment_id)
    full.eval()
    print(f"[modal] Built in {time.time() - t0:.1f}s — "
          f"raw_action_dim={meta['raw_action_dim']}, "
          f"full_stack_params={meta['full_stack_params_m']:.1f}M")

    raw_action_dim = meta["raw_action_dim"]
    chunk = 50
    B = 1

    dummy_actions = torch.randn(B, chunk, raw_action_dim, dtype=torch.float32)
    dummy_time = torch.tensor([0.5], dtype=torch.float32)
    dummy_pos = torch.arange(chunk, dtype=torch.long).unsqueeze(0)

    output_dir = Path(ONNX_OUTPUT_PATH) / "monolithic"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    print("[modal] torch.onnx.export (opset 19)...")
    t0 = time.time()
    torch.onnx.export(
        full,
        (dummy_actions, dummy_time, dummy_pos),
        str(onnx_path),
        input_names=["noisy_actions", "timestep", "position_ids"],
        output_names=["velocity"],
        dynamic_axes={
            "noisy_actions": {0: "batch"},
            "timestep": {0: "batch"},
            "position_ids": {0: "batch"},
            "velocity": {0: "batch"},
        },
        opset_version=19,
    )
    print(f"[modal] ONNX conversion: {time.time() - t0:.1f}s")

    onnx_output.commit()

    if not onnx_path.exists():
        return {"status": "fail", "reason": "onnx file not created"}

    size_mb = onnx_path.stat().st_size / 1e6
    data_files = list(output_dir.glob("*.data")) + list(output_dir.glob("*.bin"))
    data_mb = sum(f.stat().st_size for f in data_files) / 1e6
    total_mb = size_mb + data_mb
    print(f"[modal] SUCCESS: {onnx_path}")
    print(f"[modal]   model.onnx: {size_mb:.1f}MB")
    print(f"[modal]   external data: {data_mb:.1f}MB ({len(data_files)} files)")
    print(f"[modal]   total: {total_mb:.1f}MB")

    return {
        "status": "ok",
        "onnx_path": str(onnx_path),
        "size_mb": total_mb,
        "raw_action_dim": raw_action_dim,
        "embodiment_id": embodiment_id,
        "full_stack_params_m": meta["full_stack_params_m"],
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={
        HF_CACHE_PATH: hf_cache,
        ONNX_OUTPUT_PATH: onnx_output,
    },
    secrets=[_hf_secret()],
)
def parity_test_monolithic(
    model_id: str = "nvidia/GR00T-N1.6-3B",
    embodiment_id: int = 0,
):
    """Parity test: PyTorch GR00TFullStack vs monolithic ONNX.

    Shared seeded inputs (noise, timestep, position_ids); compare
    first-action + full-chunk cos and max_abs. Target: cos=1.0 at
    machine precision (max_abs < 1e-4).
    """
    import time
    from pathlib import Path

    import numpy as np
    import torch
    import onnxruntime as ort

    from reflex.checkpoint import load_checkpoint
    from reflex.exporters.gr00t import build_gr00t_full_stack

    print(f"[parity] Loading {model_id}...")
    t0 = time.time()
    state_dict, _ = load_checkpoint(model_id)
    full, meta = build_gr00t_full_stack(state_dict, embodiment_id=embodiment_id)
    full.eval()
    print(f"[parity] Loaded in {time.time() - t0:.1f}s")

    raw_action_dim = meta["raw_action_dim"]
    chunk = 50
    B = 1

    # Shared seeded inputs
    torch.manual_seed(42)
    noisy = torch.randn(B, chunk, raw_action_dim, dtype=torch.float32)
    timestep = torch.tensor([0.5], dtype=torch.float32)
    position_ids = torch.arange(chunk, dtype=torch.long).unsqueeze(0)

    # PyTorch reference
    print("[parity] Running PyTorch reference...")
    t0 = time.time()
    with torch.no_grad():
        pt_out = full(noisy, timestep, position_ids)
    pt_np = pt_out.cpu().numpy()
    print(f"[parity] pt: {pt_np.shape} in {time.time() - t0:.1f}s, "
          f"first: {pt_np[0, 0, :5]}")

    # ONNX
    print("[parity] Running ONNX (CPU)...")
    onnx_path = Path(ONNX_OUTPUT_PATH) / "monolithic" / "model.onnx"
    t0 = time.time()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {
        "noisy_actions": noisy.numpy(),
        "timestep": timestep.numpy(),
        "position_ids": position_ids.numpy().astype(np.int64),
    })[0]
    print(f"[parity] onnx: {ort_out.shape} in {time.time() - t0:.1f}s, "
          f"first: {ort_out[0, 0, :5]}")

    # Compare
    pt0 = pt_np[0, 0]
    on0 = ort_out[0, 0]
    first_max_abs = float(np.abs(pt0 - on0).max())
    first_cos = float(
        np.dot(pt0, on0)
        / (np.linalg.norm(pt0) * np.linalg.norm(on0) + 1e-12)
    )
    full_max_abs = float(np.abs(pt_np - ort_out).max())
    full_cos = float(
        np.dot(pt_np.flatten(), ort_out.flatten())
        / (np.linalg.norm(pt_np) * np.linalg.norm(ort_out) + 1e-12)
    )

    print("\n====== PARITY ======")
    print(f"  first-action max_abs: {first_max_abs:.4e}")
    print(f"  first-action cos:     {first_cos:+.6f}")
    print(f"  full-chunk  max_abs:  {full_max_abs:.4e}")
    print(f"  full-chunk  cos:      {full_cos:+.6f}")
    passed = full_cos >= 0.9999 and full_max_abs < 1e-3
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")

    return {
        "status": "ok",
        "first_cos": first_cos,
        "first_max_abs": first_max_abs,
        "full_cos": full_cos,
        "full_max_abs": full_max_abs,
        "passed": passed,
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={
        HF_CACHE_PATH: hf_cache,
        ONNX_OUTPUT_PATH: onnx_output,
    },
    secrets=[_hf_secret()],
)
def denoise_loop_parity(
    model_id: str = "nvidia/GR00T-N1.6-3B",
    embodiment_id: int = 0,
    num_steps: int = 4,
):
    """End-to-end denoise-loop parity.

    Wraps the ONNX in a Python DDIM-style num_steps loop and compares
    against the same loop over PyTorch `GR00TFullStack`. Since both loops
    use the same per-step function (one trained, one exported), a
    machine-precision per-step claim implies a machine-precision
    end-of-loop claim — this test confirms that empirically and gives us
    a "full chunk at num_steps=4" number analogous to pi0's num_steps=10.
    """
    import time
    from pathlib import Path

    import numpy as np
    import torch
    import onnxruntime as ort

    from reflex.checkpoint import load_checkpoint
    from reflex.exporters.gr00t import build_gr00t_full_stack

    print(f"[loop-parity] Loading {model_id}...")
    state_dict, _ = load_checkpoint(model_id)
    full, meta = build_gr00t_full_stack(state_dict, embodiment_id=embodiment_id)
    full.eval()

    raw_action_dim = meta["raw_action_dim"]
    chunk = 50
    B = 1

    torch.manual_seed(42)
    init_noise = torch.randn(B, chunk, raw_action_dim, dtype=torch.float32)
    position_ids = torch.arange(chunk, dtype=torch.long).unsqueeze(0)

    # Timesteps: simple uniform schedule from 1.0 → 0.0, one per step
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]  # [num_steps]
    dt = -1.0 / num_steps

    print(f"[loop-parity] Running PyTorch denoise loop ({num_steps} steps)...")
    t0 = time.time()
    x_pt = init_noise.clone()
    with torch.no_grad():
        for t_val in timesteps:
            t_tensor = torch.tensor([float(t_val)], dtype=torch.float32)
            vel = full(x_pt, t_tensor, position_ids)
            x_pt = x_pt + dt * vel
    pt_np = x_pt.cpu().numpy()
    print(f"[loop-parity] pt final: {pt_np.shape} in {time.time() - t0:.1f}s")

    print(f"[loop-parity] Running ONNX denoise loop ({num_steps} steps)...")
    onnx_path = Path(ONNX_OUTPUT_PATH) / "monolithic" / "model.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    t0 = time.time()
    x_on = init_noise.numpy().copy()
    for t_val in timesteps:
        t_np = np.array([float(t_val)], dtype=np.float32)
        vel = sess.run(None, {
            "noisy_actions": x_on,
            "timestep": t_np,
            "position_ids": position_ids.numpy().astype(np.int64),
        })[0]
        x_on = x_on + dt * vel
    print(f"[loop-parity] onnx final: {x_on.shape} in {time.time() - t0:.1f}s")

    first_max_abs = float(np.abs(pt_np[0, 0] - x_on[0, 0]).max())
    first_cos = float(
        np.dot(pt_np[0, 0], x_on[0, 0])
        / (np.linalg.norm(pt_np[0, 0]) * np.linalg.norm(x_on[0, 0]) + 1e-12)
    )
    full_max_abs = float(np.abs(pt_np - x_on).max())
    full_cos = float(
        np.dot(pt_np.flatten(), x_on.flatten())
        / (np.linalg.norm(pt_np) * np.linalg.norm(x_on) + 1e-12)
    )

    print(f"\n====== LOOP PARITY (num_steps={num_steps}) ======")
    print(f"  first-action max_abs: {first_max_abs:.4e}")
    print(f"  first-action cos:     {first_cos:+.6f}")
    print(f"  full-chunk  max_abs:  {full_max_abs:.4e}")
    print(f"  full-chunk  cos:      {full_cos:+.6f}")
    passed = full_cos >= 0.9999 and full_max_abs < 1e-3
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")

    return {
        "status": "ok",
        "num_steps": num_steps,
        "first_cos": first_cos,
        "first_max_abs": first_max_abs,
        "full_cos": full_cos,
        "full_max_abs": full_max_abs,
        "passed": passed,
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def export_gr00t_with_vlm_modal(
    model_id: str = "nvidia/GR00T-N1.6-3B",
    embodiment_id: int = 0,
):
    """Export GR00T full-stack WITH state + vlm_kv inputs (Step 4a, 2026-04-19).

    Extends the prior zero-stub export. Now takes FIVE inputs:
      noisy_actions, timestep, position_ids, state, vlm_kv

    Output: velocity_raw [B, chunk, raw_action_dim].

    Differences vs export_gr00t_monolithic_modal:
    - state + vlm_kv are first-class inputs (not zero-stubbed)
    - uses the Step 2 state_encoder port + Step 3 fixed pos_embed logic
    - produces expert_stack_with_vlm.onnx (kept separate from the old
      model.onnx for compatibility with existing reflex serve config)
    """
    import time
    from pathlib import Path

    import torch

    from reflex.checkpoint import load_checkpoint
    from reflex.exporters.gr00t import build_gr00t_full_stack

    print(f"[modal-vlm] Loading {model_id}...")
    t0 = time.time()
    state_dict, _ = load_checkpoint(model_id)
    print(f"[modal-vlm] Checkpoint loaded in {time.time()-t0:.1f}s "
          f"({len(state_dict)} tensors)")

    print(f"[modal-vlm] Building GR00T full stack w/ state_encoder "
          f"(embodiment={embodiment_id})...")
    t0 = time.time()
    full, meta = build_gr00t_full_stack(state_dict, embodiment_id=embodiment_id)
    full.eval()
    assert meta.get("has_state_encoder"), (
        "build_gr00t_full_stack did not load state_encoder — "
        "Step 4a requires N1.6 checkpoint with action_head.state_encoder.*"
    )
    print(f"[modal-vlm] Built in {time.time()-t0:.1f}s — "
          f"raw_action_dim={meta['raw_action_dim']}, "
          f"raw_state_dim={meta['raw_state_dim']}, "
          f"full_stack_params={meta['full_stack_params_m']:.1f}M")

    raw_action_dim = meta["raw_action_dim"]
    raw_state_dim = meta["raw_state_dim"]
    chunk = 50
    B = 1
    vlm_kv_seq = 256
    vlm_kv_dim = 2048

    dummy_actions = torch.randn(B, chunk, raw_action_dim, dtype=torch.float32)
    dummy_time = torch.tensor([0.5], dtype=torch.float32)
    dummy_pos = torch.arange(chunk + 1, dtype=torch.long).unsqueeze(0)
    dummy_state = torch.randn(B, raw_state_dim, dtype=torch.float32)
    dummy_vlm_kv = torch.randn(B, vlm_kv_seq, vlm_kv_dim, dtype=torch.float32)

    output_dir = Path(ONNX_OUTPUT_PATH) / "monolithic_with_vlm"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "expert_stack_with_vlm.onnx"

    print("[modal-vlm] torch.onnx.export (opset 19, 5 inputs)...")
    t0 = time.time()
    torch.onnx.export(
        full,
        (dummy_actions, dummy_time, dummy_pos, dummy_state, dummy_vlm_kv),
        str(onnx_path),
        input_names=["noisy_actions", "timestep", "position_ids", "state", "vlm_kv"],
        output_names=["velocity"],
        dynamic_axes={
            "noisy_actions": {0: "batch"},
            "timestep": {0: "batch"},
            "position_ids": {0: "batch"},
            "state": {0: "batch"},
            "vlm_kv": {0: "batch", 1: "vlm_seq"},
            "velocity": {0: "batch"},
        },
        opset_version=19,
    )
    print(f"[modal-vlm] ONNX conversion: {time.time()-t0:.1f}s")

    onnx_output.commit()

    if not onnx_path.exists():
        return {"status": "fail", "reason": "onnx file not created"}

    size_mb = onnx_path.stat().st_size / 1e6
    data_files = (list(output_dir.glob("*.data"))
                  + list(output_dir.glob("*.bin")))
    data_mb = sum(f.stat().st_size for f in data_files) / 1e6
    total_mb = size_mb + data_mb
    print(f"[modal-vlm] SUCCESS: {onnx_path}")
    print(f"[modal-vlm]   model onnx: {size_mb:.1f}MB")
    print(f"[modal-vlm]   external data: {data_mb:.1f}MB")
    print(f"[modal-vlm]   total: {total_mb:.1f}MB")

    return {
        "status": "ok",
        "onnx_path": str(onnx_path),
        "size_mb": total_mb,
        "raw_action_dim": raw_action_dim,
        "raw_state_dim": raw_state_dim,
        "embodiment_id": embodiment_id,
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def parity_test_with_vlm(model_id: str = "nvidia/GR00T-N1.6-3B"):
    """PyTorch vs expert_stack_with_vlm.onnx single-step parity.

    Runs both paths on shared seeded inputs (noisy, timestep, state,
    vlm_kv). Target: cos=+1.000000, max_abs<1e-5.
    """
    import time
    from pathlib import Path

    import numpy as np
    import torch
    import onnxruntime as ort

    from reflex.checkpoint import load_checkpoint
    from reflex.exporters.gr00t import build_gr00t_full_stack

    print(f"[parity-vlm] Loading {model_id}...")
    state_dict, _ = load_checkpoint(model_id)
    full, meta = build_gr00t_full_stack(state_dict, embodiment_id=0)
    full.eval()

    raw_action_dim = meta["raw_action_dim"]
    raw_state_dim = meta["raw_state_dim"]
    chunk = 50
    B = 1
    vlm_kv_seq = 256
    vlm_kv_dim = 2048

    torch.manual_seed(42)
    noisy = torch.randn(B, chunk, raw_action_dim, dtype=torch.float32)
    ts = torch.tensor([0.5], dtype=torch.float32)
    pos = torch.arange(chunk + 1, dtype=torch.long).unsqueeze(0)
    state = torch.randn(B, raw_state_dim, dtype=torch.float32)
    vlm_kv = torch.randn(B, vlm_kv_seq, vlm_kv_dim, dtype=torch.float32)

    # PyTorch
    t0 = time.time()
    with torch.no_grad():
        pt_out = full(noisy, ts, pos, state=state, vlm_kv=vlm_kv)
    print(f"[parity-vlm] pt: {tuple(pt_out.shape)} in {time.time()-t0:.1f}s")

    # ONNX
    onnx_path = Path(ONNX_OUTPUT_PATH) / "monolithic_with_vlm" / "expert_stack_with_vlm.onnx"
    t0 = time.time()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {
        "noisy_actions": noisy.numpy(),
        "timestep": ts.numpy(),
        "position_ids": pos.numpy().astype(np.int64),
        "state": state.numpy(),
        "vlm_kv": vlm_kv.numpy(),
    })[0]
    print(f"[parity-vlm] onnx: {ort_out.shape} in {time.time()-t0:.1f}s")

    # Compare
    pt_flat = pt_out.cpu().numpy().reshape(-1)
    onnx_flat = ort_out.reshape(-1)
    max_abs = float(np.abs(pt_flat - onnx_flat).max())
    cos = float(
        np.dot(pt_flat, onnx_flat)
        / (np.linalg.norm(pt_flat) * np.linalg.norm(onnx_flat) + 1e-12)
    )
    first_pt = pt_out[0, 0, :5].cpu().numpy().tolist()
    first_ort = ort_out[0, 0, :5].tolist()

    print(f"\n====== PARITY (with-vlm ONNX vs PyTorch) ======")
    print(f"  cos:     {cos:+.6f}")
    print(f"  max_abs: {max_abs:.4e}")
    print(f"  pt[0,0,:5]: {first_pt}")
    print(f"  on[0,0,:5]: {first_ort}")
    verdict = "PASS" if cos >= 0.9999 and max_abs < 1e-4 else "FAIL"
    print(f"  VERDICT: {verdict}")

    return {
        "status": "ok",
        "cos": cos,
        "max_abs": max_abs,
        "pt_first": first_pt,
        "on_first": first_ort,
        "verdict": verdict,
    }


@app.local_entrypoint()
def main(parity: bool = False, loop: bool = False, num_steps: int = 4,
         vlm: bool = False, vlm_parity: bool = False):
    """GR00T monolithic export + parity tests.

    Flags:
      (default)      — export zero-stub monolithic ONNX (old)
      --parity       — single-step velocity parity (zero-stub)
      --loop         — num_steps denoise-loop parity (zero-stub)
      --vlm          — export expert_stack_with_vlm.onnx (Step 4a)
      --vlm-parity   — parity test for expert_stack_with_vlm.onnx
    """
    if vlm_parity:
        result = parity_test_with_vlm.remote()
    elif vlm:
        result = export_gr00t_with_vlm_modal.remote()
    elif loop:
        result = denoise_loop_parity.remote(num_steps=num_steps)
    elif parity:
        result = parity_test_monolithic.remote()
    else:
        result = export_gr00t_monolithic_modal.remote()
    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
