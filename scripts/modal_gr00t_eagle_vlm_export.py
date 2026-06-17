"""Step 4b — Eagle VLM ONNX export for GR00T N1.6.

Builds `EagleExportStack` from nvidia/GR00T-N1.6-3B's `backbone.model.*`
state_dict slice, exports to `eagle_vlm.onnx` (~2-3 GB). The ONNX
produces `[B, seq, 2048]` hidden states that chain into
`expert_stack_with_vlm.onnx`'s `vlm_kv` input.

This is harder than Step 4a because Qwen2's decoder has a causal mask +
potential DynamicCache touchpoints that torch.export may stumble on.
Since our forward is single-shot (no loop, use_cache=False), we expect
the simplest path to work; if not, we port the 3-patch stack from
pi0/pi0.5.

Usage:
    modal run scripts/modal_gr00t_eagle_vlm_export.py             # quick PyTorch smoke test
    modal run scripts/modal_gr00t_eagle_vlm_export.py --export    # export ONNX
    modal run scripts/modal_gr00t_eagle_vlm_export.py --parity    # ONNX vs PyTorch parity
"""
import os
import subprocess
import modal

app = modal.App("tether-gr00t-eagle-vlm")


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


hf_cache = modal.Volume.from_name("gr00t-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("gr00t-onnx-outputs", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
ONNX_OUTPUT_PATH = "/onnx_out"


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
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def smoke_test(model_id: str = "nvidia/GR00T-N1.6-3B"):
    """PyTorch-only smoke test: build + forward + report shapes.

    Answers whether EagleExportStack instantiates and runs at all,
    BEFORE we attempt the ONNX export. Cheap sanity check.
    """
    import time
    import numpy as np
    import torch

    from tether.checkpoint import load_checkpoint
    from tether.exporters.eagle_export_stack import build_eagle_export_stack

    print(f"[smoke] Loading {model_id}...")
    t0 = time.time()
    state_dict, _ = load_checkpoint(model_id)
    print(f"[smoke] {len(state_dict)} tensors in {time.time()-t0:.1f}s")

    print(f"[smoke] Building EagleExportStack...")
    t0 = time.time()
    stack, meta = build_eagle_export_stack(state_dict)
    stack = stack.to("cuda")
    print(f"[smoke] Built in {time.time()-t0:.1f}s")
    for k, v in meta.items():
        print(f"         {k}: {v}")

    # Dummy inputs — use the image_size + token count derived from the
    # checkpoint (N1.6 SigLIP was trained at 224×224 per position_embedding).
    B = 1
    # Can't pull from `meta` directly inside modal container — re-derive
    # from state_dict or use published defaults. N1.6 uses 224×224.
    import math as _math
    pos_embed_key = (
        "backbone.model.vision_model.vision_model.embeddings.position_embedding.weight"
    )
    num_positions = state_dict[pos_embed_key].shape[0]
    patches_per_side = int(_math.sqrt(num_positions))
    patch_size = 14
    H = W = patches_per_side * patch_size  # 16 * 14 = 224
    # Pixel shuffle scale_factor=0.5 concatenates 2×2 patches into 1 token.
    n_image_tokens = num_positions // 4
    seq = n_image_tokens + 16  # image tokens + short text prompt
    vocab = meta["vocab_size"]
    img_tok = meta["image_token_index"]

    torch.manual_seed(42)
    pixel_values = torch.randn(B, 3, H, W, device="cuda", dtype=torch.float32)
    # Remaining slots: text tokens. Use token id 0 as a trivial filler.
    assert seq > n_image_tokens, f"seq={seq} must exceed image_tokens={n_image_tokens}"
    input_ids = torch.zeros(B, seq, dtype=torch.long, device="cuda")
    input_ids[:, :n_image_tokens] = img_tok
    # Give remaining positions random text tokens < vocab
    input_ids[:, n_image_tokens:] = torch.randint(
        low=1, high=min(32000, vocab), size=(B, seq - n_image_tokens),
        device="cuda",
    )
    attention_mask = torch.ones(B, seq, dtype=torch.long, device="cuda")
    image_flags = torch.tensor([1], device="cuda", dtype=torch.long)

    print(f"[smoke] Dummy inputs: pixel_values={tuple(pixel_values.shape)}, "
          f"input_ids={tuple(input_ids.shape)} (image_tokens={n_image_tokens})")

    print(f"[smoke] Forward...")
    t0 = time.time()
    with torch.no_grad():
        hidden = stack(pixel_values, input_ids, attention_mask, image_flags)
    print(f"[smoke] Forward done in {time.time()-t0:.2f}s")
    print(f"[smoke] output shape: {tuple(hidden.shape)}")
    print(f"[smoke]   mean={hidden.mean().item():+.4f}  std={hidden.std().item():.4f}")

    return {
        "status": "ok",
        "output_shape": list(hidden.shape),
        "mean": float(hidden.mean()),
        "std": float(hidden.std()),
        **{k: v for k, v in meta.items() if isinstance(v, (int, float, str))},
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def export(model_id: str = "nvidia/GR00T-N1.6-3B"):
    """Export EagleExportStack to eagle_vlm.onnx."""
    import time
    from pathlib import Path

    import torch

    from tether.checkpoint import load_checkpoint
    from tether.exporters.eagle_export_stack import build_eagle_export_stack

    print(f"[export] Loading {model_id}...")
    t0 = time.time()
    state_dict, _ = load_checkpoint(model_id)
    stack, meta = build_eagle_export_stack(state_dict)
    stack = stack.to("cuda")
    print(f"[export] Built in {time.time()-t0:.1f}s  params={meta['total_params_m']:.1f}M")

    # Derive image_size + image-token count from checkpoint (N1.6 uses 224×224).
    import math as _math
    pos_embed_key = (
        "backbone.model.vision_model.vision_model.embeddings.position_embedding.weight"
    )
    num_positions = state_dict[pos_embed_key].shape[0]
    patches_per_side = int(_math.sqrt(num_positions))
    patch_size = 14
    B = 1
    H = W = patches_per_side * patch_size
    n_image_tokens = num_positions // 4  # pixel_shuffle 0.5^2
    seq = n_image_tokens + 16
    img_tok = meta["image_token_index"]

    torch.manual_seed(42)
    pixel_values = torch.randn(B, 3, H, W, device="cuda", dtype=torch.float32)
    input_ids = torch.zeros(B, seq, dtype=torch.long, device="cuda")
    input_ids[:, :n_image_tokens] = img_tok
    input_ids[:, n_image_tokens:] = torch.randint(
        low=1, high=32000, size=(B, seq - n_image_tokens), device="cuda",
    )
    attention_mask = torch.ones(B, seq, dtype=torch.long, device="cuda")
    image_flags = torch.tensor([1], device="cuda", dtype=torch.long)

    output_dir = Path(ONNX_OUTPUT_PATH) / "eagle_vlm"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "eagle_vlm.onnx"

    print(f"[export] torch.onnx.export (opset 19)...")
    t0 = time.time()
    try:
        torch.onnx.export(
            stack,
            (pixel_values, input_ids, attention_mask, image_flags),
            str(onnx_path),
            input_names=["pixel_values", "input_ids", "attention_mask", "image_flags"],
            output_names=["hidden_states"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "image_flags": {0: "batch"},
                "hidden_states": {0: "batch", 1: "seq"},
            },
            opset_version=19,
        )
    except Exception as e:
        import traceback
        print(f"[export] FAILED: {type(e).__name__}: {e}")
        print(traceback.format_exc()[-2000:])
        return {"status": "fail", "reason": str(e)[:500]}
    print(f"[export] ONNX conversion: {time.time()-t0:.1f}s")

    onnx_output.commit()

    if not onnx_path.exists():
        return {"status": "fail", "reason": "onnx file not created"}

    size_mb = onnx_path.stat().st_size / 1e6
    data_files = (list(output_dir.glob("*.data")) + list(output_dir.glob("*.bin")))
    data_mb = sum(f.stat().st_size for f in data_files) / 1e6
    total_mb = size_mb + data_mb
    print(f"[export] SUCCESS: {onnx_path}")
    print(f"[export]   model.onnx: {size_mb:.1f}MB")
    print(f"[export]   external data: {data_mb:.1f}MB ({len(data_files)} files)")
    print(f"[export]   total: {total_mb:.1f}MB")

    return {
        "status": "ok",
        "onnx_path": str(onnx_path),
        "size_mb": total_mb,
        "params_m": meta["total_params_m"],
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def parity_test(model_id: str = "nvidia/GR00T-N1.6-3B"):
    """Verify eagle_vlm.onnx output matches PyTorch EagleExportStack."""
    import math as _math
    import time
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort
    import torch

    from tether.checkpoint import load_checkpoint
    from tether.exporters.eagle_export_stack import build_eagle_export_stack

    onnx_path = Path(ONNX_OUTPUT_PATH) / "eagle_vlm" / "eagle_vlm.onnx"
    if not onnx_path.exists():
        return {"status": "fail", "reason": f"{onnx_path} not found — run --export first"}

    print(f"[parity] Loading {model_id}...")
    state_dict, _ = load_checkpoint(model_id)
    stack, meta = build_eagle_export_stack(state_dict)
    stack = stack.to("cuda").eval()
    print(f"[parity] PyTorch built: {meta['total_params_m']:.1f}M params")

    pos_embed_key = (
        "backbone.model.vision_model.vision_model.embeddings.position_embedding.weight"
    )
    num_positions = state_dict[pos_embed_key].shape[0]
    patches_per_side = int(_math.sqrt(num_positions))
    patch_size = 14
    B = 1
    H = W = patches_per_side * patch_size
    n_image_tokens = num_positions // 4
    seq = n_image_tokens + 16
    img_tok = meta["image_token_index"]
    vocab = meta["vocab_size"]

    torch.manual_seed(1337)
    pixel_values = torch.randn(B, 3, H, W, device="cuda", dtype=torch.float32)
    input_ids = torch.zeros(B, seq, dtype=torch.long, device="cuda")
    input_ids[:, :n_image_tokens] = img_tok
    input_ids[:, n_image_tokens:] = torch.randint(
        low=1, high=min(32000, vocab), size=(B, seq - n_image_tokens), device="cuda",
    )
    attention_mask = torch.ones(B, seq, dtype=torch.long, device="cuda")
    image_flags = torch.tensor([1], device="cuda", dtype=torch.long)

    print(f"[parity] PyTorch forward...")
    t0 = time.time()
    with torch.no_grad():
        pt_out = stack(pixel_values, input_ids, attention_mask, image_flags)
    pt_out_np = pt_out.detach().cpu().numpy().astype(np.float32)
    print(f"[parity] PyTorch: shape={pt_out_np.shape}  "
          f"mean={pt_out_np.mean():+.4f}  std={pt_out_np.std():.4f}  "
          f"[{time.time()-t0:.2f}s]")

    del stack
    torch.cuda.empty_cache()

    print(f"[parity] Loading ONNX (ORT CUDA)...")
    t0 = time.time()
    providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f"[parity] ORT loaded in {time.time()-t0:.1f}s")
    print(f"[parity] ONNX forward...")
    t0 = time.time()
    onnx_out = sess.run(
        ["hidden_states"],
        {
            "pixel_values": pixel_values.cpu().numpy().astype(np.float32),
            "input_ids": input_ids.cpu().numpy().astype(np.int64),
            "attention_mask": attention_mask.cpu().numpy().astype(np.int64),
            "image_flags": image_flags.cpu().numpy().astype(np.int64),
        },
    )[0].astype(np.float32)
    print(f"[parity] ONNX: shape={onnx_out.shape}  "
          f"mean={onnx_out.mean():+.4f}  std={onnx_out.std():.4f}  "
          f"[{time.time()-t0:.2f}s]")

    diff = np.abs(pt_out_np - onnx_out)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    a = pt_out_np.flatten().astype(np.float64)
    b = onnx_out.flatten().astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"[parity] max_abs={max_abs:.6e}  mean_abs={mean_abs:.6e}  cos={cos:+.6f}")

    verdict = "PASS" if (cos > 0.9999 and max_abs < 5e-3) else "FAIL"
    print(f"[parity] {verdict}")

    return {
        "status": "ok",
        "verdict": verdict,
        "cos_sim": cos,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "output_shape": list(pt_out_np.shape),
    }


@app.local_entrypoint()
def main(export_onnx: bool = False, parity: bool = False):
    """
      (default)   — PyTorch smoke test (build + forward + shapes)
      --export    — torch.onnx.export to eagle_vlm.onnx
      --parity    — ONNX vs PyTorch parity
    """
    if export_onnx:
        r = export.remote()
    elif parity:
        r = parity_test.remote()
    else:
        r = smoke_test.remote()
    print(f"\n=== RESULT ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
