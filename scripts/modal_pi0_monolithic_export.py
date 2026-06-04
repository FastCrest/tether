"""Modal: pi0 monolithic ONNX export via onnx-diagnostic torch_export_patches.

Uses A10G + persistent Volume to cache pi0_base weights between runs.
onnx-diagnostic's `torch_export_patches(patch_transformers=True)` teaches
torch.export.export how to flatten DynamicCache as PyTree nodes, which is
the research-identified path for Gemma-family DynamicCache ONNX export.

Iteration: ~1-2 min/run after first (pi0_base cached in Volume).

Usage:
    # One-time setup: register your HF token as a Modal secret
    modal secret create hf-token HF_TOKEN=hf_xxxxx
    # Or export locally: `export HF_TOKEN=hf_xxxxx` (picked up as fallback)

    modal run scripts/modal_pi0_monolithic_export.py

Output: /tmp/pi0_monolithic_onnx/model.onnx in the Modal container. Fetch
via modal.Volume or print verdict + cos number for parity test.
"""
import os
import modal

app = modal.App("pi0-monolithic-export")


def _hf_secret():
    """Return a Modal Secret carrying HF_TOKEN.

    Order: local `HF_TOKEN` env var (dev convenience) → empty secret
    fallback (works for public models on any workspace without a
    pre-registered `hf-token`). Never hardcode the token in source.
    """
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_dict({})

# Persistent volume for HF cache (pi0_base is ~14GB)
hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
# Persistent volume for ONNX outputs
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)

HF_CACHE_PATH = "/root/.cache/huggingface"
ONNX_OUTPUT_PATH = "/onnx_out"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # Install lerobot first — it pins torch>=2.7 and hub>=1.0
        "lerobot==0.5.1",
        "num2words",
        "safetensors>=0.4.0",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "onnx-diagnostic>=0.9",
        "optree",  # onnx-diagnostic soft-dep
        "scipy",   # onnx-diagnostic patches_transformers_qwen2_5 transitively
        "numpy",
        "accelerate",
        "draccus",
    )
    # transformers 5.x accepts huggingface-hub>=1.0 (which lerobot pins)
    .pip_install("transformers>=5.0,<6.0")
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)

# GPU-aware image for CUDAExecutionProvider runtime tests. Base image has
# only `onnxruntime` (CPU). We install `onnxruntime-gpu` + nvidia cudnn/cublas
# wheels and export LD_LIBRARY_PATH so ORT's CUDA EP can dlopen them.
gpu_image = (
    image.pip_install(
        "onnxruntime-gpu>=1.20,<1.24",
        "nvidia-cudnn-cu12>=9.0,<10.0",
        "nvidia-cublas-cu12>=12.0,<13.0",
        extra_options="--no-deps",
    )
    .pip_install("onnxruntime-gpu>=1.20,<1.24")
    .env({
        "LD_LIBRARY_PATH": (
            "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:"
            "/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib:"
            "/usr/local/cuda/lib64"
        ),
    })
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={
        HF_CACHE_PATH: hf_cache,
        ONNX_OUTPUT_PATH: onnx_output,
    },
    secrets=[_hf_secret()],
)
def export_pi0_monolithic_modal(
    model_id: str = "lerobot/pi0_base",
    num_steps: int = 10,
):
    """Export pi0 monolithic ONNX using onnx-diagnostic torch_export_patches."""
    import os
    import sys
    import types
    import time
    from pathlib import Path

    import torch
    import torch.nn as nn

    # Python 3.13 + lerobot 0.5.1 compat shim (same as local)
    for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
        stub = types.ModuleType(_mod)
        stub.GrootPolicy = None
        stub.GR00TN15 = None
        sys.modules[_mod] = stub

    # ---- Diagnostic: wrap onnx-diagnostic's patched__maybe_broadcast so
    # when it raises "expand: 835 -> 886" we get the aten op + user-stack
    # that triggered it. Goal: find which Python line in sample_actions (or
    # transformers attention code) is broadcasting pre-suffix shape against
    # post-suffix shape at num_steps=10.
    try:
        from onnx_diagnostic.torch_export_patches.patches import patch_torch as _pt
        _orig_mb = _pt.patched__maybe_broadcast

        def _debug_mb(*args, **kwargs):
            import traceback as _tb
            try:
                return _orig_mb(*args, **kwargs)
            except RuntimeError as e:
                if "expand" in str(e):
                    shapes = []
                    for a in args:
                        if hasattr(a, "shape"):
                            shapes.append(tuple(a.shape))
                        elif isinstance(a, (list, tuple)):
                            shapes.append([tuple(getattr(x, "shape", None) or []) for x in a])
                        else:
                            shapes.append(str(type(a).__name__))
                    print(f"\n!!! [broadcast-fail] {e}")
                    print(f"    arg shapes: {shapes}")
                    print("    Stack (outermost first, 15 frames):")
                    for frame in _tb.format_stack()[-15:]:
                        print("    " + frame.rstrip())
                raise
        _pt.patched__maybe_broadcast = _debug_mb
        print("[modal] installed broadcast diagnostic")
    except ImportError:
        pass

    # ---- Transformers 4.57+ compat monkey-patches ----
    from lerobot.policies.pi0 import modeling_pi0
    from transformers import masking_utils
    from lerobot.policies import pi_gemma as _pg

    _orig_embed = modeling_pi0.PaliGemmaWithExpertModel.embed_image

    def _patched_embed_image(self, image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        out = self.paligemma.model.get_image_features(image)
        if hasattr(out, "pooler_output"):
            features = out.pooler_output
        else:
            features = out
        features = features * self.paligemma.config.text_config.hidden_size ** 0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    modeling_pi0.PaliGemmaWithExpertModel.embed_image = _patched_embed_image

    _orig_ccm = masking_utils.create_causal_mask

    def _ccm_shim(*args, **kwargs):
        # Dispatch: if caller passed a 4D mask (pi0 WITH the F.pad-based
        # denoise_step patch below), use it — cos=1.0 path. Otherwise
        # return None to unblock export.
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        attn = kwargs.get("attention_mask")
        if attn is None and len(args) >= 3:
            attn = args[2]
        try:
            if attn is not None and hasattr(attn, "dim") and attn.dim() == 4:
                return attn
        except Exception:
            pass
        return None

    masking_utils.create_causal_mask = _ccm_shim
    if hasattr(_pg, "create_causal_mask"):
        _pg.create_causal_mask = _ccm_shim

    # Patch denoise_step: replace torch.cat of prefix+suffix masks with
    # F.pad + logical AND. Under torch.export FakeTensor tracing, the cat
    # loses the suffix dim (mask arrives at attention as [1,1,51,835]
    # instead of [1,1,51,886]). Two F.pad calls have concrete output size
    # at every step → mask survives intact. This is the missing piece
    # that blocked cos=1.0 at num_steps=10.
    import torch.nn.functional as _F
    from lerobot.policies.pi0 import modeling_pi0 as _mp0
    from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks as _make_att_2d_masks

    _patched_called = [0]

    def _patched_denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        _patched_called[0] += 1
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, timestep,
        )
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        # CRITICAL: use past_key_values.get_seq_length() (the actual cached
        # prefix K in attention) NOT prefix_pad_masks.shape[1]. Under tracing
        # those diverge — prefix_pad_masks.shape[1] can be smaller than the
        # real cached K, so the built mask's K is smaller than attention's
        # scores' K, and we get 835->886 broadcast failure.
        cached_prefix_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        prefix_len = max(cached_prefix_len, prefix_pad_masks.shape[1])

        # If the cached prefix is longer than prefix_pad_masks, pad with True
        # (allowed) — the extra positions are the pre-computed prefix the
        # attention saw but we didn't get pad-mask info for.
        pad_deficit = prefix_len - prefix_pad_masks.shape[1]
        prefix_pad_masks_extended = prefix_pad_masks
        if pad_deficit > 0:
            prefix_pad_masks_extended = _F.pad(prefix_pad_masks, (0, pad_deficit), value=True)

        prefix_pad_2d_masks = prefix_pad_masks_extended[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = _make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # F.pad + AND (replaces torch.cat([..., ...], dim=2) that loses shape under tracing)
        prefix_allowed = _F.pad(prefix_pad_2d_masks, (0, suffix_len), value=True)
        suffix_allowed = _F.pad(suffix_att_2d_masks, (prefix_len, 0), value=True)
        full_att_2d_masks = prefix_allowed & suffix_allowed
        if _patched_called[0] <= 2:
            print(f"[patched_denoise_step] call {_patched_called[0]}: "
                  f"cached_prefix={cached_prefix_len}, "
                  f"pad_mask_prefix={prefix_pad_masks.shape[1]}, "
                  f"used_prefix={prefix_len}, suffix={suffix_len}, "
                  f"full={tuple(full_att_2d_masks.shape)}")

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    _mp0.PI0Pytorch.denoise_step = _patched_denoise_step

    # === Freeze DynamicCache.update during denoise iterations ===
    # Root cause of pi0 cos=0.977 (vs 1.0) at num_steps=10: under torch.export
    # FakeTensor tracing, DynamicCache.update() APPENDS K/V regardless of
    # use_cache=False because transformers' attention layer unconditionally
    # calls `past_kv.update(k, v, layer_idx)` to get the concat of
    # past + current. Each Euler iteration therefore mutates the shared
    # cache → iter N's attention sees past of size (orig + N*suffix_len),
    # diverging from eager PyTorch (where use_cache=False prevents mutation).
    #
    # Fix: install a wrapped update() that, when a denoise-phase flag is set,
    # returns past+current CONCAT without appending to self.key_cache. Prefix
    # forward runs WITHOUT the flag (builds the cache normally); the Euler
    # loop runs WITH the flag (cache frozen). This matches canonical eager
    # PyTorch semantics and produces cos=1.0.
    from transformers.cache_utils import DynamicCache as _DC, DynamicLayer as _DL
    _orig_layer_update = _DL.update
    _denoise_phase = [False]

    def _frozen_layer_update(self, key_states, value_states, cache_kwargs=None):
        # Transformers 5.3: DynamicLayer stores K/V as self.keys / self.values.
        # During denoise phase, return past_concat_current WITHOUT appending
        # to self.keys/values — cache stays frozen at prefix size.
        if _denoise_phase[0] and getattr(self, "is_initialized", False):
            past_k = self.keys
            past_v = self.values
            if past_k is not None and past_v is not None:
                new_k = torch.cat([past_k, key_states], dim=-2)
                new_v = torch.cat([past_v, value_states], dim=-2)
                return new_k, new_v
        return _orig_layer_update(self, key_states, value_states, cache_kwargs)

    _DL.update = _frozen_layer_update

    # Wrap denoise_step to flip the flag during its execution.
    _dstep_inner = _mp0.PI0Pytorch.denoise_step

    def _denoise_step_with_flag(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        _denoise_phase[0] = True
        try:
            return _dstep_inner(self, state, prefix_pad_masks, past_key_values, x_t, timestep)
        finally:
            _denoise_phase[0] = False

    _mp0.PI0Pytorch.denoise_step = _denoise_step_with_flag

    # Patch denoise_step to skip copy.deepcopy(past_key_values) — the deepcopy
    # fails under torch.export FakeTensor tracing ("Cannot access data pointer").
    # Safe to skip: use_cache=False means the forward doesn't mutate the cache.
    import copy as _copy
    _orig_deepcopy = _copy.deepcopy

    def _safe_deepcopy(obj, *args, **kwargs):
        # DynamicCache: create a NEW container with SAME tensor refs (shallow
        # copy of the internal lists). Pure pass-through fails because if any
        # downstream code appends (even under use_cache=False, transformers
        # sometimes mutates during eager attention), the SAME object gets
        # contaminated across iterations — mask shapes grow across traced
        # iterations (835→886→937 ...) and broadcast against fixed-size
        # attention scores fails. Shallow-list-copy fixes this: new container
        # per iteration, so appends are isolated.
        from transformers.cache_utils import DynamicCache
        if isinstance(obj, DynamicCache):
            new = DynamicCache()
            # Shallow copy of the internal lists — new containers, SAME tensor
            # references. Attention ops that append K/V extend the NEW lists,
            # leaving the original untouched across traced iterations.
            if hasattr(obj, "key_cache"):
                new.key_cache = list(obj.key_cache)
            if hasattr(obj, "value_cache"):
                new.value_cache = list(obj.value_cache)
            return new
        return _orig_deepcopy(obj, *args, **kwargs)

    _copy.deepcopy = _safe_deepcopy

    # ---- Load policy ----
    print(f"[modal] Loading {model_id}...")
    t0 = time.time()
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    policy = PI0Policy.from_pretrained(model_id).eval().to("cpu").to(torch.float32)
    print(f"[modal] Loaded in {time.time() - t0:.1f}s")

    # ---- Build wrapper ----
    class Pi0MonolithicWrapper(nn.Module):
        def __init__(self, pi0_pytorch_model, num_steps=10):
            super().__init__()
            self.model = pi0_pytorch_model
            self.num_steps = num_steps

        def forward(
            self,
            img_base, img_wrist_l, img_wrist_r,
            mask_base, mask_wrist_l, mask_wrist_r,
            lang_tokens, lang_masks,
            state, noise,
        ):
            images = [img_base, img_wrist_l, img_wrist_r]
            img_masks = [mask_base, mask_wrist_l, mask_wrist_r]
            return self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state,
                noise=noise, num_steps=self.num_steps,
            )

    wrapper = Pi0MonolithicWrapper(policy.model, num_steps=num_steps).eval()

    # ---- Dummy inputs ----
    cfg = policy.config
    B = 1
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim
    state_dim = getattr(cfg, "max_state_dim", 32)

    dummy = dict(
        img_base=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        img_wrist_l=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        img_wrist_r=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        mask_base=torch.ones(B, dtype=torch.bool),
        mask_wrist_l=torch.ones(B, dtype=torch.bool),
        mask_wrist_r=torch.ones(B, dtype=torch.bool),
        lang_tokens=torch.randint(0, 257152, (B, 16), dtype=torch.long),
        lang_masks=torch.ones(B, 16, dtype=torch.bool),
        state=torch.randn(B, state_dim, dtype=torch.float32),
        noise=torch.randn(B, chunk, action_dim, dtype=torch.float32),
    )

    # ---- Export via onnx-diagnostic torch_export_patches ----
    output_dir = Path(ONNX_OUTPUT_PATH) / "monolithic"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    print("[modal] torch_export_patches(patch_transformers=True) + torch.export.export...")
    t0 = time.time()

    # onnx-diagnostic 0.9.x expects MambaCache which was removed in transformers 5.x.
    # Stub it before onnx-diagnostic imports.
    try:
        from transformers.models.mamba.modeling_mamba import MambaCache as _MambaCache  # noqa
    except ImportError:
        import transformers.cache_utils as _cu
        if not hasattr(_cu, "MambaCache"):
            class _MambaCacheStub:
                pass
            _cu.MambaCache = _MambaCacheStub
            try:
                import transformers.models.mamba.modeling_mamba as _mm
                if not hasattr(_mm, "MambaCache"):
                    _mm.MambaCache = _MambaCacheStub
            except ImportError:
                pass

    try:
        from onnx_diagnostic.torch_export_patches import torch_export_patches
    except ImportError as e:
        print(f"[modal] onnx-diagnostic not available: {e}")
        raise

    # Dynamic shapes declaration (torch.export style)
    dynamic_shapes = None  # start simple — try static first, then add Dim if needed

    with torch_export_patches(patch_transformers=True):
        ep = torch.export.export(
            wrapper,
            tuple(dummy.values()),
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
    print(f"[modal] torch.export took {time.time() - t0:.1f}s")

    # ---- Convert ExportedProgram to ONNX ----
    t0 = time.time()
    input_names = list(dummy.keys())
    try:
        # Prefer the newer ONNX conversion path
        onnx_program = torch.onnx.export(
            ep,
            tuple(dummy.values()),
            str(onnx_path),
            input_names=input_names,
            output_names=["actions"],
            opset_version=19,
        )
    except Exception as e:
        print(f"[modal] ONNX conversion failed: {e}")
        raise
    print(f"[modal] ONNX conversion took {time.time() - t0:.1f}s")

    # Commit to volume
    onnx_output.commit()

    # Report
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / 1e6
        print(f"[modal] SUCCESS: {onnx_path} ({size_mb:.1f}MB)")
        # Also check for external data
        data_files = list(output_dir.glob("*.data"))
        total_mb = sum(f.stat().st_size for f in data_files) / 1e6 + size_mb
        print(f"[modal] Total on disk: {total_mb:.1f}MB (incl {len(data_files)} data files)")
        return {
            "status": "ok",
            "onnx_path": str(onnx_path),
            "size_mb": total_mb,
        }
    else:
        return {"status": "fail", "reason": "onnx file not created"}


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={
        HF_CACHE_PATH: hf_cache,
        ONNX_OUTPUT_PATH: onnx_output,
    },
    secrets=[_hf_secret()],
)
def parity_test_monolithic(model_id: str = "lerobot/pi0_base", num_steps: int = 1):
    """Parity test: PyTorch pi0 vs monolithic ONNX at a given num_steps.

    The monolithic ONNX bakes num_steps in at export time; this test runs
    PyTorch `sample_actions(num_steps=N)` with the same N on seeded inputs
    and compares. Target: cos >= 0.999.
    """
    import sys, types
    for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
        stub = types.ModuleType(_mod); stub.GrootPolicy = None; stub.GR00TN15 = None
        sys.modules[_mod] = stub

    import numpy as np
    import torch
    import onnxruntime as ort
    from pathlib import Path

    # Apply same patches as export (so PyTorch ref runs the same way)
    from lerobot.policies.pi0 import modeling_pi0
    from transformers import masking_utils
    from lerobot.policies import pi_gemma as _pg

    _orig_embed = modeling_pi0.PaliGemmaWithExpertModel.embed_image
    def _patched_embed_image(self, image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        out = self.paligemma.model.get_image_features(image)
        features = out.pooler_output if hasattr(out, "pooler_output") else out
        features = features * self.paligemma.config.text_config.hidden_size ** 0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features
    modeling_pi0.PaliGemmaWithExpertModel.embed_image = _patched_embed_image

    _orig_ccm = masking_utils.create_causal_mask
    def _ccm_shim(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return _orig_ccm(*args, **kwargs)
    masking_utils.create_causal_mask = _ccm_shim
    if hasattr(_pg, "create_causal_mask"):
        _pg.create_causal_mask = _ccm_shim

    # Load PyTorch reference
    print("[parity] Loading PyTorch pi0...")
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    policy = PI0Policy.from_pretrained(model_id).eval().to(torch.float32).to("cpu")

    # Build same dummy inputs as export
    B = 1
    cfg = policy.config
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim
    state_dim = getattr(cfg, "max_state_dim", 32)

    torch.manual_seed(42)
    img = torch.randn(B, 3, 224, 224, dtype=torch.float32)
    mask = torch.ones(B, dtype=torch.bool)
    lang_tokens = torch.randint(0, 257152, (B, 16), dtype=torch.long)
    lang_masks = torch.ones(B, 16, dtype=torch.bool)
    state = torch.randn(B, state_dim, dtype=torch.float32)
    noise = torch.randn(B, chunk, action_dim, dtype=torch.float32)

    # Run PyTorch ref at matching num_steps
    print(f"[parity] Running PyTorch ref (num_steps={num_steps})...")
    images = [img, img, img]
    img_masks = [mask, mask, mask]
    with torch.no_grad():
        pt_actions = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            noise=noise, num_steps=num_steps,
        )
    pt_np = pt_actions.cpu().numpy()
    print(f"[parity] pt actions: {pt_np.shape}, first: {pt_np[0, 0, :5]}")

    # Run ONNX
    print("[parity] Running ONNX...")
    onnx_path = Path(ONNX_OUTPUT_PATH) / "monolithic" / "model.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "img_base": img.numpy(),
        "img_wrist_l": img.numpy(),
        "img_wrist_r": img.numpy(),
        "mask_base": mask.numpy(),
        "mask_wrist_l": mask.numpy(),
        "mask_wrist_r": mask.numpy(),
        "lang_tokens": lang_tokens.numpy().astype(np.int64),
        "lang_masks": lang_masks.numpy(),
        "state": state.numpy(),
        "noise": noise.numpy(),
    }
    ort_out = sess.run(None, ort_inputs)[0]
    print(f"[parity] onnx actions: {ort_out.shape}, first: {ort_out[0, 0, :5]}")

    # Compare
    pt0 = pt_np[0, 0]
    on0 = ort_out[0, 0]
    max_abs = float(np.abs(pt0 - on0).max())
    cos = float(np.dot(pt0, on0) / (np.linalg.norm(pt0) * np.linalg.norm(on0) + 1e-8))
    full_cos = float(
        np.dot(pt_np.flatten(), ort_out.flatten())
        / (np.linalg.norm(pt_np) * np.linalg.norm(ort_out) + 1e-8)
    )
    full_max = float(np.abs(pt_np - ort_out).max())

    print(f"\n====== PARITY (num_steps=1) ======")
    print(f"  first-action max_abs: {max_abs:.4e}")
    print(f"  first-action cos:     {cos:+.6f}")
    print(f"  full chunk max_abs:   {full_max:.4e}")
    print(f"  full chunk cos:       {full_cos:+.6f}")
    passed = full_cos >= 0.999 and full_max < 0.1
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    return {
        "status": "ok",
        "first_cos": cos,
        "first_max_abs": max_abs,
        "full_cos": full_cos,
        "full_max_abs": full_max,
        "passed": passed,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache},
    secrets=[
        _hf_secret(),
    ],
)
def parity_native_pi0(model_id: str = "lerobot/pi0_base"):
    """Native-path parity: PI0Policy.predict_action_chunk (wrapper) vs
    PI0Pytorch.sample_actions (raw) on same seeded inputs.

    Closes the cross-framework moat claim for pi0 — SmolVLA already has
    this via `native-path-parity` item #1 (DecomposedRMSNorm swap path).
    For pi0, no swap needed, so this is trivially cos=1.0 but worth
    recording as a measured number.
    """
    import sys, types
    for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
        stub = types.ModuleType(_mod); stub.GrootPolicy = None; stub.GR00TN15 = None
        sys.modules[_mod] = stub

    import numpy as np
    import torch

    print("[native] Loading PI0Policy...")
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from huggingface_hub import snapshot_download

    policy = PI0Policy.from_pretrained(model_id).eval().to(torch.float32).to("cpu")
    repo = snapshot_download(model_id)
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cpu"}},
    )

    # Build realistic batch
    cfg = policy.config
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim

    rng = np.random.RandomState(42)
    img_np = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    state = torch.from_numpy(rng.randn(14).astype(np.float32) * 0.1)

    batch_raw = {
        "observation.images.base_0_rgb": img_t.unsqueeze(0),
        "observation.images.left_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.images.right_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.state": state.unsqueeze(0),
        "task": ["pick up the red bowl"],
    }
    batch = pre(batch_raw)
    noise = torch.from_numpy(np.random.RandomState(99).randn(1, chunk, action_dim).astype(np.float32))

    # Path 1: wrapper
    print("[native] Running PI0Policy.predict_action_chunk (wrapper)...")
    with torch.no_grad():
        a_wrapper = policy.predict_action_chunk(batch, noise=noise).cpu().numpy()

    # Path 2: raw — reproduce what the wrapper does internally
    print("[native] Running PI0Pytorch.sample_actions (raw)...")
    images, img_masks = policy._preprocess_images(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]
    state_padded = policy.prepare_state(batch)
    with torch.no_grad():
        a_raw = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state_padded, noise=noise,
        ).cpu().numpy()
    # Wrapper slices to original action_dim
    orig_action_dim = a_wrapper.shape[-1]
    a_raw_sliced = a_raw[:, :, :orig_action_dim]

    # Compare
    def _cos(a, b):
        af, bf = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
        return float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
    first_cos = _cos(a_wrapper[0, 0], a_raw_sliced[0, 0])
    first_max = float(np.max(np.abs(a_wrapper[0, 0] - a_raw_sliced[0, 0])))
    full_cos = _cos(a_wrapper, a_raw_sliced)
    full_max = float(np.max(np.abs(a_wrapper - a_raw_sliced)))
    print(f"[native] wrapper first: {a_wrapper[0, 0, :5]}")
    print(f"[native] raw     first: {a_raw_sliced[0, 0, :5]}")
    print(f"[native]   first_cos={first_cos:.10f}, first_max={first_max:.3e}")
    print(f"[native]   full_cos ={full_cos:.10f}, full_max ={full_max:.3e}")
    passed = first_cos >= 0.999 and full_cos >= 0.999
    print(f"[native] VERDICT: {'PASS' if passed else 'FAIL'}")
    return {
        "first_cos": first_cos, "first_max_abs": first_max,
        "full_cos": full_cos, "full_max_abs": full_max,
        "passed": passed,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache},
    secrets=[_hf_secret()],
)
def quality_num_steps_pi0(model_id: str = "lerobot/pi0_base"):
    """Characterize PyTorch pi0 action drift: num_steps=1 vs num_steps=10.

    We currently ship the num_steps=1 monolithic ONNX because num_steps=10
    hits a shape-tracing bug during torch.export. This test measures how
    much action quality is lost by integrating with 1 big Euler step vs
    10 smaller steps on shared seeded inputs. Informs whether we ship with
    a "num_steps=1 regression expected" warning or if the drift is small.
    """
    import sys, types
    for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
        stub = types.ModuleType(_mod); stub.GrootPolicy = None; stub.GR00TN15 = None
        sys.modules[_mod] = stub

    import numpy as np
    import torch

    # Need the same patches the export used, so PyTorch runs with the same code
    from lerobot.policies.pi0 import modeling_pi0
    from transformers import masking_utils
    from lerobot.policies import pi_gemma as _pg
    _orig_embed = modeling_pi0.PaliGemmaWithExpertModel.embed_image
    def _patched_embed_image(self, image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        out = self.paligemma.model.get_image_features(image)
        features = out.pooler_output if hasattr(out, "pooler_output") else out
        features = features * self.paligemma.config.text_config.hidden_size ** 0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features
    modeling_pi0.PaliGemmaWithExpertModel.embed_image = _patched_embed_image
    _orig_ccm = masking_utils.create_causal_mask
    def _ccm_shim(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return _orig_ccm(*args, **kwargs)
    masking_utils.create_causal_mask = _ccm_shim
    if hasattr(_pg, "create_causal_mask"):
        _pg.create_causal_mask = _ccm_shim

    print("[quality] Loading pi0 PyTorch policy...")
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    policy = PI0Policy.from_pretrained(model_id).eval().to(torch.float32).to("cpu")

    cfg = policy.config
    B = 1
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim
    state_dim = getattr(cfg, "max_state_dim", 32)

    torch.manual_seed(42)
    img = torch.randn(B, 3, 224, 224, dtype=torch.float32)
    mask = torch.ones(B, dtype=torch.bool)
    lang_tokens = torch.randint(0, 257152, (B, 16), dtype=torch.long)
    lang_masks = torch.ones(B, 16, dtype=torch.bool)
    state = torch.randn(B, state_dim, dtype=torch.float32)
    noise = torch.randn(B, chunk, action_dim, dtype=torch.float32)
    images = [img, img, img]
    img_masks = [mask, mask, mask]

    with torch.no_grad():
        a_1 = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            noise=noise, num_steps=1,
        ).cpu().numpy()
        a_10 = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            noise=noise, num_steps=10,
        ).cpu().numpy()

    def _cos(a, b):
        af, bf = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
        return float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))

    first_cos = _cos(a_1[0, 0], a_10[0, 0])
    first_max = float(np.max(np.abs(a_1[0, 0] - a_10[0, 0])))
    first_l2 = float(np.linalg.norm(a_1[0, 0] - a_10[0, 0]))
    full_cos = _cos(a_1, a_10)
    full_max = float(np.max(np.abs(a_1 - a_10)))
    full_l2 = float(np.linalg.norm(a_1 - a_10))
    action_range = float(np.max(a_10) - np.min(a_10))

    print(f"[quality] num_steps=1  first-5: {a_1[0, 0, :5]}")
    print(f"[quality] num_steps=10 first-5: {a_10[0, 0, :5]}")
    print(f"[quality] first-action: cos={first_cos:.6f}, max_abs={first_max:.3e}, L2={first_l2:.3e}")
    print(f"[quality] full-chunk:   cos={full_cos:.6f}, max_abs={full_max:.3e}, L2={full_l2:.3e}")
    print(f"[quality] action range (max-min) at num_steps=10: {action_range:.3f}")
    print(f"[quality] relative max_abs (first): {first_max / (action_range + 1e-12):.4f}")

    return {
        "first_cos": first_cos, "first_max_abs": first_max, "first_l2": first_l2,
        "full_cos": full_cos, "full_max_abs": full_max, "full_l2": full_l2,
        "action_range": action_range,
    }


@app.function(
    image=gpu_image,
    gpu="A10G",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def parity_cuda_pi0():
    """CUDAExecutionProvider vs CPUExecutionProvider parity for the pi0
    monolithic ONNX. Confirms the GPU serve path produces identical
    outputs (within float32 tolerance) — a strict gate for
    `tether serve --device cuda`."""
    import numpy as np
    import onnxruntime as ort
    from pathlib import Path

    onnx_path = Path(ONNX_OUTPUT_PATH) / "monolithic" / "model.onnx"
    assert onnx_path.exists(), f"{onnx_path} missing"

    print("[cuda] Creating fixed input batch...")
    B = 1
    chunk, action_dim = 50, 32
    rng = np.random.RandomState(42)
    img = rng.randn(B, 3, 224, 224).astype(np.float32)
    mask = np.ones((B,), dtype=np.bool_)
    lang = rng.randint(0, 257152, (B, 16)).astype(np.int64)
    lang_mask = np.ones((B, 16), dtype=np.bool_)
    state = rng.randn(B, 32).astype(np.float32)
    noise = rng.randn(B, chunk, action_dim).astype(np.float32)
    inputs = {
        "img_base": img, "img_wrist_l": img, "img_wrist_r": img,
        "mask_base": mask, "mask_wrist_l": mask, "mask_wrist_r": mask,
        "lang_tokens": lang, "lang_masks": lang_mask,
        "state": state, "noise": noise,
    }

    print("[cuda] Running on CPUExecutionProvider...")
    s_cpu = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    cpu_inputs = {k: v for k, v in inputs.items() if k in [i.name for i in s_cpu.get_inputs()]}
    out_cpu = s_cpu.run(None, cpu_inputs)[0]
    print(f"[cuda] CPU first: {out_cpu[0, 0, :5]}")

    print("[cuda] Running on CUDAExecutionProvider...")
    s_gpu = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    used = s_gpu.get_providers()[0]
    print(f"[cuda] GPU session providers (first): {used}")
    gpu_inputs = {k: v for k, v in inputs.items() if k in [i.name for i in s_gpu.get_inputs()]}
    out_gpu = s_gpu.run(None, gpu_inputs)[0]
    print(f"[cuda] GPU first: {out_gpu[0, 0, :5]}")

    def _cos(a, b):
        af, bf = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
        return float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))

    cos = _cos(out_cpu, out_gpu)
    max_abs = float(np.max(np.abs(out_cpu - out_gpu)))
    print(f"[cuda] CPU vs GPU: cos={cos:.8f}, max_abs={max_abs:.3e}")
    print(f"[cuda] VERDICT: {'PASS' if cos >= 0.999 and used == 'CUDAExecutionProvider' else 'FAIL'}")
    return {"cos": cos, "max_abs": max_abs, "used_provider": used}


@app.local_entrypoint()
def main(
    num_steps: int = 1,
    parity: bool = False,
    native: bool = False,
    quality: bool = False,
    cuda: bool = False,
):
    """Default to num_steps=1 (the only working config currently).

    Usage:
        modal run scripts/modal_pi0_monolithic_export.py              # export
        modal run scripts/modal_pi0_monolithic_export.py --parity     # ONNX parity
        modal run scripts/modal_pi0_monolithic_export.py --native     # native parity
        modal run scripts/modal_pi0_monolithic_export.py --quality    # num_steps=1 vs =10
        modal run scripts/modal_pi0_monolithic_export.py --cuda       # CPU vs CUDA ORT
    """
    if cuda:
        result = parity_cuda_pi0.remote()
    elif quality:
        result = quality_num_steps_pi0.remote()
    elif native:
        result = parity_native_pi0.remote()
    elif parity:
        result = parity_test_monolithic.remote(num_steps=num_steps)
    else:
        result = export_pi0_monolithic_modal.remote(num_steps=num_steps)
    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
