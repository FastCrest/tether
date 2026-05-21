"""Lift #1 Day 4h — Pi0VLA.predict_action vs lerobot PI0Policy parity gate.

Validates that Pi0VLA.predict_action (the new BaseVLA-spine path, shipped
Day 4g Phase B) produces bit-identical actions to lerobot's PI0Policy
upstream reference on the same inputs + same noise seed.

Pass criteria:
    max abs error  <  1e-4   (bit-identical for fp32 inference)
    p95 abs error  <  1e-5

If divergence is observed, investigate root cause (per CLAUDE.md "no
band-aids") — do NOT widen the tolerance.

Usage:
    modal run scripts/modal_pi0_predict_action_parity.py
    modal run scripts/modal_pi0_predict_action_parity.py --num-steps 10 --chunk-size 50

Hardware:    A10G (~$1.10/hr)
Cold start:  ~2 min (PaliGemma 3B + lerobot + LIBERO-less deps)
Wall clock:  ~3-5 min total
Spend:       ~$1.50 per run
"""
from __future__ import annotations

import os
import subprocess
import sys
import types

import modal


def _hf_secret():
    """HF token secret (PaliGemma + lerobot/pi0_base are gated)."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
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


# Image: lerobot==0.5.1 (upstream PI0Policy) + reflex-vla (Pi0VLA spine).
# No LIBERO / MuJoCo needed — we're doing one forward pass, not a rollout.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "transformers<5.4,>=4.40",  # match modal_libero_lerobot_native.py pin
        "numpy",
        "Pillow",
        "pydantic>=2.0",
        "pyyaml",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "lerobot==0.5.1",
        "num2words",
    )
    .run_commands(
        f'pip install "reflex-vla @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


app = modal.App("reflex-pi0-spine-parity")
_hf_cache_volume = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    secrets=[_hf_secret()],
    volumes={"/root/.cache/huggingface": _hf_cache_volume},
)
def run_parity(
    num_steps: int = 10,
    chunk_size: int = 50,
    noise_seed: int = 99,
    input_seed: int = 42,
) -> dict:
    """Single-shot parity: PI0Policy.predict_action_chunk vs Pi0VLA.predict_action.

    Both pipelines fed identical preprocessed inputs + identical noise.
    Returns the error distribution + verdict.
    """
    import time

    import numpy as np
    import torch

    results: dict = {"num_steps": num_steps, "chunk_size": chunk_size, "steps": []}

    def step(name: str, status: str, detail: str = ""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else ("FAIL" if status == "fail" else "INFO")
        print(f"[{tag}] {name} — {detail}", flush=True)

    # ─── transformers-version patches for lerobot PI0Policy ────────────
    # Mirror local_pi0_monolithic_parity.py:11-58 — required for lerobot
    # to load on transformers 4.51+.
    for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
        _stub = types.ModuleType(_mod)
        _stub.GrootPolicy = None
        _stub.GR00TN15 = None
        sys.modules[_mod] = _stub

    def _patch_pi0_for_transformers_457():
        from lerobot.policies.pi0 import modeling_pi0

        def patched_embed_image(self, image):
            out_dtype = image.dtype
            if image.dtype != torch.float32:
                image = image.to(torch.float32)
            image_outputs = self.paligemma.model.get_image_features(image)
            if hasattr(image_outputs, "pooler_output"):
                features = image_outputs.pooler_output
            else:
                features = image_outputs
            features = features * self.paligemma.config.text_config.hidden_size ** 0.5
            if features.dtype != out_dtype:
                features = features.to(out_dtype)
            return features

        modeling_pi0.PaliGemmaWithExpertModel.embed_image = patched_embed_image

    def _patch_create_causal_mask_kwarg():
        from transformers import masking_utils
        original = masking_utils.create_causal_mask

        def shim(*args, **kwargs):
            if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
                kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
            return original(*args, **kwargs)

        masking_utils.create_causal_mask = shim
        try:
            from lerobot.policies import pi_gemma
            if hasattr(pi_gemma, "create_causal_mask"):
                pi_gemma.create_causal_mask = shim
        except ImportError:
            pass

    _patch_pi0_for_transformers_457()
    _patch_create_causal_mask_kwarg()
    step("patches", "pass", "lerobot patched for transformers 4.51+")

    # ─── 1. Load lerobot PI0Policy (the oracle) ────────────────────────
    print("\n=== Step 1: Load lerobot PI0Policy ===", flush=True)
    start = time.time()
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from huggingface_hub import snapshot_download

    policy = PI0Policy.from_pretrained("lerobot/pi0_base").eval()
    policy = policy.to(dtype=torch.float32).to("cpu")  # CPU + fp32 for max determinism
    step("load_lerobot", "pass", f"{time.time() - start:.1f}s, params={sum(p.numel() for p in policy.parameters())/1e9:.2f}B")

    # ─── 2. Build the input batch + preprocess ─────────────────────────
    print("\n=== Step 2: Build deterministic input batch ===", flush=True)
    rng = np.random.RandomState(input_seed)
    img_np = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    img_t = img_t * 2.0 - 1.0  # [-1, 1] SigLIP normalization
    state = torch.from_numpy(rng.randn(14).astype(np.float32) * 0.1)

    batch_raw = {
        "observation.images.base_0_rgb": img_t.unsqueeze(0),
        "observation.images.left_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.images.right_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.state": state.unsqueeze(0),
        "task": ["pick up the red bowl"],
    }
    repo = snapshot_download("lerobot/pi0_base")
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cpu"}},
    )
    batch_pp = pre(batch_raw)
    step("preprocess", "pass", f"seed={input_seed}, state.shape={state.shape}")

    # ─── 3. Generate shared noise (the only stochastic input) ──────────
    cfg = policy.config
    action_dim = cfg.max_action_dim  # pi0 padded action dim = 32
    noise_np = np.random.RandomState(noise_seed).randn(1, chunk_size, action_dim).astype(np.float32)
    noise = torch.from_numpy(noise_np)
    step("noise", "pass", f"seed={noise_seed}, shape={tuple(noise.shape)}")

    # ─── 4. Run lerobot PI0Policy (oracle) ─────────────────────────────
    print("\n=== Step 4: lerobot PI0Policy.predict_action_chunk (oracle) ===", flush=True)
    start = time.time()
    with torch.no_grad():
        oracle_actions = policy.predict_action_chunk(batch_pp, noise=noise.clone())
    oracle_actions = oracle_actions.cpu().numpy() if hasattr(oracle_actions, "cpu") else np.asarray(oracle_actions)
    step("lerobot_forward", "pass", f"{time.time() - start:.1f}s, shape={oracle_actions.shape}, first={oracle_actions[0, 0, :5]}")

    # Extract the SAME tensors PI0Policy uses internally — we feed these to Pi0VLA.
    images, img_masks = policy._preprocess_images(batch_pp)
    lang_tokens = batch_pp["observation.language.tokens"]
    lang_masks = batch_pp["observation.language.attention_mask"]
    state_tensor = policy.prepare_state(batch_pp)
    step("extract_inputs", "pass",
         f"images={[tuple(i.shape) for i in images]}, lang_tokens={tuple(lang_tokens.shape)}, "
         f"state={tuple(state_tensor.shape)}")

    # ─── 5. Build Pi0VLA from the SAME loaded lerobot policy ──────────
    # We deliberately DON'T call Pi0VLA.from_pretrained("lerobot/pi0_base")
    # here — that's broken (it calls PaliGemmaForConditionalGeneration.from_pretrained
    # on the lerobot/pi0_base repo, which fails to map any PaliGemma weights
    # because lerobot's checkpoint nests them under paligemma_with_expert.paligemma.*).
    # The parity gate's purpose is to validate Pi0VLA's INFERENCE MATH matches
    # lerobot's, not the checkpoint loading path — so we build Pi0VLA from the
    # already-loaded lerobot policy's components. Checkpoint-loading correctness
    # is a separate concern, fixed later.
    print("\n=== Step 5: Build Pi0VLA (from loaded lerobot weights) ===", flush=True)
    start = time.time()
    from reflex.models.vlas.pi0 import Pi0VLA
    from reflex.models.vision.siglip_backbone import SigLIPBackbone
    from reflex.models.llm.paligemma_backbone import PaliGemmaBackbone
    from reflex.models.projectors.linear_projector import LinearProjector
    from reflex.models.heads.flow_matching_head import FlowMatchingHead
    from reflex.exporters.pi0_prefix_exporter import build_pi0_expert_with_prefix

    paligemma = policy.model.paligemma_with_expert.paligemma
    vision = SigLIPBackbone(model=paligemma.model.vision_tower)
    llm = PaliGemmaBackbone(model=paligemma)

    state_proj_lerobot = policy.model.state_proj
    projector = LinearProjector(in_dim=32, out_dim=1024)
    with torch.no_grad():
        projector.linear.weight.copy_(state_proj_lerobot.weight)
        projector.linear.bias.copy_(state_proj_lerobot.bias)

    flowmatch_state_dict = policy.model.state_dict()
    expert, _ = build_pi0_expert_with_prefix(flowmatch_state_dict)
    head = FlowMatchingHead(expert_stack=expert)

    vla = Pi0VLA(
        vision_backbone=vision,
        llm_backbone=llm,
        projector=projector,
        vla_head=head,
    )
    # Ensure dtype/device match (lerobot policy already on cpu+fp32)
    for module in [vla.vision_backbone, vla.llm_backbone, vla.projector, vla.vla_head]:
        module.to(dtype=torch.float32).to("cpu")
    step("build_vla", "pass",
         f"{time.time() - start:.1f}s, paligemma+expert+state_proj inherited from lerobot policy")

    # NOW free lerobot — we have references via vla
    del policy
    import gc
    gc.collect()

    # ─── 5b. Intermediate-tensor parity diff ──────────────────────────
    # Compare my pipeline vs lerobot's STEP-BY-STEP — embedding, prefix
    # prefill, state_emb, first denoise step. The first row that diverges
    # is the bug location.
    print("\n=== Step 5b: Intermediate-tensor parity ===", flush=True)
    with torch.no_grad():
        # Re-load lerobot to compute reference intermediates (policy was deleted).
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        _temp_policy = PI0Policy.from_pretrained("lerobot/pi0_base").eval()
        _temp_policy = _temp_policy.to(dtype=torch.float32).to("cpu")
        ler_prefix_embs, ler_prefix_pad, ler_prefix_att = _temp_policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        # My pipeline's embed (must mirror pi0.py:258-285 — text pre-scaled, image raw)
        my_image_embs = []
        text_hidden = vla.llm_backbone.text_hidden_size
        sqrt_h = text_hidden ** 0.5
        for img in images:
            e = vla.vision_backbone(img)
            e = vla.llm_backbone.multi_modal_projector(e)
            my_image_embs.append(e)
        my_text_emb = vla.llm_backbone.embed_tokens(lang_tokens) * sqrt_h
        my_prefix_embs = torch.cat([*my_image_embs, my_text_emb], dim=1)

        # Per-region norms (image tokens, text tokens)
        img_token_count = ler_prefix_embs.shape[1] - lang_tokens.shape[1]  # 768
        print(f"  Prefix shape: lerobot {ler_prefix_embs.shape}, mine {my_prefix_embs.shape}")
        print(f"  Prefix total norm:    lerobot {ler_prefix_embs.norm():.4f}  mine {my_prefix_embs.norm():.4f}")
        print(f"  Image-region norm:    lerobot {ler_prefix_embs[:, :img_token_count].norm():.4f}  mine {my_prefix_embs[:, :img_token_count].norm():.4f}")
        print(f"  Text-region norm:     lerobot {ler_prefix_embs[:, img_token_count:].norm():.4f}  mine {my_prefix_embs[:, img_token_count:].norm():.4f}")
        print(f"  Sample img tok[0, 0, :5]:  lerobot {ler_prefix_embs[0, 0, :5]}  mine {my_prefix_embs[0, 0, :5]}")
        print(f"  Sample text tok[0, {img_token_count}, :5]: lerobot {ler_prefix_embs[0, img_token_count, :5]}  mine {my_prefix_embs[0, img_token_count, :5]}")

        embed_diff = (ler_prefix_embs - my_prefix_embs).abs()
        print(f"  Embed diff: max {embed_diff.max():.4e}  mean {embed_diff.mean():.4e}")

        # Compare state_emb
        ler_state_emb = _temp_policy.model.state_proj(state_tensor.to(torch.float32))
        my_state_emb = vla.projector(state_tensor)
        print(f"\n  State emb shape: lerobot {ler_state_emb.shape}, mine {my_state_emb.shape}")
        state_diff = (ler_state_emb - my_state_emb).abs()
        print(f"  State emb diff: max {state_diff.max():.4e}  mean {state_diff.mean():.4e}")

        # Compare prefix prefill K/V — this is the bridge between prefix embedding
        # and the expert's attention. If embeds match but PKV diverges, the bug is
        # in the prefill (attention mask, position_ids, attn impl, dtype path).
        print(f"\n  --- Prefix prefill K/V comparison ---")
        # Lerobot side: invoke paligemma_with_expert.forward with the full block mask
        # the way denoise_step would set it up (without the suffix part).
        import math as _math
        ler_prefix_pad = ler_prefix_pad.to(torch.bool)
        # Build lerobot's prefix att 2d mask (bidirectional within prefix)
        from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks
        ler_prefix_2d = make_att_2d_masks(ler_prefix_pad, ler_prefix_att)  # [B, p, p]
        # 4D format for paligemma
        neg_inf = torch.finfo(ler_prefix_embs.dtype).min
        ler_prefix_4d = torch.where(ler_prefix_2d.unsqueeze(1),
                                    torch.zeros((), dtype=ler_prefix_embs.dtype),
                                    torch.full((), neg_inf, dtype=ler_prefix_embs.dtype))
        ler_pos = torch.cumsum(ler_prefix_pad.long(), dim=1) - 1
        _temp_policy.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        ler_prefix_out, ler_pkv = _temp_policy.model.paligemma_with_expert.forward(
            inputs_embeds=[ler_prefix_embs, None],
            past_key_values=None,
            attention_mask=ler_prefix_4d,
            position_ids=ler_pos,
            use_cache=True,
            adarms_cond=[None, None],
        )
        ler_pkv_l0_k = ler_pkv.layers[0].keys
        print(f"  Lerobot prefill: pkv layers={len(ler_pkv.layers)}, layer-0 K shape={ler_pkv_l0_k.shape}")

        # My side: same flow as pi0.py:295-356
        valid_pair = ler_prefix_pad[:, :, None] & ler_prefix_pad[:, None, :]
        my_prefix_4d = torch.where(valid_pair.unsqueeze(1),
                                   torch.zeros((), dtype=my_prefix_embs.dtype),
                                   torch.full((), neg_inf, dtype=my_prefix_embs.dtype))
        my_pos = torch.cumsum(ler_prefix_pad.long(), dim=1) - 1
        vla.llm_backbone.language_model.config._attn_implementation = "eager"
        my_prefill = vla.llm_backbone(
            inputs_embeds=my_prefix_embs,
            attention_mask=my_prefix_4d,
            position_ids=my_pos,
            use_cache=True,
        )
        my_pkv = my_prefill.past_key_values
        my_pkv_l0_k = my_pkv.layers[0].keys if hasattr(my_pkv, "layers") else my_pkv.key_cache[0]
        print(f"  Mine prefill:    pkv layers={len(my_pkv.layers if hasattr(my_pkv,'layers') else my_pkv.key_cache)}, layer-0 K shape={my_pkv_l0_k.shape}")

        pkv_diff = (ler_pkv_l0_k - my_pkv_l0_k).abs()
        print(f"  Layer-0 K diff: max {pkv_diff.max():.4e}  mean {pkv_diff.mean():.4e}")
        print(f"  Layer-0 K norm: lerobot {ler_pkv_l0_k.norm():.4f}  mine {my_pkv_l0_k.norm():.4f}")

        # Also compare a deeper layer (layer 8) and the last one
        for li in (8, len(ler_pkv.layers) - 1):
            ler_li = ler_pkv.layers[li].keys
            my_li = my_pkv.layers[li].keys if hasattr(my_pkv, "layers") else my_pkv.key_cache[li]
            d = (ler_li - my_li).abs()
            print(f"  Layer-{li} K diff: max {d.max():.4e}  mean {d.mean():.4e}  (ler norm {ler_li.norm():.2f} vs mine {my_li.norm():.2f})")

        # ─── Expert one-step v_t comparison ────────────────────────────
        # Same input noise + same prefix-KV; compare lerobot's denoise_step
        # vs my expert's forward. Isolates the expert bug.
        print(f"\n  --- Expert one-step v_t comparison (chunk_size={chunk_size}) ---")
        import copy as _copy
        # CRITICAL: lerobot's denoise_step returns v_t directly (NOT x_t_new).
        # See modeling_pi0.py:931 `return self.action_out_proj(suffix_out)`.
        # The Euler step `x_t += dt * v_t` happens in sample_actions, not in
        # denoise_step.
        ler_pkv_copy = _copy.deepcopy(ler_pkv)
        v_t_ler = _temp_policy.model.denoise_step(
            state_tensor.to(torch.float32),
            ler_prefix_pad,
            ler_pkv_copy,
            noise.clone(),
            torch.tensor([1.0], dtype=torch.float32),
        )

        # Mine: rebuild my masks + run expert
        # (Mirrors pi0.py:303-394 for one step)
        prefix_len_per_batch = ler_prefix_pad.long().sum(dim=-1, keepdim=True)
        suffix_pad_mask = torch.ones(1, chunk_size + 1, dtype=torch.long)
        suffix_position_ids = prefix_len_per_batch + torch.cumsum(suffix_pad_mask, dim=1) - 1
        # Build attn mask (lerobot block pattern)
        prefix_len_int = ler_prefix_pad.shape[1]
        total_len = prefix_len_int + chunk_size + 1
        full_att = torch.zeros(1, total_len, dtype=torch.long)
        full_att[:, prefix_len_int] = 1
        full_att[:, prefix_len_int + 1] = 1
        cumsum_full = torch.cumsum(full_att, dim=1)
        att_2d = cumsum_full[:, None, :] <= cumsum_full[:, :, None]
        full_pad = torch.cat([ler_prefix_pad, suffix_pad_mask.bool()], dim=1)
        pad_2d = full_pad[:, None, :] & full_pad[:, :, None]
        suffix_2d = (att_2d & pad_2d)[:, prefix_len_int:, :].unsqueeze(1)

        # Stack my pkv into [L, B, nkv, prefix_len, hd]
        my_pk_list = [layer.keys for layer in (my_pkv.layers if hasattr(my_pkv, "layers") else my_pkv.key_cache)]
        my_pv_list = [layer.values for layer in (my_pkv.layers if hasattr(my_pkv, "layers") else my_pkv.value_cache)]
        my_prefix_k = torch.stack(my_pk_list, dim=0)
        my_prefix_v = torch.stack(my_pv_list, dim=0)

        v_t_mine = vla.vla_head(
            noisy_actions=noise.clone(),
            timestep=torch.tensor([1.0], dtype=torch.float32),
            position_ids=suffix_position_ids,
            prefix_k=my_prefix_k,
            prefix_v=my_prefix_v,
            state_emb=my_state_emb.unsqueeze(1),
            attn_mask=suffix_2d,
        )

        v_t_diff = (v_t_ler - v_t_mine).abs()
        print(f"  v_t shapes: lerobot {v_t_ler.shape}, mine {v_t_mine.shape}")
        print(f"  v_t diff: max {v_t_diff.max():.4e}  mean {v_t_diff.mean():.4e}")
        print(f"  v_t norm: lerobot {v_t_ler.norm():.4f}  mine {v_t_mine.norm():.4f}")
        print(f"  v_t[0, 0, :8]: lerobot {v_t_ler[0, 0, :8]}  mine {v_t_mine[0, 0, :8]}")

        del _temp_policy
        gc.collect()
    step("intermediate_parity", "pass", "see prints above")

    # ─── 6. Run Pi0VLA.predict_action ──────────────────────────────────
    print("\n=== Step 6: Pi0VLA.predict_action ===", flush=True)
    start = time.time()
    with torch.no_grad():
        vla_actions = vla.predict_action(
            images=images,
            image_masks=img_masks,
            state=state_tensor,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            noise=noise.clone(),
            num_steps=num_steps,
            chunk_size=chunk_size,
        )
    vla_actions = vla_actions.cpu().numpy()
    step("vla_forward", "pass", f"{time.time() - start:.1f}s, shape={vla_actions.shape}, first={vla_actions[0, 0, :5]}")

    # ─── 7. Compare ────────────────────────────────────────────────────
    print("\n=== Step 7: Parity comparison ===", flush=True)
    if oracle_actions.shape != vla_actions.shape:
        step("compare", "fail", f"shape mismatch: oracle={oracle_actions.shape}, vla={vla_actions.shape}")
        return results

    diff = oracle_actions - vla_actions
    abs_diff = np.abs(diff).flatten()
    err_mean = float(abs_diff.mean())
    err_p50 = float(np.percentile(abs_diff, 50))
    err_p95 = float(np.percentile(abs_diff, 95))
    err_p99 = float(np.percentile(abs_diff, 99))
    err_max = float(abs_diff.max())

    # Cosine similarity on first action (for sanity)
    first_oracle = oracle_actions[0, 0]
    first_vla = vla_actions[0, 0]
    cos = float(
        np.dot(first_oracle, first_vla)
        / (np.linalg.norm(first_oracle) * np.linalg.norm(first_vla) + 1e-8)
    )

    metrics = {
        "err_mean": err_mean, "err_p50": err_p50, "err_p95": err_p95,
        "err_p99": err_p99, "err_max": err_max, "first_action_cos": cos,
    }
    results["metrics"] = metrics

    print(f"\n  err_mean = {err_mean:.4e}", flush=True)
    print(f"  err_p50  = {err_p50:.4e}", flush=True)
    print(f"  err_p95  = {err_p95:.4e}", flush=True)
    print(f"  err_p99  = {err_p99:.4e}", flush=True)
    print(f"  err_max  = {err_max:.4e}", flush=True)
    print(f"  first_action_cos = {cos:+.6f}", flush=True)

    # ─── Diagnostic localization ─────────────────────────────────────
    # Per-action-position error (across all 32 features), per-feature error
    # (across all 50 positions). Identifies whether divergence is concentrated
    # at a specific timestep or feature dim.
    diff_3d = oracle_actions - vla_actions  # [B, chunk_size, action_dim]
    per_pos_err = np.abs(diff_3d).mean(axis=(0, 2))  # [chunk_size]
    per_feat_err = np.abs(diff_3d).mean(axis=(0, 1))  # [action_dim]

    print(f"\n  per-position errors (mean across features):")
    print(f"    first 5:  {per_pos_err[:5]}")
    print(f"    last 5:   {per_pos_err[-5:]}")
    print(f"    argmax pos: idx={per_pos_err.argmax()}, val={per_pos_err.max():.4e}")
    print(f"    argmin pos: idx={per_pos_err.argmin()}, val={per_pos_err.min():.4e}")

    print(f"\n  per-feature errors (mean across positions):")
    print(f"    first 5:  {per_feat_err[:5]}")
    print(f"    last 5:   {per_feat_err[-5:]}")
    print(f"    argmax feat: idx={per_feat_err.argmax()}, val={per_feat_err.max():.4e}")

    # First action vs last action — direction comparison
    print(f"\n  oracle action[0, 0, :8] = {oracle_actions[0, 0, :8]}")
    print(f"  vla    action[0, 0, :8] = {vla_actions[0, 0, :8]}")
    print(f"  oracle action[0, -1, :8] = {oracle_actions[0, -1, :8]}")
    print(f"  vla    action[0, -1, :8] = {vla_actions[0, -1, :8]}")

    # Cosine across all 50 actions
    cos_per_pos = []
    for t in range(oracle_actions.shape[1]):
        o = oracle_actions[0, t]
        v = vla_actions[0, t]
        c = float(np.dot(o, v) / (np.linalg.norm(o) * np.linalg.norm(v) + 1e-8))
        cos_per_pos.append(c)
    print(f"\n  per-position cosine sim (first/min/mean/max):")
    print(f"    first 5: {[f'{c:+.3f}' for c in cos_per_pos[:5]]}")
    print(f"    last 5:  {[f'{c:+.3f}' for c in cos_per_pos[-5:]]}")
    print(f"    mean: {np.mean(cos_per_pos):+.4f}, min: {np.min(cos_per_pos):+.4f}, max: {np.max(cos_per_pos):+.4f}")

    metrics["per_pos_err_max_idx"] = int(per_pos_err.argmax())
    metrics["per_pos_err_max_val"] = float(per_pos_err.max())
    metrics["mean_cos_across_positions"] = float(np.mean(cos_per_pos))

    # ─── 8. Verdict ────────────────────────────────────────────────────
    passed = (err_max < 1e-4) and (err_p95 < 1e-5)
    results["passed"] = passed
    if passed:
        step("VERDICT", "pass", f"max={err_max:.2e} < 1e-4 ✓, p95={err_p95:.2e} < 1e-5 ✓")
    else:
        step("VERDICT", "fail",
             f"max={err_max:.2e} (need < 1e-4), p95={err_p95:.2e} (need < 1e-5) — "
             f"investigate root cause per CLAUDE.md")

    return results


@app.local_entrypoint()
def main(
    num_steps: int = 10,
    chunk_size: int = 50,
    noise_seed: int = 99,
    input_seed: int = 42,
):
    print(f"=== Pi0VLA vs lerobot PI0Policy parity gate ===")
    print(f"num_steps={num_steps}, chunk_size={chunk_size}")
    print(f"noise_seed={noise_seed}, input_seed={input_seed}")
    print()

    results = run_parity.remote(
        num_steps=num_steps, chunk_size=chunk_size,
        noise_seed=noise_seed, input_seed=input_seed,
    )

    print("\n========== FINAL ==========")
    for s in results["steps"]:
        tag = "PASS" if s["status"] == "pass" else ("FAIL" if s["status"] == "fail" else "INFO")
        print(f"  [{tag}] {s['step']} — {s['detail']}")

    if "metrics" in results:
        m = results["metrics"]
        print(f"\nError distribution:")
        print(f"  max  = {m['err_max']:.4e}    (gate: < 1e-4)")
        print(f"  p95  = {m['err_p95']:.4e}    (gate: < 1e-5)")
        print(f"  p99  = {m['err_p99']:.4e}")
        print(f"  cos  = {m['first_action_cos']:+.6f}")

    verdict = "PASS" if results.get("passed", False) else "FAIL"
    print(f"\nVerdict: {verdict}")
    sys.exit(0 if results.get("passed", False) else 1)
