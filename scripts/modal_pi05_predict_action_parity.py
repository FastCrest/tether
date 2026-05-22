"""Lift #1 Day 5 Phase B — Pi05VLA.predict_action vs lerobot PI05Policy parity gate.

Mirrors `scripts/modal_pi0_predict_action_parity.py` (Day 4h) but for pi0.5.
Diagnostic harness built in from the start per Day 4h learning: hooks layer-0
input/output + intra-layer-0 sub-modules to localize bugs fast if parity fails.

Pass criteria (same as Day 4h):
    max abs error  <  1e-4
    p95 abs error  <  1e-5

If divergence is observed, investigate root cause per CLAUDE.md "no band-aids".

Usage:
    modal run scripts/modal_pi05_predict_action_parity.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import types

import modal


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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
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
        "lerobot==0.5.1",
        "num2words",
    )
    .run_commands(
        f'pip install "reflex-vla @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


app = modal.App("reflex-pi05-spine-parity")
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
    hf_id: str = "lerobot/pi05_libero_finetuned_v044",
) -> dict:
    import time
    import copy as _copy

    import numpy as np
    import torch

    results: dict = {"num_steps": num_steps, "chunk_size": chunk_size, "steps": []}

    def step(name: str, status: str, detail: str = ""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else ("FAIL" if status == "fail" else "INFO")
        print(f"[{tag}] {name} — {detail}", flush=True)

    # transformers-version patches for lerobot pi0.5 (same as pi0)
    for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
        _stub = types.ModuleType(_mod)
        _stub.GrootPolicy = None
        _stub.GR00TN15 = None
        sys.modules[_mod] = _stub

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

    _patch_create_causal_mask_kwarg()
    step("patches", "pass", "lerobot patched for transformers 4.51+")

    # ─── 1. Load lerobot PI05Policy ────────────────────────────────────
    print("\n=== Step 1: Load lerobot PI05Policy ===", flush=True)
    start = time.time()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from huggingface_hub import snapshot_download

    policy = PI05Policy.from_pretrained(hf_id).eval()
    policy = policy.to(dtype=torch.float32).to("cpu")
    step("load_lerobot", "pass", f"{time.time() - start:.1f}s, params={sum(p.numel() for p in policy.parameters())/1e9:.2f}B")

    # ─── 2. Build deterministic input batch ────────────────────────────
    print("\n=== Step 2: Build deterministic input batch ===", flush=True)
    rng = np.random.RandomState(input_seed)
    # pi0.5_libero expects 256x256 images for image+image2, 224x224 for
    # empty_camera_0 (the placeholder camera slot). Per the model's
    # input_features config from PI05Config.
    img256_np = rng.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img256_t = torch.from_numpy(img256_np).permute(2, 0, 1).float() / 255.0
    img256_t = img256_t * 2.0 - 1.0
    img224_np = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img224_t = torch.from_numpy(img224_np).permute(2, 0, 1).float() / 255.0
    img224_t = img224_t * 2.0 - 1.0
    # pi0.5_libero uses Franka 8-dim state (vs pi0_base's 14-dim Aloha state).
    state = torch.from_numpy(rng.randn(8).astype(np.float32) * 0.1)

    batch_raw = {
        "observation.images.image": img256_t.unsqueeze(0),
        "observation.images.image2": img256_t.unsqueeze(0),
        "observation.images.empty_camera_0": img224_t.unsqueeze(0),
        "observation.state": state.unsqueeze(0),
        "task": ["pick up the red bowl"],
    }
    repo = snapshot_download(hf_id)
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cpu"}},
    )
    batch_pp = pre(batch_raw)
    step("preprocess", "pass", f"seed={input_seed}, state.shape={state.shape}")

    # ─── 3. Shared noise ──────────────────────────────────────────────
    cfg = policy.config
    action_dim = cfg.max_action_dim
    noise_np = np.random.RandomState(noise_seed).randn(1, chunk_size, action_dim).astype(np.float32)
    noise = torch.from_numpy(noise_np)
    step("noise", "pass", f"seed={noise_seed}, shape={tuple(noise.shape)}")

    # ─── 4. Run lerobot PI05Policy (oracle) ───────────────────────────
    print("\n=== Step 4: lerobot PI05Policy.predict_action_chunk (oracle) ===", flush=True)
    start = time.time()
    with torch.no_grad():
        oracle_actions = policy.predict_action_chunk(batch_pp, noise=noise.clone())
    oracle_actions = oracle_actions.cpu().numpy() if hasattr(oracle_actions, "cpu") else np.asarray(oracle_actions)
    step("lerobot_forward", "pass", f"{time.time() - start:.1f}s, shape={oracle_actions.shape}, first={oracle_actions[0, 0, :5]}")

    # Extract internals for Pi05VLA
    images, img_masks = policy._preprocess_images(batch_pp)
    lang_tokens = batch_pp["observation.language.tokens"]
    lang_masks = batch_pp["observation.language.attention_mask"]
    step("extract_inputs", "pass",
         f"images={[tuple(i.shape) for i in images]}, lang_tokens={tuple(lang_tokens.shape)}")

    # ─── 5. Build Pi05VLA from loaded lerobot policy ──────────────────
    # Same workaround pattern as Day 4h Pi0VLA — from_pretrained's PaliGemma
    # loader can't find weights nested under paligemma_with_expert.paligemma.*.
    print("\n=== Step 5: Build Pi05VLA (from loaded lerobot weights) ===", flush=True)
    start = time.time()
    from reflex.models.vlas.pi05 import Pi05VLA
    from reflex.models.vision.siglip_backbone import SigLIPBackbone
    from reflex.models.llm.paligemma_backbone import PaliGemmaBackbone
    from reflex.models.heads.flow_matching_head import FlowMatchingHead
    from reflex.exporters.pi0_prefix import build_pi05_expert_with_prefix

    paligemma = policy.model.paligemma_with_expert.paligemma
    vision = SigLIPBackbone(model=paligemma.model.vision_tower)
    llm = PaliGemmaBackbone(model=paligemma)
    flowmatch_state_dict = policy.model.state_dict()
    expert, _ = build_pi05_expert_with_prefix(flowmatch_state_dict)
    head = FlowMatchingHead(expert_stack=expert)
    vla = Pi05VLA(vision_backbone=vision, llm_backbone=llm, vla_head=head)
    for module in [vla.vision_backbone, vla.llm_backbone, vla.vla_head]:
        module.to(dtype=torch.float32).to("cpu")
    step("build_vla", "pass",
         f"{time.time() - start:.1f}s, paligemma+expert inherited from lerobot policy")

    # ─── 5b. Intermediate-tensor parity diff ──────────────────────────
    # Re-use loaded policy (no second load — Day 4h learning).
    print("\n=== Step 5b: Intermediate-tensor parity ===", flush=True)
    import gc
    with torch.no_grad():
        ler_prefix_embs, ler_prefix_pad, ler_prefix_att = policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        text_hidden = vla.llm_backbone.text_hidden_size
        sqrt_h = text_hidden ** 0.5
        my_image_embs = [vla.llm_backbone.multi_modal_projector(vla.vision_backbone(img)) for img in images]
        my_text_emb = vla.llm_backbone.embed_tokens(lang_tokens) * sqrt_h
        my_prefix_embs = torch.cat([*my_image_embs, my_text_emb], dim=1)

        img_token_count = ler_prefix_embs.shape[1] - lang_tokens.shape[1]
        print(f"  Prefix shape: lerobot {ler_prefix_embs.shape}, mine {my_prefix_embs.shape}")
        print(f"  Total norm:   lerobot {ler_prefix_embs.norm():.4f}  mine {my_prefix_embs.norm():.4f}")
        print(f"  Image norm:   lerobot {ler_prefix_embs[:, :img_token_count].norm():.4f}  mine {my_prefix_embs[:, :img_token_count].norm():.4f}")
        print(f"  Text norm:    lerobot {ler_prefix_embs[:, img_token_count:].norm():.4f}  mine {my_prefix_embs[:, img_token_count:].norm():.4f}")
        embed_diff = (ler_prefix_embs - my_prefix_embs).abs()
        print(f"  Embed diff: max {embed_diff.max():.4e}  mean {embed_diff.mean():.4e}")

        # PaliGemma prefill K/V parity
        from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks
        ler_prefix_pad = ler_prefix_pad.to(torch.bool)
        ler_prefix_2d = make_att_2d_masks(ler_prefix_pad, ler_prefix_att)
        neg_inf = torch.finfo(ler_prefix_embs.dtype).min
        ler_prefix_4d = torch.where(ler_prefix_2d.unsqueeze(1),
                                    torch.zeros((), dtype=ler_prefix_embs.dtype),
                                    torch.full((), neg_inf, dtype=ler_prefix_embs.dtype))
        ler_pos = torch.cumsum(ler_prefix_pad.long(), dim=1) - 1
        policy.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, ler_pkv = policy.model.paligemma_with_expert.forward(
            inputs_embeds=[ler_prefix_embs, None],
            past_key_values=None,
            attention_mask=ler_prefix_4d,
            position_ids=ler_pos,
            use_cache=True,
            adarms_cond=[None, None],
        )

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

        print(f"\n  PKV: lerobot {len(ler_pkv.layers)} layers, mine {len(my_pkv.layers)} layers")
        for li in (0, 8, 17):
            ler_li = ler_pkv.layers[li].keys
            my_li = my_pkv.layers[li].keys
            d = (ler_li - my_li).abs()
            print(f"  Layer-{li} K diff: max {d.max():.4e}  mean {d.mean():.4e}")

        # ─── Expert one-step v_t comparison ────────────────────────────
        print(f"\n  --- Expert one-step v_t (chunk_size={chunk_size}, no state for pi0.5) ---")
        ler_pkv_copy = _copy.deepcopy(ler_pkv)
        v_t_ler = policy.model.denoise_step(
            ler_prefix_pad, ler_pkv_copy, noise.clone(),
            torch.tensor([1.0], dtype=torch.float32),
        )

        # Build my v_t inputs
        prefix_len_per_batch = ler_prefix_pad.long().sum(dim=-1, keepdim=True)
        suffix_pad_mask = torch.ones(1, chunk_size, dtype=torch.long)
        suffix_position_ids = prefix_len_per_batch + torch.cumsum(suffix_pad_mask, dim=1) - 1
        prefix_len_int = ler_prefix_pad.shape[1]
        total_len = prefix_len_int + chunk_size
        full_att = torch.zeros(1, total_len, dtype=torch.long)
        full_att[:, prefix_len_int] = 1
        cumsum_full = torch.cumsum(full_att, dim=1)
        att_2d = cumsum_full[:, None, :] <= cumsum_full[:, :, None]
        full_pad = torch.cat([ler_prefix_pad, suffix_pad_mask.bool()], dim=1)
        pad_2d = full_pad[:, None, :] & full_pad[:, :, None]
        suffix_2d = (att_2d & pad_2d)[:, prefix_len_int:, :].unsqueeze(1)

        my_pk_list = [layer.keys for layer in my_pkv.layers]
        my_pv_list = [layer.values for layer in my_pkv.layers]
        my_prefix_k = torch.stack(my_pk_list, dim=0)
        my_prefix_v = torch.stack(my_pv_list, dim=0)

        v_t_mine = vla.vla_head(
            noisy_actions=noise.clone(),
            timestep=torch.tensor([1.0], dtype=torch.float32),
            position_ids=suffix_position_ids,
            prefix_k=my_prefix_k, prefix_v=my_prefix_v,
            attn_mask=suffix_2d,
        )

        v_t_diff = (v_t_ler - v_t_mine).abs()
        print(f"  v_t shapes: lerobot {v_t_ler.shape}, mine {v_t_mine.shape}")
        print(f"  v_t diff: max {v_t_diff.max():.4e}  mean {v_t_diff.mean():.4e}")
        print(f"  v_t norm: lerobot {v_t_ler.norm():.4f}  mine {v_t_mine.norm():.4f}")
        print(f"  v_t[0, 0, :8]: lerobot {v_t_ler[0, 0, :8]}  mine {v_t_mine[0, 0, :8]}")

        # ─── Intra-attention diff via layer's debug_captures ───────────
        from reflex.models.heads.expert_stack import Pi05ExpertGQALayer
        Pi05ExpertGQALayer.debug_captures = {}
        Pi05ExpertGQALayer.debug_layer_id = 0
        ler_eager_captures: dict = {}
        from transformers.models.gemma import modeling_gemma as _stock_gemma
        _orig_eager = _stock_gemma.eager_attention_forward
        def _patched_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
            if not ler_eager_captures:  # capture only layer 0's first call
                ler_eager_captures["q"] = query.detach().clone()
                ler_eager_captures["k"] = key.detach().clone()
                ler_eager_captures["v"] = value.detach().clone()
                from transformers.models.gemma.modeling_gemma import repeat_kv
                k_full = repeat_kv(key, module.num_key_value_groups)
                v_full = repeat_kv(value, module.num_key_value_groups)
                scores = torch.matmul(query, k_full.transpose(2, 3)) * scaling
                if attention_mask is not None:
                    cm = attention_mask[:, :, :, :k_full.shape[-2]]
                    scores = scores + cm
                attn = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
                ler_eager_captures["attn_weights"] = attn.detach().clone()
            return _orig_eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kw)
        _stock_gemma.eager_attention_forward = _patched_eager
        try:
            ler_pkv_copy3 = _copy.deepcopy(ler_pkv)
            _ = policy.model.denoise_step(
                ler_prefix_pad, ler_pkv_copy3,
                noise.clone(), torch.tensor([1.0], dtype=torch.float32),
            )
            _ = vla.vla_head(
                noisy_actions=noise.clone(),
                timestep=torch.tensor([1.0], dtype=torch.float32),
                position_ids=suffix_position_ids,
                prefix_k=my_prefix_k, prefix_v=my_prefix_v,
                attn_mask=suffix_2d,
            )
            print(f"\n  --- Intra-attention diff (layer 0) ---")
            # Also print per-layer norms for L0, L8, L17 to see if drift compounds
            print(f"  Per-layer my captures (q_post_rope norm):")
            for lid in [0, 4, 8, 12, 17]:
                key = f"L{lid}_q_post_rope"
                if key in Pi05ExpertGQALayer.debug_captures:
                    t = Pi05ExpertGQALayer.debug_captures[key]
                    print(f"    L{lid}: norm {t.norm():.4f}, [0,0,0,:5]={t[0,0,0,:5]}")
            print()
            for name, my_key, ler_key in [
                ("Q_pre_rope_action", "L0_q_pre_rope", None),
                ("Q_post_rope_action", "L0_q_post_rope", "q"),
                ("K_post_rope_action_only", "L0_k_post_rope", None),
                ("V_action_only", "L0_v", None),
                ("K_full_concat", "L0_k_concat", "k"),
                ("ATTN_WEIGHTS", "L0_attn_weights", "attn_weights"),
            ]:
                my_x = Pi05ExpertGQALayer.debug_captures.get(my_key)
                ler_x = ler_eager_captures.get(ler_key) if ler_key else None
                if my_x is None:
                    print(f"  {name}: my missing")
                    continue
                if ler_x is None:
                    print(f"  {name}: shape mine={tuple(my_x.shape)} norm {my_x.norm():.4f}")
                    continue
                if ler_x.shape != my_x.shape:
                    print(f"  {name}: shape mismatch ler={tuple(ler_x.shape)} my={tuple(my_x.shape)}")
                    continue
                d = (ler_x.float() - my_x.float()).abs()
                print(f"  {name}: shape {tuple(ler_x.shape)}, ler norm {ler_x.norm():.4f}, my norm {my_x.norm():.4f}, diff max {d.max():.4e}, mean {d.mean():.4e}")
                # Print first 8 values of [0, 0, 0, :] for direct comparison
                print(f"    ler[0,0,0,:8] = {ler_x[0, 0, 0, :8] if ler_x.ndim >= 4 else ler_x[0, 0, :8]}")
                print(f"    my [0,0,0,:8] = {my_x[0, 0, 0, :8] if my_x.ndim >= 4 else my_x[0, 0, :8]}")

            # Defer RoPE bisection to AFTER sub-module hooks populate captured.
        finally:
            _stock_gemma.eager_attention_forward = _orig_eager
            Pi05ExpertGQALayer.debug_captures = None
        # Save ler_eager_captures + ler q post-RoPE for the bisection below.
        _saved_ler_eager_captures = dict(ler_eager_captures)

        # ─── Layer-0 + sub-module diff (Day 4h methodology) ─────────────
        print(f"\n  --- Layer-0 + intra-layer-0 sub-module diff ---")
        captured: dict = {}
        def make_hook(name):
            def hook(module, inp, out):
                captured[name] = out[0] if isinstance(out, tuple) else out
            return hook

        ler_layers = policy.model.paligemma_with_expert.gemma_expert.model.layers
        my_layers = vla.vla_head.expert_stack.layers
        ler_l0 = ler_layers[0]
        my_l0 = my_layers[0]
        # Also hook layer 17 (last) to localize where the chain diverges.
        ler_l17 = ler_layers[17]
        my_l17 = my_layers[17]
        sub_hooks = [
            ler_l0.register_forward_hook(make_hook("ler_layer_out")),
            my_l0.register_forward_hook(make_hook("my_layer_out")),
            ler_l17.register_forward_hook(make_hook("ler_layer17_out")),
            my_l17.register_forward_hook(make_hook("my_layer17_out")),
            # Pre-hooks for layer 17 INPUT (= layer 16 output) to see if divergence
            # is in layer 17's INPUT or in its forward.
            ler_l17.register_forward_pre_hook(
                lambda mod, inp: captured.__setitem__("ler_layer17_in", inp[0] if isinstance(inp, tuple) else inp)
            ),
            my_l17.register_forward_pre_hook(
                lambda mod, inp: captured.__setitem__("my_layer17_in", inp[0] if isinstance(inp, tuple) else inp)
            ),
            ler_l0.input_layernorm.register_forward_hook(make_hook("ler_input_ln")),
            my_l0.input_layernorm.register_forward_hook(make_hook("my_input_ln")),
            ler_l0.self_attn.q_proj.register_forward_hook(make_hook("ler_q_proj")),
            my_l0.q_proj.register_forward_hook(make_hook("my_q_proj")),
            ler_l0.self_attn.k_proj.register_forward_hook(make_hook("ler_k_proj")),
            my_l0.k_proj.register_forward_hook(make_hook("my_k_proj")),
            ler_l0.self_attn.v_proj.register_forward_hook(make_hook("ler_v_proj")),
            my_l0.v_proj.register_forward_hook(make_hook("my_v_proj")),
            # Also hook self_attn itself to capture the attention output BEFORE o_proj
            ler_l0.self_attn.register_forward_hook(make_hook("ler_attn_out")),
            ler_l0.self_attn.o_proj.register_forward_hook(make_hook("ler_o_proj")),
            my_l0.o_proj.register_forward_hook(make_hook("my_o_proj")),
            # Pre-hooks on o_proj to capture its INPUT = attention output before o_proj
            ler_l0.self_attn.o_proj.register_forward_pre_hook(
                lambda module, inp: captured.__setitem__("ler_o_proj_in", inp[0] if isinstance(inp, tuple) else inp)
            ),
            my_l0.o_proj.register_forward_pre_hook(
                lambda module, inp: captured.__setitem__("my_o_proj_in", inp[0] if isinstance(inp, tuple) else inp)
            ),
            ler_l0.post_attention_layernorm.register_forward_hook(make_hook("ler_post_ln")),
            my_l0.post_attention_layernorm.register_forward_hook(make_hook("my_post_ln")),
            ler_l0.mlp.gate_proj.register_forward_hook(make_hook("ler_gate")),
            my_l0.gate_proj.register_forward_hook(make_hook("my_gate")),
            ler_l0.mlp.up_proj.register_forward_hook(make_hook("ler_up")),
            my_l0.up_proj.register_forward_hook(make_hook("my_up")),
            ler_l0.mlp.down_proj.register_forward_hook(make_hook("ler_down")),
            my_l0.down_proj.register_forward_hook(make_hook("my_down")),
        ]
        try:
            ler_pkv_copy2 = _copy.deepcopy(ler_pkv)
            policy.model.denoise_step(
                ler_prefix_pad, ler_pkv_copy2,
                noise.clone(), torch.tensor([1.0], dtype=torch.float32),
            )
            _ = vla.vla_head(
                noisy_actions=noise.clone(),
                timestep=torch.tensor([1.0], dtype=torch.float32),
                position_ids=suffix_position_ids,
                prefix_k=my_prefix_k, prefix_v=my_prefix_v,
                attn_mask=suffix_2d,
            )
            for name in ["layer_out", "layer17_in", "layer17_out", "input_ln",
                          "q_proj", "k_proj", "v_proj",
                          "o_proj", "post_ln", "gate", "up", "down"]:
                ler_x = captured.get(f"ler_{name}")
                my_x = captured.get(f"my_{name}")
                if ler_x is None or my_x is None:
                    continue
                if ler_x.shape != my_x.shape:
                    print(f"  {name}: shape mismatch ler={ler_x.shape} my={my_x.shape}")
                    continue
                d = (ler_x.float() - my_x.float()).abs()
                print(f"  {name}: shape {ler_x.shape}, ler norm {ler_x.norm():.4f}, my norm {my_x.norm():.4f}, diff max {d.max():.4e}, mean {d.mean():.4e}")
            # Attention output BEFORE o_proj — captured via forward_pre_hook
            # on o_proj. If Q, K, V all bit-identical but THIS diverges, the
            # softmax/matmul attention computation has a bug.
            ler_a = captured.get("ler_o_proj_in")
            my_a = captured.get("my_o_proj_in")
            if ler_a is not None and my_a is not None:
                d_a = (ler_a.float() - my_a.float()).abs()
                print(f"  ATTN_OUT (pre-o_proj): shape {ler_a.shape}, ler norm {ler_a.norm():.4f}, my norm {my_a.norm():.4f}, diff max {d_a.max():.4e}, mean {d_a.mean():.4e}")
                print(f"    sample[0, 0, :8]: ler {ler_a[0, 0, :8]}  my {my_a[0, 0, :8]}")
        finally:
            for h in sub_hooks:
                h.remove()

        # ─── Bisection: apply MY RoPE to LEROBOT's pre-RoPE q ───────
        # If my RoPE math matches stock, this should produce stock's q_post_rope
        # bit-identically. If not, my RoPE has a bug. `captured` was populated
        # by the sub-module hooks just above.
        sub_q_proj_ler = captured.get("ler_q_proj")
        if sub_q_proj_ler is not None and "q" in _saved_ler_eager_captures:
            ler_q_pre_rope = sub_q_proj_ler.view(1, 50, 8, 256).transpose(1, 2)
            my_rope = vla.vla_head.expert_stack.layers[0].rope
            my_applied_to_ler_q = my_rope.apply(ler_q_pre_rope.contiguous(), suffix_position_ids)
            ler_q_post_rope = _saved_ler_eager_captures["q"]
            bisect_diff = (my_applied_to_ler_q - ler_q_post_rope).abs()
            print(f"\n  --- RoPE bisection ---")
            print(f"  My RoPE applied to LER q_pre_rope vs LER q_post_rope:")
            print(f"    diff max {bisect_diff.max():.4e}  mean {bisect_diff.mean():.4e}")
            print(f"    pos_ids = {suffix_position_ids[0, :5]}...")
            if bisect_diff.max() < 1e-3:
                print(f"    → my RoPE BIT-IDENTICAL to stock; bug is elsewhere")
            else:
                print(f"    → my RoPE DIVERGES from stock at runtime")

        # ─── Direct cos/sin diff ───────────────────────────────────────
        # Extract cos/sin from LEROBOT's actual GemmaRotaryEmbedding instance
        # (which is what stock attention uses) and compare against my RoPE's.
        ler_rotary_emb = policy.model.paligemma_with_expert.gemma_expert.model.rotary_emb
        with torch.no_grad():
            # Stock returns (cos, sin) for given position_ids
            dummy_hidden = ler_q_pre_rope[:, 0]  # shape [B, S, head_dim]; only used for dtype/device
            ler_cos, ler_sin = ler_rotary_emb(dummy_hidden, suffix_position_ids)
        print(f"\n  --- Direct cos/sin comparison ---")
        print(f"  ler_cos shape {tuple(ler_cos.shape)}, norm {ler_cos.norm():.4f}")
        my_cos_runtime = my_rope.cos_cached[suffix_position_ids]  # [B, S, head_dim]
        my_sin_runtime = my_rope.sin_cached[suffix_position_ids]
        print(f"  my_cos shape {tuple(my_cos_runtime.shape)}, norm {my_cos_runtime.norm():.4f}")
        if ler_cos.shape == my_cos_runtime.shape:
            cd = (ler_cos - my_cos_runtime).abs()
            sd = (ler_sin - my_sin_runtime).abs()
            print(f"  cos diff max {cd.max():.4e} mean {cd.mean():.4e}")
            print(f"  sin diff max {sd.max():.4e} mean {sd.mean():.4e}")
            print(f"  ler_cos[0, 0, :8] = {ler_cos[0, 0, :8]}")
            print(f"  my_cos[0, 0, :8]  = {my_cos_runtime[0, 0, :8]}")
        # ler attention_scaling:
        print(f"  ler rotary_emb.attention_scaling = {ler_rotary_emb.attention_scaling}")
        print(f"  ler rotary_emb.rope_type = {ler_rotary_emb.rope_type}")
        print(f"  ler rotary_emb.inv_freq[:5] = {ler_rotary_emb.inv_freq[:5]}")
        print(f"  ler rotary_emb.inv_freq.shape = {ler_rotary_emb.inv_freq.shape}")
        print(f"  my rotary_emb.inv_freq[:5] = {my_rope.inv_freq[:5]}")
        print(f"  my rotary_emb.inv_freq.shape = {my_rope.inv_freq.shape}")
        cfg = policy.model.paligemma_with_expert.gemma_expert.model.config
        print(f"  ler config.head_dim = {cfg.head_dim}")
        print(f"  ler config attrs = {[a for a in dir(cfg) if 'rope' in a.lower() or 'theta' in a.lower() or 'partial' in a.lower()]}")

        del policy
        gc.collect()
    step("intermediate_parity", "pass", "see prints above")

    # ─── 6. Run Pi05VLA.predict_action ────────────────────────────────
    print("\n=== Step 6: Pi05VLA.predict_action ===", flush=True)
    start = time.time()
    with torch.no_grad():
        vla_actions = vla.predict_action(
            images=images,
            image_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            noise=noise.clone(),
            num_steps=num_steps,
            chunk_size=chunk_size,
            action_dim=action_dim,
        )
    vla_actions = vla_actions.cpu().numpy()
    step("vla_forward", "pass", f"{time.time() - start:.1f}s, shape={vla_actions.shape}, first={vla_actions[0, 0, :5]}")

    # ─── 7. Compare ────────────────────────────────────────────────────
    print("\n=== Step 7: Parity comparison ===", flush=True)
    # lerobot's predict_action_chunk unpads the action to action_dim (7 for
    # Franka). My Pi05VLA returns the padded max_action_dim (32). Trim mine
    # to the oracle's action_dim for direct comparison.
    if oracle_actions.shape[-1] != vla_actions.shape[-1]:
        trim_dim = oracle_actions.shape[-1]
        print(f"  Trimming vla actions from {vla_actions.shape[-1]} → {trim_dim} for parity (lerobot unpads)")
        vla_actions = vla_actions[..., :trim_dim]
    if oracle_actions.shape != vla_actions.shape:
        step("compare", "fail", f"shape mismatch after trim: oracle={oracle_actions.shape}, vla={vla_actions.shape}")
        return results

    diff = oracle_actions - vla_actions
    abs_diff = np.abs(diff).flatten()
    err_mean = float(abs_diff.mean())
    err_p50 = float(np.percentile(abs_diff, 50))
    err_p95 = float(np.percentile(abs_diff, 95))
    err_p99 = float(np.percentile(abs_diff, 99))
    err_max = float(abs_diff.max())

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

    diff_3d = oracle_actions - vla_actions
    per_pos_err = np.abs(diff_3d).mean(axis=(0, 2))
    per_feat_err = np.abs(diff_3d).mean(axis=(0, 1))
    print(f"\n  per-position err first 5: {per_pos_err[:5]}, argmax pos {per_pos_err.argmax()}")
    print(f"  per-feature err first 5:  {per_feat_err[:5]}, argmax feat {per_feat_err.argmax()}={per_feat_err.max():.4e}")
    print(f"  oracle action[0, 0, :8] = {oracle_actions[0, 0, :8]}")
    print(f"  vla    action[0, 0, :8] = {vla_actions[0, 0, :8]}")

    passed = (err_max < 1e-4) and (err_p95 < 1e-5)
    results["passed"] = passed
    if passed:
        step("VERDICT", "pass", f"max={err_max:.2e} < 1e-4 ✓, p95={err_p95:.2e} < 1e-5 ✓")
    else:
        step("VERDICT", "fail",
             f"max={err_max:.2e} (need < 1e-4), p95={err_p95:.2e} (need < 1e-5)")

    return results


@app.local_entrypoint()
def main(
    num_steps: int = 10,
    chunk_size: int = 50,
    noise_seed: int = 99,
    input_seed: int = 42,
):
    print(f"=== Pi05VLA vs lerobot PI05Policy parity gate ===")
    print(f"num_steps={num_steps}, chunk_size={chunk_size}, noise_seed={noise_seed}, input_seed={input_seed}\n")

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
        print(f"  cos  = {m['first_action_cos']:+.6f}")

    verdict = "PASS" if results.get("passed", False) else "FAIL"
    print(f"\nVerdict: {verdict}")
    sys.exit(0 if results.get("passed", False) else 1)
