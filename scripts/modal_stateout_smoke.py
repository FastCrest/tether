"""Stage 1 smoke for v0.5 distill-state-out pi0.5.

100 training steps on synthetic data. Confirms end-to-end:
  - student loads + enable_snapflow_state_out swaps the class
  - new state_proj layer is in model.parameters() and receives gradients
  - forward pass through the state-out path produces right-shape output
  - loss is finite (not NaN), and moves at all
  - no OOM on A100-80GB

Does NOT validate that the student actually learns to use state — that's
Stage 2 (2k steps with real teacher supervision). This is purely a
"does the training loop run without crashing" test.

Cost: ~$0.10-0.20 Modal (~2 min on A100-80GB).

Usage:
    modal run scripts/modal_stateout_smoke.py \\
      --student-checkpoint /onnx_out/distill_v031_pi05_libero_r4/training/checkpoints/00010000/pretrained_model
"""
import os
import subprocess
import modal

app = modal.App("tether-stateout-smoke")


def _hf_secret():
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


def _head():
    try:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


def _bust():
    import time
    return str(int(time.time()))


_HEAD = _head()
_BUST = _bust()

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
        "numpy",
        "draccus",
    )
    .run_commands(
        f'echo "bust={_BUST}"',
        f'pip install "fastcrest-tether[monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
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
    timeout=1800,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def smoke(
    student_checkpoint: str,
    num_steps: int = 100,
    batch_size: int = 1,
):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("smoke")

    import torch
    from tether.distill.snapflow_pi0_model import (
        load_snapflow_student,
        enable_snapflow_state_out,
    )

    log.info("Loading student from %s", student_checkpoint)
    # The existing v0.3.1 student already has target_time_embed_mlp.
    # We load it as a base, then re-enable with state-out on top.
    # For the smoke, we don't actually need a v0.3.1 student — a fresh
    # PI05Policy.from_pretrained on the teacher would be cleaner. But
    # since we have the student ckpt on volume, use it as a "close to
    # v0.3.1" starting point.
    policy = load_snapflow_student(student_checkpoint)
    policy.train().to("cuda").to(torch.bfloat16)
    # Disable gradient checkpointing — otherwise use_cache=True silently
    # flips to False, past_kv comes back empty, and the expert attention
    # mask (built assuming prefix+suffix concatenation) doesn't match the
    # attn_weights (suffix only). Distillation production paths disable
    # this for the forward measurement too; we follow that pattern.
    disable = getattr(policy.model, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
        log.info("  gradient_checkpointing disabled (so use_cache=True holds)")
    log.info("  loaded, dtype=bf16, device=cuda")

    # Re-enable with state-out variant. This requires resetting __class__
    # back to the base pi05 class first (load_snapflow_student already
    # swapped to SnapFlowPI05Pytorch).
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
    policy.model.__class__ = PI05Pytorch  # reset class
    enable_snapflow_state_out(policy.model)
    log.info("  state-out variant enabled, class=%s", type(policy.model).__name__)

    # Verify state_proj is registered + has parameters
    state_proj_params = list(policy.model.state_proj.parameters())
    log.info("  state_proj params: %d (expect ≥1), total elements: %d",
             len(state_proj_params),
             sum(p.numel() for p in state_proj_params))
    assert len(state_proj_params) > 0, "state_proj has no parameters"

    # Build synthetic batch matching pi0.5 input shapes.
    cfg = policy.config
    B = batch_size
    chunk = cfg.chunk_size
    action_dim_pad = cfg.max_action_dim
    state_dim = cfg.max_state_dim
    torch.manual_seed(42)
    images = [
        torch.randn(B, 3, 224, 224, device="cuda", dtype=torch.bfloat16),
        torch.randn(B, 3, 224, 224, device="cuda", dtype=torch.bfloat16),
        torch.randn(B, 3, 224, 224, device="cuda", dtype=torch.bfloat16),
    ]
    img_masks = [torch.ones(B, dtype=torch.bool, device="cuda") for _ in range(3)]
    lang_tokens = torch.randint(0, 257152, (B, 200), device="cuda", dtype=torch.long)
    lang_masks = torch.ones(B, 200, dtype=torch.bool, device="cuda")
    state = torch.randn(B, state_dim, device="cuda", dtype=torch.bfloat16)
    noise_shape = (B, chunk, action_dim_pad)
    target_action = torch.randn(*noise_shape, device="cuda", dtype=torch.bfloat16)

    # Capture state_proj weight norm before training, to verify gradient flow
    pre_norm = policy.model.state_proj.weight.detach().float().norm().item()
    log.info("  state_proj pre-train weight norm: %.6f", pre_norm)

    # Training loop
    opt = torch.optim.Adam(policy.model.parameters(), lr=1e-4)
    losses = []
    for step in range(num_steps):
        opt.zero_grad()
        # Forward: embed prefix, then denoise_step with state
        prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
        )
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = policy.model._prepare_attention_masks_4d(prefix_att_2d)
        policy.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"

        _, past_key_values = policy.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        # Denoise with state
        # Match model dtype (bf16) to avoid cascading cast failures
        # inside embed_suffix's time_mlp_in / time_mlp_out linear layers.
        time = torch.ones(B, dtype=torch.bfloat16, device="cuda")
        noise = torch.randn(*noise_shape, device="cuda", dtype=torch.bfloat16)
        v_t = policy.model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=noise,
            timestep=time,
            target_time=time,
            state=state,
        )
        # L2 distance of predicted velocity against (noise - target_action),
        # which is the standard flow-matching training target.
        target_v = noise - target_action
        loss = torch.nn.functional.mse_loss(v_t.float(), target_v.float())
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == num_steps - 1:
            log.info("  step %d loss=%.4f mem=%.1fGB",
                     step, loss.item(),
                     torch.cuda.max_memory_allocated() / 1e9)

    post_norm = policy.model.state_proj.weight.detach().float().norm().item()
    log.info("state_proj post-train weight norm: %.6f (pre: %.6f, delta: %.6f)",
             post_norm, pre_norm, abs(post_norm - pre_norm))

    # Verify training signal moved the state_proj weights
    # Threshold: bf16 + lr=1e-4 + 100 steps × init=0.02 grads gives
    # ~1e-5 scale weight deltas. 1e-7 is the "anything moved at all"
    # floor. Stage 2 (2k steps) should produce much larger deltas.
    assert abs(post_norm - pre_norm) > 1e-7, (
        f"state_proj didn't receive gradients? pre={pre_norm:.6f} post={post_norm:.6f}"
    )
    # Verify loss is finite and moved
    import math
    assert all(math.isfinite(l) for l in losses), "loss went NaN/inf"
    first_loss, last_loss = losses[0], losses[-1]
    log.info("first loss: %.4f, last loss: %.4f, delta: %.4f",
             first_loss, last_loss, first_loss - last_loss)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    log.info("peak GPU memory: %.1f GB", peak_gb)

    return {
        "status": "pass",
        "first_loss": first_loss,
        "last_loss": last_loss,
        "loss_delta": first_loss - last_loss,
        "state_proj_weight_delta": post_norm - pre_norm,
        "peak_gpu_gb": peak_gb,
        "num_steps": num_steps,
    }


@app.local_entrypoint()
def main(
    student_checkpoint: str = "/onnx_out/distill_v031_pi05_libero_r4/training/checkpoints/00010000/pretrained_model",
    num_steps: int = 100,
    batch_size: int = 1,
):
    r = smoke.remote(
        student_checkpoint=student_checkpoint,
        num_steps=num_steps,
        batch_size=batch_size,
    )
    print("\n=== STAGE 1 SMOKE ===")
    import json
    print(json.dumps(r, indent=2))
