"""v0.5 Stage 2: 2k-step distill on real LeRobotDataset for state-out student.

Stage 1 (100-step synthetic) confirmed the architecture runs end-to-end.
Stage 2 validates the student actually LEARNS to use state_vec when
lang has no state — by training against real teacher supervision from
demonstrations.

Setup:
- Teacher: pi05_libero_finetuned_v044, state-in-lang preprocessor,
  frozen, produces target actions on each batch.
- Student: v0.3.1 SnapFlow student + enable_snapflow_state_out +
  state-out preprocessor (no state in lang).
- Loss: MSE between student velocity and (noise - demo_action) at
  target_time=1 — SnapFlow consistency loss. Teacher output is the
  gold, demos are what teacher was trained on.

Success criteria (go/no-go for Stage 3):
- Loss drops ≥3× from initial to step 2000
- state_proj.weight.norm delta ≥ 1e-3 (real learning, not noise)
- No OOM, no NaN

Cost: ~$2 on A100-80GB.

Usage:
    modal run scripts/modal_stateout_distill_stage2.py \\
      --teacher-repo lerobot/pi05_libero_finetuned_v044 \\
      --dataset lerobot/libero \\
      --num-steps 2000
"""
import os
import subprocess
import modal

app = modal.App("tether-stateout-distill-stage2")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name("huggingface")


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
    .apt_install("git", "ffmpeg")
    .pip_install(
        "lerobot==0.5.1",
        "transformers==5.3.0",
        "num2words",
        "safetensors>=0.4.0",
        "numpy",
        "draccus",
        "datasets",
    )
    .run_commands(
        f'echo "bust={_BUST}"',
        f'pip install "tether[monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether-vla@{_HEAD}"',
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
    timeout=7200,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def distill(
    teacher_repo: str = "lerobot/pi05_libero_finetuned_v044",
    student_checkpoint: str = "/onnx_out/distill_v031_pi05_libero_r4/training/checkpoints/00010000/pretrained_model",
    dataset: str = "lerobot/libero",
    num_steps: int = 2000,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    log_every: int = 50,
):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("stage2")

    import torch
    import torch.nn.functional as F
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch, make_att_2d_masks
    from tether.distill.snapflow_pi0_model import (
        load_snapflow_student,
        enable_snapflow_state_out,
    )
    from tether.distill.pi05_state_out_processor import (
        make_pi05_state_out_preprocessor,
    )
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
    from lerobot.utils.constants import (
        OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE, ACTION,
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # ─── Teacher ──────────────────────────────────────────────────────
    log.info("Loading TEACHER: %s", teacher_repo)
    # Suppress torch.compile + grad-ckpt interference during load
    _orig_compile = torch.compile
    torch.compile = lambda fn=None, *a, **kw: (fn if fn is not None else (lambda f: f))
    try:
        teacher = PI05Policy.from_pretrained(teacher_repo)
    finally:
        torch.compile = _orig_compile
    teacher.eval().to("cuda").to(torch.bfloat16)
    gc_disable = getattr(teacher.model, "gradient_checkpointing_disable", None)
    if callable(gc_disable):
        gc_disable()
    for p in teacher.model.parameters():
        p.requires_grad = False
    log.info("  teacher loaded + frozen")

    # ─── Student ──────────────────────────────────────────────────────
    log.info("Loading STUDENT from %s", student_checkpoint)
    student = load_snapflow_student(student_checkpoint)
    student.train().to("cuda").to(torch.bfloat16)
    # Reset class before enabling state-out (load_snapflow_student swapped
    # to SnapFlowPI05Pytorch; state-out subclass expects the parent chain)
    student.model.__class__ = PI05Pytorch
    enable_snapflow_state_out(student.model)
    gc_disable = getattr(student.model, "gradient_checkpointing_disable", None)
    if callable(gc_disable):
        gc_disable()
    log.info("  student: %s", type(student.model).__name__)

    # ─── Preprocessors (same config, different state handling) ────────
    from huggingface_hub import snapshot_download
    teacher_dir = snapshot_download(teacher_repo)
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    # Teacher: default pi0.5 preprocessor (state in lang)
    teacher_preproc = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=teacher_dir,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cuda"}},
    )
    # Student: state-out preprocessor (state NOT in lang)
    student_preproc = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=teacher_dir,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cuda"}},
    )
    from tether.distill.pi05_state_out_processor import swap_prepare_step_in_pipeline
    swap_prepare_step_in_pipeline(student_preproc, max_state_dim=teacher.config.max_state_dim)
    log.info("  preprocessors ready (teacher=state-in-lang, student=state-out)")

    # ─── Dataset ──────────────────────────────────────────────────────
    log.info("Loading dataset: %s", dataset)
    cfg = teacher.config
    delta_timestamps = {
        ACTION: [i / 10 for i in range(cfg.chunk_size)],  # 10 Hz approx
    }
    ds = LeRobotDataset(dataset, delta_timestamps=delta_timestamps)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    log.info("  dataset: %d samples", len(ds))

    # ─── Optimizer ────────────────────────────────────────────────────
    opt = torch.optim.Adam(
        [p for p in student.model.parameters() if p.requires_grad],
        lr=learning_rate,
    )
    pre_norm = student.model.state_proj.weight.detach().float().norm().item()
    log.info("state_proj pre-train weight norm: %.6f", pre_norm)

    # ─── Training loop ────────────────────────────────────────────────
    step = 0
    loss_history = []
    import itertools
    batch_iter = itertools.cycle(loader)
    while step < num_steps:
        batch = next(batch_iter)
        batch = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in batch.items()}

        # Teacher forward (no grad, state-in-lang)
        teacher_batch = teacher_preproc(batch)
        with torch.no_grad():
            teacher_actions = teacher.model.sample_actions(
                [teacher_batch["observation.images.image"],
                 teacher_batch.get("observation.images.image2",
                                   teacher_batch["observation.images.image"]),
                 teacher_batch.get("observation.images.image3",
                                   teacher_batch["observation.images.image"])],
                [torch.ones(batch_size, dtype=torch.bool, device="cuda")] * 3,
                teacher_batch[OBS_LANGUAGE_TOKENS],
                teacher_batch[OBS_LANGUAGE_ATTENTION_MASK],
            )

        # Student: state-out preprocessing
        student_batch = student_preproc(batch)
        lang_tokens = student_batch[OBS_LANGUAGE_TOKENS]
        lang_masks = student_batch[OBS_LANGUAGE_ATTENTION_MASK]
        state_vec = student_batch[OBS_STATE]

        # Flow-matching training step: predict velocity at target_time=1
        opt.zero_grad()
        noise = torch.randn_like(teacher_actions)
        time = torch.ones(batch_size, dtype=torch.bfloat16, device="cuda")

        # Student forward (with state)
        images = [student_batch["observation.images.image"],
                  student_batch.get("observation.images.image2",
                                    student_batch["observation.images.image"]),
                  student_batch.get("observation.images.image3",
                                    student_batch["observation.images.image"])]
        img_masks = [torch.ones(batch_size, dtype=torch.bool, device="cuda")] * 3
        prefix_embs, prefix_pad_masks, prefix_att_masks = student.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = student.model._prepare_attention_masks_4d(prefix_att_2d)
        student.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, past_key_values = student.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        predicted_velocity = student.model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=noise,
            timestep=time,
            target_time=time,
            state=state_vec,
        )

        # SnapFlow loss: velocity should match (noise - teacher_actions)
        target_v = noise - teacher_actions.to(noise.dtype)
        loss = F.mse_loss(predicted_velocity.float(), target_v.float())
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        if step % log_every == 0 or step == num_steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1e9
            log.info("step %4d  loss=%.4f  mem=%.1fGB", step, loss.item(), mem)
        step += 1

    post_norm = student.model.state_proj.weight.detach().float().norm().item()
    log.info("state_proj post-train norm: %.6f (delta: %+.6f)", post_norm, post_norm - pre_norm)
    log.info("first 5 losses: %s", [f"{l:.3f}" for l in loss_history[:5]])
    log.info("last 5 losses:  %s", [f"{l:.3f}" for l in loss_history[-5:]])

    first_avg = sum(loss_history[:20]) / 20
    last_avg = sum(loss_history[-20:]) / 20
    ratio = first_avg / max(last_avg, 1e-8)
    log.info("loss reduction: %.2fx (first-20 avg %.3f -> last-20 avg %.3f)", ratio, first_avg, last_avg)

    return {
        "status": "ok",
        "num_steps": num_steps,
        "first_20_avg_loss": first_avg,
        "last_20_avg_loss": last_avg,
        "loss_reduction_x": ratio,
        "state_proj_pre_norm": pre_norm,
        "state_proj_post_norm": post_norm,
        "state_proj_delta": post_norm - pre_norm,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


@app.local_entrypoint()
def main(
    teacher_repo: str = "lerobot/pi05_libero_finetuned_v044",
    student_checkpoint: str = "/onnx_out/distill_v031_pi05_libero_r4/training/checkpoints/00010000/pretrained_model",
    dataset: str = "lerobot/libero",
    num_steps: int = 2000,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
):
    r = distill.remote(
        teacher_repo=teacher_repo,
        student_checkpoint=student_checkpoint,
        dataset=dataset,
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    print("\n=== STAGE 2 RESULT ===")
    import json
    print(json.dumps(r, indent=2))
