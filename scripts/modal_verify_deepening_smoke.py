"""GPU smoke for the `tether verify` deepening (tap + MMD + embodied).

Validates on a REAL LIBERO rollout that:
1. the default-off `capture_trajectories` tap captures per-step applied actions +
   EEF positions (counts > 0),
2. the MMD/energy two-sample test computes on the real per-step action matrices
   (7-dim applied actions — identical layout for the native and Triton arms,
   unlike the model-internal chunk tensors, whose widths differ 7 vs 350),
3. the embodied parity (jerk / motion / completion) computes on the real EEF
   trajectories.

It runs the ORIGINAL (native pi05) and OPTIMIZED (Triton export of the same
weights) arms via `tether.verify.gather_paired_samples` — so the EXPECTED result
is parity (distributions_differ False), since the export preserves the policy.

Small N (1 task, a few episodes): this is a plumbing/tap validation, NOT a
statistically-powered cert (that needs >= 30 episodes through `tether verify`).

Reuses the PROVEN LIBERO+CUDA image from
`scripts/modal_fast_kernels_l3_side_by_side.py` verbatim, installing tether
from git @ local HEAD (which must be pushed). Run:

    ( sleep 2400 && modal app stop tether-verify-deepening-smoke ) &   # watchdog
    modal run scripts/modal_verify_deepening_smoke.py --n-episodes 4
"""
import os
import subprocess

import modal

app = modal.App("tether-verify-deepening-smoke")


def _repo_head_sha() -> str:
    pin = os.environ.get("TETHER_PIN_SHA", "").strip()
    if pin:
        return pin
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        # No git inside the Modal container; _HEAD is only used at build time
        # (the image is already built + cached), so any non-crashing value works.
        return "main"


_HEAD = _repo_head_sha()


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name("huggingface")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "ninja-build", "clang", "build-essential",
        "libgl1-mesa-glx", "libglib2.0-0", "libegl1-mesa", "libglvnd0", "ffmpeg",
        "cmake", "libosmesa6", "libosmesa6-dev",
        "gnupg", "wget",
    )
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
        " && dpkg -i cuda-keyring_1.1-1_all.deb"
        " && apt-get update"
        " && apt-get install -y cuda-toolkit-12-4 --no-install-recommends"
        " && rm cuda-keyring_1.1-1_all.deb",
    )
    .pip_install(
        "safetensors>=0.4.0", "huggingface_hub",
        "transformers<5.4,>=4.40",
        "numpy", "Pillow", "pydantic>=2.0", "pyyaml",
        "psutil", "typer", "rich",
        "triton>=3.1", "ninja",
        "mujoco==3.3.2", "robosuite==1.4.1",
        "h5py", "bddl==1.0.1", "future", "robomimic",
        "hydra-core>=1.1", "easydict", "einops",
        "opencv-python-headless", "gym", "gymnasium",
        "lerobot==0.5.1", "num2words", "imageio",
    )
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO"
        " && cd /opt/LIBERO && pip install . --no-deps"
    )
    .add_local_file("scripts/patch_libero.py", "/root/patch_libero.py", copy=True)
    .run_commands("python /root/patch_libero.py")
    .run_commands(
        f'pip install "fastcrest-tether @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "LIBERO_DATA_DIR": "/tmp/libero_data",
        "LIBERO_ASSET_DIR": "/opt/LIBERO/libero/libero/assets",
        "LIBERO_BASE": "/tmp/libero_data",
        "PYTHONPATH": "/opt/LIBERO",
    })
    .run_commands("mkdir -p /tmp/libero_data")
)


@app.function(image=image, gpu="A100-40GB", timeout=1800, secrets=[_hf_secret()])
def validate(model_id: str, n_episodes: int, task_idx: int) -> dict:
    import json

    import numpy as np
    import torch

    # lerobot checkpoints pickle with weights_only=False.
    _orig_load = torch.load
    def _patched_load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)
    torch.load = _patched_load

    from tether.verify import (
        _collect_eef_and_steps,
        _collect_step_actions,
        gather_paired_samples,
    )
    from tether.verify_metrics import aggregate_embodied, two_sample_test

    orig, opt = gather_paired_samples(
        optimized_ref=model_id,
        original_ref=None,
        suite="libero",
        task_suite_name="libero_10",
        num_episodes=n_episodes,
        task_indices=[task_idx],
        seed=7,
    )

    def _counts(res):
        act = sum(len(ep.get("actions", []) or [])
                  for tk in res.get("per_task", []) for ep in tk.get("episodes", []))
        eef = sum(len(ep.get("eef_positions", []) or [])
                  for tk in res.get("per_task", []) for ep in tk.get("episodes", []))
        return act, eef

    o_act, o_eef = _counts(orig)
    c_act, c_eef = _counts(opt)
    base_actions, base_groups = _collect_step_actions(orig)
    cand_actions, cand_groups = _collect_step_actions(opt)
    ts = None
    if (
        base_actions.size
        and cand_actions.size
        and base_actions.shape[1] == cand_actions.shape[1]
    ):
        ts = two_sample_test(
            base_actions, cand_actions,
            baseline_groups=base_groups, candidate_groups=cand_groups,
        ).to_dict()

    bp, bs = _collect_eef_and_steps(orig)
    cp, cs = _collect_eef_and_steps(opt)
    emb = None
    if bp and cp:
        emb = aggregate_embodied(
            baseline_positions=bp, candidate_positions=cp,
            baseline_velocities=[np.diff(p, axis=0) for p in bp],
            candidate_velocities=[np.diff(p, axis=0) for p in cp],
            baseline_completion_steps=bs, candidate_completion_steps=cs,
        ).to_dict()

    out = {
        "model_id": model_id,
        "n_episodes": n_episodes,
        "task_idx": task_idx,
        "tap_captured": {
            "orig_actions": o_act, "orig_eef_steps": o_eef,
            "opt_actions": c_act, "opt_eef_steps": c_eef,
        },
        "action_matrix_shapes": {"baseline": list(base_actions.shape), "candidate": list(cand_actions.shape)},
        "two_sample": ts,
        "embodied": emb,
    }
    print("VALIDATION_RESULT " + json.dumps(out))
    return out


@app.local_entrypoint()
def main(
    n_episodes: int = 4,
    task_idx: int = 0,
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
):
    import json
    res = validate.remote(model_id=model_id, n_episodes=n_episodes, task_idx=task_idx)
    print(json.dumps(res, indent=2))
