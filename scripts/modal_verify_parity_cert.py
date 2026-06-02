"""GPU parity CERT for `reflex verify`: native pi05 vs Triton export.

The N=4 smoke (modal_verify_deepening_smoke.py) validated the gate MECHANICS and
caught two bugs. This is the statistically meaningful verdict: does the Triton/bf16
export behave equivalently to native fp32 pi05 on real LIBERO rollouts, judged by
the *fixed* episode-block MMD gate (PR #201) + embodied parity?

N=30 episodes/arm × 1 task = the gate's design floor (>=30). Runs the ORIGINAL
(native) and OPTIMIZED (Triton export of the same weights) arms via
`reflex.verify.gather_paired_samples`, then the calibrated episode-block two-sample
test + embodied parity. Installs reflex-vla from git @ local HEAD (must be pushed;
should be main with the #201 fix).

This is a ~2-hour single-container run (native arm ~1 hr). It therefore sets a long
function timeout, prints a 60s heartbeat so timing pressure can't hide, and MUST be
launched with a wall-clock watchdog:

    ( sleep 11400 && MODAL_PROFILE=romirj-16723 modal app stop <app-id> ) &   # 3h10m cap
    MODAL_PROFILE=romirj-16723 modal run scripts/modal_verify_parity_cert.py --n-episodes 30
"""
import os
import subprocess

import modal

app = modal.App("reflex-verify-parity-cert")

# Persist raw per-episode actions/eef/success so future verdict-logic experiments
# (conditioning, parity_pass thresholds) run OFFLINE on this data — no GPU re-run.
# Retrieve: modal volume get reflex-verify-cert-data <file> .
cert_data_vol = modal.Volume.from_name("reflex-verify-cert-data", create_if_missing=True)


def _repo_head_sha() -> str:
    pin = os.environ.get("REFLEX_PIN_SHA", "").strip()
    if pin:
        return pin
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        # No git inside the container; _HEAD is build-time only (image cached).
        return "main"


_HEAD = _repo_head_sha()


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name("huggingface")


# Same proven LIBERO+CUDA image as the smoke / L3 harness.
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
        f'pip install "reflex-vla @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla@{_HEAD}"',
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


@app.function(image=image, gpu="A100-40GB", timeout=10800, secrets=[_hf_secret()],
              volumes={"/data": cert_data_vol})
def cert(model_id: str, n_episodes: int, task_idx: int) -> dict:
    import json
    import threading
    import time

    import numpy as np
    import torch

    # lerobot checkpoints pickle with weights_only=False.
    _orig_load = torch.load
    def _patched_load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)
    torch.load = _patched_load

    # Heartbeat: a 60s elapsed tick so a stalled/slow arm can't hide timing
    # pressure behind sparse per-episode prints (the 2026-04-30 $44 lesson).
    _start = time.time()
    _stop = threading.Event()

    def _hb():
        while not _stop.wait(60):
            print(f"[HEARTBEAT] elapsed={int(time.time() - _start)}s", flush=True)

    threading.Thread(target=_hb, daemon=True).start()

    from reflex.verify import (
        _collect_eef_and_steps,
        _collect_paired_succeeded_step_actions,
        gather_paired_samples,
    )
    from reflex.verify_metrics import aggregate_embodied, two_sample_test

    orig, opt = gather_paired_samples(
        optimized_ref=model_id,
        original_ref=None,
        suite="libero",
        task_suite_name="libero_10",
        num_episodes=n_episodes,
        task_indices=[task_idx],
        seed=7,
    )
    _stop.set()

    # Persist the raw per-episode results (actions/eef/success) so any future
    # verdict-logic experiment runs OFFLINE — this is the LAST GPU run we should
    # need to tune conditioning / parity_pass thresholds.
    # Best-effort: a persistence failure must NEVER lose the verdict after a ~2h
    # run. Wrap it; default=str makes json robust to any stray numpy scalar.
    raw_path = f"/data/cert_raw_n{n_episodes}_task{task_idx}.json"
    try:
        with open(raw_path, "w") as fh:
            json.dump({"original": orig, "optimized": opt}, fh, default=str)
        cert_data_vol.commit()
        print(f"[persist] raw results -> {raw_path}", flush=True)
    except Exception as exc:  # noqa: BLE001 — never fatal
        raw_path = None
        print(f"[persist] WARNING raw-results dump failed (non-fatal): {exc}", flush=True)

    def _succ(res):
        s = sum(tk.get("success", 0) for tk in res.get("per_task", []))
        t = sum(tk.get("total", 0) for tk in res.get("per_task", []))
        return s, t

    o_s, o_t = _succ(orig)
    c_s, c_t = _succ(opt)

    # Outcome-conditioned: compare per-step actions only on episodes BOTH arms
    # succeeded — isolates the policy shift from the outcome shift (an arm that
    # fails more injects flailing actions that differ for a non-policy reason).
    base_actions, base_groups, cand_actions, cand_groups, n_common = (
        _collect_paired_succeeded_step_actions(orig, opt)
    )

    # Thin each episode to <= STEP_CAP steps (uniform stride) before the test.
    # The full N=30 pooled matrix is ~18k rows => a ~2.6 GB (N,N) kernel + slow
    # per-permutation indexing; thinning keeps the episode-block structure, cuts
    # autocorrelation further, and leaves ~60 x 64 = ample samples. Standard
    # time-series thinning, not a shortcut that changes the verdict.
    STEP_CAP = 64

    def _thin(acts, grps):
        if acts.size == 0:
            return acts, grps
        rows, gs = [], []
        for g in np.unique(grps):
            idx = np.flatnonzero(grps == g)
            sel = (
                idx[np.linspace(0, idx.size - 1, STEP_CAP).astype(int)]
                if idx.size > STEP_CAP else idx
            )
            rows.append(acts[sel])
            gs.append(grps[sel])
        return np.vstack(rows), np.concatenate(gs)

    base_actions, base_groups = _thin(base_actions, base_groups)
    cand_actions, cand_groups = _thin(cand_actions, cand_groups)

    ts = None
    if (
        base_actions.size
        and cand_actions.size
        and base_actions.shape[1] == cand_actions.shape[1]
    ):
        ts = two_sample_test(
            base_actions, cand_actions, n_permutations=500,
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

    embodied_ok = emb is not None and not emb["embodied_regressed"]
    cand_better_or_equal = (c_s / c_t if c_t else 0.0) >= (o_s / o_t if o_t else 0.0)
    # Rich verdict: a "different but better" export (>= success, no embodied
    # regression) is legible at a glance, even when the distributional test flags
    # a shift. parity_pass stays STRICT for now (any distributional differ =>
    # fail); the effect-size/direction-aware threshold is deferred pending a
    # design-partner signal that defines what "parity" means commercially.
    candidate_not_worse = cand_better_or_equal and embodied_ok
    verdict_pass = (
        ts is not None
        and not ts["distributions_differ"]
        and embodied_ok
    )

    out = {
        "model_id": model_id,
        "n_episodes": n_episodes,
        "task_idx": task_idx,
        "elapsed_s": int(time.time() - _start),
        "success": {
            "native": f"{o_s}/{o_t}", "optimized": f"{c_s}/{c_t}",
            "native_rate": (o_s / o_t) if o_t else None,
            "optimized_rate": (c_s / c_t) if c_t else None,
        },
        "two_sample_conditioned_on_episodes": int(n_common),  # both arms succeeded
        "step_cap_per_episode": STEP_CAP,
        "action_matrix_shapes": {
            "baseline": list(base_actions.shape),
            "candidate": list(cand_actions.shape),
        },
        "two_sample": ts,
        "embodied": emb,
        "candidate_not_worse": candidate_not_worse,
        "parity_pass": verdict_pass,
        "raw_results_volume": "reflex-verify-cert-data",
        "raw_results_path": raw_path,
    }
    # Persist the VERDICT to the Volume — this is how we recover the result when
    # the run is spawn()ed and the client is long gone (laptop sleep). Never fatal.
    verdict_path = f"/data/cert_verdict_n{n_episodes}_task{task_idx}.json"
    try:
        with open(verdict_path, "w") as fh:
            json.dump(out, fh, default=str)
        cert_data_vol.commit()
        print(f"[persist] verdict -> {verdict_path}", flush=True)
    except Exception as exc:  # noqa: BLE001 — never fatal
        print(f"[persist] WARNING verdict dump failed (non-fatal): {exc}", flush=True)
    print("CERT_RESULT " + json.dumps(out), flush=True)
    return out


@app.local_entrypoint()
def main(n_episodes: int = 30, task_idx: int = 0,
         model_id: str = "lerobot/pi05_libero_finetuned_v044"):
    # SPAWN (not .remote): the function runs server-side INDEPENDENT of this
    # client, so a laptop sleep / lid-close can't cancel it (the failure mode that
    # killed 3 prior --detach runs — --detach keeps the app alive but the blocking
    # .remote() input dies with the client). Fire-and-exit; the verdict is written
    # to the Volume. Recover with:
    #   modal volume get reflex-verify-cert-data cert_verdict_n<N>_task<T>.json -
    call = cert.spawn(model_id, n_episodes, task_idx)
    print(f"SPAWNED call_id={call.object_id}")
    print(f"verdict will land at Volume reflex-verify-cert-data:/"
          f"cert_verdict_n{n_episodes}_task{task_idx}.json (~70 min)")
