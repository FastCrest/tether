"""Lift #5 L3 side-by-side — native lerobot vs Triton on the SAME proven loop.

Quality-parity gate for the `--fast-kernels` Triton path. Runs BOTH arms through
`reflex.eval.libero_rollout.run_libero_rollout` — the exact primitive the proven
native eval (`modal_libero_lerobot_native.py`, 80%+ on libero_10 task 0 at N=20)
uses — so the ONLY thing that differs between arms is the inference backend:

  ARM A (native): use_native=True  → policy.select_action (proven path)
  ARM B (triton): inference=TritonLIBEROAdapter, use_native=False

Identical preprocessing (cv2 INTER_AREA resize + 180° flip), identical seed,
identical task set, identical centrally-generated noise, identical bool(done)
success criterion. This REPLACES the earlier bespoke loop whose PIL-BILINEAR
resize + seed-42 + hard-task-set [3,4,6] produced a spurious 0/9 "baseline"
(see 03_experiments/2026-05-24-lift5-l3-sbs-baseline-zero.md). There was never
a proven native run at that config to compare against.

Usage:
    modal profile activate romirj-16723
    # FORMAL kill-trigger-3 gate (default): N=100/task × 3 tasks × both arms,
    # sharded into 6 parallel A100 cells (~3 hr wall = slowest native cell,
    # retryable per cell; 4 hr per-cell timeout):
    modal run scripts/modal_fast_kernels_l3_side_by_side.py
    # cheap plumbing smoke (~$0.7): N=2, task 0, both arms, sharded path:
    modal run scripts/modal_fast_kernels_l3_side_by_side.py --smoke
    # legacy single-container directional run (no shard, capped by 2 hr timeout):
    modal run scripts/modal_fast_kernels_l3_side_by_side.py --no-shard \
        --num-episodes 10 --task-indices 0,1,2

The formal gate writes its full result dict to $L3_RESULT_JSON if that env var
is set, so the monthly launchd runner (~/_gate_l3_parity_monthly.py) can read the
verdict without scraping stdout. kill_trigger_3_fires := native_rate − triton_rate
> 5.0 pp (see 01_decisions/2026-05-20-fast-kernels-kill-triggers.md, Trigger 3).
"""
import os
import subprocess

import modal

app = modal.App("reflex-fast-kernels-l3-side-by-side")


def _repo_head_sha() -> str:
    # REFLEX_PIN_SHA lets the monthly launchd runner pin origin/main explicitly,
    # so the gate always tests SHIPPED code regardless of the working-tree branch
    # (and without clobbering it). Unset → local HEAD, today's dev behavior.
    pin = os.environ.get("REFLEX_PIN_SHA", "").strip()
    if pin:
        return pin
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,  # quiet the expected "not a git repo" inside containers
        ).decode().strip()[:12]
    except Exception:
        return "lift/5-day1-2-vendor-triton-kernels"


_HEAD = _repo_head_sha()


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


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


def _run_one_arm(
    arm: str,
    model_id: str,
    task_suite_name: str,
    task_indices: list[int],
    num_episodes: int,
    seed: int,
) -> dict:
    """Run ONE arm (native|triton) over task_indices on the shared proven loop.

    Plain helper (not a Modal function) so it can run either in its own sharded
    container (run_cell) or twice inside one container (legacy run_side_by_side).
    Loads its own policy + processors, runs the rollout, then frees GPU memory so
    the legacy single-container path can run a second arm without 2× resident.

    The ONLY thing differing between arms is the inference backend; preprocessing,
    seed, task set and success criterion are identical (see module docstring).
    """
    import time

    import torch

    # PyTorch 2.6+ defaults torch.load to weights_only=True; LIBERO init-state
    # pickles need weights_only=False. (run_libero_rollout patches this too, but
    # the policy load below happens first.)
    _orig_torch_load = torch.load

    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _compat_load

    # PyTorch Inductor autotuner blocks the GPU 30+s per matmul shape on first
    # call → Modal kills the task for "failed to respond to cancellation".
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True

    print(
        f"[arm:{arm}] suite={task_suite_name} tasks={task_indices} "
        f"N={num_episodes} seed={seed}",
        flush=True,
    )
    print(f"[arm:{arm}] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    t0 = time.time()

    from reflex.eval.libero_rollout import run_libero_rollout

    # ── Load policy (fp32 cuda — native baseline quality + shared preprocessing)
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained(model_id).to(dtype=torch.float32).to("cuda")
    policy.eval()
    print(f"[arm:{arm}] [{time.time()-t0:.1f}s] PI05Policy loaded (cuda fp32)", flush=True)

    # ── Pre/post processors ───────────────────────────────────────────
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from huggingface_hub import snapshot_download
    repo_dir = snapshot_download(repo_id=model_id)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_preprocessor.json",
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo_dir,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    print(f"[arm:{arm}] pre/post processors loaded", flush=True)

    common = dict(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task_suite_name=task_suite_name,
        num_episodes=num_episodes,
        task_indices=task_indices,
        seed=seed,
        replan_steps=5,
        num_steps_wait=10,
    )

    adapter = None
    if arm == "native":
        print(f"\n[arm:{arm}] native lerobot (select_action, fp32)", flush=True)
        res = run_libero_rollout(
            inference=None, use_native=True, label="NATIVE", **common,
        )
    elif arm == "triton":
        print(f"\n[arm:{arm}] Triton fast-kernels (Pi05FastKernelsInference)", flush=True)
        from reflex.runtime.fast_inference.libero_adapter import TritonLIBEROAdapter
        adapter = TritonLIBEROAdapter.from_policy(policy, capture=True)
        res = run_libero_rollout(
            inference=adapter, use_native=False, label="TRITON", **common,
        )
    else:
        raise ValueError(f"unknown arm {arm!r} (expected 'native' | 'triton')")

    res["arm"] = arm
    res["task_indices_run"] = list(task_indices)
    res["cell_seconds"] = time.time() - t0
    print(
        f"[arm:{arm}] {res['total_success']}/{res['total_eps']} "
        f"({res.get('success_rate_pct', 0.0):.1f}%) in {res['cell_seconds']:.1f}s",
        flush=True,
    )

    # Free GPU memory so the legacy single-container path can run arm B after A.
    del adapter
    del policy
    torch.cuda.empty_cache()
    return res


def _merge_arm(cell_results: list[dict]) -> dict:
    """Merge per-(arm,task) sharded cells into one arm-level result dict.

    Shape-compatible with a single _run_one_arm() multi-task result so _assemble
    and the verdict printing don't care whether the run was sharded or not.
    """
    if not cell_results:
        return {}
    arm = cell_results[0].get("arm", "")
    per_task: list = []
    total_success = 0
    total_eps = 0
    cell_seconds = 0.0
    for c in cell_results:
        per_task.extend(c.get("per_task", []))
        total_success += int(c.get("total_success", 0))
        total_eps += int(c.get("total_eps", 0))
        cell_seconds += float(c.get("cell_seconds", 0.0))
    rate = (100.0 * total_success / total_eps) if total_eps else 0.0
    return {
        "arm": arm,
        "per_task": per_task,
        "total_success": total_success,
        "total_eps": total_eps,
        "success_rate_pct": rate,
        "cell_seconds": cell_seconds,
    }


def _assemble(
    native: dict | None,
    triton: dict | None,
    task_indices: list[int],
    num_episodes: int,
    seed: int,
    arms: str,
) -> dict:
    """Build the comparison dict + kill-trigger-3 verdict from arm-level results.

    kill_trigger_3_fires := native_rate − triton_rate > 5.0pp  (a >5pp Triton
    REGRESSION vs native fires the gate — see Trigger 3 in the kill-trigger ADR).
    """
    out: dict = {
        "native": native,
        "triton": triton,
        "task_indices": task_indices,
        "num_episodes": num_episodes,
        "seed": seed,
        "arms": arms,
    }
    if native is not None:
        out["native_rate_pct"] = native.get("success_rate_pct", 0.0)
    if triton is not None:
        out["triton_rate_pct"] = triton.get("success_rate_pct", 0.0)
    if native is not None and triton is not None:
        nmt = out["native_rate_pct"] - out["triton_rate_pct"]
        out["delta_pp"] = out["triton_rate_pct"] - out["native_rate_pct"]  # triton − native
        out["native_minus_triton_pp"] = nmt
        out["kill_trigger_3_fires"] = nmt > 5.0
    return out


@app.function(
    image=image, gpu="A100-40GB", timeout=14400,
    secrets=[_hf_secret()],
)
def run_cell(
    arm: str,
    task_index: int,
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    task_suite_name: str = "libero_10",
    num_episodes: int = 100,
    seed: int = 7,
) -> dict:
    """One shard = one (arm × task). Thin Modal wrapper over _run_one_arm.

    timeout=14400 (4 hr). The blended directional rate is ~89 s/ep (N=30/arm both
    arms in one container, 2026-05-28: 60 eps / 5354.9 s), but that average hides
    the per-cell long pole: the NATIVE arm runs ~1.5× slower per ep (fp32+TF32, and
    its extra failures run to max episode length), so a native N=100 single-task
    cell projects to ~2.95–3.05 hr — at or just over a 3 hr (10800 s) ceiling. For
    an unattended monthly launchd gate that thin margin is a real timeout-kill risk
    on the native arm, so the ceiling is 4 hr (triton cells land ~2 hr, well under).
    Cost is unaffected — Modal bills GPU-seconds consumed, not the timeout ceiling.
    Spawned in parallel (one A100 container per (arm,task)); the SLOWEST cell sets
    the ~3 hr wall, and any single cell is retryable in isolation. First formal
    N=100 fire should still confirm the native cells at the 30-min progress mark.
    """
    return _run_one_arm(
        arm=arm,
        model_id=model_id,
        task_suite_name=task_suite_name,
        task_indices=[task_index],
        num_episodes=num_episodes,
        seed=seed,
    )


@app.function(
    image=image, gpu="A100-40GB", timeout=7200,
    secrets=[_hf_secret()],
)
def run_side_by_side(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    task_suite_name: str = "libero_10",
    task_indices: list[int] | None = None,
    num_episodes: int = 10,
    seed: int = 7,
    arms: str = "both",  # "native" | "triton" | "both"
) -> dict:
    """Legacy single-container native vs Triton on the shared proven loop.

    Both arms run sequentially in one A100 (capped by the 2 hr timeout, so only
    suitable for directional N≈10 runs). The formal N=100/task gate uses the
    sharded run_cell path via the local entrypoint instead. See module docstring.
    """
    import time

    if task_indices is None:
        # Anchor on task 0 (proven ~80%+ native at N=20) + 1,2 for breadth.
        task_indices = [0, 1, 2]

    print(
        f"[sbs] suite={task_suite_name} tasks={task_indices} "
        f"N={num_episodes} seed={seed} arms={arms}",
        flush=True,
    )
    t_total = time.time()

    native = None
    triton = None
    if arms in ("native", "both"):
        native = _run_one_arm(
            "native", model_id, task_suite_name, task_indices, num_episodes, seed,
        )
    if arms in ("triton", "both"):
        triton = _run_one_arm(
            "triton", model_id, task_suite_name, task_indices, num_episodes, seed,
        )

    out = _assemble(native, triton, task_indices, num_episodes, seed, arms)
    print(f"\n[sbs] {'='*60}", flush=True)
    if native is not None:
        print(f"[sbs] NATIVE:  {native['total_success']}/{native['total_eps']} ({out['native_rate_pct']:.1f}%)", flush=True)
    if triton is not None:
        print(f"[sbs] TRITON:  {triton['total_success']}/{triton['total_eps']} ({out['triton_rate_pct']:.1f}%)", flush=True)
    if "delta_pp" in out:
        print(
            f"[sbs] Delta:   {out['delta_pp']:+.1f}pp  "
            f"(kill-trigger-3 native−triton>5pp fires: {out['kill_trigger_3_fires']})",
            flush=True,
        )
    print(f"[sbs] Total time: {time.time()-t_total:.1f}s", flush=True)
    print(f"[sbs] {'='*60}", flush=True)
    return out


@app.local_entrypoint()
def main(
    suite: str = "libero_10",
    task_indices: str = "0,1,2",
    num_episodes: int = 100,
    seed: int = 7,
    arms: str = "both",
    shard: bool = True,
    smoke: bool = False,
):
    """Formal kill-trigger-3 L3 parity gate, sharded across parallel A100 cells.

    Default = the monthly gate: N=100/task × tasks {0,1,2} × {native,triton} = 6
    cells spawned in parallel (~2.5 hr wall). --smoke runs N=2 on task 0 to validate
    the spawn/gather/aggregate plumbing for ~$0.7. --no-shard falls back to the
    legacy single-container run_side_by_side (directional N≈10 only). Writes the
    full result dict to $L3_RESULT_JSON when set so the launchd monthly runner can
    read the verdict without scraping stdout.
    """
    import json
    import threading
    import time

    print("=" * 70)
    print("Lift #5 L3 parity gate: native lerobot vs Triton (proven rollout loop)")
    print("=" * 70)

    if smoke:
        idx = [0]
        num_episodes = 2
        print("[gate] SMOKE: N=2, task 0 — plumbing validation only (~$0.7)")
    else:
        idx = [int(x) for x in task_indices.split(",") if x.strip()] or [0, 1, 2]

    # ── Legacy single-container path ───────────────────────────────────
    if not shard:
        print(f"[gate] --no-shard: single-container run_side_by_side N={num_episodes} tasks={idx}")
        result = run_side_by_side.remote(
            task_suite_name=suite,
            task_indices=idx,
            num_episodes=num_episodes,
            seed=seed,
            arms=arms,
        )
    else:
        # ── Sharded path: one A100 cell per (arm × task), spawned in parallel ─
        arm_list = ["native", "triton"] if arms == "both" else [arms]
        cells = [(a, t) for a in arm_list for t in idx]
        print(
            f"[gate] SHARDED: {len(cells)} cells = {arm_list} × tasks {idx}, "
            f"N={num_episodes}/cell, seed={seed}",
            flush=True,
        )

        t_wall = time.time()
        handles = []
        for (a, t) in cells:
            h = run_cell.spawn(
                arm=a,
                task_index=t,
                task_suite_name=suite,
                num_episodes=num_episodes,
                seed=seed,
            )
            handles.append((a, t, h))
            print(f"[gate] spawned cell arm={a} task={t}", flush=True)

        # 60s heartbeat so a multi-hour blocking gather still emits progress
        # (satisfies the >30-min Modal progress-logging discipline).
        stop = threading.Event()

        def _heartbeat():
            while not stop.wait(60):
                el = time.time() - t_wall
                print(f"[gate] …{el/60:.1f} min elapsed, awaiting {len(cells)} cells", flush=True)

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()

        # Gather. .get() blocks per handle; a failed cell's exception propagates
        # (fail-loud — no silent partial-gate result).
        per_arm: dict[str, list] = {a: [] for a in arm_list}
        per_cell = []
        try:
            for (a, t, h) in handles:
                res = h.get()
                per_arm[a].append(res)
                per_cell.append({
                    "arm": a,
                    "task_index": t,
                    "success": res.get("total_success"),
                    "eps": res.get("total_eps"),
                    "rate_pct": res.get("success_rate_pct"),
                    "seconds": res.get("cell_seconds"),
                })
                print(
                    f"[gate] cell done arm={a} task={t}: "
                    f"{res.get('total_success')}/{res.get('total_eps')} "
                    f"({res.get('success_rate_pct', 0.0):.1f}%)",
                    flush=True,
                )
        finally:
            stop.set()
            hb.join(timeout=2)

        native = _merge_arm(per_arm.get("native", [])) if "native" in arm_list else None
        triton = _merge_arm(per_arm.get("triton", [])) if "triton" in arm_list else None
        result = _assemble(native, triton, idx, num_episodes, seed, arms)
        result["per_cell"] = per_cell
        result["wall_seconds"] = time.time() - t_wall
        print(f"[gate] wall time: {result['wall_seconds']/60:.1f} min", flush=True)

    # ── Verdict ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    n = result.get("native")
    tr = result.get("triton")
    if n is not None:
        print(f"NATIVE: {n['total_success']}/{n['total_eps']} ({result.get('native_rate_pct', 0.0):.1f}%)")
    if tr is not None:
        print(f"TRITON: {tr['total_success']}/{tr['total_eps']} ({result.get('triton_rate_pct', 0.0):.1f}%)")
    if "native_minus_triton_pp" in result:
        print(f"DELTA:  triton−native {result['delta_pp']:+.1f}pp  |  native−triton {result['native_minus_triton_pp']:+.1f}pp")
        fires = result["kill_trigger_3_fires"]
        print(f"KILL-TRIGGER-3 (native−triton > 5pp): {'FIRES — FAIL' if fires else 'clear — PASS'}")
    print("=" * 70)

    result_json = os.environ.get("L3_RESULT_JSON")
    if result_json:
        with open(result_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[gate] wrote result JSON → {result_json}")
