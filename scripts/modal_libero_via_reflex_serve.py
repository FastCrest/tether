"""LIBERO eval routed through tether serve HTTP -- closes B.5 Day 5
+ b4 gate methodology gap.

The existing scripts/modal_libero_monolithic_onnx.py BYPASSES tether
serve and calls the ONNX session directly. That validates the model
correctness but skips ALL the wedges (action_guard, RTC, A2C2,
record-replay, prometheus, policy-versioning, chunk-budget-batching).
For the B.5 Day 5 deliverable ("LIBERO success delta with A2C2-on vs
A2C2-off"), we need LIBERO actions to flow through the actual /act
handler -- otherwise the A2C2 hook isn't exercised.

Per ADR 2026-04-25-decomposed-dispatch-via-tether-serve §"Validation
experiments to file" #2: this is the LIBERO-via-tether-serve harness.

Architecture:
- Single Modal A100 container
- Spawn `tether serve <decomposed_export>` as subprocess (uses the
  Pi05DecomposedServer dispatch shipped in commit cfb846d today)
- Wait for /health
- Boot LIBERO + run episodes; for each step, POST /act with
  base64-encoded image + instruction + state, parse response, step env
- Two runs: A2C2-off (baseline) + A2C2-on (--a2c2-checkpoint <head>)
- Compare success rates

Phase 1 status: SUBSTRATE only. Each piece (subprocess spawn, HTTP
client, LIBERO episode loop) is validated independently in code; the
end-to-end Modal run is the next-session validation. ~$5-10 to
validate when fired.

Usage (when ready to fire):
    modal run scripts/modal_libero_via_tether_serve.py
    modal run scripts/modal_libero_via_tether_serve.py --num-episodes 3 --tasks 0
    modal run scripts/modal_libero_via_tether_serve.py --a2c2-checkpoint /a2c2/head.npz
"""
from __future__ import annotations

import os
import subprocess
import modal

app = modal.App("tether-libero-via-tether-serve")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


def _build_bust() -> str:
    import time
    return str(int(time.time()))


_HEAD = _repo_head_sha()
_BUILD_BUST = _build_bust()


# Same volume layout as modal_libero_monolithic_onnx.py + the gate output
# volume so a2c2 heads from modal_b4_gate_fire are reachable.
hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
gate_output = modal.Volume.from_name("a2c2-gate-output", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
ONNX_OUTPUT_PATH = "/onnx_out"
GATE_OUTPUT_PATH = "/gate_out"


# Image: same shape as modal_libero (LIBERO + osmesa + MuJoCo) + the
# tether-vla install + httpx for the HTTP client.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "libegl1-mesa",
        "libglvnd0", "ffmpeg", "cmake", "build-essential",
        "libosmesa6", "libosmesa6-dev", "clang",
    )
    .pip_install(
        "torch", "safetensors>=0.4.0", "huggingface_hub",
        "transformers<5.4,>=4.40", "numpy", "Pillow", "pydantic>=2.0",
        "pyyaml", "onnx>=1.16", "onnxruntime-gpu>=1.20",
        "onnxscript>=0.1", "mujoco==3.3.2", "robosuite==1.4.1", "h5py",
        "bddl==1.0.1", "future", "robomimic", "hydra-core>=1.1",
        "easydict", "einops", "opencv-python-headless",
        "gym", "gymnasium", "lerobot==0.5.1", "num2words", "imageio",
        "httpx>=0.24",  # for the HTTP client to tether serve
    )
    .run_commands(
        "git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO"
        " && cd /opt/LIBERO && pip install . --no-deps"
    )
    .add_local_file("scripts/patch_libero.py", "/root/patch_libero.py", copy=True)
    .run_commands("python /root/patch_libero.py")
    .env({
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "LIBERO_DATA_DIR": "/tmp/libero_data",
        "LIBERO_ASSET_DIR": "/opt/LIBERO/libero/libero/assets",
        "LIBERO_BASE": "/tmp/libero_data",
        "PYTHONPATH": "/opt/LIBERO",
    })
    # Mount local tether-vla source instead of pulling from git+https. Removes
    # the dependency on a workspace-level github-token Modal secret (every
    # fresh Modal account otherwise needs the secret added at
    # https://modal.com/secrets/<workspace>/main/create?secret_name=github-token
    # before the first run can succeed). add_local_dir picks up local edits
    # immediately — no git push needed before modal run.
    .add_local_dir(
        os.path.join(REPO_ROOT, "src"),
        remote_path="/root/tether-vla/src",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_file(
        os.path.join(REPO_ROOT, "pyproject.toml"),
        remote_path="/root/tether-vla/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        os.path.join(REPO_ROOT, "README.md"),
        remote_path="/root/tether-vla/README.md",
        copy=True,
    )
    .add_local_file(
        os.path.join(REPO_ROOT, "LICENSE"),
        remote_path="/root/tether-vla/LICENSE",
        copy=True,
    )
    .run_commands(
        "mkdir -p /tmp/libero_data",
        f'echo "build_bust={_BUILD_BUST}"',
        'pip install -e "/root/tether-vla[serve,gpu]"',
    )
)


# Constants mirror modal_libero_monolithic_onnx.py
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=86400,  # 24 hr Modal max (was 3 hr — bit us 2026-04-30: gate-6 N=20 × 5 tasks × 2 conditions both hit the 3hr cap mid-task-2 with no warning). Per-event logs hide timing pressure; pair with the periodic [progress] emission added below.
    volumes={
        HF_CACHE_PATH: hf_cache,
        ONNX_OUTPUT_PATH: onnx_output,
        GATE_OUTPUT_PATH: gate_output,
    },
    secrets=[_hf_secret()],
)
def libero_via_serve(
    export_subdir: str = "distill_v050r2_decomposed",
    num_episodes: int = 1,
    task_suite_name: str = "libero_10",
    task_indices: list[int] | None = None,
    a2c2_checkpoint: str = "",
    a2c2_latency_threshold_ms: float = 40.0,
    a2c2_success_threshold: float = 0.90,
    inject_latency_ms: float = 0.0,
    bid_n_candidates: int = 0,
    bid_coherence_window: int = 5,
    bid_coherence_metric: str = "l2",
    rtc: bool = False,
    rtc_execution_horizon: int = 0,
    seed: int = 7,
    serve_health_timeout_s: int = 480,
    serve_port: int = 8000,
) -> dict:
    """Run LIBERO episodes against a tether serve subprocess.

    Args:
        export_subdir: subfolder under /onnx_out/ -- the decomposed export
            to serve (e.g., distill_v050r2_decomposed).
        num_episodes: episodes per task (default 1 for smoke; 50 for full).
        task_suite_name: libero_10 | libero_spatial | libero_object | libero_goal | libero_90.
        task_indices: list of task ids; None = all tasks in suite.
        a2c2_checkpoint: path to A2C2 head .npz (e.g.,
            /gate_out/run_001/a2c2_head.npz). Empty = baseline (no A2C2).
        seed: RNG seed.
        serve_health_timeout_s: how long to wait for tether serve /health.
        serve_port: port for the tether serve subprocess.

    Returns:
        {"status": "ok"|"fail", "success_rate_pct": float, "per_task": [...],
         "errors": [...], "a2c2_active": bool, "total_steps": int}
    """
    import base64
    import io
    import json
    import math
    import sys
    import time
    import traceback
    import urllib.request
    import urllib.error
    from pathlib import Path

    import numpy as np
    from PIL import Image

    # PyTorch 2.6+ weights_only=True refuses LIBERO init states.
    import torch
    _orig_torch_load = torch.load
    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _compat_load

    export_dir = f"{ONNX_OUTPUT_PATH}/{export_subdir}"
    print(f"[libero_via_serve] export_dir={export_dir} a2c2={a2c2_checkpoint or '(none)'}")

    # ---- Spawn tether serve subprocess ----
    serve_cmd = [
        "tether", "serve", export_dir,
        "--port", str(serve_port),
        "--device", "cuda",
        "--no-strict-providers",
        "--no-prewarm",  # /health flips ready immediately; first /act pays JIT cost
    ]
    if a2c2_checkpoint:
        serve_cmd.extend(["--a2c2-checkpoint", a2c2_checkpoint])
        serve_cmd.extend(["--a2c2-latency-threshold-ms", str(a2c2_latency_threshold_ms)])
        serve_cmd.extend(["--a2c2-success-threshold", str(a2c2_success_threshold)])
    if inject_latency_ms > 0:
        serve_cmd.extend(["--inject-latency-ms", str(int(inject_latency_ms))])
    if bid_n_candidates > 0:
        serve_cmd.extend(["--bid-num-candidates", str(int(bid_n_candidates))])
        serve_cmd.extend(["--bid-coherence-window", str(int(bid_coherence_window))])
        serve_cmd.extend(["--bid-coherence-metric", bid_coherence_metric])
    if rtc:
        serve_cmd.append("--rtc")
        if rtc_execution_horizon > 0:
            serve_cmd.extend(["--rtc-execution-horizon", str(int(rtc_execution_horizon))])

    print(f"[libero_via_serve] starting: {' '.join(serve_cmd)}")
    serve_proc = subprocess.Popen(
        serve_cmd,
        env={**os.environ},
    )

    def _wait_for_health(url: str, timeout_s: int) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        elapsed = time.time() - t0
                        print(f"  /health OK after {elapsed:.1f}s")
                        return True
            except Exception:
                pass
            time.sleep(2)
        return False

    health_url = f"http://127.0.0.1:{serve_port}/health"
    if not _wait_for_health(health_url, serve_health_timeout_s):
        serve_proc.terminate()
        return {
            "status": "fail",
            "reason": f"tether serve didn't come up in {serve_health_timeout_s}s",
        }

    # ---- HTTP client helper ----
    def _b64png(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _post_act(image_arr: np.ndarray, instruction: str, state: list[float], episode_id: str, wrist_arr: np.ndarray | None = None) -> dict:
        """POST one /act request + return the parsed response."""
        payload: dict = {
            "image": _b64png(image_arr),
            "instruction": instruction,
            "state": state,
            "episode_id": episode_id,
        }
        if wrist_arr is not None:
            payload["image_wrist"] = _b64png(wrist_arr)
        req = urllib.request.Request(
            f"http://127.0.0.1:{serve_port}/act",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    try:
        # ---- LIBERO setup ----
        np.random.seed(seed)
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        max_steps = TASK_SUITE_MAX_STEPS[task_suite_name]
        print(f"[libero_via_serve] suite={task_suite_name}, num_tasks={num_tasks_in_suite}, max_steps={max_steps}")

        def _quat2axisangle(quat):
            if quat[3] > 1.0: quat[3] = 1.0
            elif quat[3] < -1.0: quat[3] = -1.0
            den = np.sqrt(1.0 - quat[3] * quat[3])
            if math.isclose(den, 0.0):
                return np.zeros(3)
            return (quat[:3] * 2.0 * math.acos(quat[3])) / den

        def _build_env(task):
            task_bddl_file = (
                Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            )
            env = OffScreenRenderEnv(
                bddl_file_name=str(task_bddl_file),
                camera_heights=LIBERO_ENV_RESOLUTION,
                camera_widths=LIBERO_ENV_RESOLUTION,
            )
            env.seed(seed)
            return env

        results = {
            "status": "ok",
            "export_subdir": export_subdir,
            "a2c2_active": bool(a2c2_checkpoint),
            "a2c2_checkpoint": a2c2_checkpoint,
            "suite": task_suite_name,
            "num_episodes_per_task": num_episodes,
            "max_steps": max_steps,
            "per_task": [],
            "total_success": 0,
            "total_eps": 0,
            "total_steps": 0,
            "errors": [],
        }

        tasks_to_run = task_indices if task_indices is not None else list(range(num_tasks_in_suite))
        print(f"[libero_via_serve] tasks: {tasks_to_run}")

        # Periodic-progress watchdog. After each episode, emit a one-line
        # [progress] stat with elapsed-vs-budget so timeout/preemption issues
        # surface BEFORE the run ends. Per-event 'ep N done' lines hide timing
        # pressure; this gives Monitor / observers a single-line pace check.
        # Lesson from 2026-04-30: gate-6 N=20 × 5 tasks × 2 conditions both
        # hit the 3hr Modal timeout cap mid-task-2 with no warning.
        # See feedback_modal_periodic_progress_logging memory.
        run_start_ts = time.time()
        timeout_budget_s = 86400  # match @app.function(timeout=...)
        total_expected_eps = num_episodes * len(tasks_to_run)

        for task_idx in tasks_to_run:
            task = task_suite.get_task(task_idx)
            task_description = task.language
            print(f"\n[libero_via_serve] TASK {task_idx}: {task_description!r}")
            initial_states = task_suite.get_task_init_states(task_idx)

            env = _build_env(task)
            task_start = time.time()
            task_result = {
                "task_idx": task_idx,
                "task_description": task_description,
                "episodes": [],
                "success": 0, "total": 0,
            }

            for ep in range(num_episodes):
                try:
                    env.reset()
                    init_idx = ep % len(initial_states)
                    obs = env.set_init_state(initial_states[init_idx])
                    episode_id = f"ep_{task_idx:02d}_{ep:02d}"
                    t = 0
                    done = False
                    n_steps = 0
                    while t < max_steps:
                        try:
                            agent_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                            wrist_img = None
                            if "robot0_eye_in_hand_image" in obs:
                                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                            state_vec = np.concatenate([
                                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                                _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy()),
                                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
                            ]).astype(np.float32).tolist()
                            resp = _post_act(
                                agent_img, task_description, state_vec, episode_id,
                                wrist_arr=wrist_img,
                            )
                            if "error" in resp:
                                raise RuntimeError(f"/act error: {resp['error']}")
                            actions = resp.get("actions") or []
                            if not actions:
                                raise RuntimeError("/act returned empty actions")
                            # Diagnostic: log A2C2 decision once per ep at step 0
                            # so we can see whether the hook actually applied or
                            # auto-skipped (low_latency / high_success / cold_start).
                            if t == 0 and "a2c2_applied" in resp:
                                print(
                                    f"  [a2c2 diag] step=0: applied={resp.get('a2c2_applied')} "
                                    f"reason={resp.get('a2c2_reason')!r} "
                                    f"magnitude={resp.get('a2c2_correction_magnitude')}"
                                )
                            # Take the first action of the chunk (chunk-by-chunk replan).
                            # Pi05DecomposedServer returns action_dim per export config
                            # (32 for pi05 padded). LIBERO franka env expects 7-DOF.
                            # Truncate client-side -- mirrors the LIBERO convention in
                            # modal_libero_monolithic_onnx.py:_onnx_predict_chunk
                            # (chunk_np[:, :7]). Caught 2026-04-25 first smoke.
                            action = np.asarray(actions[0], dtype=np.float32)[:7]
                            obs, _, done, info = env.step(action.tolist())
                            n_steps += 1
                            t += 1
                            if done:
                                task_result["success"] += 1
                                results["total_success"] += 1
                                break
                        except Exception as exc:
                            err_tb = traceback.format_exc()
                            print(f"  step error: {exc}")
                            print(err_tb[-500:])
                            results["errors"].append({
                                "task": task_idx, "ep": ep,
                                "error": str(exc), "tb": err_tb[-400:],
                            })
                            break
                    task_result["episodes"].append({
                        "ep": int(ep),
                        "init_idx": int(init_idx),
                        "steps": int(t),
                        "success": bool(done),
                    })
                    task_result["total"] += 1
                    results["total_eps"] += 1
                    results["total_steps"] += n_steps
                    print(f"  ep {ep} (init={init_idx}): {'SUCCESS' if done else 'fail'} at {t} steps ({time.time()-task_start:.1f}s total)")
                    # Periodic progress watchdog: emit pace check vs Modal timeout budget.
                    elapsed_s = time.time() - run_start_ts
                    eps_done = results["total_eps"]
                    if eps_done > 0:
                        eta_total_s = elapsed_s * total_expected_eps / eps_done
                        eta_remaining_min = (eta_total_s - elapsed_s) / 60.0
                        time_pct = 100.0 * elapsed_s / timeout_budget_s
                        ep_pct = 100.0 * eps_done / total_expected_eps
                        warn = "WARN_OVER_BUDGET" if (time_pct > 70 and ep_pct < time_pct - 10) else ""
                        print(
                            f"  [progress] elapsed={elapsed_s/60:.1f}min ({time_pct:.0f}% of budget) "
                            f"| done={eps_done}/{total_expected_eps} ({ep_pct:.0f}%) "
                            f"| ETA={eta_remaining_min:.0f}min {warn}".strip()
                        )
                except Exception as exc:
                    err_tb = traceback.format_exc()
                    print(f"  episode error: {exc}")
                    print(err_tb[-500:])
                    results["errors"].append({
                        "task": task_idx, "ep": ep,
                        "error": str(exc), "tb": err_tb[-400:],
                    })
                    task_result["total"] += 1
                    results["total_eps"] += 1
            results["per_task"].append(task_result)
            print(f"[libero_via_serve] task {task_idx} done: {task_result['success']}/{task_result['total']}")
            try:
                env.close()
            except Exception:
                pass

        success_rate = (
            100.0 * results["total_success"] / results["total_eps"]
            if results["total_eps"] else 0.0
        )
        results["success_rate_pct"] = round(success_rate, 1)
        print(f"\n====== {task_suite_name} via tether serve (a2c2={a2c2_checkpoint or 'off'}) ======")
        print(f"  Export:  {export_subdir}")
        print(f"  Success: {results['total_success']}/{results['total_eps']} = {success_rate:.1f}%")
        return results
    finally:
        # Always terminate the serve subprocess.
        print("[libero_via_serve] shutting down tether serve...")
        serve_proc.terminate()
        try:
            serve_proc.wait(20)
        except subprocess.TimeoutExpired:
            serve_proc.kill()


@app.local_entrypoint()
def main(
    export_subdir: str = "distill_v050r2_decomposed",
    num_episodes: int = 1,
    tasks: str = "0",
    suite: str = "libero_10",
    a2c2_checkpoint: str = "",
    a2c2_latency_threshold_ms: float = 40.0,
    a2c2_success_threshold: float = 0.90,
    inject_latency_ms: float = 0.0,
    bid_n_candidates: int = 0,
    bid_coherence_window: int = 5,
    bid_coherence_metric: str = "l2",
    rtc: bool = False,
    rtc_execution_horizon: int = 0,
):
    """Run LIBERO via tether serve.

    --tasks "0"          single task
    --tasks "0,1,2"      3 tasks
    --tasks "all"        full suite
    --a2c2-checkpoint /gate_out/.../a2c2_head.npz  enable A2C2
    --a2c2-success-threshold 1.01  disable success-based skip (for measurement)
    --a2c2-latency-threshold-ms 0  disable latency-based skip (force-apply)
    --inject-latency-ms 100  synthetic latency for A2C2 measurement (ADR 2026-04-24)
    --bid-n-candidates 8  enable BID chunk selection (mutex with a2c2_checkpoint in Phase 1)
    """
    if tasks == "all":
        task_list = None
    else:
        task_list = [int(t) for t in tasks.split(",")]
    print(f"Running LIBERO {suite} via tether serve: tasks={task_list or 'all'}, "
          f"{num_episodes} eps each, a2c2={a2c2_checkpoint or '(off)'}, "
          f"inject_latency_ms={inject_latency_ms}")
    r = libero_via_serve.remote(
        export_subdir=export_subdir,
        num_episodes=num_episodes,
        task_suite_name=suite,
        task_indices=task_list,
        a2c2_checkpoint=a2c2_checkpoint,
        a2c2_latency_threshold_ms=a2c2_latency_threshold_ms,
        a2c2_success_threshold=a2c2_success_threshold,
        inject_latency_ms=inject_latency_ms,
        bid_n_candidates=bid_n_candidates,
        bid_coherence_window=bid_coherence_window,
        bid_coherence_metric=bid_coherence_metric,
        rtc=rtc,
        rtc_execution_horizon=rtc_execution_horizon,
    )
    print("\n=== RESULT ===")
    if r.get("status") == "fail":
        print(f"  status: FAIL")
        print(f"  reason: {r.get('reason', '(no reason)')}")
        return
    print(f"  status:        {r.get('status')}")
    print(f"  export:        {r.get('export_subdir')}")
    print(f"  a2c2_active:   {r.get('a2c2_active')}")
    print(f"  success_rate:  {r.get('success_rate_pct', '?')}%")
    print(f"  total:         {r.get('total_success', '?')}/{r.get('total_eps', '?')}")
    print(f"  total_steps:   {r.get('total_steps', '?')}")
    print(f"  errors:        {len(r.get('errors', []))}")
    for task in r.get("per_task", []):
        print(f"  task {task['task_idx']}: {task['success']}/{task['total']} -- {task['task_description'][:60]}")
