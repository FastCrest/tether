"""Collect real LIBERO traces with full chunks per step + train A2C2 head on
the (stale, fresh) gap. Methodology fix per A2C2 paper (arxiv 2509.23224).

Replaces the b4 gate's zero-target trainer: that one trains on
target_residual = zeros (magnitude proxy) which produces a head whose output
is identically zero (zero-init * zero-target -> zero-gradient -> zero forever).

This script does:
1. Boot LIBERO + tether serve (decomposed-export)
2. For each step t in each episode: POST /act -> get full chunk[t] (50 actions);
   record (state_t, image_hash_t, chunk_t, latency_t) to a per-step JSONL on
   the gate volume; env.step with chunk[t][0][:7] (FRESH-policy-every-step
   execution -- standard A2C2 collection protocol)
3. After all episodes: SYNTHESIZE training pairs from the collected chunks:
   for each (t, i) with t+i < trajectory_length:
     stale = chunk[t][i]               (the i-th action in the chunk emitted at t)
     fresh = chunk[t+i][0]             (the 0-th action in the chunk emitted at t+i)
     target_residual = fresh - stale   (the gap A2C2 head learns to predict)
     obs_features = state[t+i]         (state at execution time)
     latency_estimate = i * dt_step    (proxy for "how stale is this action?")
4. Train numpy A2C2 head on the synthesized pairs (Path A unification trainer)
5. Save .npz to /gate_out/<label>/a2c2_head_real.npz

Cost: ~$1-2 (10 eps collection ~30 min A100 + train ~1 min CPU; image cache
warm after first run). One-shot; the produced .npz is the input for the
B.5 Day 5 LIBERO success-delta regression run.

Usage:
    modal run scripts/modal_a2c2_real_traces.py --label real-traces-v1
    modal run scripts/modal_a2c2_real_traces.py --label small --num-collect 3
"""
from __future__ import annotations

import os
import subprocess
import modal

app = modal.App("tether-a2c2-real-traces")


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
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


def _build_bust() -> str:
    import time
    return str(int(time.time()))


_HEAD = _repo_head_sha()
_BUILD_BUST = _build_bust()


# Volumes mirror modal_libero_via_tether_serve
HF_CACHE_PATH = "/hf_cache"
ONNX_OUTPUT_PATH = "/onnx_out"
GATE_OUTPUT_PATH = "/gate_out"
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
gate_output = modal.Volume.from_name("a2c2-gate-output", create_if_missing=True)


# Mirror modal_libero_via_tether_serve.py exactly so we share the cached image.
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
        "httpx>=0.24",
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
    .run_commands(
        "mkdir -p /tmp/libero_data",
        f'echo "build_bust={_BUILD_BUST}"',
        f'pip install "fastcrest-tether[serve,gpu] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
)


LIBERO_ENV_RESOLUTION = 256
TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@app.function(
    image=image, gpu="A100-80GB", timeout=7200,
    volumes={
        HF_CACHE_PATH: hf_cache,
        ONNX_OUTPUT_PATH: onnx_output,
        GATE_OUTPUT_PATH: gate_output,
    },
    secrets=[_hf_secret()],
)
def collect_and_train(
    export_subdir: str = "distill_v050r2_decomposed",
    output_label: str = "real-traces-v1",
    num_collect_episodes: int = 10,
    task_suite_name: str = "libero_10",
    train_epochs: int = 20,
    train_batch_size: int = 64,
    train_lr: float = 1e-3,
    seed: int = 11,
    serve_health_timeout_s: int = 480,
    serve_port: int = 8000,
    # Phase 2 fixes per 2026-04-29-a2c2-correction_research_revisit:
    collect_inject_latency_ms: float = 0.0,  # pass --inject-latency-ms to tether serve at collect time
    l2_penalty_override: float = -1.0,  # if > 0, override L2_MAGNITUDE_PENALTY constant
    use_actual_latency: bool = False,  # use record's measured latency_ms instead of synthesized i*median_step_ms
    # Phase 3: per-head saturation scale (default 3.0 = Phase 1 behavior;
    # 1.5 allows L2 penalty to actually steer training without saturation cliff)
    output_saturation_scale: float = 3.0,
) -> dict:
    """Run collection + training in one Modal call. Returns paths + summary."""
    import base64
    import io
    import json
    import math
    import time
    import traceback
    import urllib.request
    from pathlib import Path

    import numpy as np
    from PIL import Image

    # PyTorch 2.6+ weights_only=True refuses LIBERO init states
    import torch
    _orig_torch_load = torch.load
    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _compat_load

    out_root = Path(GATE_OUTPUT_PATH) / output_label
    out_root.mkdir(parents=True, exist_ok=True)
    traces_path = out_root / "real_traces.jsonl"
    head_path = out_root / "a2c2_head_real.npz"
    metrics_path = out_root / "real_train_metrics.json"
    print(f"[real_traces] output: {out_root}")

    # ---- 1. Spawn tether serve ----
    export_dir = f"{ONNX_OUTPUT_PATH}/{export_subdir}"
    serve_cmd = [
        "tether", "serve", export_dir,
        "--port", str(serve_port),
        "--device", "cuda",
        "--no-strict-providers",
        "--no-prewarm",
    ]
    if collect_inject_latency_ms > 0:
        # Phase 2: collect at runtime-like latency so training distribution
        # matches eval distribution. Lens 3 of the 2026-04-29 research-revisit
        # showed training-vs-eval distribution mismatch was 40-50% of failure.
        serve_cmd += ["--inject-latency-ms", str(collect_inject_latency_ms)]
        print(f"[real_traces] Phase 2 collection: --inject-latency-ms {collect_inject_latency_ms}")
    print(f"[real_traces] starting: {' '.join(serve_cmd)}")
    serve_proc = subprocess.Popen(serve_cmd, env={**os.environ})

    def _wait_for_health(url: str, timeout_s: int) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        print(f"  /health OK after {time.time() - t0:.1f}s")
                        return True
            except Exception:
                pass
            time.sleep(2)
        return False

    health_url = f"http://127.0.0.1:{serve_port}/health"
    if not _wait_for_health(health_url, serve_health_timeout_s):
        serve_proc.terminate()
        return {"status": "fail", "reason": f"serve didn't come up in {serve_health_timeout_s}s"}

    def _post_act(image_arr: np.ndarray, instruction: str, state: list[float], episode_id: str) -> dict:
        buf = io.BytesIO()
        Image.fromarray(image_arr).save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        payload = {
            "image": img_b64, "instruction": instruction,
            "state": state, "episode_id": episode_id,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{serve_port}/act",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    # ---- 2. Collect traces ----
    try:
        np.random.seed(seed)
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        max_steps = TASK_SUITE_MAX_STEPS[task_suite_name]
        num_tasks_in_suite = task_suite.n_tasks
        print(f"[real_traces] suite={task_suite_name} num_tasks={num_tasks_in_suite} max_steps={max_steps}")

        def _quat2axisangle(quat):
            if quat[3] > 1.0: quat[3] = 1.0
            elif quat[3] < -1.0: quat[3] = -1.0
            den = np.sqrt(1.0 - quat[3] * quat[3])
            if math.isclose(den, 0.0):
                return np.zeros(3)
            return (quat[:3] * 2.0 * math.acos(quat[3])) / den

        # Distribute episodes across tasks: round-robin so we get diversity
        per_task_eps = [0] * num_tasks_in_suite
        for i in range(num_collect_episodes):
            per_task_eps[i % num_tasks_in_suite] += 1

        records_written = 0
        with traces_path.open("w") as out_f:
            for task_idx in range(num_tasks_in_suite):
                if per_task_eps[task_idx] == 0:
                    continue
                task = task_suite.get_task(task_idx)
                task_description = task.language
                initial_states = task_suite.get_task_init_states(task_idx)
                print(f"[real_traces] TASK {task_idx} ({per_task_eps[task_idx]} eps): {task_description!r}")
                task_bddl_file = (
                    Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                )
                env = OffScreenRenderEnv(
                    bddl_file_name=str(task_bddl_file),
                    camera_heights=LIBERO_ENV_RESOLUTION,
                    camera_widths=LIBERO_ENV_RESOLUTION,
                )
                env.seed(seed)

                for ep in range(per_task_eps[task_idx]):
                    try:
                        env.reset()
                        init_idx = ep % len(initial_states)
                        obs = env.set_init_state(initial_states[init_idx])
                        episode_id = f"collect_{task_idx:02d}_{ep:02d}"
                        t = 0
                        ep_records = 0
                        while t < max_steps:
                            try:
                                agent_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                                state_vec = np.concatenate([
                                    np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                                    _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy()),
                                    np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
                                ]).astype(np.float32).tolist()
                                resp_t0 = time.time()
                                resp = _post_act(agent_img, task_description, state_vec, episode_id)
                                latency_ms = (time.time() - resp_t0) * 1000.0
                                if "error" in resp:
                                    raise RuntimeError(f"/act error: {resp['error']}")
                                actions = resp.get("actions") or []
                                if not actions:
                                    raise RuntimeError("/act returned empty actions")

                                # Record (state_t, full chunk[t], latency, ep, t)
                                rec = {
                                    "episode_id": episode_id,
                                    "task_idx": task_idx,
                                    "step": t,
                                    "state": state_vec,
                                    "chunk": [list(map(float, a)) for a in actions],
                                    "latency_ms": latency_ms,
                                }
                                out_f.write(json.dumps(rec) + "\n")
                                ep_records += 1
                                records_written += 1

                                # Step env with chunk[0][:7] (LIBERO franka 7-DOF)
                                action = np.asarray(actions[0], dtype=np.float32)[:7]
                                obs, _, done, _ = env.step(action.tolist())
                                t += 1
                                if done:
                                    break
                            except Exception as exc:
                                print(f"  step error t={t}: {exc}")
                                break
                        print(f"  ep {ep}: {ep_records} steps recorded")
                    except Exception as exc:
                        print(f"  episode error: {exc}")
                        traceback.print_exc()
                env.close()

        print(f"[real_traces] collection done: {records_written} (state, chunk) records -> {traces_path}")

    finally:
        print("[real_traces] shutting down tether serve...")
        serve_proc.terminate()
        try:
            serve_proc.wait(20)
        except subprocess.TimeoutExpired:
            serve_proc.kill()

    # ---- 3. Synthesize training pairs from chunks ----
    print(f"\n[real_traces] synthesizing (stale, fresh) pairs from {traces_path}")
    from collections import defaultdict
    by_ep: dict[str, list[dict]] = defaultdict(list)
    with traces_path.open() as f:
        for line in f:
            rec = json.loads(line.strip())
            by_ep[rec["episode_id"]].append(rec)
    for k in by_ep:
        by_ep[k].sort(key=lambda r: r["step"])

    if not by_ep:
        return {"status": "fail", "reason": "no traces collected"}

    # Probe shapes from first record
    first_rec = next(iter(by_ep.values()))[0]
    chunk_t0 = first_rec["chunk"]
    chunk_size = len(chunk_t0)
    full_action_dim = len(chunk_t0[0]) if chunk_t0 else 0
    state_dim = len(first_rec["state"])
    # LIBERO franka uses 7-DOF; pi05 export pads to 32. Truncate.
    target_action_dim = 7
    obs_dim = state_dim  # use state as obs_features (Phase 1)

    print(f"[real_traces] chunk_size={chunk_size} full_action_dim={full_action_dim} "
          f"target_action_dim={target_action_dim} state_dim={state_dim}")

    # Estimate per-step interval (~latency_ms_per_step) from collected latencies
    all_lats = []
    for ep, recs in by_ep.items():
        for r in recs:
            all_lats.append(r.get("latency_ms", 30.0))
    median_step_ms = float(np.median(all_lats)) if all_lats else 30.0
    print(f"[real_traces] median_step_ms={median_step_ms:.1f} (proxy for staleness latency)")

    base_rows: list[np.ndarray] = []
    obs_rows: list[np.ndarray] = []
    chunk_pos_rows: list[int] = []
    latency_rows: list[float] = []
    target_rows: list[np.ndarray] = []

    n_pairs_per_ep_max = chunk_size  # cap to avoid explosion on very long eps
    for ep_id, recs in by_ep.items():
        T = len(recs)
        # For each (t, i) with t+i < T and i < chunk_size, build a training pair.
        # i = how stale the action would be when executed (chunk_position).
        for t in range(T):
            chunk_t = recs[t]["chunk"]
            i_max = min(chunk_size, T - t, n_pairs_per_ep_max)
            for i in range(i_max):
                stale = np.asarray(chunk_t[i], dtype=np.float32)[:target_action_dim]
                fresh = np.asarray(recs[t + i]["chunk"][0], dtype=np.float32)[:target_action_dim]
                target = fresh - stale
                state_at_exec = np.asarray(recs[t + i]["state"], dtype=np.float32)
                obs_pad = np.zeros(obs_dim, dtype=np.float32)
                obs_pad[: state_at_exec.shape[0]] = state_at_exec
                base_rows.append(stale)
                obs_rows.append(obs_pad)
                chunk_pos_rows.append(i)
                if use_actual_latency:
                    # Phase 2: use the actual measured latency at the record
                    # where the action would be executed. With --inject-latency-ms
                    # at collection, this matches runtime distribution.
                    latency_rows.append(float(recs[t + i].get("latency_ms", median_step_ms * (i + 1))))
                else:
                    latency_rows.append(i * median_step_ms)
                target_rows.append(target)
    n_pairs = len(base_rows)
    print(f"[real_traces] synthesized {n_pairs} (stale, fresh) training pairs")
    if n_pairs == 0:
        return {"status": "fail", "reason": "no training pairs"}

    base_arr = np.asarray(base_rows, dtype=np.float32)
    obs_arr = np.asarray(obs_rows, dtype=np.float32)
    chunk_pos_arr = np.asarray(chunk_pos_rows, dtype=np.int64)
    lat_arr = np.asarray(latency_rows, dtype=np.float32)
    target_arr = np.asarray(target_rows, dtype=np.float32)

    # Sanity: how much actual signal is there?
    mean_target_l2 = float(np.mean(np.linalg.norm(target_arr, axis=-1)))
    print(f"[real_traces] mean ||target|| = {mean_target_l2:.6f} "
          f"(if ~0, head will learn nothing meaningful; if >0, real signal)")

    # ---- 4. Train numpy A2C2 head ----
    from tether.correction import A2C2Config, train_a2c2_head, evaluate_mse
    from tether.correction import a2c2_training as _a2c2_training_module

    # Phase 2: relax L2 magnitude penalty if requested. Phase 1 default
    # 0.01 was conservative — head learned magnitude ~0.1/step (chunk 0.755).
    # Phase 2 with multi-latency training data may need 0.001 to allow
    # actually-useful magnitude ~0.5-1.0/step.
    if l2_penalty_override > 0:
        _orig_l2 = _a2c2_training_module.L2_MAGNITUDE_PENALTY
        _a2c2_training_module.L2_MAGNITUDE_PENALTY = l2_penalty_override
        print(f"[real_traces] Phase 2: L2_MAGNITUDE_PENALTY override "
              f"{_orig_l2} -> {l2_penalty_override}")

    cfg = A2C2Config(
        action_dim=target_action_dim,
        obs_dim=obs_dim,
        chunk_size=chunk_size,
        hidden_dim=128,
        num_hidden_layers=3,
        position_encoding_dim=32,
        output_saturation_scale=output_saturation_scale,
    )
    print(f"[real_traces] A2C2Config: input={cfg.input_dim} hidden={cfg.hidden_dim} × {cfg.num_hidden_layers} "
          f"output={cfg.action_dim} sat_scale={cfg.output_saturation_scale} "
          f"(~{cfg.estimated_size_bytes()/1024:.1f} KB FP32)")

    print(f"[real_traces] training {train_epochs} epochs on {n_pairs} pairs (batch={train_batch_size}, lr={train_lr})")
    result = train_a2c2_head(
        base_actions=base_arr,
        observations=obs_arr,
        chunk_positions=chunk_pos_arr,
        latency_ms_per_step=lat_arr,
        target_residuals=target_arr,
        cfg=cfg,
        epochs=train_epochs,
        batch_size=train_batch_size,
        lr=train_lr,
        val_split=0.1,
        seed=seed,
    )
    head = result.head

    # Print first/last/best epoch stats
    epochs_log = result.metrics["epochs"]
    print(f"[real_traces] training summary:")
    print(f"  epoch 0   train={epochs_log[0]['train_loss']:.6f} val={epochs_log[0]['val_loss']:.6f}")
    if len(epochs_log) > 1:
        print(f"  epoch {len(epochs_log)-1:3d} train={epochs_log[-1]['train_loss']:.6f} val={epochs_log[-1]['val_loss']:.6f}")

    head.save(head_path)
    n_params = sum(w.size + b.size for w, b in zip(head._weights, head._biases))
    print(f"[real_traces] checkpoint saved: {head_path} ({head_path.stat().st_size / 1024:.1f} KB, {n_params} params)")

    Path(metrics_path).write_text(json.dumps({
        "n_collect_episodes": num_collect_episodes,
        "n_records_collected": records_written,
        "n_training_pairs": n_pairs,
        "mean_target_l2": mean_target_l2,
        "median_step_ms": median_step_ms,
        "chunk_size": chunk_size,
        "target_action_dim": target_action_dim,
        "config": result.metrics["config"],
        "epochs": epochs_log,
        "head_path": str(head_path),
        "traces_path": str(traces_path),
    }, indent=2))
    print(f"[real_traces] metrics written: {metrics_path}")

    gate_output.commit()

    return {
        "status": "ok",
        "head_path": str(head_path),
        "traces_path": str(traces_path),
        "metrics_path": str(metrics_path),
        "n_collect_episodes": num_collect_episodes,
        "n_records_collected": records_written,
        "n_training_pairs": n_pairs,
        "mean_target_l2": mean_target_l2,
        "final_val_mse": epochs_log[-1]["val_loss"] if epochs_log else float("nan"),
    }


@app.local_entrypoint()
def main(
    label: str = "real-traces-v1",
    num_collect: int = 10,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    # Phase 2 flags per 2026-04-29-a2c2-correction_research_revisit:
    collect_inject_latency_ms: float = 0.0,
    l2_penalty_override: float = -1.0,
    use_actual_latency: bool = False,
    # Phase 3:
    output_saturation_scale: float = 3.0,
):
    """Collect real LIBERO traces + train A2C2 head on (stale, fresh) gap.

    Output: /gate_out/<label>/a2c2_head_real.npz (loadable by tether serve --a2c2-checkpoint)

    Phase 2 invocation:
        modal run scripts/modal_a2c2_real_traces.py \\
            --label phase2-multilat-2ep \\
            --num-collect 2 \\
            --epochs 5 \\
            --collect-inject-latency-ms 100 \\
            --l2-penalty-override 0.001 \\
            --use-actual-latency
    """
    print("=" * 70)
    print("A2C2 real-trace collect + train")
    print("=" * 70)
    print(f"label: {label}, num_collect: {num_collect}, epochs: {epochs}")
    if collect_inject_latency_ms > 0:
        print(f"Phase 2: collect_inject_latency_ms={collect_inject_latency_ms}, "
              f"l2_penalty={l2_penalty_override}, use_actual_latency={use_actual_latency}")
    r = collect_and_train.remote(
        output_label=label,
        num_collect_episodes=num_collect,
        train_epochs=epochs,
        train_batch_size=batch_size,
        train_lr=lr,
        collect_inject_latency_ms=collect_inject_latency_ms,
        l2_penalty_override=l2_penalty_override,
        use_actual_latency=use_actual_latency,
        output_saturation_scale=output_saturation_scale,
    )
    print("\n=== RESULT ===")
    if r.get("status") == "fail":
        print(f"  status: FAIL")
        print(f"  reason: {r.get('reason')}")
        return
    print(f"  status:               {r.get('status')}")
    print(f"  head_path:            {r.get('head_path')}")
    print(f"  traces_path:          {r.get('traces_path')}")
    print(f"  metrics_path:         {r.get('metrics_path')}")
    print(f"  n_collect_episodes:   {r.get('n_collect_episodes')}")
    print(f"  n_records_collected:  {r.get('n_records_collected')}")
    print(f"  n_training_pairs:     {r.get('n_training_pairs')}")
    print(f"  mean_target_l2:       {r.get('mean_target_l2'):.6f}")
    print(f"  final_val_mse:        {r.get('final_val_mse'):.6f}")
    print()
    print(f"Next: modal run scripts/modal_libero_via_tether_serve.py \\")
    print(f"        --tasks all --num-episodes 5 \\")
    print(f"        --a2c2-checkpoint {r.get('head_path')}")
