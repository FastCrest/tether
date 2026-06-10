"""Modal: fire the B.4 A2C2 transfer-validation gate (MSE arm).

Single Modal A10G run. Spins up `tether serve` twice (low and high injected
latency), records traces with `--inject-latency-ms` + `--record`, trains an
A2C2 head on the low-latency split, and fires the gate against both.

Per ADR 2026-04-24-a2c2-gate-modal-synthetic-latency: this is the MSE arm of
the gate. Task-success arm requires B.5 minimal wiring + LIBERO eval — that is
its own follow-up. This script gives a real PROCEED / PAUSE / ABORT call on
the transfer-error question.

Usage:
    modal run scripts/modal_b4_gate_fire.py
    modal run scripts/modal_b4_gate_fire.py --n-episodes 50 --high-ms 150
    modal run scripts/modal_b4_gate_fire.py --label run_002 --low-ms 30 --high-ms 200

Cost: ~$2-4 on A10G (model load 60-90s × 2 + ~5 min request loop × 2 + train).
Wall clock: 15-25 min.
"""
import os
import subprocess
import modal

app = modal.App("tether-b4-gate-fire")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


def _gh_secret():
    try:
        return modal.Secret.from_name("github-token")
    except Exception:
        return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()

hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs")
gate_output = modal.Volume.from_name("a2c2-gate-output", create_if_missing=True)

HF_CACHE = "/root/.cache/huggingface"
ONNX_OUT = "/onnx_out"
GATE_OUT = "/gate_out"

image = (
    modal.Image.from_registry("nvcr.io/nvidia/tensorrt:24.10-py3", add_python="3.12")
    .apt_install("git", "curl", "clang")
    .pip_install("Pillow>=10.0.0")  # for synthetic input image generation
    .run_commands(
        # SHA-pinned so the image rebuilds on every commit (avoids stale-image cache).
        f'pip install "fastcrest-tether[serve,gpu] @ '
        f'git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
        secrets=[_gh_secret()],
    )
    .env({
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": f"{HF_CACHE}/transformers",
    })
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output, GATE_OUT: gate_output},
    secrets=[_hf_secret()],
)
def fire_gate(
    export_dir: str = f"{ONNX_OUT}/distill_v050r2_decomposed",
    n_episodes_per_run: int = 30,
    chunk_size_per_episode: int = 8,
    low_latency_ms: float = 20.0,
    high_latency_ms: float = 100.0,
    output_label: str = "run_001",
    train_epochs: int = 5,
    train_batch_size: int = 32,
    action_dim: int = 7,
    obs_dim: int = 32,
):
    """Fire the gate end-to-end. Returns dict with decision + report contents."""
    import base64
    import gzip
    import json
    import signal
    import time
    import urllib.error
    import urllib.request
    from io import BytesIO
    from pathlib import Path

    import numpy as np

    out_root = Path(GATE_OUT) / output_label
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_root}")
    print(f"export: {export_dir}")
    print(f"head SHA on Modal image: {_HEAD}")

    # ---- Helpers --------------------------------------------------------------

    def wait_for_health(url: str, timeout_s: int = 240) -> bool:
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

    # Synthetic but consistent inputs — same RNG seed across both latency runs.
    try:
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (224, 224), color=(128, 128, 128))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        grey_jpeg_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        grey_jpeg_b64 = ""

    instructions = [
        "pick up the cup", "place the block", "open the drawer",
        "close the gripper", "move to home", "rotate left",
    ]

    def make_inputs(seed: int = 42):
        rng = np.random.default_rng(seed)
        for ep in range(n_episodes_per_run):
            for chunk in range(chunk_size_per_episode):
                yield {
                    "image": grey_jpeg_b64,
                    "instruction": instructions[(ep + chunk) % len(instructions)],
                    "state": rng.uniform(-1, 1, size=8).tolist(),
                    "episode_id": f"gate_ep_{ep:03d}",
                }

    def run_serve_and_record(latency_ms: float, traces_dir: Path):
        traces_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "tether", "serve", export_dir,
            "--port", "8000",
            "--device", "cuda",
            "--no-strict-providers",
            "--no-prewarm",  # gate fires synthetic requests; full prewarm
                             # adds 60-90s + risks /health timeout per
                             # 2026-04-25 re-fire failure log
            "--inject-latency-ms", str(latency_ms),
            "--record", str(traces_dir),
            "--record-no-gzip",
        ]
        print(f"\n[latency={latency_ms}ms] starting: {' '.join(cmd)}")
        env = {**os.environ}
        proc = subprocess.Popen(cmd, env=env)
        try:
            # 480s budget covers cold-container model load (~30-60s) +
            # ORT session creation (~30s) + lifespan startup (~5s).
            # Was 240s; bumped 2026-04-25 after re-fire failure (server
            # came up but exceeded the old budget).
            if not wait_for_health("http://127.0.0.1:8000/health", 480):
                proc.terminate()
                raise RuntimeError(f"server didn't come up in 480s for latency={latency_ms}")
            n_total = n_episodes_per_run * chunk_size_per_episode
            print(f"[latency={latency_ms}ms] sending {n_total} requests")
            t0 = time.time()
            n_ok = n_err = 0
            for i, inp in enumerate(make_inputs(seed=42)):
                req = urllib.request.Request(
                    "http://127.0.0.1:8000/act",
                    data=json.dumps(inp).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        body = r.read()
                        n_ok += 1
                        if i == 0:
                            try:
                                first = json.loads(body)
                                print(f"  first response: actions_len={len(first.get('actions', []))} "
                                      f"latency_ms={first.get('latency_ms')} "
                                      f"injected={first.get('injected_latency_ms')}")
                            except Exception:
                                pass
                except Exception as e:
                    n_err += 1
                    if n_err <= 3:
                        print(f"  request {i} failed: {e}")
            elapsed = time.time() - t0
            print(f"[latency={latency_ms}ms] sent {n_ok}/{n_ok+n_err} in {elapsed:.1f}s "
                  f"({(n_ok / max(elapsed, 0.01)):.1f} req/s)")
        finally:
            proc.terminate()
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(3)
        files = sorted(list(traces_dir.glob("*.jsonl")) + list(traces_dir.glob("*.jsonl.gz")))
        print(f"[latency={latency_ms}ms] recorded {len(files)} JSONL files")
        return files

    # ---- 1. Collect traces at LOW latency (training distribution) -----------
    traces_low = out_root / "traces_low"
    low_files = run_serve_and_record(low_latency_ms, traces_low)
    if not low_files:
        return {"decision": "ABORT", "exit_code": 2, "error": "no low-latency traces recorded",
                "report_md": "", "report_json": ""}

    # ---- 2. Collect traces at HIGH latency (held-out distribution) ----------
    traces_high = out_root / "traces_high"
    high_files = run_serve_and_record(high_latency_ms, traces_high)
    if not high_files:
        return {"decision": "ABORT", "exit_code": 2, "error": "no high-latency traces recorded",
                "report_md": "", "report_json": ""}

    # ---- 3. Train A2C2 head on low-latency traces ---------------------------
    # Path A unification (2026-04-25): pure-numpy trainer; produces .npz that
    # runtime/a2c2_hook.py loads directly. No torch on this path.
    from tether.correction import (
        A2C2Config, train_a2c2_head, evaluate_mse,
    )

    cfg = A2C2Config(
        action_dim=action_dim, obs_dim=obs_dim,
        chunk_size=50, hidden_dim=128, num_hidden_layers=3,
        position_encoding_dim=32,
    )
    print(f"\nA2C2Config: input={cfg.input_dim} hidden={cfg.hidden_dim} × {cfg.num_hidden_layers} "
          f"output={cfg.action_dim} (~{cfg.estimated_size_bytes()/1024:.1f} KB FP32)")

    def _flatten(records: list) -> tuple:
        base_rows, obs_rows, chunk_idx_rows, latency_rows = [], [], [], []
        n_skipped_dim = 0
        n_skipped_empty = 0
        for rec in records:
            actions = rec.get("actions") or rec.get("response", {}).get("actions")
            state = rec.get("state") or rec.get("request", {}).get("state")
            if not actions or not state:
                n_skipped_empty += 1
                continue
            base_lat = float(rec.get("latency_ms") or rec.get("latency_total_ms") or 0.0)
            inj = float(rec.get("injected_latency_ms") or 0.0)
            obs_lat = base_lat + inj
            for ci, action in enumerate(actions):
                a = np.asarray(action, dtype=np.float32)
                # Pi05 exports return 32-dim padded actions; we only care
                # about the first action_dim. TRUNCATE rather than reject
                # (caught 2026-04-25 b4 gate v6: every record skipped because
                # decomposed actions were 32-dim and gate default was 7).
                if a.shape[0] >= action_dim:
                    a = a[:action_dim]
                else:
                    n_skipped_dim += 1
                    continue
                obs_pad = np.zeros(obs_dim, dtype=np.float32)
                s = np.asarray(state[:obs_dim], dtype=np.float32)
                obs_pad[: s.shape[0]] = s
                base_rows.append(a)
                obs_rows.append(obs_pad)
                chunk_idx_rows.append(ci)
                latency_rows.append(obs_lat)
        if n_skipped_empty or n_skipped_dim:
            print(
                f"  flatten: skipped {n_skipped_empty} empty + "
                f"{n_skipped_dim} dim-too-small records"
            )
        if not base_rows:
            return None
        return (np.asarray(base_rows, dtype=np.float32),
                np.asarray(obs_rows, dtype=np.float32),
                np.asarray(chunk_idx_rows, dtype=np.int64),
                np.asarray(latency_rows, dtype=np.float32))

    def _load_split(files: list[Path]) -> tuple:
        recs = []
        for p in files:
            opener = gzip.open if p.suffix == ".gz" else open
            with opener(p, "rt") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        print(f"  loaded {len(recs)} records from {len(files)} file(s)")
        return _flatten(recs)

    low_split = _load_split(low_files)
    if low_split is None:
        return {"decision": "ABORT", "exit_code": 2,
                "error": "low-latency traces had no usable rows",
                "report_md": "", "report_json": ""}
    base_l, obs_l, ci_l, lat_l = low_split
    high_split = _load_split(high_files)
    if high_split is None:
        return {"decision": "ABORT", "exit_code": 2,
                "error": "high-latency traces had no usable rows",
                "report_md": "", "report_json": ""}
    base_h, obs_h, ci_h, lat_h = high_split

    print(f"\ntrain rows (low): {base_l.shape[0]}; held-out rows (high): {base_h.shape[0]}")

    target_residual_l = np.zeros_like(base_l)  # magnitude-proxy; ADR documents

    print("\ntraining A2C2 head (numpy + Adam):")
    train_result = train_a2c2_head(
        base_actions=base_l,
        observations=obs_l,
        chunk_positions=ci_l,
        latency_ms_per_step=lat_l,
        target_residuals=target_residual_l,
        cfg=cfg,
        epochs=train_epochs,
        batch_size=train_batch_size,
        lr=1e-3,
        val_split=0.1,
        seed=42,
    )
    head = train_result.head
    for em in train_result.metrics["epochs"]:
        print(f"  epoch {em['epoch']} train={em['train_loss']:.6f} val={em['val_loss']:.6f}")

    # Path A unification: emit .npz directly (loadable by runtime/a2c2_hook.py)
    head_path = out_root / "a2c2_head.npz"
    head.save(head_path)
    n_params = sum(w.size + b.size for w, b in zip(head._weights, head._biases))
    print(f"checkpoint saved: {head_path} ({head_path.stat().st_size / 1024:.1f} KB, {n_params} params)")

    # ---- 4. Fire gate -------------------------------------------------------
    from tether.correction.transfer_gate import GateThresholds, compute_gate_report

    # Evaluate against zero-target (magnitude-of-correction proxy on real data).
    in_dist_mse = evaluate_mse(
        head,
        base_actions=base_l, observations=obs_l,
        chunk_positions=ci_l, latency_ms_per_step=lat_l,
        target_residuals=target_residual_l,
    )
    held_out_mse = evaluate_mse(
        head,
        base_actions=base_h, observations=obs_h,
        chunk_positions=ci_h, latency_ms_per_step=lat_h,
        target_residuals=np.zeros_like(base_h),
    )
    print(f"\nin-distribution MSE (low, val split):  {in_dist_mse:.6f}")
    print(f"held-out MSE (high latency):           {held_out_mse:.6f}")
    print(f"ratio: {held_out_mse / max(in_dist_mse, 1e-12):.3f}")

    n_train_rows = base_l.shape[0] - max(1, int(base_l.shape[0] * 0.1))
    n_val_rows = max(1, int(base_l.shape[0] * 0.1))
    notes = [
        f"B.4 MSE-arm fire on Modal A10G; tether SHA={_HEAD}",
        f"low={low_latency_ms}ms vs high={high_latency_ms}ms",
        f"head: action_dim={action_dim} obs_dim={obs_dim} params={n_params}",
        f"train rows={n_train_rows} val rows={n_val_rows} held-out rows={base_h.shape[0]}",
        f"export: {export_dir}",
        "task-success arm DEFERRED — requires B.5 minimal /act wiring + LIBERO eval",
        "MSE is magnitude proxy on real data (per ADR 2026-04-24-a2c2-gate-modal-synthetic-latency)",
        "trainer: numpy + Adam (Path A unification 2026-04-25 — single A2C2Head shared with serve runtime)",
    ]
    report = compute_gate_report(
        in_dist_mse=in_dist_mse,
        held_out_mses=[held_out_mse],
        task_success_on=0.5,  # not measured; arm deferred
        task_success_off=0.5,
        eval_latency_ms=high_latency_ms,
        thresholds=GateThresholds(),
        notes=notes,
    )

    report_md = out_root / "b4_gate_report.md"
    report_json = out_root / "b4_gate_report.json"
    report.write(report_md)
    report.write(report_json)
    exit_code = {"PROCEED": 0, "PAUSE": 1, "ABORT": 2}[report.decision.value]
    print(f"\n{'='*70}\nDECISION: {report.decision.value} (exit={exit_code})")
    if report.failure_reasons:
        for r in report.failure_reasons:
            print(f"  reason: {r}")
    print(f"{'='*70}")

    gate_output.commit()

    return {
        "report_md_path": str(report_md),
        "report_json_path": str(report_json),
        "exit_code": exit_code,
        "decision": report.decision.value,
        "report_md": report_md.read_text(),
        "report_json": report_json.read_text(),
        "in_dist_mse": in_dist_mse,
        "held_out_mse": held_out_mse,
        "ratio": held_out_mse / max(in_dist_mse, 1e-12),
    }


@app.local_entrypoint()
def main(
    n_episodes: int = 30,
    chunk_size: int = 8,
    low_ms: float = 20.0,
    high_ms: float = 100.0,
    label: str = "run_001",
    train_epochs: int = 5,
):
    result = fire_gate.remote(
        n_episodes_per_run=n_episodes,
        chunk_size_per_episode=chunk_size,
        low_latency_ms=low_ms,
        high_latency_ms=high_ms,
        output_label=label,
        train_epochs=train_epochs,
    )
    print("\n" + "=" * 70)
    print(f"GATE DECISION: {result.get('decision')}")
    # Defensive formatting: when the underlying gate aborted on bad data
    # (e.g., empty action chunks from missing ONNX), the MSE/ratio fields
    # come back as None. Don't crash the printer.
    def _fmt(v, spec):
        if v is None:
            return "(none -- gate aborted before measurement)"
        try:
            return format(v, spec)
        except (TypeError, ValueError):
            return repr(v)

    print(f"in-dist MSE:  {_fmt(result.get('in_dist_mse'), '.6f')}")
    print(f"held-out MSE: {_fmt(result.get('held_out_mse'), '.6f')}")
    print(f"ratio:        {_fmt(result.get('ratio'), '.3f')}")
    print(f"exit code:    {result.get('exit_code')}")
    print("=" * 70)
    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        print("\n--- REPORT (markdown) ---")
        print(result.get("report_md", "")[:5000])
