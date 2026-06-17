"""Modal: cuda-graphs A/B benchmark on real pi0.5 decomposed export (Day 8-9).

Per cuda-graphs_plan.md Day 8-9, runs the full 6-gate ship-rule A/B:
  OFF: build_cuda_graph_providers(enabled=False), N iterations, measure latency
  ON:  try_capture_or_fall_back(...), N iterations, measure latency
  → compare_reports() with the 6-gate ship rule

Each session (vlm_prefix + expert_denoise) is measured independently because
they have different shapes + memory footprints (vlm_prefix may OOM on A10G
per the Day-0 spike; expert_denoise captures cleanly on both A10G + A100).

Outputs JSON + MD to the pi0-onnx-outputs volume under
ab_results/{hw}/{export_subdir}/{session}_{mode}.json so the comparison
can be re-evaluated post-hoc without re-running.

Usage:
    modal run scripts/modal_cuda_graphs_ab.py --hw a10g
    modal run scripts/modal_cuda_graphs_ab.py --hw a100 --n-iters 200

Reference:
- features/01_serve/subfeatures/_perf_compound/cuda-graphs/cuda-graphs_plan.md (Day 8-9)
- ADR 01_decisions/2026-04-24-cuda-graphs-architecture.md
- Day-0 spike: 03_experiments/2026-04-25-cuda-graphs-ort-spike-modal.md

Cost: ~$2-3 on A10G, ~$4-5 on A100-80GB, ~10-15 min wall-clock each.
"""
from __future__ import annotations

import os
import subprocess
import modal

app = modal.App("tether-cuda-graphs-ab")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_bust() -> str:
    import time
    return str(int(time.time()))


def _repo_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()
_BUST = _build_bust()

onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=False)
ONNX_OUT = "/onnx_out"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "onnxruntime-gpu==1.20.1",
        "numpy<2.0",
        "onnx>=1.16",
        "prometheus-client>=0.19",
        "fastapi>=0.110",
        "torch==2.5.1",
        "transformers>=4.40",
    )
    .env({
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu",
    })
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
        f'echo "build_bust={_BUST}"',
        'pip install -e "/root/tether-vla" --no-deps',
    )
)


def _run_one_session_ab(
    model_path,
    session_name: str,
    n_warmup: int,
    n_iters: int,
    seed: int = 0,
):
    """OFF + ON A/B for ONE session (vlm_prefix or expert_denoise).

    Returns a dict with both LatencyStats + the BenchCompareResult verdict.
    """
    import time
    import numpy as np
    import onnxruntime as ort

    from tether.runtime.cuda_graphs import (
        build_cuda_graph_providers,
        try_capture_or_fall_back,
    )
    from tether.bench.methodology import compute_stats

    _ORT_TO_NP = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }

    def _make_feed(session, seed_val: int):
        rng = np.random.default_rng(seed_val)
        feed = {}
        for inp in session.get_inputs():
            shape = [1 if (isinstance(d, str) or d is None) else int(d) for d in inp.shape]
            dtype = _ORT_TO_NP.get(inp.type, np.float32)
            if np.issubdtype(dtype, np.floating):
                feed[inp.name] = rng.standard_normal(shape).astype(dtype)
            elif dtype == np.bool_:
                feed[inp.name] = rng.integers(0, 2, size=shape) > 0
            else:
                feed[inp.name] = rng.integers(0, 100, size=shape, dtype=dtype)
        return feed

    print(f"\n=== {session_name} ({model_path.name}) ===")

    # --- OFF run (eager CUDA EP, no graph capture) ----------------------
    off_providers = build_cuda_graph_providers(enabled=False)
    off_session = ort.InferenceSession(str(model_path), providers=off_providers)
    off_active = off_session.get_providers()
    if "CUDAExecutionProvider" not in off_active:
        return {
            "status": "fail",
            "reason": "cuda_ep_not_active_off",
            "active_providers": off_active,
        }
    print(f"  OFF active providers: {off_active}", flush=True)
    feed_off = _make_feed(off_session, seed_val=seed)

    off_latencies_ms = []
    last_log_t = time.perf_counter()
    total = n_warmup + n_iters
    for i in range(total):
        t0 = time.perf_counter()
        _ = off_session.run(None, feed_off)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        off_latencies_ms.append(elapsed_ms)
        now = time.perf_counter()
        if (now - last_log_t) >= 30.0 or i == total - 1:
            done = i + 1
            recent = off_latencies_ms[-min(20, done):]
            recent_mean = sum(recent) / len(recent)
            print(
                f"  OFF progress: {done}/{total} (recent mean ~{recent_mean:.2f}ms)",
                flush=True,
            )
            last_log_t = now
    off_stats = compute_stats(off_latencies_ms, warmup_n=n_warmup)
    print(
        f"  OFF: mean={off_stats.mean_ms:.2f}ms p99={off_stats.p99_ms:.2f}ms "
        f"jitter={off_stats.jitter:.4f} n={off_stats.n}",
        flush=True,
    )

    # --- ON run (cuda_graph=1 via try_capture_or_fall_back) -------------
    def _factory(cg_enabled: bool):
        return ort.InferenceSession(
            str(model_path),
            providers=build_cuda_graph_providers(enabled=cg_enabled),
        )

    feed_on = _make_feed(off_session, seed_val=seed)
    wrapped = try_capture_or_fall_back(
        _factory,
        session_name=session_name,
        embodiment="franka",
        model_id=model_path.parent.name,
        probe_feed=feed_on,
    )
    captured = wrapped.captured
    print(f"  ON capture_succeeded={captured} (wrapper={type(wrapped).__name__})", flush=True)

    on_latencies_ms = []
    last_log_t = time.perf_counter()
    total = n_warmup + n_iters
    for i in range(total):
        t0 = time.perf_counter()
        _ = wrapped.run(None, feed_on)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        on_latencies_ms.append(elapsed_ms)
        now = time.perf_counter()
        if (now - last_log_t) >= 30.0 or i == total - 1:
            done = i + 1
            recent = on_latencies_ms[-min(20, done):]
            recent_mean = sum(recent) / len(recent)
            print(
                f"  ON progress:  {done}/{total} (recent mean ~{recent_mean:.2f}ms)",
                flush=True,
            )
            last_log_t = now
    on_stats = compute_stats(on_latencies_ms, warmup_n=n_warmup)
    print(
        f"  ON:  mean={on_stats.mean_ms:.2f}ms p99={on_stats.p99_ms:.2f}ms "
        f"jitter={on_stats.jitter:.4f} n={on_stats.n}",
        flush=True,
    )

    # --- Parity check (captured matches eager bit-identically?) ---------
    parity_cos = None
    try:
        out_off = off_session.run(None, feed_off)
        out_on = wrapped.run(None, feed_on)
        if len(out_off) == len(out_on) and out_off and out_on:
            a = out_off[0].astype(np.float64).ravel()
            b = out_on[0].astype(np.float64).ravel()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            parity_cos = float(np.dot(a, b) / denom) if denom > 0 else None
    except Exception as exc:
        print(f"  parity check failed: {exc}")

    print(f"  parity cos: {parity_cos}")

    return {
        "status": "ok",
        "session": session_name,
        "captured": bool(captured),
        "wrapper_type": type(wrapped).__name__,
        "off_stats": off_stats.to_dict(),
        "on_stats": on_stats.to_dict(),
        "parity_cos": parity_cos,
        "n_warmup": n_warmup,
        "n_iters": n_iters,
    }


def _run_ab_for_export(export_subdir: str, hw_label: str, n_warmup: int, n_iters: int):
    """Run OFF/ON A/B for both vlm_prefix + expert_denoise of a single export."""
    import json
    from pathlib import Path

    import onnxruntime as ort

    print("=" * 60)
    print(f"cuda-graphs A/B  hw={hw_label}  export={export_subdir}")
    print(f"ORT={ort.__version__}  providers={ort.get_available_providers()}")
    print(f"warmup={n_warmup}  iters={n_iters}")
    print("=" * 60)

    export_dir = Path(ONNX_OUT) / export_subdir
    if not export_dir.exists():
        return {"status": "fail", "reason": f"export_dir_missing: {export_dir}"}

    cfg_path = export_dir / "tether_config.json"
    if not cfg_path.exists():
        return {"status": "fail", "reason": "tether_config.json missing"}
    cfg = json.loads(cfg_path.read_text())
    prefix_path = export_dir / cfg["decomposed"]["vlm_prefix_onnx"]
    expert_path = export_dir / cfg["decomposed"]["expert_denoise_onnx"]

    results = {}
    for session_name, model_path in (
        ("vlm_prefix", prefix_path),
        ("expert_denoise", expert_path),
    ):
        if not model_path.exists():
            results[session_name] = {"status": "skip", "reason": f"missing: {model_path}"}
            continue
        try:
            results[session_name] = _run_one_session_ab(
                model_path, session_name, n_warmup=n_warmup, n_iters=n_iters,
            )
        except Exception as exc:
            results[session_name] = {"status": "fail", "exc": repr(exc)}
            print(f"  EXC: {exc!r}")

    out_dir = Path(ONNX_OUT) / "ab_results" / hw_label / export_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ab_summary.json"
    out_path.write_text(json.dumps({
        "hw": hw_label,
        "export_subdir": export_subdir,
        "ort_version": ort.__version__,
        "n_warmup": n_warmup,
        "n_iters": n_iters,
        "sessions": results,
    }, indent=2))
    print(f"\nSaved: {out_path}")

    return {"status": "ok", "out_path": str(out_path), "sessions": results}


@app.function(
    image=image,
    gpu="A10G",
    volumes={ONNX_OUT: onnx_output},
    timeout=1800,
)
def run_a10g(export_subdir: str, n_warmup: int, n_iters: int):
    return _run_ab_for_export(export_subdir, "a10g", n_warmup, n_iters)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={ONNX_OUT: onnx_output},
    timeout=1800,
)
def run_a100(export_subdir: str, n_warmup: int, n_iters: int):
    return _run_ab_for_export(export_subdir, "a100", n_warmup, n_iters)


@app.local_entrypoint()
def main(
    hw: str = "a10g",
    export_subdir: str = "pi05_decomposed_smoke_local_auto",
    n_warmup: int = 5,
    n_iters: int = 100,
):
    """Entrypoint: pick hw + scoped export. Default A10G + 100 iterations."""
    if hw == "a10g":
        result = run_a10g.remote(export_subdir, n_warmup, n_iters)
    elif hw == "a100":
        result = run_a100.remote(export_subdir, n_warmup, n_iters)
    else:
        raise ValueError(f"hw must be 'a10g' or 'a100', got {hw!r}")

    print("\n" + "=" * 60)
    print(f"AB RESULT  hw={hw}  export={export_subdir}")
    print("=" * 60)
    if result.get("status") != "ok":
        print(f"FAIL: {result}")
        return

    for sname, sres in result["sessions"].items():
        print(f"\n--- {sname} ---")
        if sres.get("status") != "ok":
            print(f"  {sres.get('status')}: {sres.get('reason') or sres.get('exc')}")
            continue
        off = sres["off_stats"]
        on = sres["on_stats"]
        captured = sres["captured"]
        parity = sres.get("parity_cos")
        speedup = off["mean_ms"] / max(on["mean_ms"], 1e-9)
        delta_pct = (on["mean_ms"] - off["mean_ms"]) / max(off["mean_ms"], 1e-9) * 100
        print(f"  captured: {captured}  wrapper: {sres['wrapper_type']}")
        print(f"  OFF mean {off['mean_ms']:.2f}ms p99 {off['p99_ms']:.2f}ms jitter {off['jitter']:.4f}")
        print(f"  ON  mean {on['mean_ms']:.2f}ms p99 {on['p99_ms']:.2f}ms jitter {on['jitter']:.4f}")
        print(f"  speedup: {speedup:.2f}x  mean_delta: {delta_pct:+.2f}%  parity_cos: {parity}")

    print(f"\nFull JSON: {result['out_path']}")
