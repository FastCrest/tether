"""Verify `reflex bench` works for all 4 supported VLAs with auto-TRT FP16.

The smolvla install-path test confirmed auto-TRT works for that model. This
script extends to pi0, pi0.5, gr00t — bigger models with longer engine builds
that are more likely to surface edge cases.

For each model:
  reflex export <hf_id> --target desktop
  reflex bench <export_dir> --iterations 50 --warmup 10
Capture inference_mode and per-chunk latency from the bench output.

Usage:
    modal run scripts/modal_verify_bench_all.py
    modal run scripts/modal_verify_bench_all.py --gpu L40S --models smolvla,pi0
    modal run scripts/modal_verify_bench_all.py --gpu L40S --source local --spawn-only
"""

import modal
from pathlib import Path

app = modal.App("reflex-bench-all-verify")

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/tensorrt:24.10-py3",
        add_python="3.12",
    )
    .apt_install("git", "clang", "build-essential")
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
local_image = image.add_local_dir(
    _REPO_ROOT,
    "/workspace/reflex-vla",
    copy=True,
    ignore=[
        ".git",
        ".venv",
        ".mypy_cache",
        "dist",
        "build",
        "build_binary",
        "reference",
        ".pytest_cache",
        ".ruff_cache",
        "**/__pycache__",
        "*.pyc",
    ],
)


def _run_streamed(cmd, *, timeout: int):
    import os
    import selectors
    import subprocess
    import time

    started = time.time()
    heartbeat_s = 60.0
    last_heartbeat = started
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )
    lines: list[str] = []
    timed_out = False
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    os.set_blocking(fd, False)
    selector = selectors.DefaultSelector()
    selector.register(fd, selectors.EVENT_READ)
    try:
        while True:
            now = time.time()
            elapsed = now - started
            if elapsed > timeout:
                timed_out = True
                proc.kill()
                returncode = proc.wait()
                break

            if proc.poll() is not None:
                while True:
                    try:
                        chunk = os.read(fd, 65536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    print(text, end="", flush=True)
                    lines.append(text)
                returncode = proc.returncode
                break

            events = selector.select(timeout=1.0)
            if events:
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    chunk = b""
                if chunk:
                    text = chunk.decode("utf-8", errors="replace")
                    print(text, end="", flush=True)
                    lines.append(text)
                continue

            now = time.time()
            if now - last_heartbeat >= heartbeat_s:
                print(
                    f"[modal-verify] still running ({now - started:.0f}s/{timeout}s): "
                    + " ".join(str(part) for part in cmd[:4]),
                    flush=True,
                )
                last_heartbeat = now
    finally:
        selector.close()

    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout": "".join(lines),
        "elapsed_s": time.time() - started,
    }


def _run_all_impl(
    gpu_label: str = "A10G",
    models_csv: str = "",
    export_timeout: int = 1800,
    artifact_dir: str = "/tmp/reflex_bench_artifacts",
    source: str = "git",
):
    import os
    import json
    import re
    import time
    from pathlib import Path

    # Install reflex. "git" validates the public install path; "local" validates
    # the exact dirty worktree mounted into the Modal image.
    source_norm = source.lower()
    # This script runs on NVIDIA's TensorRT NGC image, so libnvinfer is already
    # present. Use gpu-min to avoid rebuilding/downloading PyPI's multi-GB
    # TensorRT wheel while still installing ORT-GPU and CUDA libs.
    extras = "serve,gpu-min,monolithic"
    install_cmd = [
        "pip", "install",
        f"reflex-vla[{extras}] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/reflex-vla",
    ]
    if source_norm == "local":
        install_cmd = ["pip", "install", "-e", f"/workspace/reflex-vla[{extras}]"]
    print(
        f"=== Installing reflex-vla[{extras}] from {source_norm} ===",
        flush=True,
    )
    r = _run_streamed(
        install_cmd,
        timeout=900,
    )
    if r["returncode"] != 0:
        return {
            "error": "pip install failed",
            "timed_out": r["timed_out"],
            "stdout_tail": r["stdout"][-4000:],
        }
    print("  install ok", flush=True)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    all_models = [
        ("smolvla", "lerobot/smolvla_base"),
        ("pi0", "lerobot/pi0_base"),
        ("pi05", "lerobot/pi05_base"),
        ("gr00t", "nvidia/GR00T-N1.6-3B"),
    ]
    if models_csv:
        wanted = {m.strip() for m in models_csv.split(",") if m.strip()}
        models = [m for m in all_models if m[0] in wanted]
    else:
        models = all_models

    results = {}
    for tag, hf_id in models:
        print(f"\n{'='*60}\n{tag} ({hf_id})\n{'='*60}", flush=True)
        export_dir = f"/tmp/{tag}"

        # Export
        t0 = time.time()
        r = _run_streamed(
            ["reflex", "export", hf_id, "--target", "desktop", "--output", export_dir],
            timeout=export_timeout,
        )
        export_s = time.time() - t0
        if r["returncode"] != 0:
            results[tag] = {
                "export_s": round(export_s, 1),
                "export_error": r["stdout"][-4000:],
                "export_timed_out": r["timed_out"],
            }
            reason = "TIMEOUT" if r["timed_out"] else "FAIL"
            print(f"  EXPORT {reason} ({export_s:.1f}s)", flush=True)
            continue
        files = os.listdir(export_dir)
        print(f"  export ok ({export_s:.1f}s, files={files})", flush=True)

        # Bench
        t0 = time.time()
        report_json = str(Path(artifact_dir) / f"{tag}_{gpu_label.lower()}_bench.json")
        report_md = str(Path(artifact_dir) / f"{tag}_{gpu_label.lower()}_bench.md")
        r = _run_streamed(
            ["reflex", "bench", export_dir, "--iterations", "50", "--warmup", "10",
             "--device", "cuda", "--report-json", report_json, "--report", report_md],
            timeout=900,
        )
        bench_s = time.time() - t0
        if r["returncode"] != 0:
            results[tag] = {
                "export_s": round(export_s, 1),
                "bench_error": r["stdout"][-4000:],
                "bench_timed_out": r["timed_out"],
            }
            reason = "TIMEOUT" if r["timed_out"] else "FAIL"
            print(f"  BENCH {reason} ({bench_s:.1f}s)", flush=True)
            continue

        # Parse the bench output. The CLI prints lines like:
        #   mean    11.52 ms
        #   p50     11.50 ms
        #   p95     11.85 ms
        #   p99     12.01 ms
        #   hz       86.8
        #   Inference mode: onnx_trt_fp16
        out = r["stdout"]
        def _parse(label):
            m = re.search(rf"^\s*{re.escape(label)}\s+([\d.]+)\s*ms?", out, re.MULTILINE)
            return float(m.group(1)) if m else None

        mode_match = re.search(r"Inference mode:\s*(\S+)", out)
        mode = mode_match.group(1) if mode_match else "?"
        provider_match = re.search(r"Provider:\s*(\S+)", out)
        provider_mode = provider_match.group(1) if provider_match else "?"
        active_match = re.search(r"Active EP:\s*(.+)", out)
        active_providers = (
            [p.strip() for p in active_match.group(1).split(",")]
            if active_match else []
        )

        report_payload = None
        if Path(report_json).exists():
            try:
                report_payload = json.loads(Path(report_json).read_text())
                stats = report_payload.get("stats", {})
                env = report_payload.get("environment", {})
                provider_mode = env.get("provider_mode") or provider_mode
                active_providers = env.get("active_providers") or active_providers
            except Exception as exc:
                print(f"  report parse skipped: {exc}", flush=True)
                stats = {}
        else:
            stats = {}

        results[tag] = {
            "export_s": round(export_s, 1),
            "bench_s": round(bench_s, 1),
            "mean_ms": stats.get("mean_ms") or _parse("mean"),
            "p50_ms": stats.get("p50_ms") or _parse("p50"),
            "p95_ms": stats.get("p95_ms") or _parse("p95"),
            "p99_ms": stats.get("p99_ms") or _parse("p99"),
            "p99_9_ms": stats.get("p99_9_ms"),
            "min_ms": stats.get("min_ms") or _parse("min"),
            "inference_mode": mode,
            "provider_mode": provider_mode,
            "active_providers": active_providers,
            "report_json": report_json if Path(report_json).exists() else None,
            "report_md": report_md if Path(report_md).exists() else None,
            "report": report_payload,
        }
        print(f"  bench ok ({bench_s:.1f}s): mean={results[tag]['mean_ms']}ms "
              f"p95={results[tag]['p95_ms']}ms mode={mode} provider={provider_mode}",
              flush=True)

        # Free disk between models — checkpoints are huge
        _run_streamed(["rm", "-rf", export_dir], timeout=60)

    print(f"\n{'='*80}", flush=True)
    print(f"VERDICT — `reflex bench` on {gpu_label}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Model':<10} {'mean_ms':>10} {'p95_ms':>10} {'provider':>16} {'mode':>20} {'export_s':>10}", flush=True)
    for tag, r in results.items():
        if "export_error" in r:
            print(f"{tag:<10} EXPORT FAILED", flush=True)
        elif "bench_error" in r:
            print(f"{tag:<10} BENCH FAILED ({r.get('export_s', 0)}s export)", flush=True)
        else:
            mean = r.get("mean_ms", "—")
            p95 = r.get("p95_ms", "—")
            mode = r.get("inference_mode", "?")
            provider = r.get("provider_mode", "?")
            exp = f"{r.get('export_s', 0)}s"
            print(f"{tag:<10} {str(mean):>10} {str(p95):>10} {provider:>16} {mode:>20} {exp:>10}", flush=True)
    print("\n=== JSON ===", flush=True)
    print(json.dumps(results, indent=2, default=str), flush=True)
    return results


@app.function(image=image, gpu="A10G", timeout=7200)
def run_all_a10g(models_csv: str = "", export_timeout: int = 1800, artifact_dir: str = "/tmp/reflex_bench_artifacts"):
    return _run_all_impl(
        gpu_label="A10G",
        models_csv=models_csv,
        export_timeout=export_timeout,
        artifact_dir=artifact_dir,
        source="git",
    )


@app.function(image=image, gpu="L40S", timeout=7200)
def run_all_l40s(models_csv: str = "", export_timeout: int = 1800, artifact_dir: str = "/tmp/reflex_bench_artifacts"):
    return _run_all_impl(
        gpu_label="L40S",
        models_csv=models_csv,
        export_timeout=export_timeout,
        artifact_dir=artifact_dir,
        source="git",
    )


@app.function(image=local_image, gpu="A10G", timeout=7200)
def run_all_a10g_local(models_csv: str = "", export_timeout: int = 1800, artifact_dir: str = "/tmp/reflex_bench_artifacts"):
    return _run_all_impl(
        gpu_label="A10G",
        models_csv=models_csv,
        export_timeout=export_timeout,
        artifact_dir=artifact_dir,
        source="local",
    )


@app.function(image=local_image, gpu="L40S", timeout=7200)
def run_all_l40s_local(models_csv: str = "", export_timeout: int = 1800, artifact_dir: str = "/tmp/reflex_bench_artifacts"):
    return _run_all_impl(
        gpu_label="L40S",
        models_csv=models_csv,
        export_timeout=export_timeout,
        artifact_dir=artifact_dir,
        source="local",
    )


@app.local_entrypoint()
def main(
    gpu: str = "A10G",
    models: str = "",
    export_timeout: int = 1800,
    artifact_dir: str = "/tmp/reflex_bench_artifacts",
    source: str = "git",
    spawn_only: bool = False,
):
    import json

    gpu_norm = gpu.upper()
    source_norm = source.lower()
    print(f"Verifying `reflex bench` works on {gpu_norm} (auto-TRT, source={source_norm})\n")
    if gpu_norm == "A10G" and source_norm == "git":
        fn = run_all_a10g
    elif gpu_norm == "L40S" and source_norm == "git":
        fn = run_all_l40s
    elif gpu_norm == "A10G" and source_norm == "local":
        fn = run_all_a10g_local
    elif gpu_norm == "L40S" and source_norm == "local":
        fn = run_all_l40s_local
    else:
        raise ValueError(
            f"Unsupported --gpu/source {gpu!r}/{source!r}; expected gpu A10G|L40S "
            f"and source git|local"
        )

    kwargs = {
        "models_csv": models,
        "export_timeout": export_timeout,
        "artifact_dir": artifact_dir,
    }
    if spawn_only:
        call = fn.spawn(**kwargs)
        call.hydrate()
        payload = {
            "function_call_id": call.object_id,
            "gpu": gpu_norm,
            "source": source_norm,
            "models": models,
            "artifact_dir": artifact_dir,
        }
        print("\n=== SPAWNED ===")
        print(json.dumps(payload, indent=2, default=str))
        print("\nUse `modal app logs reflex-bench-all-verify -f` to stream remote logs.")
        print("Use `modal.FunctionCall.from_id(<id>).get()` to fetch the result later.")
        return

    r = fn.remote(**kwargs)
    print("\n=== JSON ===")
    print(json.dumps(r, indent=2, default=str))
