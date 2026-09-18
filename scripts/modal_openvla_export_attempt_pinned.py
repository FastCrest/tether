"""Modal: OpenVLA export-parity attempt (Tether Studio M4 #54).

Attempts the documented OpenVLA export + receipt-grade parity path on Modal
Linux + CUDA, pinned to exact commits. Unlike pi0.5/GR00T this family has no
working exporter: `optimum-cli export onnx` cannot dispatch on
`model_type: "openvla"` (remote-code `auto_map`), `tether export` raises
`NotImplementedError`, and the monolithic exporter has no OpenVLA branch.
This job produces first-hand GPU-side evidence of that state instead of a
receipt:

1. clones the exact Tether implementation commit,
2. snapshots `openvla/openvla-7b` at the exact pinned revision,
3. attempts `optimum-cli export onnx` from that snapshot and records the
   outcome (expected: failure at architecture dispatch),
4. runs `scripts/local_openvla_monolithic_parity.py` to show it refuses
   loudly (no model.onnx, no receipt, nonzero exit) rather than passing
   vacuously,
5. writes an attempt record (explicitly NOT a parity receipt) to the
   shared output volume and prints the full evidence to stdout.

A self-kill wall-clock watchdog bounds GPU burn; the export attempt is
expected to fail fast at dispatch, so the watchdog is short.

Usage:
    modal run scripts/modal_openvla_export_attempt_pinned.py
"""

import os

import modal


# `openvla/openvla-7b` HEAD (2026-02-17 README update; weights/code stable
# since 2024-09-16). Pinned exactly; floating HEAD is refused.
OPENVLA_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"

# Exact Tether implementation commit under test. This vehicle adds only new
# files; src/ and the harness are byte-identical to this commit.
TETHER_REVISION = "8f464ebdff9cda09d9f30fca0af5b409cfd794b6"

TETHER_REPO = "https://github.com/FastCrest/tether.git"

app = modal.App("tether-openvla-export-attempt-pinned")
_hf_cache_volume = modal.Volume.from_name(
    "openvla-hf-cache", create_if_missing=True
)
_parity_out_volume = modal.Volume.from_name(
    "parity-gpu-receipts", create_if_missing=True
)

HF_CACHE_PATH = "/root/.cache/huggingface"
PARITY_OUT_PATH = "/parity_out"


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
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=4.40,<5.4",
        "optimum[onnxruntime]",
        "huggingface_hub",
        "safetensors>=0.4.0",
        "numpy",
        "Pillow",
        "accelerate",
    )
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={
        HF_CACHE_PATH: _hf_cache_volume,
        PARITY_OUT_PATH: _parity_out_volume,
    },
    secrets=[_hf_secret()],
)
def run_export_attempt(
    model_revision: str = OPENVLA_REVISION,
    tether_revision: str = TETHER_REVISION,
    optimum_task: str = "image-text-to-text",
    watchdog_seconds: float = 1500,
) -> dict:
    import json as _json
    import subprocess
    import sys
    import time
    from pathlib import Path

    import os as _os
    import signal as _signal
    import threading as _threading

    _WATCHDOG_SECONDS = float(
        _os.environ.get("PARITY_WATCHDOG_SECONDS", str(watchdog_seconds))
    )

    def _self_kill() -> None:
        print(
            f"[watchdog] deadline of {_WATCHDOG_SECONDS:.0f}s reached; killing this "
            "container rather than burning GPU time on a hung step.",
            flush=True,
        )
        _os.kill(_os.getpid(), _signal.SIGKILL)

    _watchdog = _threading.Timer(_WATCHDOG_SECONDS, _self_kill)
    _watchdog.daemon = True
    _watchdog.start()

    _started_at = time.monotonic()

    def _progress(stage: str) -> None:
        print(f"[progress] {time.monotonic() - _started_at:7.1f}s  {stage}", flush=True)

    _progress("container up, watchdog armed")
    record: dict = {
        "kind": "openvla-export-attempt",
        "is_parity_receipt": False,
        "model_source": "openvla/openvla-7b",
        "model_revision": model_revision,
        "tether_revision": tether_revision,
    }

    pin_dir = Path("/root/tether-pin")
    if pin_dir.exists():
        subprocess.run(["rm", "-rf", str(pin_dir)], check=True)
    subprocess.run(["git", "clone", "--quiet", TETHER_REPO, str(pin_dir)], check=True)
    subprocess.run(
        ["git", "-C", str(pin_dir), "checkout", "--quiet", tether_revision],
        check=True,
    )
    got = subprocess.check_output(
        ["git", "-C", str(pin_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    assert got == tether_revision, f"tether pin mismatch: {got} != {tether_revision}"
    record["tether_commit"] = got
    _progress(f"tether pinned at {got[:12]}")

    from huggingface_hub import snapshot_download

    _progress("snapshot_download openvla/openvla-7b@" + model_revision[:12])
    snap = snapshot_download("openvla/openvla-7b", revision=model_revision)
    record["snapshot_dir"] = snap
    _progress("snapshot complete")

    export_dir = Path(PARITY_OUT_PATH) / "openvla" / "export-attempt"
    export_dir.mkdir(parents=True, exist_ok=True)
    _progress("attempting optimum-cli export onnx from pinned snapshot")
    # --task is required: Optimum cannot infer a task from a local snapshot
    # dir. With the task given, export must then dispatch on
    # config.model_type ("openvla") -- the step that has no mapping.
    export_proc = subprocess.run(
        [
            "optimum-cli", "export", "onnx",
            "--model", snap,
            "--task", optimum_task,
            str(export_dir),
        ],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    record["optimum_task"] = optimum_task
    record["export_returncode"] = export_proc.returncode
    record["export_stdout_tail"] = export_proc.stdout[-3000:]
    record["export_stderr_tail"] = export_proc.stderr[-3000:]
    record["model_onnx_present"] = (export_dir / "model.onnx").is_file()
    print("--- optimum-cli stdout (tail) ---", flush=True)
    print(record["export_stdout_tail"], flush=True)
    print("--- optimum-cli stderr (tail) ---", flush=True)
    print(record["export_stderr_tail"], flush=True)
    _progress(
        f"export attempt rc={export_proc.returncode} "
        f"model.onnx present={record['model_onnx_present']}"
    )

    harness_proc = subprocess.run(
        [
            sys.executable,
            str(pin_dir / "scripts" / "local_openvla_monolithic_parity.py"),
        ]
        + [
            "--model-source", "openvla/openvla-7b",
            "--model-revision", model_revision,
            "--onnx-dir", str(export_dir),
            "--provider", "CUDAExecutionProvider",
            "--action-dim", "7",
            "--trust-remote-code",
            "--receipt",
            str(Path(PARITY_OUT_PATH) / "openvla" / "openvla-export-parity-receipt.json"),
        ],
        cwd=str(pin_dir),
        capture_output=True,
        text=True,
        env={
            **_os.environ,
            "PYTHONPATH": str(pin_dir / "src")
            + _os.pathsep
            + _os.environ.get("PYTHONPATH", ""),
        },
    )
    record["harness_returncode"] = harness_proc.returncode
    record["harness_stdout_tail"] = harness_proc.stdout[-3000:]
    record["harness_stderr_tail"] = harness_proc.stderr[-3000:]
    print("--- parity harness stdout (tail) ---", flush=True)
    print(record["harness_stdout_tail"], flush=True)
    print("--- parity harness stderr (tail) ---", flush=True)
    print(record["harness_stderr_tail"], flush=True)
    _progress(f"harness rc={harness_proc.returncode} (nonzero = loud refusal)")

    record_path = Path(PARITY_OUT_PATH) / "openvla" / "openvla-export-attempt.json"
    record_path.write_text(_json.dumps(record, sort_keys=True, indent=2) + "\n")
    _parity_out_volume.commit()
    print("[attempt-record] " + _json.dumps(record, sort_keys=True), flush=True)

    _watchdog.cancel()
    _progress("done")
    return record


@app.local_entrypoint()
def main() -> None:
    result = run_export_attempt.remote()
    print("\n=== RESULT ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
