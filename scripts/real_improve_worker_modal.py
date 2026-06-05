"""Run the FastCrest Improve worker contract on Modal.

This is the credentialed smoke wrapper for the Cloud Improve worker path. It
does not talk to the Cloud database and it does not write Registry/Fleet state.
Cloud builds ``fastcrest.improve.worker_input.v1``; this script runs the
Tether-owned worker in Modal and returns a ``fastcrest.improve.worker_result.v1``
envelope that Cloud can ingest through the existing Improve APIs.

Usage:
    modal run scripts/real_improve_worker_modal.py \\
        --worker-input worker_input.json \\
        --output-uri s3://<bucket>/improve/<job_id>/ \\
        --gpu A10G
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

app = modal.App("fastcrest-real-improve-worker")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REMOTE_REPO = "/workspace/tether"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "boto3>=1.34",
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "rich>=13.0",
        "typer>=0.9.0",
    )
    .add_local_dir(
        _REPO_ROOT,
        _REMOTE_REPO,
        copy=True,
        ignore=[
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "build",
            "dist",
            "**/__pycache__",
            "*.pyc",
        ],
    )
    .run_commands(f"cd {_REMOTE_REPO} && pip install -e .")
)


@app.function(image=image, timeout=3600, scaledown_window=60)
def run_worker_remote(
    worker_input: dict[str, Any],
    *,
    output_uri: str,
    result_pretty: bool = False,
) -> dict[str, Any]:
    import os
    import time
    from pathlib import Path

    from tether.finetune.improve_worker import run_improve_worker

    job_id = str(worker_input.get("job_id") or "unknown_job")
    run_dir = Path("/tmp/fastcrest_improve_worker") / job_id
    started = time.time()
    result = run_improve_worker(
        worker_input,
        output_dir=run_dir,
        output_uri=output_uri,
        now=started,
    )
    result.setdefault("metadata", {})
    result["metadata"] = {
        **dict(result.get("metadata") or {}),
        "modal_app": "fastcrest-real-improve-worker",
        "modal_smoke": True,
        "modal_gpu_requested": os.environ.get("MODAL_GPU", "unknown"),
        "elapsed_s": time.time() - started,
    }
    if result_pretty:
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    else:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return result


@app.local_entrypoint()
def main(
    worker_input: str,
    output_uri: str,
    gpu: str = "A10G",
    timeout_seconds: int = 3600,
    result_output: str = "",
    pretty: bool = False,
) -> None:
    if not output_uri:
        raise SystemExit("--output-uri is required")
    payload = json.loads(Path(worker_input).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("--worker-input must be a JSON object")

    runner = run_worker_remote.with_options(gpu=gpu, timeout=timeout_seconds)
    result = runner.remote(payload, output_uri=output_uri, result_pretty=pretty)
    encoded = json.dumps(
        result,
        indent=2 if pretty else None,
        sort_keys=True,
        separators=None if pretty else (",", ":"),
    )
    if result_output:
        Path(result_output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
