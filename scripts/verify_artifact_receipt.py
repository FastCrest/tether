#!/usr/bin/env python3
"""Execute a real generated export and write a provenance receipt for #267."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np

from tether.runtime.verify_inference import load_verification_inference
from tether.smoke import create_smoke_export


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_smoke_actions(actions: np.ndarray, noise: np.ndarray) -> None:
    """Fail the workflow unless the generated Constant export truly executed."""
    expected = np.zeros_like(noise)
    if not np.array_equal(actions, expected):
        raise RuntimeError("smoke export actions do not match its declared constant graph")


def build_receipt(output: Path) -> dict[str, object]:
    event_name = os.environ.get("GITHUB_EVENT_NAME")
    ref = os.environ.get("GITHUB_REF")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        if event_name not in {"push", "workflow_dispatch"}:
            raise RuntimeError(f"receipt event is not allowlisted: {event_name!r}")
        if ref != "refs/heads/main":
            raise RuntimeError(f"receipt must execute from protected main, got {ref!r}")
        if os.environ.get("GITHUB_REF_PROTECTED") != "true":
            raise RuntimeError("main is not reported as a protected ref")
    with tempfile.TemporaryDirectory(prefix="tether-verify-artifact-") as temporary:
        export_dir = create_smoke_export(Path(temporary) / "export")
        inference = load_verification_inference(export_dir, device="cpu")
        noise = np.arange(50 * 32, dtype=np.float32).reshape(1, 50, 32)
        image = np.zeros((1, 3, 512, 512), dtype=np.float32)
        mask = np.ones((1,), dtype=np.bool_)
        actions = inference.predict_action_chunk(
            img_base=image,
            img_wrist_l=image,
            img_wrist_r=image,
            mask_base=mask,
            mask_wrist_l=mask,
            mask_wrist_r=mask,
            lang_tokens=np.zeros((1, 16), dtype=np.int64),
            lang_masks=np.ones((1, 16), dtype=np.bool_),
            noise=noise,
            state=np.zeros((1, 32), dtype=np.float32),
            episode_id="receipt",
        )
        if actions.shape != noise.shape:
            raise RuntimeError(f"unexpected action shape {actions.shape}; expected {noise.shape}")
        _validate_smoke_actions(actions, noise)
        receipt: dict[str, object] = {
            "schema_version": 1,
            "kind": "tether.verify_artifact_execution",
            "source_sha": os.environ.get("GITHUB_SHA", "local"),
            "workflow": os.environ.get("GITHUB_WORKFLOW", "local"),
            "workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF", "local"),
            "repository": os.environ.get("GITHUB_REPOSITORY", "local"),
            "event_name": event_name or "local",
            "ref": ref or "local",
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_type": "smolvla",
            "export_kind": "monolithic_onnx",
            "backend": inference.get_stats().get("backend"),
            "model_sha256": _sha256(export_dir / "model.onnx"),
            "config_sha256": _sha256(export_dir / "tether_config.json"),
            "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            "python": platform.python_version(),
            "passed": True,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("verify-artifact-receipt.json"))
    args = parser.parse_args()
    print(json.dumps(build_receipt(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
