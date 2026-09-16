"""Qualification-grade GR00T monolithic ONNX parity harness.

GR00T's exported graph is the per-denoise-step velocity function. This harness
compares that exact hashed ONNX artifact against an exact upstream checkpoint,
exact embodiment and exact shared tensors. The receipt is the public prerequisite
for Tether Studio M4 #53.

A CPU run is local evidence only. External acceptance requires a passing Linux
CUDAExecutionProvider run with ONNX Runtime CPU EP fallback disabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from tether.checkpoint import load_checkpoint
from tether.eval.artifact_identity import build_artifact_identity
from tether.eval.groot_export_parity import MARKER, build_groot_export_parity_receipt
from tether.exporters.gr00t import build_gr00t_full_stack


def _repo_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "Exact Tether Git commit is required for a parity receipt. Run from a Git checkout."
        ) from exc


def _shared_input_digest(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(values[name])
        metadata = json.dumps(
            {"name": name, "dtype": str(array.dtype), "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        payload = array.tobytes(order="C")
        digest.update(len(metadata).to_bytes(4, "big"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.reshape(-1).astype(np.float64)
    b64 = b.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(a64) * np.linalg.norm(b64)
    if denominator == 0:
        return 1.0 if np.array_equal(a64, b64) else 0.0
    return float(np.dot(a64, b64) / denominator)


def _write_receipt(path: Path, receipt: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare an exact GR00T ONNX export with an exact upstream reference."
    )
    parser.add_argument("--model-source", default="nvidia/GR00T-N1.6-3B")
    parser.add_argument(
        "--model-revision",
        required=True,
        help="Exact 40-hex Hugging Face checkpoint commit; floating HEAD/main is refused.",
    )
    parser.add_argument("--embodiment-id", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--onnx-dir",
        default="/tmp/groot_monolithic_onnx",
        help="Directory containing model.onnx and any external ONNX data files.",
    )
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help=(
            "Exact ONNX Runtime provider to request. External acceptance requires "
            "CUDAExecutionProvider."
        ),
    )
    parser.add_argument("--input-seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=99)
    parser.add_argument(
        "--receipt",
        default="groot-export-parity-receipt.json",
        help="Receipt path. Must be outside --onnx-dir so it does not change artifact identity.",
    )
    args = parser.parse_args(argv)

    if args.embodiment_id < 0:
        raise ValueError("--embodiment-id must be non-negative.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    onnx_dir = Path(args.onnx_dir).expanduser().resolve()
    onnx_path = onnx_dir / "model.onnx"
    receipt_path = Path(args.receipt).expanduser().resolve()
    if onnx_dir == receipt_path or onnx_dir in receipt_path.parents:
        raise ValueError("Parity receipt must be stored outside --onnx-dir.")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Missing GR00T ONNX export: {onnx_path}")

    import onnxruntime as ort
    from huggingface_hub import snapshot_download

    tether_commit = _repo_commit()
    print(
        f"Loading exact GR00T reference {args.model_source}@{args.model_revision}...",
        flush=True,
    )
    checkpoint_dir = snapshot_download(args.model_source, revision=args.model_revision)
    state_dict, _ = load_checkpoint(checkpoint_dir)
    reference_model, metadata = build_gr00t_full_stack(
        state_dict,
        embodiment_id=args.embodiment_id,
    )
    reference_model.eval()
    raw_action_dim = int(metadata["raw_action_dim"])

    input_rng = np.random.RandomState(args.input_seed)
    noise_rng = np.random.RandomState(args.noise_seed)
    shared = {
        "noisy_actions": noise_rng.randn(1, args.chunk_size, raw_action_dim).astype(np.float32),
        "timestep": np.array([input_rng.uniform(0.05, 0.95)], dtype=np.float32),
        "position_ids": np.arange(args.chunk_size, dtype=np.int64)[None, :],
    }
    shared_input_sha256 = _shared_input_digest(shared)

    noisy_actions = torch.from_numpy(shared["noisy_actions"])
    timestep = torch.from_numpy(shared["timestep"])
    position_ids = torch.from_numpy(shared["position_ids"])

    print("Running PyTorch GR00TFullStack reference...", flush=True)
    with torch.no_grad():
        reference = reference_model(noisy_actions, timestep, position_ids)
    reference_np = reference.cpu().numpy().astype(np.float32)

    print(f"Running exact ONNX artifact with {args.provider}...", flush=True)
    cpu_fallback_disabled = args.provider == "CUDAExecutionProvider"
    session_options = ort.SessionOptions()
    if cpu_fallback_disabled:
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=[args.provider],
    )
    active_providers = list(session.get_providers())
    export_np = np.asarray(session.run(None, shared)[0], dtype=np.float32)

    first_reference = reference_np[0, 0]
    first_export = export_np[0, 0]
    first_cosine = _cosine(first_reference, first_export)
    first_max_abs = float(np.max(np.abs(first_reference - first_export)))
    full_cosine = _cosine(reference_np, export_np)
    full_max_abs = float(np.max(np.abs(reference_np - export_np)))

    export_identity = build_artifact_identity(
        onnx_dir,
        source="groot-monolithic-export",
        revision=tether_commit,
        kind="groot-monolithic-onnx",
    )
    receipt = build_groot_export_parity_receipt(
        model_source=args.model_source,
        model_revision=args.model_revision,
        tether_commit=tether_commit,
        embodiment_id=args.embodiment_id,
        export_identity=export_identity,
        platform_system=platform.system(),
        ort_providers=active_providers,
        requested_provider=args.provider,
        cpu_fallback_disabled=cpu_fallback_disabled,
        input_seed=args.input_seed,
        noise_seed=args.noise_seed,
        shared_input_sha256=shared_input_sha256,
        reference_shape=list(reference_np.shape),
        export_shape=list(export_np.shape),
        first_cosine=first_cosine,
        first_max_abs=first_max_abs,
        full_cosine=full_cosine,
        full_max_abs=full_max_abs,
    )
    _write_receipt(receipt_path, receipt)

    print("\n====== GR00T EXPORT PARITY ======")
    print(f"  exact model revision: {args.model_revision}")
    print(f"  embodiment id:        {args.embodiment_id}")
    print(f"  export artifact sha:  {export_identity['artifact_sha256']}")
    print(f"  shared input sha:     {shared_input_sha256}")
    print(f"  providers:            {active_providers}")
    print(f"  CPU fallback off:     {cpu_fallback_disabled}")
    print(f"  first cosine:         {first_cosine:+.8f}")
    print(f"  first max_abs:        {first_max_abs:.4e}")
    print(f"  full cosine:          {full_cosine:+.8f}")
    print(f"  full max_abs:         {full_max_abs:.4e}")
    print(f"  verdict:              {receipt['verdict']}")
    print(f"  external acceptance:  {receipt['external_acceptance']}")
    print(f"  receipt:              {receipt_path}")
    print(MARKER + json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt["verdict"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
