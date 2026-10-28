"""Qualification-grade SmolVLA monolithic ONNX parity harness.

Compares one exact hashed ONNX export against one exact LeRobot SmolVLA
checkpoint revision. Each case builds a real-shaped LeRobot batch (three
256x256 uint8 cameras, a 6-dim state, a LIBERO-10 instruction) and runs it
through the checkpoint's own ``policy_preprocessor`` (newline, tokenizer
padded to ``tokenizer_max_length``, MEAN_STD state normalisation) and
``SmolVLAPolicy.prepare_images``/``prepare_state``. The reference is
``VLAFlowMatching.sample_actions`` on those exact tensors and a seeded noise;
the export receives the same tensors.

The monolithic graph has a static language length (16 at this exporter). The
harness feeds it the reference's tokens cut to that length only when every
cut position is padding (mask False), so the reference sees LeRobot's full
padded sequence and the export sees the same real tokens. Instructions that
do not fit are refused as cases and listed in the receipt; they are never
silently truncated.

The receipt records external acceptance only for a passing Linux run with an
active CUDAExecutionProvider and ONNX Runtime CPU EP fallback disabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

from tether.eval.artifact_identity import build_artifact_identity
from tether.eval.smolvla_export_parity import (
    MARKER,
    MAX_ABS_ERROR,
    abs_error_distribution,
    build_smolvla_export_parity_receipt,
)

for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
    _stub = types.ModuleType(_mod)
    _stub.GrootPolicy = None
    _stub.GR00TN15 = None
    sys.modules[_mod] = _stub


# The ten LIBERO-10 instructions, as logged by the pinned LIBERO runner in
# Studio docs/evidence/smolvla-qualification-20261027/{job-A,positive-control}/*.log.
LIBERO_10_INSTRUCTIONS = (
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
)
CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3")


def _repo_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=5
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "Exact Tether Git commit is required for a parity receipt. Run from a Git checkout."
        ) from exc
    if dirty:
        raise RuntimeError(f"Tether checkout has uncommitted tracked changes:\n{dirty}")
    return value


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
    denom = np.linalg.norm(a64) * np.linalg.norm(b64)
    if denom == 0:
        return 1.0 if np.array_equal(a64, b64) else 0.0
    return float(np.dot(a64, b64) / denom)


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
        description="Compare an exact SmolVLA ONNX export with an exact LeRobot reference revision."
    )
    parser.add_argument("--model-source", default="lerobot/smolvla_base")
    parser.add_argument(
        "--model-revision",
        required=True,
        help="Exact 40-hex Hugging Face checkpoint commit; floating HEAD/main is refused.",
    )
    parser.add_argument("--onnx-dir", required=True, help="Directory containing model.onnx.")
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="Exact ONNX Runtime provider. External acceptance requires CUDAExecutionProvider.",
    )
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--input-seed", type=int, default=42, help="Case k uses input_seed + k.")
    parser.add_argument("--noise-seed", type=int, default=99, help="Case k uses noise_seed + k.")
    parser.add_argument("--receipt", default="smolvla-export-parity-receipt.json")
    args = parser.parse_args(argv)

    onnx_dir = Path(args.onnx_dir).expanduser().resolve()
    onnx_path = onnx_dir / "model.onnx"
    receipt_path = Path(args.receipt).expanduser().resolve()
    if onnx_dir == receipt_path or onnx_dir in receipt_path.parents:
        raise ValueError("Parity receipt must be stored outside --onnx-dir.")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Missing SmolVLA ONNX export: {onnx_path}")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")

    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from lerobot.policies.smolvla import processor_smolvla  # noqa: F401  (registers steps)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from lerobot.processor.pipeline import PolicyProcessorPipeline

    tether_commit = _repo_commit()
    print(f"Loading exact PyTorch reference {args.model_source}@{args.model_revision}...", flush=True)
    repo = snapshot_download(args.model_source, revision=args.model_revision)
    policy = SmolVLAPolicy.from_pretrained(repo).eval().to(dtype=torch.float32).to("cpu")
    cfg = policy.config
    if cfg.num_steps != args.num_steps:
        print(f"Overriding checkpoint num_steps {cfg.num_steps} -> {args.num_steps}", flush=True)
        policy.model.config.num_steps = args.num_steps
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cpu"}},
    )

    session_options = ort.SessionOptions()
    cpu_fallback_disabled = args.provider == "CUDAExecutionProvider"
    if cpu_fallback_disabled:
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=[args.provider])
    active_providers = list(session.get_providers())
    graph_inputs = {item.name: item.shape for item in session.get_inputs()}
    graph_lang_len = graph_inputs["lang_tokens"][1]
    if not isinstance(graph_lang_len, int):
        raise TypeError(f"Expected a static lang_tokens length, got {graph_inputs['lang_tokens']}.")

    # The serve path (SmolVLAOnnxServer) must tokenize every instruction exactly
    # as the checkpoint's LeRobot preprocessor does, with the export's bundled
    # tokenizer; each case checks that before comparing actions.
    from tether.runtime.smolvla_onnx_server import tokenize_smolvla_instruction
    from tether.runtime.tokenizers import load_export_tokenizer

    export_config = json.loads((onnx_dir / "tether_config.json").read_text())
    serve_tokenizer = load_export_tokenizer(
        onnx_dir, export_config, default_ref=cfg.vlm_model_name, allow_remote=False
    )
    if serve_tokenizer is None:
        raise RuntimeError(f"No bundled tokenizer loads from {onnx_dir}.")

    state_dim = cfg.input_features["observation.state"].shape[0]
    image_shape = cfg.input_features[CAMERA_KEYS[0]].shape  # (3, H, W)
    shared: dict[str, np.ndarray] = {}
    case_rows: list[dict] = []
    refused: list[dict] = []
    references: list[np.ndarray] = []
    exports: list[np.ndarray] = []
    for index, instruction in enumerate(LIBERO_10_INSTRUCTIONS):
        seed = args.input_seed + index
        rng = np.random.RandomState(seed)
        batch_raw: dict = {
            key: torch.from_numpy(
                rng.randint(0, 256, (image_shape[1], image_shape[2], image_shape[0]), dtype=np.uint8)
            ).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            for key in CAMERA_KEYS
        }
        batch_raw["observation.state"] = torch.from_numpy(
            rng.randn(state_dim).astype(np.float32)
        ).unsqueeze(0)
        batch_raw["task"] = [instruction]
        batch_pp = pre(batch_raw)

        lang_tokens = batch_pp["observation.language.tokens"]
        lang_masks = batch_pp["observation.language.attention_mask"].bool()
        real_tokens = int(lang_masks.sum())
        serve_tokens, serve_masks, serve_report = tokenize_smolvla_instruction(
            serve_tokenizer, instruction, graph_lang_len
        )
        serve_equal = (
            serve_tokens.shape[1] == lang_tokens.shape[1]
            and np.array_equal(serve_tokens, lang_tokens.cpu().numpy())
            and np.array_equal(serve_masks, lang_masks.cpu().numpy())
        )
        if not serve_equal:
            raise RuntimeError(
                f"case {index}: serve tokenization differs from LeRobot's preprocessor for {instruction!r}"
            )
        if bool(lang_masks[:, graph_lang_len:].any()):
            refused.append({"instruction": instruction, "tokens": real_tokens, "graph_lang_len": graph_lang_len})
            print(f"  case {index}: REFUSED, {real_tokens} tokens > graph length {graph_lang_len}", flush=True)
            continue

        images, img_masks = policy.prepare_images(batch_pp)
        state_tensor = policy.prepare_state(batch_pp)
        noise_seed = args.noise_seed + index
        noise = np.random.RandomState(noise_seed).randn(1, cfg.chunk_size, cfg.max_action_dim).astype(np.float32)

        with torch.no_grad():
            reference = policy.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state_tensor, noise=torch.from_numpy(noise)
            )
        reference_np = reference.cpu().numpy().astype(np.float32)

        feed = {
            "img_cam1": images[0].cpu().numpy().astype(np.float32),
            "img_cam2": images[1].cpu().numpy().astype(np.float32),
            "img_cam3": images[2].cpu().numpy().astype(np.float32),
            "mask_cam1": img_masks[0].cpu().numpy(),
            "mask_cam2": img_masks[1].cpu().numpy(),
            "mask_cam3": img_masks[2].cpu().numpy(),
            "lang_tokens": lang_tokens[:, :graph_lang_len].cpu().numpy().astype(np.int64),
            "lang_masks": lang_masks[:, :graph_lang_len].cpu().numpy(),
            "state": state_tensor.cpu().numpy().astype(np.float32),
            "noise": noise,
        }
        export_np = np.asarray(session.run(None, feed)[0], dtype=np.float32)
        for name, value in feed.items():
            shared[f"case{index:02d}/{name}"] = value
        shared[f"case{index:02d}/reference_lang_tokens"] = lang_tokens.cpu().numpy().astype(np.int64)
        shared[f"case{index:02d}/reference_lang_masks"] = lang_masks.cpu().numpy()

        if reference_np.shape != export_np.shape:
            raise RuntimeError(f"case {index}: reference {reference_np.shape} != export {export_np.shape}")
        action_dim = cfg.output_features["action"].shape[0]
        row = {
            "index": index,
            "instruction": instruction,
            "real_tokens": real_tokens,
            "serve_tokenization_equals_lerobot": serve_equal,
            "serve_truncated": serve_report["truncated"],
            "input_seed": seed,
            "noise_seed": noise_seed,
            "first_cosine": _cosine(reference_np[0, 0], export_np[0, 0]),
            "first_max_abs": float(np.max(np.abs(reference_np[0, 0] - export_np[0, 0]))),
            "full_cosine": _cosine(reference_np, export_np),
            "full_max_abs": float(np.max(np.abs(reference_np - export_np))),
            "unpadded_action_max_abs": float(
                np.max(np.abs(reference_np[..., :action_dim] - export_np[..., :action_dim]))
            ),
            "reference_abs_max": float(np.max(np.abs(reference_np))),
        }
        case_rows.append(row)
        references.append(reference_np)
        exports.append(export_np)
        print(
            f"  case {index}: tokens={real_tokens} full_cos={row['full_cosine']:+.8f} "
            f"full_max_abs={row['full_max_abs']:.3e}",
            flush=True,
        )

    if not case_rows:
        raise RuntimeError("Every instruction was refused; no parity case ran.")
    pooled_ref = np.stack(references)
    pooled_exp = np.stack(exports)
    distribution = abs_error_distribution(pooled_ref, pooled_exp, tolerance=MAX_ABS_ERROR)
    action_dim = cfg.output_features["action"].shape[0]
    distribution_unpadded = abs_error_distribution(
        pooled_ref[..., :action_dim], pooled_exp[..., :action_dim], tolerance=MAX_ABS_ERROR
    )
    shared_input_sha256 = _shared_input_digest(shared)

    export_identity = build_artifact_identity(
        onnx_dir, source="smolvla-monolithic-export", revision=tether_commit, kind="smolvla-monolithic-onnx"
    )
    from importlib.metadata import version as _version

    receipt = build_smolvla_export_parity_receipt(
        model_source=args.model_source,
        model_revision=args.model_revision,
        tether_commit=tether_commit,
        export_identity=export_identity,
        platform_system=platform.system(),
        ort_providers=active_providers,
        requested_provider=args.provider,
        cpu_fallback_disabled=cpu_fallback_disabled,
        input_seed=args.input_seed,
        noise_seed=args.noise_seed,
        num_steps=args.num_steps,
        shared_input_sha256=shared_input_sha256,
        reference_shape=list(references[0].shape),
        export_shape=list(exports[0].shape),
        first_cosine=min(row["first_cosine"] for row in case_rows),
        first_max_abs=max(row["first_max_abs"] for row in case_rows),
        full_cosine=min(row["full_cosine"] for row in case_rows),
        full_max_abs=max(row["full_max_abs"] for row in case_rows),
        error_distribution=distribution,
        cases=case_rows,
        subject_context={
            "graph_lang_len": graph_lang_len,
            "reference_lang_len": int(cfg.tokenizer_max_length),
            "instructions_refused_exceed_graph_lang_len": refused,
            "unpadded_action_dim": action_dim,
            "error_distribution_unpadded_action_dims": distribution_unpadded,
            "runtime_versions": {
                p: _version(p)
                for p in ("lerobot", "transformers", "torch", "numpy")
            }
            | {"onnxruntime": ort.__version__},
            "reference_device": "cpu-fp32",
            # ORT's CUDA EP uses TF32 math unless this is "0"; recorded so a
            # strict-fp32 diagnostic receipt is distinguishable from the default.
            "environment": {"NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE")},
        },
    )
    _write_receipt(receipt_path, receipt)

    print("\n====== SMOLVLA EXPORT PARITY ======")
    print(f"  exact model revision: {args.model_revision}")
    print(f"  export artifact sha:  {export_identity['artifact_sha256']}")
    print(f"  shared input sha:     {shared_input_sha256}")
    print(f"  providers:            {active_providers}")
    print(f"  cases run / refused:  {len(case_rows)} / {len(refused)}")
    print(f"  worst full cosine:    {receipt['metrics']['full_cosine']:+.8f}")
    print(f"  abs error dist:       {json.dumps(distribution)}")
    print(f"  verdict:              {receipt['verdict']}")
    print(f"  external acceptance:  {receipt['external_acceptance']}")
    print(f"  receipt:              {receipt_path}")
    print(MARKER + json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt["verdict"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
