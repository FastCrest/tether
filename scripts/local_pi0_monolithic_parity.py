"""Qualification-grade pi0 monolithic ONNX parity harness.

Compares one exact hashed ONNX export against one exact LeRobot pi0 checkpoint
revision on mechanically shared preprocessed inputs and diffusion noise. The
result is emitted as the versioned public receipt consumed by Tether Studio M4
#51 qualification work.

This script can run on CPU for local debugging, but the receipt records external
acceptance only for a passing Linux run with an active CUDAExecutionProvider.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import types

import numpy as np
import torch

from tether.eval.artifact_identity import build_artifact_identity
from tether.eval.pi0_export_parity import MARKER, build_pi0_export_parity_receipt


for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
    _stub = types.ModuleType(_mod)
    _stub.GrootPolicy = None
    _stub.GR00TN15 = None
    sys.modules[_mod] = _stub


def _patch_pi0_for_transformers_457() -> None:
    from lerobot.policies.pi0 import modeling_pi0

    def patched_embed_image(self, image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = (
            image_outputs.pooler_output
            if hasattr(image_outputs, "pooler_output")
            else image_outputs
        )
        features = features * self.paligemma.config.text_config.hidden_size**0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    modeling_pi0.PaliGemmaWithExpertModel.embed_image = patched_embed_image


def _patch_create_causal_mask_kwarg() -> None:
    from transformers import masking_utils

    original = masking_utils.create_causal_mask

    def shim(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return original(*args, **kwargs)

    masking_utils.create_causal_mask = shim
    from lerobot.policies import pi_gemma

    if hasattr(pi_gemma, "create_causal_mask"):
        pi_gemma.create_causal_mask = shim


def _repo_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "Exact Tether Git commit is required for a parity receipt. Run from a Git checkout."
        ) from exc
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
        description="Compare an exact pi0 ONNX export with an exact LeRobot reference revision."
    )
    parser.add_argument("--model-source", default="lerobot/pi0_base")
    parser.add_argument(
        "--model-revision",
        required=True,
        help="Exact 40-hex Hugging Face checkpoint commit; floating HEAD/main is refused.",
    )
    parser.add_argument(
        "--onnx-dir",
        default="/tmp/pi0_monolithic_onnx",
        help="Directory containing model.onnx and any external ONNX data files.",
    )
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="Exact ONNX Runtime provider to request. External acceptance requires CUDAExecutionProvider.",
    )
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--input-seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=99)
    parser.add_argument(
        "--receipt",
        default="pi0-export-parity-receipt.json",
        help="Receipt path. Must be outside --onnx-dir so it does not change artifact identity.",
    )
    args = parser.parse_args(argv)

    onnx_dir = Path(args.onnx_dir).expanduser().resolve()
    onnx_path = onnx_dir / "model.onnx"
    receipt_path = Path(args.receipt).expanduser().resolve()
    if onnx_dir == receipt_path or onnx_dir in receipt_path.parents:
        raise ValueError("Parity receipt must be stored outside --onnx-dir.")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Missing pi0 ONNX export: {onnx_path}")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")

    _patch_pi0_for_transformers_457()
    _patch_create_causal_mask_kwarg()

    from huggingface_hub import snapshot_download
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    import onnxruntime as ort

    print(
        f"Loading exact PyTorch reference {args.model_source}@{args.model_revision}...",
        flush=True,
    )
    repo = snapshot_download(args.model_source, revision=args.model_revision)
    policy = PI0Policy.from_pretrained(repo).eval().to(dtype=torch.float32).to("cpu")

    rng = np.random.RandomState(args.input_seed)
    img_np = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    img_t = img_t * 2.0 - 1.0
    state = torch.from_numpy(rng.randn(14).astype(np.float32) * 0.1)
    batch_raw = {
        "observation.images.base_0_rgb": img_t.unsqueeze(0),
        "observation.images.left_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.images.right_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.state": state.unsqueeze(0),
        "task": ["pick up the red bowl"],
    }
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cpu"}},
    )
    batch_pp = pre(batch_raw)

    images, img_masks = policy._preprocess_images(batch_pp)
    lang_tokens = batch_pp["observation.language.tokens"]
    lang_masks = batch_pp["observation.language.attention_mask"]
    state_tensor = policy.prepare_state(batch_pp)
    cfg = policy.config
    noise = np.random.RandomState(args.noise_seed).randn(
        1, cfg.chunk_size, cfg.max_action_dim
    ).astype(np.float32)

    shared = {
        "img_base": images[0].cpu().numpy().astype(np.float32),
        "img_wrist_l": images[1].cpu().numpy().astype(np.float32),
        "img_wrist_r": images[2].cpu().numpy().astype(np.float32),
        "mask_base": img_masks[0].cpu().numpy(),
        "mask_wrist_l": img_masks[1].cpu().numpy(),
        "mask_wrist_r": img_masks[2].cpu().numpy(),
        "lang_tokens": lang_tokens.cpu().numpy().astype(np.int64),
        "lang_masks": lang_masks.cpu().numpy(),
        "state": state_tensor.cpu().numpy().astype(np.float32),
        "noise": noise,
    }
    shared_input_sha256 = _shared_input_digest(shared)

    print(f"Running PyTorch reference with num_steps={args.num_steps}...", flush=True)
    with torch.no_grad():
        reference = policy.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state_tensor,
            noise=torch.from_numpy(noise),
            num_steps=args.num_steps,
        )
    reference_np = reference.cpu().numpy().astype(np.float32)

    print(f"Running exact ONNX artifact with {args.provider}...", flush=True)
    session = ort.InferenceSession(str(onnx_path), providers=[args.provider])
    active_providers = list(session.get_providers())
    export_np = np.asarray(session.run(None, shared)[0], dtype=np.float32)

    first_ref = reference_np[0, 0]
    first_export = export_np[0, 0]
    first_cosine = _cosine(first_ref, first_export)
    first_max_abs = float(np.max(np.abs(first_ref - first_export)))
    full_cosine = _cosine(reference_np, export_np)
    full_max_abs = float(np.max(np.abs(reference_np - export_np)))

    export_identity = build_artifact_identity(
        onnx_dir,
        source="pi0-monolithic-export",
        revision=_repo_commit(),
        kind="pi0-monolithic-onnx",
    )
    receipt = build_pi0_export_parity_receipt(
        model_source=args.model_source,
        model_revision=args.model_revision,
        tether_commit=_repo_commit(),
        export_identity=export_identity,
        platform_system=platform.system(),
        ort_providers=active_providers,
        requested_provider=args.provider,
        input_seed=args.input_seed,
        noise_seed=args.noise_seed,
        num_steps=args.num_steps,
        shared_input_sha256=shared_input_sha256,
        reference_shape=list(reference_np.shape),
        export_shape=list(export_np.shape),
        first_cosine=first_cosine,
        first_max_abs=first_max_abs,
        full_cosine=full_cosine,
        full_max_abs=full_max_abs,
    )
    _write_receipt(receipt_path, receipt)

    print("\n====== PI0 EXPORT PARITY ======")
    print(f"  exact model revision: {args.model_revision}")
    print(f"  export artifact sha:  {export_identity['artifact_sha256']}")
    print(f"  shared input sha:     {shared_input_sha256}")
    print(f"  providers:            {active_providers}")
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
