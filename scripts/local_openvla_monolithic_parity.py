"""Qualification-grade OpenVLA ONNX + tokenized-action-decoder parity harness.

OpenVLA is deliberately not on Tether's flow-matching spine
(``src/tether/exporters/openvla.py``) and Tether owns only the bin-to-continuous
decode in ``tether.postprocess.openvla``. The subject of this receipt is
therefore the pair (exported LM graph, tokenized action decoder).

**This harness has never been run, because no exporter produces the ONNX graph
it consumes.** ``optimum-cli export onnx --model openvla/openvla-7b`` cannot
export openvla-7b (its ``model_type`` is the remote-code ``openvla``, not a
supported Optimum ONNX architecture), and ``exporters.monolithic`` has no
OpenVLA branch. See ``docs/openvla-export-parity.md``. Point ``--onnx-dir`` at
whatever a future OpenVLA exporter writes. Because the action head is ``argmax`` over the top ``n_action_bins``
vocabulary tokens, the harness gates on exact action-token agreement as well as
on the shared continuous-action cosine and max-absolute-error thresholds: one
disagreeing token is a whole-bin error that a cosine over seven dimensions can
hide.

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

from tether.eval.artifact_identity import build_artifact_identity
from tether.eval.openvla_export_parity import MARKER, build_openvla_export_parity_receipt
from tether.postprocess.openvla import decode_actions, logits_to_tokens


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


def _effective_vocab_size(config: object) -> int:
    """OpenVLA decodes against the unpadded text vocab.

    ``modeling_prismatic.py`` computes
    ``self.vocab_size = config.text_config.vocab_size - config.pad_to_multiple_of``
    (32064 - 64 = 32000 for openvla-7b). Reading it off the loaded config rather
    than hard-coding it keeps the receipt honest for fine-tunes.
    """
    text_config = getattr(config, "text_config", None)
    padded = getattr(text_config, "vocab_size", None)
    pad_to_multiple_of = getattr(config, "pad_to_multiple_of", None)
    if not isinstance(padded, int) or not isinstance(pad_to_multiple_of, int):
        raise RuntimeError(
            "Could not read text_config.vocab_size / pad_to_multiple_of from the "
            "OpenVLA config; refusing to guess the action-token decode vocabulary."
        )
    return padded - pad_to_multiple_of


def _pixel_channels(config: object) -> int:
    """Channel count ``pixel_values`` must carry for this checkpoint.

    ``openvla-7b`` sets ``use_fused_vision_backbone: true`` and
    ``vision_backbone_id: "dinosiglip-vit-so-224px"``. Its
    ``PrismaticVisionBackbone.forward`` then runs
    ``torch.split(pixel_values, [3, 3], dim=1)`` to dispatch one 3-channel image
    to the DINOv2 tower and one to the SigLIP tower, so the tensor is
    ``[bsz, 2 * 3, resolution, resolution]``. A 3-channel tensor raises inside
    that split before a single logit is produced. Read the flag rather than
    hard-coding 6, because a non-fused fine-tune takes 3.
    """
    fused = getattr(config, "use_fused_vision_backbone", None)
    if not isinstance(fused, bool):
        raise RuntimeError(
            "Could not read use_fused_vision_backbone from the OpenVLA config; "
            "refusing to guess the pixel_values channel count."
        )
    return 6 if fused else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare an exact OpenVLA ONNX export plus Tether's action decoder "
        "with the exact upstream PyTorch reference."
    )
    parser.add_argument("--model-source", default="openvla/openvla-7b")
    parser.add_argument(
        "--model-revision",
        required=True,
        help="Exact 40-hex Hugging Face checkpoint commit; floating HEAD/main is refused.",
    )
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square pixel_values edge. openvla-7b's DINOv2 + SigLIP towers are 224.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="norm_stats key to unnormalize with. Omitted: compare normalized actions.",
    )
    parser.add_argument(
        "--onnx-dir",
        default="/tmp/openvla_onnx",
        help="Directory holding the exported model.onnx. No exporter produces one for "
        "openvla-7b yet; see docs/openvla-export-parity.md.",
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
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Required. OpenVLA's modeling code lives in the checkpoint repository, "
            "so loading the reference executes code from --model-source at the exact "
            "pinned revision. Opt in explicitly."
        ),
    )
    parser.add_argument(
        "--receipt",
        default="openvla-export-parity-receipt.json",
        help="Receipt path. Must be outside --onnx-dir so it does not change artifact identity.",
    )
    args = parser.parse_args(argv)

    if args.action_dim <= 0:
        raise ValueError("--action-dim must be positive.")
    if args.prompt_tokens <= args.action_dim:
        raise ValueError("--prompt-tokens must exceed --action-dim.")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive.")
    if not args.trust_remote_code:
        raise SystemExit(
            "OpenVLA's reference implementation is remote code in the checkpoint "
            "repository. Re-run with --trust-remote-code to opt in."
        )

    onnx_dir = Path(args.onnx_dir).expanduser().resolve()
    onnx_path = onnx_dir / "model.onnx"
    receipt_path = Path(args.receipt).expanduser().resolve()
    if onnx_dir == receipt_path or onnx_dir in receipt_path.parents:
        raise ValueError("Parity receipt must be stored outside --onnx-dir.")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Missing OpenVLA ONNX export: {onnx_path}")

    import onnxruntime as ort
    from transformers import AutoConfig, AutoModelForVision2Seq

    tether_commit = _repo_commit()
    print(
        f"Loading exact OpenVLA reference {args.model_source}@{args.model_revision}...",
        flush=True,
    )
    config = AutoConfig.from_pretrained(
        args.model_source,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    vocab_size = _effective_vocab_size(config)
    n_action_bins = int(getattr(config, "n_action_bins", 256))
    pixel_channels = _pixel_channels(config)
    reference_model = AutoModelForVision2Seq.from_pretrained(
        args.model_source,
        revision=args.model_revision,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    reference_model.eval()

    norm_stats = None
    if args.dataset_name is not None:
        norm_stats = getattr(config, "norm_stats", None)
        if not isinstance(norm_stats, dict):
            raise RuntimeError(
                "--dataset-name was given but the checkpoint config carries no norm_stats."
            )

    rng = np.random.RandomState(args.input_seed)
    shared = {
        "input_ids": rng.randint(0, vocab_size, size=(1, args.prompt_tokens), dtype=np.int64),
        "attention_mask": np.ones((1, args.prompt_tokens), dtype=np.int64),
        "pixel_values": rng.randn(1, pixel_channels, args.image_size, args.image_size).astype(
            np.float32
        ),
    }
    shared_input_sha256 = _shared_input_digest(shared)

    print("Running PyTorch OpenVLA reference forward...", flush=True)
    with torch.no_grad():
        reference_logits = reference_model(
            input_ids=torch.from_numpy(shared["input_ids"]),
            attention_mask=torch.from_numpy(shared["attention_mask"]),
            pixel_values=torch.from_numpy(shared["pixel_values"]),
        ).logits
    reference_logits_np = reference_logits.cpu().numpy().astype(np.float32)

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
    required = {item.name for item in session.get_inputs()}
    missing = sorted(required - set(shared))
    if missing:
        raise RuntimeError(
            "The exported graph expects inputs this harness does not build: "
            f"{missing}. Refusing to feed a different graph than the reference saw."
        )
    feeds = {name: shared[name] for name in required}
    export_logits_np = np.asarray(session.run(None, feeds)[0], dtype=np.float32)

    reference_tokens = logits_to_tokens(reference_logits_np, args.action_dim)
    export_tokens = logits_to_tokens(export_logits_np, args.action_dim)
    total_action_tokens = int(reference_tokens.size)
    matching_action_tokens = int(np.sum(reference_tokens == export_tokens))

    decode = dict(
        action_dim=args.action_dim,
        norm_stats=norm_stats,
        dataset_name=args.dataset_name,
        vocab_size=vocab_size,
        n_bins=n_action_bins,
    )
    reference_np = np.asarray(decode_actions(reference_logits_np, **decode), dtype=np.float32)
    export_np = np.asarray(decode_actions(export_logits_np, **decode), dtype=np.float32)

    first_reference = reference_np[0]
    first_export = export_np[0]
    first_cosine = _cosine(first_reference, first_export)
    first_max_abs = float(np.max(np.abs(first_reference - first_export)))
    full_cosine = _cosine(reference_np, export_np)
    full_max_abs = float(np.max(np.abs(reference_np - export_np)))

    export_identity = build_artifact_identity(
        onnx_dir,
        source="openvla-optimum-onnx-export",
        revision=tether_commit,
        kind="openvla-onnx",
    )
    receipt = build_openvla_export_parity_receipt(
        model_source=args.model_source,
        model_revision=args.model_revision,
        tether_commit=tether_commit,
        export_identity=export_identity,
        platform_system=platform.system(),
        ort_providers=active_providers,
        requested_provider=args.provider,
        cpu_fallback_disabled=cpu_fallback_disabled,
        input_seed=args.input_seed,
        shared_input_sha256=shared_input_sha256,
        reference_shape=list(reference_np.shape),
        export_shape=list(export_np.shape),
        action_dim=args.action_dim,
        vocab_size=vocab_size,
        n_action_bins=n_action_bins,
        dataset_name=args.dataset_name,
        matching_action_tokens=matching_action_tokens,
        total_action_tokens=total_action_tokens,
        first_cosine=first_cosine,
        first_max_abs=first_max_abs,
        full_cosine=full_cosine,
        full_max_abs=full_max_abs,
    )
    _write_receipt(receipt_path, receipt)

    print("\n====== OPENVLA EXPORT PARITY ======")
    print(f"  exact model revision: {args.model_revision}")
    print(f"  decode vocab size:    {vocab_size}")
    print(f"  action bins:          {n_action_bins}")
    print(f"  pixel_values shape:   {list(shared['pixel_values'].shape)}")
    print(f"  dataset:              {args.dataset_name or '(normalized)'}")
    print(f"  export artifact sha:  {export_identity['artifact_sha256']}")
    print(f"  shared input sha:     {shared_input_sha256}")
    print(f"  providers:            {active_providers}")
    print(f"  CPU fallback off:     {cpu_fallback_disabled}")
    print(f"  action tokens agreed: {matching_action_tokens}/{total_action_tokens}")
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
