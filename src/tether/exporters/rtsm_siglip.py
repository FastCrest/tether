"""Standalone RTSM SigLIP image encoder export, independent of policy serving.

Matches the normalized pooled image output of calabi-inc/rtsm at
85b94e603b25e32fe9c3484070807081da07dca0. No policy adapter, robot authority,
TensorRT acceptance or target performance is established by this CPU export.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

MODEL = "ViT-B-16-SigLIP"
PRETRAINED = "webli"
UPSTREAM_COMMIT = "85b94e603b25e32fe9c3484070807081da07dca0"
MAX_ABSOLUTE_ERROR = 1e-4
MIN_COSINE_SIMILARITY = 0.99999


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_preprocess(checkpoint: Path | None, preprocess: str | None) -> str:
    if preprocess is None:
        if checkpoint is not None:
            raise ValueError("A local checkpoint requires --preprocess pretrained or rtsm-local")
        return "pretrained"
    if preprocess not in {"pretrained", "rtsm-local"}:
        raise ValueError("preprocess must be pretrained or rtsm-local")
    return preprocess


def _applied_preprocessing(transform: Any) -> dict[str, Any]:
    """Describe the actual transform objects, rather than an inactive config."""
    operations: list[dict[str, Any]] = []
    for operation in transform.transforms:
        entry: dict[str, Any] = {"type": type(operation).__name__}
        for name in ("size", "max_size", "antialias", "mean", "std"):
            if hasattr(operation, name):
                value = getattr(operation, name)
                entry[name] = list(value) if isinstance(value, (list, tuple)) else value
        if hasattr(operation, "interpolation"):
            interpolation = operation.interpolation
            entry["interpolation"] = getattr(interpolation, "value", str(interpolation))
        operations.append(entry)
    return {"color_mode": "RGB", "image_dtype": "uint8", "operations": operations}


def _publish_stage(stage: Path, output_dir: Path) -> None:
    """Reserve an absent destination atomically before replacing our empty slot."""
    output_dir.mkdir()
    try:
        os.replace(stage, output_dir)
    except BaseException:
        # The reservation is empty unless another writer has touched it.
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise


def _normalized_encoder(visual: Any) -> Any:
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = visual

        def forward(self, images: Any) -> Any:
            features = self.visual(images)
            return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return Encoder().eval()


def _parity_metrics(reference: Any, actual: Any) -> dict[str, float]:
    import numpy as np

    if reference.shape != (1, 768) or actual.shape != (1, 768):
        raise ValueError("SigLIP parity requires pooled embeddings shaped [1, 768]")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("SigLIP parity produced nonfinite embeddings")
    left, right = reference.astype(np.float64), actual.astype(np.float64)
    left_norm, right_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    if not math.isclose(left_norm, 1.0, abs_tol=1e-5) or not math.isclose(
        right_norm, 1.0, abs_tol=1e-5
    ):
        raise ValueError("SigLIP parity requires unit-normalized embeddings")
    metrics = {
        "max_absolute_error": float(np.max(np.abs(left - right))),
        "cosine_similarity": float(np.sum(left * right) / (left_norm * right_norm)),
        "reference_l2_norm": left_norm,
        "onnx_l2_norm": right_norm,
    }
    if (
        metrics["max_absolute_error"] > MAX_ABSOLUTE_ERROR
        or metrics["cosine_similarity"] < MIN_COSINE_SIMILARITY
    ):
        raise ValueError(f"SigLIP numerical parity failed: {metrics}")
    return metrics


def export_rtsm_siglip(
    output_dir: Path,
    *,
    checkpoint: Path | None = None,
    preprocess: str | None = None,
    seed: int = 0,
    samples: int = 4,
) -> Path:
    """Export a static batch-one FP32 encoder only after generated-image parity.

    Returns the evidence manifest path. A local checkpoint must explicitly pick
    its preprocessing path: upstream local-checkpoint defaults differ from the
    pretrained model's transforms. Existing outputs are never replaced.
    Dependencies are imported only after arguments and output paths validate.
    """
    if sys.version_info < (3, 11):
        raise ValueError(
            "SigLIP export requires Python 3.11 or newer for its CPU ONNX Runtime cohort"
        )
    mode = _select_preprocess(checkpoint, preprocess)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if isinstance(samples, bool) or not isinstance(samples, int) or not 3 <= samples <= 32:
        raise ValueError("samples must be an integer from 3 to 32")
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output already exists: {output_dir}")
    if checkpoint is not None:
        # Keep the filename extension: safetensors caches are symlinks to
        # extensionless content-addressed blobs, and open_clip dispatches by it.
        checkpoint = Path(checkpoint).absolute()
        if not checkpoint.is_file() or checkpoint.suffix not in {
            ".pt",
            ".pth",
            ".bin",
            ".safetensors",
        }:
            raise ValueError(
                "Checkpoint must be a local weights file (.pt, .pth, .bin, .safetensors)"
            )

    import numpy as np
    import onnx
    import onnxruntime as ort
    import open_clip
    import torch
    from PIL import Image

    pretrained_cfg = open_clip.get_pretrained_cfg(MODEL, PRETRAINED)
    downloaded = checkpoint is None
    if checkpoint is None:
        checkpoint = Path(open_clip.download_pretrained(pretrained_cfg)).absolute()
    checkpoint_hash = _sha256(checkpoint)
    model, _, transform = open_clip.create_model_and_transforms(
        MODEL,
        pretrained=str(checkpoint),
        precision="fp32",
        device="cpu",
        weights_only=True,
        image_mean=pretrained_cfg["mean"],
        image_std=pretrained_cfg["std"],
        image_interpolation=pretrained_cfg["interpolation"],
        image_resize_mode=pretrained_cfg["resize_mode"],
    )
    model.eval()
    if mode == "rtsm-local":
        transform = open_clip.image_transform(model.visual.image_size, is_train=False)
    image_size = model.visual.image_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    if tuple(image_size) != (224, 224):
        raise ValueError(f"Unexpected SigLIP image shape: {image_size}")
    encoder = _normalized_encoder(model.visual).float()
    del model
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        fixtures = stage / "parity-inputs"
        fixtures.mkdir()
        rng = np.random.default_rng(seed)
        cases = []
        input_tensors = []
        # Rectangles exercise resize/crop differences; include a uniform image.
        for index in range(samples):
            height, width = ((240, 320), (320, 240), (224, 224))[index % 3]
            pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            if index == 2:
                pixels.fill(127)
            image = Image.fromarray(pixels)
            image.save(fixtures / f"{index:02d}.png")
            tensor = transform(image).unsqueeze(0).float().contiguous()
            if tuple(tensor.shape) != (1, 3, 224, 224) or not torch.isfinite(tensor).all():
                raise ValueError("Preprocessing must produce finite FP32 [1, 3, 224, 224]")
            np.save(fixtures / f"{index:02d}-input.npy", tensor.numpy(), allow_pickle=False)
            input_tensors.append(tensor)
        onnx_path = stage / "image-encoder.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                encoder,
                (input_tensors[0],),
                str(onnx_path),
                input_names=["images"],
                output_names=["embedding"],
                opset_version=17,
                dynamo=False,
            )
        onnx.checker.check_model(str(onnx_path), full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        with torch.inference_mode():
            for index, tensor in enumerate(input_tensors):
                reference = encoder(tensor).cpu().numpy()
                actual = session.run(["embedding"], {"images": tensor.numpy()})[0]
                metrics = _parity_metrics(reference, actual)
                np.save(fixtures / f"{index:02d}-pytorch.npy", reference, allow_pickle=False)
                np.save(fixtures / f"{index:02d}-onnx.npy", actual, allow_pickle=False)
                cases.append({"index": index, **metrics})
        manifest = {
            "schema": "tether.rtsm-siglip-image-encoder.v1",
            "status": "VERIFIED_CPU_NUMERICAL_PARITY",
            "model": MODEL,
            "pretrained": PRETRAINED if downloaded else None,
            "architecture_config_reference": {"model": MODEL, "pretrained_config": PRETRAINED},
            "precision": "float32",
            "input": {"name": "images", "shape": [1, 3, 224, 224], "dtype": "float32"},
            "output": {"name": "embedding", "shape": [1, 768], "l2_normalized": True},
            "upstream": {
                "repository": "https://github.com/calabi-inc/rtsm",
                "commit": UPSTREAM_COMMIT,
                "loader_blob": "19b73bf780f7018a006514fea1a9702b24e21461",
                "inference_blob": "192f9db260ff5d0956e177b6ccac9b7ef22f9193",
            },
            "checkpoint": {
                "source": "public-webli" if downloaded else "local-unverified-provenance",
                "file": checkpoint.name,
                "sha256": checkpoint_hash,
                "bytes": checkpoint.stat().st_size,
            },
            "preprocessing": {
                "mode": mode,
                "transform": repr(transform),
                "applied": _applied_preprocessing(transform),
                "webli_config_reference": {
                    "applied": mode == "pretrained",
                    "config": pretrained_cfg,
                },
                "tensor_boundary": "Preprocessing runs on RGB images outside ONNX",
            },
            "libraries": {
                name: importlib.metadata.version(name)
                for name in (
                    "open-clip-torch",
                    "torch",
                    "torchvision",
                    "timm",
                    "Pillow",
                    "numpy",
                    "onnx",
                    "onnxruntime",
                )
            },
            "parity": {
                "provider": "CPUExecutionProvider",
                "seed": seed,
                "samples": samples,
                "max_absolute_error_limit": MAX_ABSOLUTE_ERROR,
                "minimum_cosine_similarity": MIN_COSINE_SIMILARITY,
                "cases": cases,
            },
            "limits": [
                "Generated images establish numerical parity, not task accuracy or customer-image parity",
                "TensorRT acceptance, target memory and latency are unmeasured",
                "No policy serving contract, RTSM snapshot consumer or robot control is included",
            ],
            "files": {
                str(path.relative_to(stage)): {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(stage.rglob("*"))
                if path.is_file()
            },
        }
        if _sha256(checkpoint) != checkpoint_hash:
            raise ValueError("Checkpoint changed during export; destination was not published")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        _publish_stage(stage, output_dir)
        return output_dir / "manifest.json"
    finally:
        if stage.exists():
            shutil.rmtree(stage)
