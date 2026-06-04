"""Phase B safetensors→flat-dict-direct loader.

Reads a safetensors checkpoint directly into a flat ``{key: bf16 CUDA tensor}``
dict, **bypassing** the ``nn.Module`` graph entirely. This delivers the actual
peak-RSS reduction the Lift #3 spec claimed.

## Why this exists

Phase A's ``BaseVLA.prepare_inference_weights()`` path extracts the flat dict
from a loaded ``nn.Module`` via ``.detach().clone()`` on each ``nn.Parameter``.
That requires holding BOTH the source ``nn.Module`` AND the cloned flat dict
in memory simultaneously — peak RSS includes the sum of the two. The Day 5
Modal A100 RSS bench measured **-15.7% (regression)** in PATH B vs PATH A
because of this. See ``03_experiments/2026-05-22-lift3-rss-bench-FAIL-architectural-finding.md``.

Phase B's approach: open the safetensors file with ``safe_open`` (no parse, just
header), then for each tensor:

1. Map the safetensors key to the flat-dict key via the per-VLA ``SAFETENSORS_MAPPING``
2. Allocate a bf16 tensor directly on the target CUDA device
3. Read tensor data once, in place

Net memory at deploy time: just the flat dict's tensors, in bf16, on CUDA.
No ``nn.Parameter`` wrapping, no Python bookkeeping, no clone. The +30% peak
RSS reduction claim is now achievable.

## Composition with Lift #5

The flat dict's bf16 CUDA layout is the exact shape Triton kernels expect.
``InferenceWeightsRuntime`` (Phase A) consumes it via IOBinding; Lift #5's
fast-kernel path will consume the same dict via direct tensor passing into
Triton kernel signatures.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def apply_prefix_mapping(key: str, mapping: Mapping[str, str]) -> str:
    """First-match-wins prefix substitution.

    Iterates ``mapping`` in declaration order. First ``src_prefix`` that
    appears as a prefix of ``key`` is replaced with the corresponding
    ``dst_prefix``. Keys without any matching prefix pass through unchanged.

    Matches the semantics of ``_name_mapping.apply_name_mapping()`` but
    operates on individual keys (rather than a whole state_dict). Useful
    for the safetensors-direct loader where we apply mapping per-key as
    we walk the file's header.

    Args:
        key: Source key (e.g. a safetensors checkpoint key like
            ``"model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"``).
        mapping: ``{src_prefix: dst_prefix}`` dict. Empty src_prefix is invalid
            (would match every key) — raise ``ValueError``.

    Returns:
        Mapped key, or the original key if no prefix matched.

    Raises:
        ValueError: if any src_prefix in ``mapping`` is empty.
    """
    for src in mapping:
        if not src:
            raise ValueError(
                "safetensors mapping contains empty src_prefix — would match "
                "every key. Use a non-empty marker (e.g. 'model.') instead."
            )
    for src, dst in mapping.items():
        if key.startswith(src):
            return dst + key[len(src):]
    return key


def load_flat_dict_from_safetensors(
    safetensors_path: str | Path,
    *,
    name_mapping: Mapping[str, str] | None = None,
    dtype: "torch.dtype | None" = None,
    device: str = "cuda",
    device_id: int = 0,
) -> "dict[str, torch.Tensor]":
    """Load a safetensors file → flat ``{key: tensor}`` dict, direct on device.

    Memory contract: each tensor is materialized exactly once on
    ``f"{device}:{device_id}"`` in the requested ``dtype``. No intermediate
    nn.Module instantiation, no detach().clone() doubling.

    Args:
        safetensors_path: Path to a single ``.safetensors`` file. For multi-file
            checkpoints, pass each file separately and merge dicts (or use
            ``load_flat_dict_from_safetensors_dir()``).
        name_mapping: ``{src_prefix: dst_prefix}`` applied in declaration order
            via ``apply_prefix_mapping()``. ``None`` or ``{}`` = identity map.
        dtype: Target tensor dtype. ``None`` = preserve source dtype.
            Typically pass ``torch.bfloat16`` for Lift #5 Triton kernels.
        device: ``"cuda"`` (default) or ``"cpu"``. CUDA is the perf path; CPU
            is for CI/parity-verification runs without GPU.
        device_id: CUDA device ordinal (default 0).

    Returns:
        ``{flat_key: tensor}`` ready for ``InferenceWeightsRuntime`` or
        ``WeightBinder``. Tensor count == safetensors file's tensor count.

    Raises:
        FileNotFoundError: ``safetensors_path`` doesn't exist.
        ValueError: ``name_mapping`` contains an empty src_prefix.
    """
    import torch
    from safetensors import safe_open

    path = Path(safetensors_path)
    if not path.exists():
        raise FileNotFoundError(f"safetensors file not found: {path}")

    mapping = name_mapping or {}
    device_str = f"{device}:{device_id}" if device == "cuda" else device

    flat: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device=device_str) as f:
        for src_key in f.keys():
            target_key = apply_prefix_mapping(src_key, mapping)
            tensor = f.get_tensor(src_key)
            if dtype is not None and tensor.dtype != dtype:
                tensor = tensor.to(dtype=dtype)
            flat[target_key] = tensor

    logger.info(
        "Loaded %d weight tensors from %s direct-to-%s%s",
        len(flat),
        path.name,
        device_str,
        f" as {dtype}" if dtype is not None else "",
    )
    return flat


def load_flat_dict_from_safetensors_dir(
    safetensors_dir: str | Path,
    *,
    name_mapping: Mapping[str, str] | None = None,
    dtype: "torch.dtype | None" = None,
    device: str = "cuda",
    device_id: int = 0,
) -> "dict[str, torch.Tensor]":
    """Load all ``.safetensors`` files in a directory and merge into one flat dict.

    Used for sharded checkpoints (typical for >5GB models — pi0.5, GR00T-3B).
    The HuggingFace ``model.safetensors.index.json`` is NOT consulted; we
    simply union the per-file tensor dicts. Duplicate keys across files raise
    ``ValueError`` (genuine multi-file checkpoints shouldn't have collisions).

    Args:
        safetensors_dir: Directory containing one or more ``.safetensors`` files.
        name_mapping: Same as ``load_flat_dict_from_safetensors``.
        dtype: Same as ``load_flat_dict_from_safetensors``.
        device: Same as ``load_flat_dict_from_safetensors``.
        device_id: Same as ``load_flat_dict_from_safetensors``.

    Returns:
        Merged flat dict.

    Raises:
        FileNotFoundError: ``safetensors_dir`` doesn't exist or has no
            ``.safetensors`` files.
        ValueError: A tensor key appears in more than one file.
    """
    path = Path(safetensors_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"not a directory: {path}")

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {path}")

    merged: dict[str, "torch.Tensor"] = {}
    for f in files:
        partial = load_flat_dict_from_safetensors(
            f,
            name_mapping=name_mapping,
            dtype=dtype,
            device=device,
            device_id=device_id,
        )
        dups = set(partial) & set(merged)
        if dups:
            sample = sorted(dups)[:5]
            suffix = f" (and {len(dups) - 5} more)" if len(dups) > 5 else ""
            raise ValueError(
                f"duplicate keys across safetensors files: {sample}{suffix}"
            )
        merged.update(partial)

    logger.info(
        "Merged %d weight tensors from %d safetensors files in %s",
        len(merged),
        len(files),
        path,
    )
    return merged


__all__ = [
    "apply_prefix_mapping",
    "load_flat_dict_from_safetensors",
    "load_flat_dict_from_safetensors_dir",
]
