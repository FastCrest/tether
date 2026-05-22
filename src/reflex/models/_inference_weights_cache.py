"""Disk cache for `prepare_inference_weights()` output.

Per ``features/01_serve/inference-only-weights_plan.md`` Day 2.

Cache key is a 5-tuple stable hash over ``(model_id, checkpoint_sha,
vla_type, reflex_version, torch_version)`` — any of these changing
invalidates the cache. The cache file lives at
``~/.cache/reflex/inference_weights/<hash>.pt`` and stores the flat
dict via ``torch.save``.

Why a five-tuple key (defensively wide):

- ``model_id``: distinct models (`lerobot/pi05_libero_finetuned_v044`
  vs base) hash differently.
- ``checkpoint_sha``: fine-tuning produces a new checkpoint with new
  weights; cache must invalidate. Caller passes the HF commit hash or
  a local fingerprint.
- ``vla_type``: composition class name (``Pi05VLA`` / ``GR00TVLA``).
  Different VLAs use different slot prefixes; the flat dict shape
  differs.
- ``reflex_version``: refactors to the spine slot layout (e.g. a
  hypothetical "split the projector into two slots" change) would
  silently shape-mismatch a cached dict. Invalidate on version bump.
- ``torch_version``: torch's tensor pickle format / dtype defaults
  change across major versions. Cheap insurance.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _cache_dir() -> Path:
    """Returns the ~/.cache/reflex/inference_weights/ directory (created on demand)."""
    base = Path.home() / ".cache" / "reflex" / "inference_weights"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _model_hash(
    model_id: str,
    checkpoint_sha: str,
    vla_type: str,
    reflex_version: str | None = None,
    torch_version: str | None = None,
) -> str:
    """Stable 16-hex-char hash of the 5-tuple cache key.

    `reflex_version` and `torch_version` default to the current process'
    values when not supplied — callers can pin them for tests.
    """
    if reflex_version is None:
        from reflex import __version__ as reflex_version  # noqa: F811
    if torch_version is None:
        torch_version = torch.__version__

    tup = (
        f"model_id={model_id}|"
        f"checkpoint_sha={checkpoint_sha}|"
        f"vla_type={vla_type}|"
        f"reflex_version={reflex_version}|"
        f"torch_version={torch_version}"
    )
    h = hashlib.sha256(tup.encode("utf-8")).hexdigest()
    return h[:16]


def cache_path(
    model_id: str,
    checkpoint_sha: str,
    vla_type: str,
    *,
    reflex_version: str | None = None,
    torch_version: str | None = None,
) -> Path:
    """Resolve the cache file path for the given 5-tuple key.

    Does NOT check whether the file exists — callers should use
    ``load_or_build`` for that.
    """
    h = _model_hash(
        model_id=model_id,
        checkpoint_sha=checkpoint_sha,
        vla_type=vla_type,
        reflex_version=reflex_version,
        torch_version=torch_version,
    )
    return _cache_dir() / f"{h}.pt"


def load_or_build(
    model_id: str,
    checkpoint_sha: str,
    vla_type: str,
    builder: Callable[[], dict[str, torch.Tensor]],
    *,
    reflex_version: str | None = None,
    torch_version: str | None = None,
) -> dict[str, torch.Tensor]:
    """Returns the cached flat-dict if present, else builds + caches it.

    Args:
        model_id: HF model id (e.g. ``"lerobot/pi05_libero_finetuned_v044"``).
        checkpoint_sha: Stable fingerprint of the underlying checkpoint
            (HF commit hash or local content hash). Required because two
            checkpoints with the same ``model_id`` (fine-tune vs base)
            must hash to different cache files.
        vla_type: Composition class name (e.g. ``"Pi05VLA"``).
        builder: Zero-arg callable that returns the flat dict to cache.
            Typically a bound ``vla.prepare_inference_weights`` call.
        reflex_version: Override reflex version (for tests).
        torch_version: Override torch version (for tests).

    Returns:
        The flat ``{key: tensor}`` dict, either freshly built or read
        from disk.
    """
    path = cache_path(
        model_id=model_id,
        checkpoint_sha=checkpoint_sha,
        vla_type=vla_type,
        reflex_version=reflex_version,
        torch_version=torch_version,
    )

    if path.exists():
        logger.info("inference-weights cache HIT: %s", path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise RuntimeError(
                f"inference-weights cache at {path} is corrupted "
                "(expected dict, got "
                f"{type(loaded).__name__}). Delete to force rebuild."
            )
        return loaded

    logger.info("inference-weights cache MISS: %s — building", path)
    weights = builder()
    if not isinstance(weights, dict):
        raise TypeError(
            f"builder() returned {type(weights).__name__}, expected dict[str, Tensor]"
        )
    torch.save(weights, path)
    logger.info(
        "inference-weights cache WROTE: %s (%d keys, %.1f MB)",
        path, len(weights), path.stat().st_size / 1e6,
    )
    return weights


__all__ = ["load_or_build", "cache_path"]
