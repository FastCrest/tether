"""Tokenizer loading helpers for exported Tether bundles."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_TOKENIZER_MARKER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "tokenizer.model",
    "special_tokens_map.json",
)


class OfflineTokenizerMissingError(RuntimeError):
    """Raised when offline mode forbids a remote tokenizer fallback."""


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def tokenizer_offline_enabled() -> bool:
    """Return True when tokenizer loading must not touch the network."""
    return (
        _env_flag("TETHER_OFFLINE")
        or _env_flag("HF_HUB_OFFLINE")
        or _env_flag("TRANSFORMERS_OFFLINE")
    )


def _looks_like_tokenizer_path(path: Path) -> bool:
    if path.is_file():
        return path.name in _TOKENIZER_MARKER_FILES
    if not path.is_dir():
        return False
    return any((path / marker).exists() for marker in _TOKENIZER_MARKER_FILES)


def find_bundled_tokenizer_path(
    export_dir: str | Path,
    config: dict[str, Any],
) -> Path | None:
    """Return the local tokenizer path in an export bundle, if present."""
    export_path = Path(export_dir)
    candidates: list[Path] = []
    rel = config.get("tokenizer_path")
    if rel:
        candidates.append(export_path / str(rel))
    candidates.append(export_path / "tokenizer")
    candidates.append(export_path)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _looks_like_tokenizer_path(candidate):
            return candidate
    return None


def ensure_offline_tokenizer_bundle(
    export_dir: str | Path,
    config: dict[str, Any],
    *,
    default_ref: str,
) -> None:
    """Fail early when offline serving would need a remote tokenizer."""
    if not tokenizer_offline_enabled():
        return
    if find_bundled_tokenizer_path(export_dir, config) is not None:
        return
    tokenizer_ref = config.get("tokenizer_ref") or default_ref
    raise OfflineTokenizerMissingError(
        "offline tokenizer assets missing: TETHER_OFFLINE/HF offline mode "
        f"prevents downloading {tokenizer_ref!r}. Re-run `tether export` with "
        "network access so it writes export_dir/tokenizer, or copy a compatible "
        "Hugging Face tokenizer bundle into the export and set "
        "`tokenizer_path` in tether_config.json."
    )


def load_export_tokenizer(
    export_dir: str | Path,
    config: dict[str, Any],
    *,
    default_ref: str,
    set_pad_to_eos: bool = False,
    allow_remote: bool | None = None,
) -> Any | None:
    """Load tokenizer from an export bundle, then fall back to HF.

    Preferred order:
      1. ``tether_config.json:tokenizer_path`` relative to export_dir
      2. ``export_dir/tokenizer``
      3. export_dir itself if tokenizer files were written at the root
      4. ``tether_config.json:tokenizer_ref`` or the provided default HF ref

    Local sources use ``local_files_only=True`` so offline deployments do not
    accidentally call Hugging Face during startup.
    """
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        logger.warning("transformers unavailable; tokenizer cannot load: %s", exc)
        return None

    export_path = Path(export_dir)
    sources: list[tuple[str | Path, bool]] = []
    bundled = find_bundled_tokenizer_path(export_path, config)
    if bundled is not None:
        sources.append((bundled, True))
    if allow_remote is None:
        allow_remote = not tokenizer_offline_enabled()
    if allow_remote:
        sources.append((str(config.get("tokenizer_ref") or default_ref), False))

    seen: set[str] = set()
    errors: list[str] = []
    for source, local_only in sources:
        key = str(source)
        if key in seen:
            continue
        seen.add(key)
        if local_only and not Path(source).exists():
            continue
        try:
            tok = AutoTokenizer.from_pretrained(source, local_files_only=local_only)
            if set_pad_to_eos and getattr(tok, "pad_token", None) is None:
                tok.pad_token = getattr(tok, "eos_token", None)
            logger.info("Tokenizer loaded from %s", source)
            return tok
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    if not allow_remote:
        ref = config.get("tokenizer_ref") or default_ref
        raise OfflineTokenizerMissingError(
            "offline tokenizer load failed: no usable bundled tokenizer found "
            f"for {ref!r}. Tried: {errors or ['<no local tokenizer files>']}. "
            "Re-run `tether export` with network access so it writes "
            "export_dir/tokenizer."
        )
    logger.warning("Tokenizer load failed; tried %s", errors)
    return None
