"""Tokenizer loading helpers for exported Reflex bundles."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_export_tokenizer(
    export_dir: str | Path,
    config: dict[str, Any],
    *,
    default_ref: str,
    set_pad_to_eos: bool = False,
) -> Any | None:
    """Load tokenizer from an export bundle, then fall back to HF.

    Preferred order:
      1. ``reflex_config.json:tokenizer_path`` relative to export_dir
      2. ``export_dir/tokenizer``
      3. export_dir itself if tokenizer files were written at the root
      4. ``reflex_config.json:tokenizer_ref`` or the provided default HF ref

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
    rel = config.get("tokenizer_path")
    if rel:
        sources.append((export_path / str(rel), True))
    sources.append((export_path / "tokenizer", True))
    if any((export_path / name).exists() for name in ("tokenizer.json", "tokenizer_config.json")):
        sources.append((export_path, True))
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
    logger.warning("Tokenizer load failed; tried %s", errors)
    return None
