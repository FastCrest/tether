"""Schema-versioned readers for recorded JSONL traces.

Pattern: one module per schema version (`v1.py`, `v2.py`, ...). The
top-level `load_reader()` dispatches by reading the first header line
and selecting the matching reader.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from .v1 import ReplayReaderV1

_READERS: dict[int, type] = {
    1: ReplayReaderV1,
}


class ReplaySchemaUnknownError(ValueError):
    """Raised when a JSONL trace's schema_version isn't supported by
    any installed reader."""


def load_reader(file_path: str | Path):
    """Open a JSONL trace, peek the header, return the matching reader
    instance ready for read_header()/read_records()."""
    p = Path(file_path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        first = f.readline()
    if not first.strip():
        raise ValueError(f"Trace file is empty: {file_path}")
    try:
        head = json.loads(first)
    except json.JSONDecodeError as e:
        raise ValueError(f"Trace header is not valid JSON: {e}") from e
    schema_version = head.get("schema_version")
    if schema_version not in _READERS:
        raise ReplaySchemaUnknownError(
            f"schema_version={schema_version!r} not supported. "
            f"Installed readers: {sorted(_READERS.keys())}"
        )
    return _READERS[schema_version](p)


__all__ = ["ReplayReaderV1", "load_reader", "ReplaySchemaUnknownError"]
