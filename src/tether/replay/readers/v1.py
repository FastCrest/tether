"""Reader for JSONL trace schema v1 (per TECHNICAL_PLAN.md §D.1).

Pure stdlib. Auto-detects gzip from filename. Tolerates absence of
footer per D.1.2 ("readers MUST tolerate its absence"). Skips the final
partial line if the writer crashed mid-record (D.1.11).

Usage:
    reader = ReplayReaderV1("/path/to/trace.jsonl.gz")
    header = reader.read_header()
    print(header["model_hash"], header["model_type"])
    for kind, record in reader.read_records():
        if kind == "request":
            ...
"""
from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReplayReaderV1:
    """Streaming reader for schema v1 JSONL traces."""

    SCHEMA_VERSION = 1

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Trace file not found: {file_path}")
        self._header: dict[str, Any] | None = None
        self._lines: list[str] | None = None  # cached on first read

    def _open(self):
        if self.file_path.suffix == ".gz":
            return gzip.open(self.file_path, "rt", encoding="utf-8")
        return self.file_path.open("r", encoding="utf-8")

    def _load_lines(self) -> list[str]:
        """Read all lines into memory once. JSONL traces are small enough
        for this in v1; switch to streaming if individual files exceed
        ~1GB (which would be ~25k records at hash_only redaction)."""
        if self._lines is not None:
            return self._lines
        with self._open() as f:
            self._lines = f.readlines()
        return self._lines

    def read_header(self) -> dict[str, Any]:
        """Parse + return the header record. Cached after first call.
        Raises ValueError if the first line isn't a valid header."""
        if self._header is not None:
            return self._header
        lines = self._load_lines()
        if not lines:
            raise ValueError(f"Trace file is empty: {self.file_path}")
        first = lines[0].strip()
        if not first:
            raise ValueError(f"Trace file starts with blank line: {self.file_path}")
        try:
            head = json.loads(first)
        except json.JSONDecodeError as e:
            raise ValueError(f"Header is not valid JSON: {e}") from e
        if head.get("kind") != "header":
            raise ValueError(
                f"First line kind={head.get('kind')!r}, expected 'header'"
            )
        if head.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch: file has "
                f"{head.get('schema_version')}, reader is v{self.SCHEMA_VERSION}"
            )
        self._header = head
        return head

    def read_records(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Generator yielding (kind, record) for every line after the header.

        Skips the header line. Tolerates a missing footer. Skips a final
        partial line (per D.1.11 — writer crashed mid-record).
        """
        # Materialize header if not yet; lets caller do read_records() without
        # explicit read_header().
        self.read_header()
        lines = self._load_lines()
        if len(lines) < 2:
            return
        for i, raw in enumerate(lines[1:], start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                # Final-partial-line tolerance per D.1.11
                if i == len(lines) - 1:
                    logger.warning(
                        "ReplayReaderV1: final line at %s:%d is partial; "
                        "skipping (writer crashed mid-record)",
                        self.file_path, i + 1,
                    )
                    continue
                raise ValueError(
                    f"Trace line {i + 1} is not valid JSON in {self.file_path}"
                )
            kind = rec.get("kind", "")
            yield kind, rec

    def count_requests(self) -> int:
        """Number of request records (cheap helper for CLI summary)."""
        return sum(1 for kind, _ in self.read_records() if kind == "request")


__all__ = ["ReplayReaderV1"]
