"""Trace archive reader, filter, and aggregator.

Phase 1.5 v1 of customer-trace-archive (per spec
features/01_serve/subfeatures/_ecosystem/customer-trace-archive/).
Operates directly on the JSONL files written by RecordWriter at
src/reflex/runtime/record.py — no parquet+DuckDB migration yet.

Trace file format (per RecordWriter.write_request schema):
  Line 1: header dict (`model_hash`, `embodiment`, `hardware`, ...)
  Line 2+: one request record per /act call, each with `seq`,
           `timestamp`, `request.instruction`, `response.actions`,
           `latency.total_ms`, optional `error`, etc.

Filter dimensions (Phase 1.5):
  - time window: `--since 7d|24h|1h`
  - task substring: `--task pick-cube` (matched against request.instruction)
  - status: `--status success|failed|any` (failed = error field non-null)
  - model substring: `--model <hash>` (matched against header.model_hash)

Aggregations (Phase 1.5):
  - count
  - success_rate (= 1 - failed_rate)
  - latency p50 / p95 / p99 / max
  - fallback_count (cache.fallback_to_cpu or similar)
  - error_count

Phase 2 (deferred): parquet storage + DuckDB index for fast filter on
million-record archives; SQL surface for power users.
"""
from __future__ import annotations

import gzip
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal


# ---------------------------------------------------------------------------
# Time window parsing
# ---------------------------------------------------------------------------

_WINDOW_RE = re.compile(r"^(\d+)([dhm])$", re.IGNORECASE)
_UNIT_SECONDS = {"h": 3600, "d": 86400, "m": 60}


def _parse_window(window: str | None) -> float | None:
    """Convert '7d', '24h', '1h', '30m' to a unix-time cutoff (now - delta).
    Returns None when window is None/empty (no cutoff)."""
    if not window:
        return None
    m = _WINDOW_RE.match(window.strip())
    if not m:
        raise ValueError(
            f"invalid --since window {window!r}; expected '<N>{{d,h,m}}' "
            f"(e.g. '7d', '24h', '30m')"
        )
    n = int(m.group(1))
    unit = m.group(2).lower()
    return time.time() - n * _UNIT_SECONDS[unit]


# ---------------------------------------------------------------------------
# Records + filters
# ---------------------------------------------------------------------------

Status = Literal["success", "failed", "any"]


@dataclass
class TraceRecord:
    """Flat view of one /act request record. Pulls the fields callers
    actually filter/aggregate on; full raw record lives in `.raw`."""

    file: Path
    seq: int
    timestamp: str  # ISO8601 UTC
    instruction: str
    latency_ms: float
    error: dict[str, Any] | None
    raw: dict[str, Any] = field(repr=False)

    @property
    def is_failed(self) -> bool:
        return self.error is not None

    @property
    def is_success(self) -> bool:
        return self.error is None


@dataclass
class TraceFilter:
    """Filter spec for `query_traces` / `summarize_traces`. All fields
    optional — None means no filter on that dimension."""

    since: str | None = None  # '7d', '24h', '30m'
    task: str | None = None  # case-insensitive substring on instruction
    status: Status = "any"  # 'success' / 'failed' / 'any'
    model: str | None = None  # substring on header.model_hash
    limit: int | None = None  # max records returned (None = unbounded)

    def file_passes_time(self, file_path: Path) -> bool:
        """File-level filter: skip files older than the time window
        based on mtime, before opening them."""
        cutoff = _parse_window(self.since)
        if cutoff is None:
            return True
        try:
            return file_path.stat().st_mtime >= cutoff
        except OSError:
            return False

    def header_passes(self, header: dict[str, Any]) -> bool:
        """Header-level filter: skip the entire file if header.model_hash
        doesn't match. Cheap — runs once per file."""
        if self.model:
            mh = (header.get("model_hash") or "").lower()
            if self.model.lower() not in mh:
                return False
        return True

    def record_passes(self, record: TraceRecord) -> bool:
        """Per-record filter on task substring + status."""
        if self.task:
            if self.task.lower() not in record.instruction.lower():
                return False
        if self.status == "success" and record.is_failed:
            return False
        if self.status == "failed" and record.is_success:
            return False
        return True


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _open_trace(file_path: Path) -> Any:
    """Open a .jsonl or .jsonl.gz file for text reading. Returns either a
    text file handle (`open(...)`) or a gzip-decompressed text reader; both
    behave as context managers + line iterators. Typed as Any since gzip's
    text-mode return is a `_typeshed.SupportsReadlineN` not a TextIO."""
    if file_path.suffix == ".gz":
        return gzip.open(file_path, "rt", encoding="utf-8")
    return open(file_path, "rt", encoding="utf-8")


def _list_trace_files(dirs: Iterable[Path]) -> list[Path]:
    """Find all .jsonl + .jsonl.gz files in the given dirs, newest first."""
    files: list[Path] = []
    for d in dirs:
        if not d.exists() or not d.is_dir():
            continue
        files.extend(d.glob("*.jsonl"))
        files.extend(d.glob("*.jsonl.gz"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


class TraceReader:
    """Iterate request records across one or more trace directories.

    Lifecycle:
        reader = TraceReader([Path("/tmp/traces")])
        for rec in reader.records(filter_=TraceFilter(task="pick-cube")):
            print(rec.timestamp, rec.latency_ms, rec.is_failed)
    """

    def __init__(self, dirs: Iterable[Path | str]):
        self.dirs = [Path(d) for d in dirs]

    def files(self) -> list[Path]:
        return _list_trace_files(self.dirs)

    def records(
        self, filter_: TraceFilter | None = None,
    ) -> Iterator[TraceRecord]:
        """Yield request records that match the filter. Header records
        (kind != 'request') are dropped silently. Malformed lines are
        skipped without raising."""
        flt = filter_ or TraceFilter()
        n_yielded = 0
        for f in self.files():
            if not flt.file_passes_time(f):
                continue
            try:
                with _open_trace(f) as fh:
                    header_ok = True
                    for i, line in enumerate(fh):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # First line should be the header. Apply header
                        # filter; if it fails, skip the rest of this file.
                        if i == 0:
                            if not flt.header_passes(obj):
                                header_ok = False
                                break
                            continue
                        if not header_ok:
                            break
                        if obj.get("kind") != "request":
                            continue
                        instruction = (
                            obj.get("request", {}).get("instruction") or ""
                        )
                        latency = float(
                            obj.get("latency", {}).get("total_ms") or 0.0
                        )
                        rec = TraceRecord(
                            file=f,
                            seq=int(obj.get("seq") or 0),
                            timestamp=str(obj.get("timestamp") or ""),
                            instruction=instruction,
                            latency_ms=latency,
                            error=obj.get("error"),
                            raw=obj,
                        )
                        if not flt.record_passes(rec):
                            continue
                        yield rec
                        n_yielded += 1
                        if flt.limit and n_yielded >= flt.limit:
                            return
            except OSError:
                continue


# ---------------------------------------------------------------------------
# Query + summary
# ---------------------------------------------------------------------------


def query_traces(
    dirs: Iterable[Path | str],
    *,
    filter_: TraceFilter | None = None,
) -> list[TraceRecord]:
    """One-shot wrapper around TraceReader.records() — collects all
    matching records into a list. For very large archives, prefer the
    iterator API."""
    reader = TraceReader(dirs)
    return list(reader.records(filter_))


@dataclass
class TraceSummary:
    """Aggregation result for one bucket (e.g. one task, one model, one day).
    Surfaced fields are exactly what `reflex traces summary` reports."""

    bucket: str  # e.g. task name, model hash, or YYYY-MM-DD date
    count: int
    success_count: int
    failed_count: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float

    @property
    def success_rate(self) -> float:
        return self.success_count / self.count if self.count > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "count": self.count,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "success_rate": round(self.success_rate, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "latency_p99_ms": round(self.latency_p99_ms, 2),
            "latency_max_ms": round(self.latency_max_ms, 2),
        }


def _percentile(values: list[float], q: float) -> float:
    """Inclusive linear-interpolated percentile in [0, 1]. Empty list -> 0."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + frac * (s[hi] - s[lo])


def _bucket_for_record(rec: TraceRecord, by: str) -> str:
    """Group records by task / model / day (date-portion of timestamp)."""
    if by == "task":
        return rec.instruction[:60] if rec.instruction else "(empty)"
    if by == "model":
        # Pull from the file name (RecordWriter format embeds model_hash)
        # — fname like '20260506-013812-<hash>-<sid>.jsonl[.gz]'
        try:
            parts = rec.file.name.split("-")
            return parts[2] if len(parts) >= 3 else rec.file.name
        except Exception:  # noqa: BLE001
            return rec.file.name
    if by == "day":
        return (rec.timestamp[:10] or "unknown")
    raise ValueError(
        f"unknown summary group {by!r}; expected 'task' / 'model' / 'day'"
    )


def summarize_traces(
    dirs: Iterable[Path | str],
    *,
    filter_: TraceFilter | None = None,
    by: Literal["task", "model", "day"] = "task",
) -> list[TraceSummary]:
    """Group records by `by` and compute per-bucket aggregations. Returns
    summaries sorted by count desc."""
    if by not in ("task", "model", "day"):
        raise ValueError(
            f"unknown summary group {by!r}; expected 'task' / 'model' / 'day'"
        )
    buckets: dict[str, list[TraceRecord]] = {}
    reader = TraceReader(dirs)
    for rec in reader.records(filter_):
        key = _bucket_for_record(rec, by)
        buckets.setdefault(key, []).append(rec)

    summaries: list[TraceSummary] = []
    for bucket, recs in buckets.items():
        latencies = [r.latency_ms for r in recs]
        success_n = sum(1 for r in recs if r.is_success)
        failed_n = sum(1 for r in recs if r.is_failed)
        summaries.append(TraceSummary(
            bucket=bucket,
            count=len(recs),
            success_count=success_n,
            failed_count=failed_n,
            latency_p50_ms=_percentile(latencies, 0.5),
            latency_p95_ms=_percentile(latencies, 0.95),
            latency_p99_ms=_percentile(latencies, 0.99),
            latency_max_ms=max(latencies) if latencies else 0.0,
        ))
    summaries.sort(key=lambda s: s.count, reverse=True)
    return summaries
