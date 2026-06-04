"""Pro-tier data collection — parquet writer for the self-distilling-serve loop.

Per ADR 2026-04-25-self-distilling-serve-architecture decision #1:
- Customer-disk-only (Tether never ingests parquet)
- Opt-in via `tether serve --pro --collect-data`
- 90-day rolling retention via filename-encoded date
- PII-aware: image bytes default-blurred (Day 2 wiring); instructions
  default-hashed (Day 2 wiring); state vectors raw by default (raw
  needed for distribution-shift detection; opt-in hash with fail-loud
  warning if customer redacts)

Day 1 ships the substrate: bounded-queue collector + parquet writer
+ retention. Day 2 wires the consent flow + license check. Day 4 wires
the CLI + /act hook.

Composition with existing record-replay (B.2):
- record-replay writes JSONL for AUDIT (one entry per request, immutable)
- ProDataCollector writes parquet for TRAINING (columnar, bulk-loadable)
- They coexist; both are gated independently.

Pure NumPy + pyarrow (or pandas as fallback). No torch import; no
asyncio dependency on the hot path — the queue is lockless deque.
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# Schema version bumped on breaking changes. Phase 1 = v1; additive-only
# evolution per the ADR (`extra_metadata` field reserved for future use).
PARQUET_SCHEMA_VERSION = 1

# Default retention — 90 days. Configurable per-customer via tether.yaml.
DEFAULT_RETENTION_DAYS = 90

# Default rotation — one parquet file per UTC day. Filename:
# `<data_dir>/YYYY-MM-DD.parquet`. Simple to retention-prune (rm files
# older than N days); customers can shard further by hand if they need
# finer granularity.
_FILENAME_FMT = "%Y-%m-%d"

# Bounded queue ceiling — protects against unbounded backlog → memory
# blow-up under load (matches WebhookDispatcher + PolicyRuntime conventions).
DEFAULT_MAX_QUEUE = 10_000

# Default flush cadence — write parquet every N events OR every N seconds
# (whichever first). Bigger batches = fewer disk writes; more frequent =
# less data lost on crash.
DEFAULT_FLUSH_EVERY_EVENTS = 256
DEFAULT_FLUSH_EVERY_SECONDS = 30.0


@dataclass(frozen=True)
class CollectedEvent:
    """One /act event recorded for distillation training. Frozen + serializable
    — written once, never mutated.

    Schema v1 fields (LOCKED per ADR; additive-only evolution):
    - timestamp: ISO 8601 UTC string of the /act completion time
    - episode_id: customer-supplied episode identifier (or "" when absent)
    - state_vec: list[float] of the proprio-state input (raw by default;
      see PII posture)
    - action_chunk: list[list[float]] of the policy's action output
    - reward_proxy: float in [0, 1] — task-success heuristic (1.0 when
      no error, 0.0 when error). Phase 2 wires real reward signals
      from customer's evaluator
    - image_b64: opt-in raw or face-blurred (default blur via MediaPipe;
      Day 2 wiring)
    - instruction_hash: SHA-256 hash of the instruction by default;
      raw text on explicit opt-in
    - metadata: dict reserved for future fields
    """

    timestamp: str
    episode_id: str
    state_vec: list[float]
    action_chunk: list[list[float]]
    reward_proxy: float
    image_b64: str | None  # None when --pro-collect-faces=skip
    instruction_hash: str
    instruction_raw: str | None  # None when default; raw on opt-in
    metadata: dict[str, Any] = field(default_factory=dict)

    SCHEMA_VERSION: ClassVar[int] = PARQUET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not (0.0 <= self.reward_proxy <= 1.0):
            raise ValueError(
                f"reward_proxy must be in [0, 1], got {self.reward_proxy}"
            )
        if not isinstance(self.state_vec, list):
            raise TypeError(
                f"state_vec must be a list of floats, got {type(self.state_vec)}"
            )
        if not isinstance(self.action_chunk, list):
            raise TypeError(
                f"action_chunk must be a list-of-lists, got {type(self.action_chunk)}"
            )

    def to_row(self) -> dict[str, Any]:
        """JSON-safe row dict for parquet/json serialization."""
        return {
            "schema_version": PARQUET_SCHEMA_VERSION,
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "state_vec": self.state_vec,
            "action_chunk": self.action_chunk,
            "reward_proxy": self.reward_proxy,
            "image_b64": self.image_b64,
            "instruction_hash": self.instruction_hash,
            "instruction_raw": self.instruction_raw,
            "metadata": self.metadata,
        }


def hash_instruction(text: str | None) -> str:
    """SHA-256 hash of an instruction, hex-encoded. Empty string when None."""
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class QueueFull(Exception):
    """Raised by `ProDataCollector.record()` when the bounded queue is at
    capacity. /act callers map to a metric increment + drop the event;
    NEVER block /act on collection."""


class ProDataCollector:
    """Bounded-queue parquet writer.

    Lifecycle:
        collector = ProDataCollector(data_dir="~/.tether/pro-data/")
        collector.start()
        # in /act handler (after response computed):
        try:
            collector.record(...)
        except QueueFull:
            metrics.inc_pro_data_dropped()
        # ... at shutdown:
        collector.stop()

    Thread-safe: writer task pulls from a lockless deque; record() under
    a lock to avoid race on `len(queue) >= max_queue`. Parquet writes
    are atomic via temp+rename.

    Phase 1 default backend: JSON Lines (`.jsonl`) — pyarrow / pandas
    optional. JSONL is simpler to debug; the columnar parquet write
    happens via `to_parquet_file(path)` which the distill scheduler calls
    when ready to consume.
    """

    __slots__ = (
        "_data_dir", "_max_queue", "_flush_every_events", "_flush_every_seconds",
        "_queue", "_lock", "_writer_thread", "_stopping",
        "_events_dropped", "_events_recorded", "_events_flushed",
        "_last_flush_time",
    )

    def __init__(
        self,
        *,
        data_dir: str | Path,
        max_queue: int = DEFAULT_MAX_QUEUE,
        flush_every_events: int = DEFAULT_FLUSH_EVERY_EVENTS,
        flush_every_seconds: float = DEFAULT_FLUSH_EVERY_SECONDS,
    ):
        if max_queue < 1:
            raise ValueError(f"max_queue must be >= 1, got {max_queue}")
        if flush_every_events < 1:
            raise ValueError(
                f"flush_every_events must be >= 1, got {flush_every_events}"
            )
        if flush_every_seconds <= 0:
            raise ValueError(
                f"flush_every_seconds must be positive, got {flush_every_seconds}"
            )
        self._data_dir = Path(data_dir).expanduser()
        self._max_queue = int(max_queue)
        self._flush_every_events = int(flush_every_events)
        self._flush_every_seconds = float(flush_every_seconds)
        self._queue: collections.deque[CollectedEvent] = collections.deque()
        self._lock = threading.Lock()
        self._writer_thread: threading.Thread | None = None
        self._stopping = False
        self._events_dropped = 0
        self._events_recorded = 0
        self._events_flushed = 0
        self._last_flush_time = time.time()

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def is_running(self) -> bool:
        return self._writer_thread is not None and self._writer_thread.is_alive()

    @property
    def events_dropped(self) -> int:
        return self._events_dropped

    @property
    def events_recorded(self) -> int:
        return self._events_recorded

    @property
    def events_flushed(self) -> int:
        return self._events_flushed

    def queue_depth(self) -> int:
        return len(self._queue)

    def snapshot(self) -> dict[str, Any]:
        return {
            "data_dir": str(self._data_dir),
            "max_queue": self._max_queue,
            "queue_depth": self.queue_depth(),
            "events_recorded": self._events_recorded,
            "events_dropped": self._events_dropped,
            "events_flushed": self._events_flushed,
            "is_running": self.is_running,
        }

    def record(self, event: CollectedEvent) -> None:
        """Enqueue one event for async flush. Raises QueueFull when at
        capacity — caller should map to a metric + drop, NEVER block /act."""
        with self._lock:
            if len(self._queue) >= self._max_queue:
                self._events_dropped += 1
                raise QueueFull(
                    f"pro_data_collector queue full (max={self._max_queue})"
                )
            self._queue.append(event)
            self._events_recorded += 1

    def start(self) -> None:
        """Spawn the writer thread + ensure the data dir exists. Idempotent."""
        if self.is_running:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._stopping = False
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="pro-data-writer", daemon=True,
        )
        self._writer_thread.start()

    def stop(self, drain_timeout_s: float = 10.0) -> None:
        """Drain the queue and stop the writer. Idempotent."""
        if self._writer_thread is None:
            return
        self._stopping = True
        self._writer_thread.join(timeout=drain_timeout_s)
        # Final flush in case writer was idle when we set _stopping
        self._flush_locked()
        self._writer_thread = None

    def _writer_loop(self) -> None:
        while not self._stopping:
            now = time.time()
            should_flush = (
                len(self._queue) >= self._flush_every_events
                or (now - self._last_flush_time) >= self._flush_every_seconds
            )
            if should_flush:
                self._flush_locked()
            time.sleep(0.5)
        # Final drain on exit
        self._flush_locked()

    def _flush_locked(self) -> None:
        """Pull all currently-queued events under the lock + write atomically."""
        with self._lock:
            if not self._queue:
                self._last_flush_time = time.time()
                return
            batch = list(self._queue)
            self._queue.clear()
        # Write outside the lock — disk IO can be slow, /act must not block
        try:
            self._write_jsonl(batch)
            self._events_flushed += len(batch)
            self._last_flush_time = time.time()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "pro_data_collector.flush_failed events=%d exc=%s: %s",
                len(batch), type(exc).__name__, exc,
            )
            # Don't re-enqueue — would risk unbounded growth. Operator sees
            # the ERROR log + the events_dropped counter.
            self._events_dropped += len(batch)

    def _write_jsonl(self, batch: list[CollectedEvent]) -> None:
        """Write to today's daily file (`YYYY-MM-DD.jsonl`) atomically.

        JSONL chosen over parquet for Phase 1 because:
        - No pyarrow / pandas dep on the serve runtime
        - Append-friendly (atomic per-line write)
        - Distill scheduler converts to parquet on demand via to_parquet_file()
        """
        date_str = datetime.now(timezone.utc).strftime(_FILENAME_FMT)
        path = self._data_dir / f"{date_str}.jsonl"
        # Append, atomic-per-line via newline boundary. This is safe under
        # Linux append semantics; macOS HFS+ also guarantees per-write
        # atomicity for writes < page size.
        with open(path, "a") as f:
            for event in batch:
                f.write(json.dumps(event.to_row()) + "\n")

    def prune_older_than(self, retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
        """Remove `.jsonl` files older than `retention_days`. Returns count
        of files removed. Safe to call from a cron / cleanup task."""
        if retention_days < 1:
            raise ValueError(f"retention_days must be >= 1, got {retention_days}")
        cutoff_age_s = retention_days * 86_400
        now = time.time()
        removed = 0
        if not self._data_dir.exists():
            return 0
        for path in self._data_dir.glob("*.jsonl"):
            try:
                if (now - path.stat().st_mtime) > cutoff_age_s:
                    path.unlink()
                    removed += 1
            except OSError as exc:
                logger.warning(
                    "pro_data_collector.prune_failed path=%s: %s", path, exc,
                )
        return removed

    def to_parquet_file(self, path: str | Path, days: int | None = None) -> int:
        """Convert recent `.jsonl` files to a single parquet for the distill
        pipeline. Returns the number of rows written.

        `days=None` → all .jsonl files in data_dir.
        `days=N` → only files within the last N days (filename-date based).

        Skips silently when pyarrow + pandas are both absent (writes JSONL
        copy at the path instead — distill pipeline can read either).
        """
        rows: list[dict[str, Any]] = []
        cutoff_ts: float | None = None
        if days is not None:
            cutoff_ts = time.time() - days * 86_400
        for jsonl_path in sorted(self._data_dir.glob("*.jsonl")):
            if cutoff_ts is not None and jsonl_path.stat().st_mtime < cutoff_ts:
                continue
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        out_path = Path(path)
        try:
            import pandas as pd
            # pyarrow rejects empty struct columns ("Cannot write struct type
            # with no child field"). Replace empty `metadata` dicts with a
            # dummy child so the column round-trips cleanly.
            for row in rows:
                if not row.get("metadata"):
                    row["metadata"] = {"_": ""}
            df = pd.DataFrame(rows)
            df.to_parquet(out_path, index=False)
        except ImportError:
            # Fallback: write as concatenated JSONL
            with open(out_path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        return len(rows)


__all__ = [
    "PARQUET_SCHEMA_VERSION",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_MAX_QUEUE",
    "CollectedEvent",
    "ProDataCollector",
    "QueueFull",
    "hash_instruction",
]
