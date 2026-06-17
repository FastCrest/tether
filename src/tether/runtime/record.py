"""JSONL request/response recorder for `tether serve --record`.

Implements the wire format spec at TECHNICAL_PLAN.md §D.1 (record/replay
file format). Schema version 1. Write-only — replay readers live under
src/tether/replay/readers/v<N>.py (Day 2 deliverable of the B.2 sprint).

Design notes:
- Pure stdlib. Don't take a new runtime dep for the recorder.
- Synchronous file writes. The hook fires once per /act after inference;
  flush() per record means a writer crash leaves at most one partial line
  (reader skips it per D.1.11).
- Disk-full → degrade silently. Catches OSError, sets `self.degraded`,
  stops writing, but lets `tether serve` continue. /health surfaces this
  via `getattr(server, '_recorder', None).degraded` if a consumer wants.
- Filename convention per D.1.2: `<YYYYMMDD>-<HHMMSS>-<model_hash>-
  <session_id>.jsonl[.gz]`. UTC.

Usage:
    rec = RecordWriter(
        record_dir="/tmp/traces",
        model_hash="7a8b3c1d9f2e4a55",
        config_hash="e12f44c7b1a93802",
        export_dir="/path/to/export",
        model_type="pi0.5",
        export_kind="monolithic",
        embodiment="franka",
        providers=["CUDAExecutionProvider"],
        gpu="NVIDIA A10G",
        cuda_version="12.6",
        ort_version="1.20.1",
        image_redaction="hash_only",  # or "full", "none"
    )
    # Header emitted lazily on first write_request.
    rec.write_request(request=..., response=..., latency=..., ...)
    rec.write_footer(totals={...})
    rec.close()

Coexists with OTel/Phoenix tracing (see src/tether/runtime/tracing.py).
JSONL = bit-exact replay layer; OTel = live observability layer. Both
write to different sinks; the /act hook stamps `tether.record.seq` on
the OTel span so the two ledgers can be cross-grepped by seq.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

ImageRedaction = Literal["full", "hash_only", "none"]


def _utc_now_iso() -> str:
    """UTC timestamp with ms precision, ISO-8601, trailing 'Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + (
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"
    )


def compute_model_hash(export_dir: str | Path) -> str:
    """SHA256[:16] of all *.onnx + *.bin files in the export dir.

    Stable across runs given identical model files. Returns empty string
    if the dir doesn't exist or contains no model files.
    """
    p = Path(export_dir)
    if not p.exists():
        return ""
    files = sorted(list(p.glob("*.onnx")) + list(p.glob("*.bin")))
    if not files:
        return ""
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode("utf-8"))
        try:
            with f.open("rb") as fh:
                # Stream in 1MB chunks; ONNX files can be 10s of GB
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        except OSError as e:
            logger.warning("model_hash: skipping %s (%s)", f, e)
    return h.hexdigest()[:16]


def compute_config_hash(export_dir: str | Path) -> str:
    """SHA256[:16] of canonicalized tether_config.json. Empty string if
    the file is missing or unreadable."""
    p = Path(export_dir) / "tether_config.json"
    if not p.exists():
        return ""
    try:
        with p.open() as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("config_hash: %s unreadable (%s)", p, e)
        return ""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _redact_image(
    image_b64: str | None, mode: ImageRedaction
) -> dict[str, Any]:
    """Apply image redaction policy. Returns the request.image_* fields
    that should be in the record (varies by mode per D.1.8)."""
    out: dict[str, Any] = {}
    if not image_b64:
        return out
    if mode == "none":
        return out
    # Compute SHA always — cheap and useful for hash-keyed replay
    sha = hashlib.sha256(image_b64.encode("utf-8")).hexdigest()[:16]
    out["image_sha256"] = sha
    if mode == "full":
        out["image_b64"] = image_b64
    return out


def _chain_hash(prev_hash: str, record: dict[str, Any]) -> str:
    """``sha256(prev_hash || canonical(record without the chain fields))``.

    Excludes ``prev_record_hash`` / ``record_hash`` so the hash covers only the
    record's own content; the link to the previous record is carried by
    ``prev_hash``. Deterministic (sorted keys, no whitespace).
    """
    payload = {k: v for k, v in record.items() if k not in ("prev_record_hash", "record_hash")}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(prev_hash.encode("ascii") + canonical).hexdigest()


def _stable_payload_hash(payload: Any) -> str:
    """SHA-256[:16] over canonical JSON for compact evidence correlation."""
    try:
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        data = json.dumps(str(payload), separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def _guard_violation_count(guard: dict[str, Any] | None) -> int:
    if not guard:
        return 0
    violations = guard.get("violations")
    if isinstance(violations, list):
        return len(violations)
    count = guard.get("violation_count")
    if isinstance(count, int | float):
        return int(count)
    return 0


def _build_deployment_evidence(
    *,
    header_meta: dict[str, Any],
    request_obj: dict[str, Any],
    actions: list[list[float]],
    raw_actions: list[list[float]] | None,
    action_dim: int,
    latency: dict[str, Any],
    cache: dict[str, Any] | None,
    guard: dict[str, Any] | None,
    routing: dict[str, Any] | None,
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compact v1 evidence block for deployment proof, diff, and retraining loops.

    This is deliberately additive to the JSONL record schema: old readers ignore
    the block, while new tools can depend on stable hashes and summary fields
    without reparsing the entire response.
    """
    guarded_hash = _stable_payload_hash(actions)
    raw_hash = _stable_payload_hash(raw_actions) if raw_actions is not None else None
    return {
        "kind": "tether.rollout_evidence",
        "schema_version": 1,
        "policy": {
            "model_hash": header_meta.get("model_hash") or "",
            "config_hash": header_meta.get("config_hash") or "",
            "model_type": header_meta.get("model_type") or "",
            "export_kind": header_meta.get("export_kind") or "",
            "embodiment": header_meta.get("embodiment"),
            "routing_slot": (routing or {}).get("slot"),
        },
        "request": {
            "episode_id": request_obj.get("episode_id"),
            "request_id": request_obj.get("request_id"),
            "image_sha256": request_obj.get("image_sha256"),
            "instruction_sha256": _stable_payload_hash(request_obj.get("instruction") or ""),
            "state_dim": len(request_obj.get("state") or []),
        },
        "action": {
            "num_actions": len(actions),
            "action_dim": action_dim,
            "raw_present": raw_actions is not None,
            "raw_sha256": raw_hash,
            "guarded_sha256": guarded_hash,
            "modified_by_guard": raw_hash is not None and raw_hash != guarded_hash,
        },
        "safety": {
            "guard_present": guard is not None,
            "clamped": bool((guard or {}).get("clamped")),
            "clamp_count": int((guard or {}).get("clamp_count") or 0),
            "violation_count": _guard_violation_count(guard),
        },
        "latency": {
            "total_ms": latency.get("total_ms"),
            "rolling_p95_ms": latency.get("rolling_p95_ms"),
            "rolling_p99_ms": latency.get("rolling_p99_ms"),
        },
        "cache": {
            "status": (cache or {}).get("status", "n/a"),
        },
        "outcome": {
            "status": "failed" if error is not None else "success",
            "error_slug": (error or {}).get("slug"),
        },
    }


def verify_record_chain(records: list[dict[str, Any]]) -> tuple[bool, int | None]:
    """Verify a recorded trace's tamper-evident chain.

    Returns ``(ok, first_broken_index)``: ``ok`` is True iff every record's
    ``prev_record_hash`` links to the prior ``record_hash`` and every
    ``record_hash`` matches the recomputed content hash. Any insert / edit /
    reorder breaks it at the offending index.
    """
    prev = "0" * 64
    for i, rec in enumerate(records):
        if rec.get("prev_record_hash") != prev:
            return False, i
        if rec.get("record_hash") != _chain_hash(prev, rec):
            return False, i
        prev = rec["record_hash"]
    return True, None


class RecordWriter:
    """JSONL recorder for /act calls. One instance per recording session."""

    def __init__(
        self,
        record_dir: str | Path,
        *,
        model_hash: str,
        config_hash: str,
        export_dir: str | Path,
        model_type: str,
        export_kind: str,
        providers: list[str],
        gpu: str = "",
        cuda_version: str = "",
        ort_version: str = "",
        jetpack_version: str | None = None,
        embodiment: str | None = None,
        image_redaction: ImageRedaction = "hash_only",
        instruction_redaction: Literal["full", "hash_only"] = "full",
        sample_rate: float = 1.0,
        gzip_output: bool = True,
        notes: str = "",
        tether_version: str = "",
        policies: list[dict[str, Any]] | None = None,
        pro_customer_id: str | None = None,
        curate_collector: Any = None,
    ) -> None:
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = str(uuid.uuid4())
        self.started_at = _utc_now_iso()
        # Filename: <YYYYMMDD>-<HHMMSS>-<model_hash>-<session_id>.jsonl[.gz]
        ts = datetime.now(timezone.utc)
        fname = (
            f"{ts.strftime('%Y%m%d-%H%M%S')}-"
            f"{model_hash or 'unknownhash'}-"
            f"{self.session_id}.jsonl"
        )
        if gzip_output:
            fname += ".gz"
        self.filepath = self.record_dir / fname

        # Header bookkeeping
        self._header_meta = {
            "model_hash": model_hash,
            "config_hash": config_hash,
            "export_dir": str(Path(export_dir).resolve()),
            "model_type": model_type,
            "export_kind": export_kind,
            "embodiment": embodiment,
            "hardware": {
                "gpu": gpu,
                "cuda": cuda_version,
                "ort": ort_version,
                **({"jetpack": jetpack_version} if jetpack_version else {}),
            },
            "providers": list(providers),
            "sample_rate": sample_rate,
            "redaction": {
                "image": image_redaction,
                "instruction": instruction_redaction,
            },
            "notes": notes,
            "tether_version": tether_version,
        }
        # Day 7 policy-versioning: optional `policies` block in header,
        # one entry per loaded policy. v1 readers ignore unknown fields,
        # so this is back-compat-safe (no schema-version bump per ADR
        # 2026-04-25-policy-versioning-architecture).
        if policies:
            self._header_meta["policies"] = list(policies)

        # Pro-only fingerprint tag in the header. Identifies the source
        # customer (anonymized) of recorded traces so commercial
        # redistribution can be traced. Free-tier (pro_customer_id=None)
        # records omit this entirely. v1 readers ignore unknown fields.
        if pro_customer_id:
            try:
                from tether.pro.fingerprint import compute_fingerprint
                # Sign over the canonicalized header meta so the fingerprint
                # ties to this session's identity (not per-request).
                canonical = json.dumps(
                    self._header_meta, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                fp = compute_fingerprint(canonical, pro_customer_id)
                self._header_meta["tether_fingerprint"] = fp.to_dict()
            except Exception:  # noqa: BLE001 — never block record on fingerprint failure
                pass
        self.image_redaction: ImageRedaction = image_redaction
        self.instruction_redaction = instruction_redaction
        self.sample_rate = sample_rate

        # State
        self._fh: io.TextIOBase | None = None
        self._header_written = False
        self._seq = 0
        self._prev_record_hash = "0" * 64  # tamper-evident hash-chain head
        self.degraded = False  # set on first OSError; recorder stops writing
        # Curate dual-write: when a FreeContributorCollector is attached,
        # write_request emits to BOTH the JSONL trace (audit) AND the
        # curate queue (training corpus). Independent failure modes; if
        # the collector is broken the JSONL still records.
        self._curate_collector = curate_collector
        if curate_collector is not None:
            try:
                if not getattr(curate_collector, "is_running", False):
                    curate_collector.start()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "curate_collector.start failed (curate dual-write disabled): %s", exc,
                )
                self._curate_collector = None

        # Open file lazily on first emit so the file isn't created if
        # nothing ever gets recorded (e.g. empty test runs).

    # ---------------------------------------------------------------
    # File handle
    # ---------------------------------------------------------------

    def _open_if_needed(self) -> None:
        if self._fh is not None or self.degraded:
            return
        try:
            if self.filepath.suffix == ".gz":
                self._fh = gzip.open(self.filepath, "wt", encoding="utf-8")
            else:
                self._fh = self.filepath.open("w", encoding="utf-8")
            logger.info("RecordWriter opened: %s", self.filepath)
        except OSError as e:
            self._set_degraded(e)

    def _set_degraded(self, exc: BaseException) -> None:
        self.degraded = True
        logger.error(
            "RecordWriter degraded — recording stopped (slug=record-disk-full): %s",
            exc,
        )

    def _emit(self, record: dict[str, Any]) -> None:
        self._open_if_needed()
        if self.degraded or self._fh is None:
            return
        # Tamper-evident hash chain: each record carries the previous record's
        # hash plus a hash of its own content (excluding the chain fields), so an
        # auditor can detect any insertion, edit, or reorder of the trace.
        record["prev_record_hash"] = self._prev_record_hash
        record["record_hash"] = _chain_hash(self._prev_record_hash, record)
        self._prev_record_hash = record["record_hash"]
        try:
            self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._fh.flush()
        except OSError as e:
            self._set_degraded(e)

    # ---------------------------------------------------------------
    # Records (D.1.3 / D.1.4 / D.1.6)
    # ---------------------------------------------------------------

    def _write_header(self) -> None:
        if self._header_written:
            return
        record = {
            "kind": "header",
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "started_at": self.started_at,
            **self._header_meta,
        }
        self._emit(record)
        self._header_written = True

    def write_request(
        self,
        *,
        chunk_id: int,
        image_b64: str | None,
        instruction: str,
        state: list[float] | None,
        actions: list[list[float]],
        action_dim: int,
        latency_total_ms: float,
        episode_id: str | None = None,
        request_id: str | None = None,
        raw_actions: list[list[float]] | None = None,
        latency_stages: dict[str, float | None] | None = None,
        rolling_p50_ms: float | None = None,
        rolling_p95_ms: float | None = None,
        rolling_p99_ms: float | None = None,
        cache: dict[str, Any] | None = None,
        guard: dict[str, Any] | None = None,
        denoise: dict[str, Any] | None = None,
        mode: str = "",
        vlm_conditioning: str = "real",
        deadline: dict[str, Any] | None = None,
        rtc: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        routing: dict[str, Any] | None = None,
    ) -> int:
        """Emit one request record. Returns the seq assigned to this call.

        Returns -1 if recording was skipped (degraded or sample_rate drop).
        """
        if self.degraded:
            return -1
        # Sample rate (deterministic by seq for now; random sampling is a v2 nit)
        if self.sample_rate < 1.0 and (self._seq * self.sample_rate) % 1 >= self.sample_rate:
            self._seq += 1
            return -1

        self._write_header()

        seq = self._seq
        self._seq += 1

        request_obj: dict[str, Any] = {
            "instruction": instruction,
            "state": state,
        }
        if episode_id is not None:
            request_obj["episode_id"] = episode_id
        if request_id is not None:
            request_obj["request_id"] = request_id
        request_obj.update(_redact_image(image_b64, self.image_redaction))

        latency_obj: dict[str, Any] = {"total_ms": latency_total_ms}
        if latency_stages:
            latency_obj["stages"] = latency_stages
        if rolling_p50_ms is not None:
            latency_obj["rolling_p50_ms"] = rolling_p50_ms
        if rolling_p95_ms is not None:
            latency_obj["rolling_p95_ms"] = rolling_p95_ms
        if rolling_p99_ms is not None:
            latency_obj["rolling_p99_ms"] = rolling_p99_ms

        record: dict[str, Any] = {
            "kind": "request",
            "schema_version": SCHEMA_VERSION,
            "seq": seq,
            "chunk_id": chunk_id,
            "timestamp": _utc_now_iso(),
            "request": request_obj,
            "response": {
                "actions": actions,
                "num_actions": len(actions),
                "action_dim": action_dim,
            },
            "latency": latency_obj,
            "denoise": denoise or {
                "steps_used": 0,
                "steps_configured": 0,
                "adaptive": False,
            },
            "mode": mode,
            "vlm_conditioning": vlm_conditioning,
        }
        # Optional fields — only include when non-null (keeps line size down)
        if cache is not None:
            record["cache"] = cache
        if guard is not None:
            record["guard"] = guard
        if deadline is not None:
            record["deadline"] = deadline
        if rtc is not None:
            record["rtc"] = rtc
        if error is not None:
            record["error"] = error
        # Day 7 policy-versioning: optional `routing` block (slot, bucket,
        # routing_key, optional shadow_actions). v1 readers ignore the
        # unknown field per ADR 2026-04-25-policy-versioning-architecture
        # decision (additive evolution, no schema_version bump).
        if routing is not None:
            record["routing"] = routing

        record["evidence"] = _build_deployment_evidence(
            header_meta=self._header_meta,
            request_obj=request_obj,
            actions=actions,
            raw_actions=raw_actions,
            action_dim=action_dim,
            latency=latency_obj,
            cache=cache,
            guard=guard,
            routing=routing,
            error=error,
        )
        if raw_actions is not None:
            record["action_trace"] = {
                "raw_actions": raw_actions,
                "guarded_actions": actions,
                "raw_sha256": record["evidence"]["action"]["raw_sha256"],
                "guarded_sha256": record["evidence"]["action"]["guarded_sha256"],
                "modified_by_guard": record["evidence"]["action"]["modified_by_guard"],
            }

        self._emit(record)
        # Curate dual-write: feed the same event into the contribution queue.
        # Failures here NEVER affect the JSONL trace — collector is best-effort.
        if self._curate_collector is not None and error is None:
            try:
                from tether.pro.data_collection import (
                    CollectedEvent,
                    QueueFull,
                    hash_instruction,
                )
                event = CollectedEvent(
                    timestamp=record["timestamp"],
                    episode_id=self.session_id,
                    state_vec=list(state) if state is not None else [],
                    action_chunk=actions,
                    reward_proxy=1.0 if error is None else 0.0,
                    image_b64=image_b64 if self.image_redaction == "full" else None,
                    instruction_hash=hash_instruction(instruction),
                    instruction_raw=instruction if self.instruction_redaction == "full" else None,
                    metadata={"chunk_id": chunk_id, "seq": seq},
                )
                try:
                    self._curate_collector.record(event)
                except QueueFull:
                    pass  # queue full → drop, collector tracks via events_dropped
            except Exception as exc:  # noqa: BLE001
                logger.debug("curate dual-write skipped: %s", exc)
        return seq

    def write_shadow_result(
        self,
        *,
        seq: int,
        routing: dict[str, Any],
        actions: list[list[float]] | None = None,
        action_dim: int = 0,
        latency_total_ms: float | None = None,
        episode_id: str | None = None,
        request_id: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        """Emit append-only shadow policy evidence for an existing request seq."""
        if self.degraded:
            return
        self._write_header()
        record: dict[str, Any] = {
            "kind": "shadow_result",
            "schema_version": SCHEMA_VERSION,
            "seq": seq,
            "timestamp": _utc_now_iso(),
            "routing": routing,
        }
        request_obj: dict[str, Any] = {}
        if episode_id is not None:
            request_obj["episode_id"] = episode_id
        if request_id is not None:
            request_obj["request_id"] = request_id
        if request_obj:
            record["request"] = request_obj
        if actions is not None:
            record["response"] = {
                "actions": actions,
                "num_actions": len(actions),
                "action_dim": action_dim,
            }
        if latency_total_ms is not None:
            record["latency"] = {"total_ms": latency_total_ms}
        if error is not None:
            record["error"] = error
        self._emit(record)

    def write_footer(self, totals: dict[str, int]) -> None:
        """Emit footer on clean shutdown. Optional per D.1.6 — readers
        tolerate absence."""
        if self.degraded or not self._header_written:
            # Don't write a footer if we never wrote the header
            return
        record = {
            "kind": "footer",
            "schema_version": SCHEMA_VERSION,
            "ended_at": _utc_now_iso(),
            **totals,
        }
        self._emit(record)

    def close(self) -> None:
        # Stop the curate collector first so its drain has a chance to
        # flush queued events before the process exits.
        if self._curate_collector is not None:
            try:
                self._curate_collector.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("curate_collector.stop failed: %s", exc)
        if self._fh is None:
            return
        try:
            self._fh.flush()
            self._fh.close()
        except OSError as e:
            logger.warning("RecordWriter close failed: %s", e)
        finally:
            self._fh = None
        logger.info("RecordWriter closed: %s (seq=%d)", self.filepath, self._seq)

    # ---------------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------------

    @property
    def seq(self) -> int:
        """Current next-seq value (== number of records emitted so far)."""
        return self._seq


__all__ = [
    "RecordWriter",
    "SCHEMA_VERSION",
    "ImageRedaction",
    "compute_model_hash",
    "compute_config_hash",
]
