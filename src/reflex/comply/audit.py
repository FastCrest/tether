"""Audit-log summarization for Reflex Comply.

Supports both runtime recorder JSONL files (``kind: header/request/footer``)
and the standalone ActionGuard inference log shape.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Iterator


def _canonical_record_bytes(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _iter_jsonl_paths(path: str | Path | None) -> list[Path]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    if p.is_file():
        return [p]
    files = sorted(p.rglob("*.jsonl")) + sorted(p.rglob("*.jsonl.gz"))
    return [f for f in files if f.is_file()]


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_records(files: Iterable[Path]) -> Iterator[tuple[Path, dict[str, Any]]]:
    for file in files:
        try:
            with _open_text(file) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict):
                        yield file, rec
        except OSError:
            continue


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def summarize_audit_log(path: str | Path | None) -> dict[str, Any]:
    files = _iter_jsonl_paths(path)
    if not files:
        return {
            "present": False,
            "path": str(path) if path else "",
            "files": [],
            "record_count": 0,
            "request_count": 0,
            "safety_violation_count": 0,
            "status": "missing",
        }

    model_hashes: set[str] = set()
    config_hashes: set[str] = set()
    sessions: set[str] = set()
    schema_versions: set[str] = set()
    timestamps: list[str] = []
    latencies: list[float] = []
    safety_violation_count = 0
    clamp_count = 0
    non_finite_count = 0
    error_count = 0
    request_count = 0
    record_count = 0
    image_sha256_count = 0
    image_b64_count = 0
    raw_instruction_count = 0
    instruction_hash_count = 0
    redaction_modes: dict[str, set[str]] = {"image": set(), "instruction": set()}
    safety_samples: list[dict[str, Any]] = []

    prev_hash = "0" * 64
    for file, rec in _iter_records(files):
        record_count += 1
        schema = rec.get("schema_version")
        if schema is not None:
            schema_versions.add(str(schema))
        entry_hash = hashlib.sha256(prev_hash.encode("ascii") + _canonical_record_bytes(rec)).hexdigest()
        prev_hash = entry_hash

        kind = rec.get("kind")
        if kind == "header":
            if rec.get("session_id"):
                sessions.add(str(rec["session_id"]))
            if rec.get("model_hash"):
                model_hashes.add(str(rec["model_hash"]))
            if rec.get("config_hash"):
                config_hashes.add(str(rec["config_hash"]))
            redaction = rec.get("redaction")
            if isinstance(redaction, dict):
                for key in ("image", "instruction"):
                    if redaction.get(key):
                        redaction_modes[key].add(str(redaction[key]))
            if rec.get("started_at"):
                timestamps.append(str(rec["started_at"]))
            continue

        timestamp = rec.get("timestamp") or rec.get("started_at") or rec.get("ended_at")
        if timestamp:
            timestamps.append(str(timestamp))

        if kind == "request":
            request_count += 1
            req = rec.get("request") if isinstance(rec.get("request"), dict) else {}
            if req.get("image_sha256"):
                image_sha256_count += 1
            if req.get("image_b64"):
                image_b64_count += 1
            if req.get("instruction"):
                raw_instruction_count += 1
            if req.get("instruction_hash"):
                instruction_hash_count += 1

            latency = rec.get("latency") if isinstance(rec.get("latency"), dict) else {}
            total_ms = latency.get("total_ms")
            if isinstance(total_ms, int | float):
                latencies.append(float(total_ms))

            guard = rec.get("guard") if isinstance(rec.get("guard"), dict) else {}
            violations = guard.get("violations") if isinstance(guard.get("violations"), list) else []
            safety_violation_count += len(violations)
            clamp_count += int(guard.get("clamp_count") or 0)
            non_finite_count += sum(1 for v in violations if "non_finite" in str(v))
            if violations and len(safety_samples) < 100:
                safety_samples.append({
                    "file": file.name,
                    "seq": rec.get("seq"),
                    "timestamp": timestamp,
                    "violations": violations[:10],
                    "clamped": bool(guard.get("clamped")),
                })

            if rec.get("error"):
                error_count += 1
            continue

        # Standalone ActionGuard inference log shape.
        if "violations" in rec and "actions_safe" in rec:
            request_count += 1
            violations = rec.get("violations") if isinstance(rec.get("violations"), list) else []
            safety_violation_count += len(violations)
            if rec.get("clamped"):
                clamp_count += 1
            non_finite_count += sum(1 for v in violations if "non_finite" in str(v))
            if rec.get("model_version"):
                model_hashes.add(str(rec["model_version"]))
            latency_ms = rec.get("latency_ms")
            if isinstance(latency_ms, int | float):
                latencies.append(float(latency_ms))
            if violations and len(safety_samples) < 100:
                safety_samples.append({
                    "file": file.name,
                    "timestamp": timestamp,
                    "violations": violations[:10],
                    "clamped": bool(rec.get("clamped")),
                })

    first_ts = min(timestamps) if timestamps else ""
    last_ts = max(timestamps) if timestamps else ""
    return {
        "present": True,
        "path": str(Path(path)) if path else "",
        "files": [str(f) for f in files],
        "schema_versions": sorted(schema_versions),
        "sessions": sorted(sessions),
        "record_count": record_count,
        "request_count": request_count,
        "event_count": request_count,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "model_hashes": sorted(model_hashes),
        "config_hashes": sorted(config_hashes),
        "redaction": {
            "image_modes": sorted(redaction_modes["image"]),
            "instruction_modes": sorted(redaction_modes["instruction"]),
            "image_sha256_count": image_sha256_count,
            "image_b64_count": image_b64_count,
            "raw_instruction_count": raw_instruction_count,
            "instruction_hash_count": instruction_hash_count,
        },
        "safety_violation_count": safety_violation_count,
        "clamp_count": clamp_count,
        "non_finite_count": non_finite_count,
        "error_count": error_count,
        "latency_ms": {
            "count": len(latencies),
            "p50": median(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "tamper_evidence": {
            "alg": "sha256(prev_hash || canonical_json_record)",
            "head": prev_hash if record_count else "",
            "record_count": record_count,
        },
        "safety_violations_sample": safety_samples,
        "status": "ok",
    }


__all__ = ["summarize_audit_log"]
