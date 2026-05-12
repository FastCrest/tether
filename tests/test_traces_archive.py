"""Customer trace archive (Phase 1.5 v1).

Covers reader, filter, and aggregator behavior on synthetic JSONL traces.
Format mirrors what RecordWriter writes (see src/reflex/runtime/record.py).
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import pytest

from reflex.traces.archive import (
    TraceFilter,
    TraceReader,
    _parse_window,
    _percentile,
    query_traces,
    summarize_traces,
)


def _write_trace(
    tmp_path: Path,
    *,
    name: str,
    model_hash: str,
    records: list[dict],
    gzip_output: bool = False,
) -> Path:
    """Write a trace file (header line + one line per record)."""
    fname = name + (".jsonl.gz" if gzip_output else ".jsonl")
    fp = tmp_path / fname
    header = {
        "model_hash": model_hash,
        "embodiment": "franka",
        "hardware": {"gpu": "A100", "cuda": "12.4", "ort": "1.20"},
    }
    lines = [json.dumps(header)]
    for r in records:
        lines.append(json.dumps(r))
    body = "\n".join(lines) + "\n"
    if gzip_output:
        with gzip.open(fp, "wt", encoding="utf-8") as f:
            f.write(body)
    else:
        fp.write_text(body, encoding="utf-8")
    return fp


def _request_record(
    *, seq: int, instruction: str, latency_ms: float,
    error: dict | None = None, ts: str = "2026-05-06T12:00:00Z",
) -> dict:
    rec = {
        "kind": "request",
        "schema_version": 1,
        "seq": seq,
        "chunk_id": seq,
        "timestamp": ts,
        "request": {"instruction": instruction, "state": [0.0]},
        "response": {"actions": [[0.0]], "num_actions": 1, "action_dim": 1},
        "latency": {"total_ms": latency_ms},
        "denoise": {"steps_used": 10, "steps_configured": 10, "adaptive": False},
        "mode": "normal",
        "vlm_conditioning": "real",
    }
    if error is not None:
        rec["error"] = error
    return rec


def test_parse_window_basic():
    assert _parse_window(None) is None
    assert _parse_window("") is None
    now = time.time()
    one_hour_ago = _parse_window("1h")
    assert one_hour_ago is not None
    assert abs(one_hour_ago - (now - 3600)) < 5  # within 5s
    one_day_ago = _parse_window("1d")
    assert one_day_ago is not None
    assert abs(one_day_ago - (now - 86400)) < 5
    thirty_min = _parse_window("30m")
    assert thirty_min is not None
    assert abs(thirty_min - (now - 1800)) < 5


def test_parse_window_invalid():
    with pytest.raises(ValueError):
        _parse_window("garbage")
    with pytest.raises(ValueError):
        _parse_window("1y")  # year unit not supported
    with pytest.raises(ValueError):
        _parse_window("d")  # missing number


def test_percentile_basic():
    assert _percentile([], 0.5) == 0.0
    assert _percentile([10.0], 0.5) == 10.0
    assert _percentile([10.0, 20.0], 0.5) == 15.0
    # p99 of 1..100 is ~99
    sorted_vals = list(range(1, 101))
    p99 = _percentile([float(v) for v in sorted_vals], 0.99)
    assert 98.0 < p99 < 100.0


def test_reader_finds_files(tmp_path: Path):
    _write_trace(tmp_path, name="t1", model_hash="abc", records=[
        _request_record(seq=0, instruction="pick cube", latency_ms=50.0),
    ])
    _write_trace(tmp_path, name="t2", model_hash="abc", records=[
        _request_record(seq=0, instruction="place cube", latency_ms=60.0),
    ])
    reader = TraceReader([tmp_path])
    files = reader.files()
    assert len(files) == 2


def test_reader_empty_dir(tmp_path: Path):
    reader = TraceReader([tmp_path])
    assert reader.files() == []
    assert list(reader.records()) == []


def test_query_no_filter_returns_all(tmp_path: Path):
    _write_trace(tmp_path, name="t1", model_hash="abc", records=[
        _request_record(seq=0, instruction="pick cube", latency_ms=50.0),
        _request_record(seq=1, instruction="place cube", latency_ms=60.0),
    ])
    records = query_traces([tmp_path])
    assert len(records) == 2


def test_query_task_filter(tmp_path: Path):
    _write_trace(tmp_path, name="t1", model_hash="abc", records=[
        _request_record(seq=0, instruction="pick red cube", latency_ms=50.0),
        _request_record(seq=1, instruction="place blue block", latency_ms=60.0),
        _request_record(seq=2, instruction="pick blue cube", latency_ms=55.0),
    ])
    # Substring + case-insensitive
    records = query_traces([tmp_path], filter_=TraceFilter(task="PICK"))
    assert len(records) == 2
    assert all("pick" in r.instruction.lower() for r in records)


def test_query_status_failed_filter(tmp_path: Path):
    _write_trace(tmp_path, name="t1", model_hash="abc", records=[
        _request_record(seq=0, instruction="pick cube", latency_ms=50.0),
        _request_record(
            seq=1, instruction="pick cube", latency_ms=60.0,
            error={"reason": "timeout"},
        ),
    ])
    failed = query_traces([tmp_path], filter_=TraceFilter(status="failed"))
    assert len(failed) == 1
    assert failed[0].is_failed
    success = query_traces([tmp_path], filter_=TraceFilter(status="success"))
    assert len(success) == 1
    assert success[0].is_success


def test_query_model_filter_skips_whole_file(tmp_path: Path):
    _write_trace(tmp_path, name="t1", model_hash="aaa-good", records=[
        _request_record(seq=0, instruction="pick cube", latency_ms=50.0),
    ])
    _write_trace(tmp_path, name="t2", model_hash="bbb-other", records=[
        _request_record(seq=0, instruction="pick cube", latency_ms=60.0),
    ])
    records = query_traces([tmp_path], filter_=TraceFilter(model="aaa"))
    assert len(records) == 1


def test_query_limit(tmp_path: Path):
    _write_trace(tmp_path, name="t1", model_hash="abc", records=[
        _request_record(seq=i, instruction="pick", latency_ms=50.0)
        for i in range(10)
    ])
    records = query_traces([tmp_path], filter_=TraceFilter(limit=3))
    assert len(records) == 3


def test_query_handles_gzip(tmp_path: Path):
    _write_trace(
        tmp_path, name="t1", model_hash="abc",
        records=[_request_record(seq=0, instruction="pick", latency_ms=50.0)],
        gzip_output=True,
    )
    records = query_traces([tmp_path])
    assert len(records) == 1


def test_query_skips_malformed_lines(tmp_path: Path):
    fp = tmp_path / "broken.jsonl"
    fp.write_text(
        json.dumps({"model_hash": "abc"}) + "\n"
        + "not-json-garbage\n"
        + json.dumps(_request_record(
            seq=0, instruction="pick", latency_ms=50.0,
        )) + "\n",
        encoding="utf-8",
    )
    records = query_traces([tmp_path])
    assert len(records) == 1


def test_query_skips_non_request_lines(tmp_path: Path):
    fp = tmp_path / "mixed.jsonl"
    fp.write_text(
        json.dumps({"model_hash": "abc"}) + "\n"
        + json.dumps({"kind": "header", "schema_version": 2}) + "\n"
        + json.dumps(_request_record(
            seq=0, instruction="pick", latency_ms=50.0,
        )) + "\n",
        encoding="utf-8",
    )
    records = query_traces([tmp_path])
    assert len(records) == 1


def test_summary_by_task(tmp_path: Path):
    _write_trace(tmp_path, name="t1", model_hash="abc", records=[
        _request_record(seq=0, instruction="pick cube", latency_ms=50.0),
        _request_record(seq=1, instruction="pick cube", latency_ms=70.0),
        _request_record(
            seq=2, instruction="pick cube", latency_ms=100.0,
            error={"reason": "x"},
        ),
        _request_record(seq=3, instruction="place block", latency_ms=40.0),
    ])
    summaries = summarize_traces([tmp_path], by="task")
    by_bucket = {s.bucket: s for s in summaries}
    assert "pick cube" in by_bucket
    assert by_bucket["pick cube"].count == 3
    assert by_bucket["pick cube"].success_count == 2
    assert by_bucket["pick cube"].failed_count == 1
    assert by_bucket["pick cube"].success_rate == pytest.approx(2 / 3)
    # Sorted by count desc — pick cube (3) before place block (1)
    assert summaries[0].bucket == "pick cube"
    assert summaries[1].bucket == "place block"


def test_summary_by_model(tmp_path: Path):
    # Use realistic file naming so the model-hash extraction works
    _write_trace(tmp_path, name="20260506-120000-aaaa-sid", model_hash="aaaa", records=[
        _request_record(seq=0, instruction="pick", latency_ms=50.0),
        _request_record(seq=1, instruction="pick", latency_ms=60.0),
    ])
    _write_trace(tmp_path, name="20260506-130000-bbbb-sid", model_hash="bbbb", records=[
        _request_record(seq=0, instruction="pick", latency_ms=80.0),
    ])
    summaries = summarize_traces([tmp_path], by="model")
    by_bucket = {s.bucket: s for s in summaries}
    assert "aaaa" in by_bucket
    assert by_bucket["aaaa"].count == 2
    assert by_bucket["bbbb"].count == 1


def test_summary_by_day(tmp_path: Path):
    _write_trace(tmp_path, name="t1", model_hash="abc", records=[
        _request_record(seq=0, instruction="x", latency_ms=50.0, ts="2026-05-06T12:00:00Z"),
        _request_record(seq=1, instruction="x", latency_ms=60.0, ts="2026-05-06T13:00:00Z"),
        _request_record(seq=2, instruction="x", latency_ms=70.0, ts="2026-05-07T01:00:00Z"),
    ])
    summaries = summarize_traces([tmp_path], by="day")
    by_bucket = {s.bucket: s for s in summaries}
    assert by_bucket["2026-05-06"].count == 2
    assert by_bucket["2026-05-07"].count == 1


def test_summary_invalid_group(tmp_path: Path):
    with pytest.raises(ValueError):
        summarize_traces([tmp_path], by="invalid")  # type: ignore[arg-type]


def test_summary_latency_percentiles(tmp_path: Path):
    _write_trace(tmp_path, name="t1", model_hash="abc", records=[
        _request_record(seq=i, instruction="x", latency_ms=float(i * 10))
        for i in range(1, 11)  # 10..100 ms
    ])
    summaries = summarize_traces([tmp_path], by="task")
    assert len(summaries) == 1
    s = summaries[0]
    assert s.latency_p50_ms == pytest.approx(55.0)
    assert s.latency_max_ms == pytest.approx(100.0)
    # p99 of 10..100 by inclusive interpolation: ~99.1
    assert s.latency_p99_ms > 95.0


def test_summary_as_dict():
    from reflex.traces.archive import TraceSummary
    s = TraceSummary(
        bucket="x", count=10, success_count=8, failed_count=2,
        latency_p50_ms=50.123, latency_p95_ms=100.456,
        latency_p99_ms=200.789, latency_max_ms=300.0,
    )
    d = s.as_dict()
    assert d["bucket"] == "x"
    assert d["success_rate"] == 0.8
    assert d["latency_p50_ms"] == 50.12
    assert d["count"] == 10


def test_combined_filters(tmp_path: Path):
    _write_trace(tmp_path, name="t1", model_hash="abc", records=[
        _request_record(seq=0, instruction="pick cube", latency_ms=50.0),
        _request_record(seq=1, instruction="place cube", latency_ms=60.0),
        _request_record(
            seq=2, instruction="pick cube", latency_ms=100.0,
            error={"reason": "timeout"},
        ),
    ])
    # Failed pick = exactly 1
    records = query_traces([tmp_path], filter_=TraceFilter(
        task="pick", status="failed",
    ))
    assert len(records) == 1
    assert records[0].is_failed
    assert records[0].instruction == "pick cube"
