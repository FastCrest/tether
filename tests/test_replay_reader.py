"""Tests for the replay JSONL reader (B.2 Day 2).

Covers ReplayReaderV1 parsing, gzip auto-detect, schema dispatch via
load_reader(), partial-line tolerance (D.1.11), missing-footer tolerance
(D.1.2), and ReplaySchemaUnknownError on future schema versions.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tether.replay.readers import (
    ReplayReaderV1,
    ReplaySchemaUnknownError,
    load_reader,
)
from tether.runtime.record import RecordWriter


def _make_trace(tmp_path: Path, n_records: int = 3, *, gzip_output: bool = True,
                footer: bool = True) -> Path:
    """Produce a real JSONL trace using the writer, return its path."""
    rec = RecordWriter(
        record_dir=tmp_path,
        model_hash="deadbeefcafe0000",
        config_hash="0011223344556677",
        export_dir=str(tmp_path / "fake_export"),
        model_type="pi0.5",
        export_kind="monolithic",
        providers=["CUDAExecutionProvider"],
        gpu="test-gpu",
        cuda_version="12.6",
        ort_version="1.20.1",
        embodiment="franka",
        image_redaction="full",
        tether_version="0.0.0-test",
        gzip_output=gzip_output,
    )
    for i in range(n_records):
        rec.write_request(
            chunk_id=i,
            image_b64="aGVsbG8=",
            instruction=f"req {i}",
            state=[0.1, 0.2],
            actions=[[0.0] * 7] * 10,
            action_dim=7,
            latency_total_ms=100.0 + i,
            mode="onnx_cpu",
        )
    if footer:
        rec.write_footer({"total_requests": n_records})
    rec.close()
    return rec.filepath


# ---------------------------------------------------------------------------
# Basic reader contract
# ---------------------------------------------------------------------------


class TestReplayReaderV1Basic:
    def test_read_header_returns_dict(self, tmp_path):
        trace = _make_trace(tmp_path)
        r = ReplayReaderV1(trace)
        h = r.read_header()
        assert h["kind"] == "header"
        assert h["schema_version"] == 1

    def test_read_header_cached(self, tmp_path):
        trace = _make_trace(tmp_path)
        r = ReplayReaderV1(trace)
        h1 = r.read_header()
        h2 = r.read_header()
        assert h1 is h2  # same object, not re-parsed

    def test_read_records_yields_all_after_header(self, tmp_path):
        trace = _make_trace(tmp_path, n_records=5)
        r = ReplayReaderV1(trace)
        kinds = [k for k, _ in r.read_records()]
        assert kinds.count("request") == 5
        assert kinds.count("footer") == 1

    def test_count_requests(self, tmp_path):
        trace = _make_trace(tmp_path, n_records=7)
        assert ReplayReaderV1(trace).count_requests() == 7

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ReplayReaderV1(tmp_path / "ghost.jsonl")


# ---------------------------------------------------------------------------
# Gzip auto-detect
# ---------------------------------------------------------------------------


class TestGzipDetection:
    def test_reads_gzipped(self, tmp_path):
        trace = _make_trace(tmp_path, gzip_output=True)
        assert trace.suffix == ".gz"
        r = ReplayReaderV1(trace)
        assert r.read_header()["kind"] == "header"

    def test_reads_plain(self, tmp_path):
        trace = _make_trace(tmp_path, gzip_output=False)
        assert trace.suffix == ".jsonl"
        r = ReplayReaderV1(trace)
        assert r.read_header()["kind"] == "header"


# ---------------------------------------------------------------------------
# Schema dispatch (load_reader)
# ---------------------------------------------------------------------------


class TestLoadReaderDispatch:
    def test_dispatches_to_v1(self, tmp_path):
        trace = _make_trace(tmp_path)
        r = load_reader(trace)
        assert isinstance(r, ReplayReaderV1)

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_reader(empty)

    def test_invalid_json_header_raises(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text("this is not json\n")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_reader(bad)

    def test_unknown_schema_version_raises(self, tmp_path):
        future = tmp_path / "future.jsonl"
        future.write_text(json.dumps({"kind": "header", "schema_version": 999}) + "\n")
        with pytest.raises(ReplaySchemaUnknownError, match="999"):
            load_reader(future)


# ---------------------------------------------------------------------------
# Tolerance of partial/missing pieces (D.1.2, D.1.11)
# ---------------------------------------------------------------------------


class TestReaderTolerance:
    def test_missing_footer_ok(self, tmp_path):
        trace = _make_trace(tmp_path, n_records=3, footer=False)
        r = ReplayReaderV1(trace)
        kinds = [k for k, _ in r.read_records()]
        assert kinds == ["request", "request", "request"]
        assert "footer" not in kinds

    def test_partial_final_line_skipped(self, tmp_path):
        """Writer crashed mid-record: last line is not valid JSON.
        Reader should skip and warn, not raise."""
        trace_gz = _make_trace(tmp_path, n_records=2, footer=False)
        # Decompress, append a partial line, re-gzip
        content = gzip.decompress(trace_gz.read_bytes()).decode()
        partial = content + '{"kind":"request","schema_vers'  # truncated
        trace_gz.write_bytes(gzip.compress(partial.encode()))

        r = ReplayReaderV1(trace_gz)
        records = list(r.read_records())
        assert len(records) == 2  # 2 requests, the partial is dropped

    def test_corrupt_middle_line_raises(self, tmp_path):
        """Corrupt line that isn't the last one is a real error, not
        tolerated."""
        trace_gz = _make_trace(tmp_path, n_records=3, footer=False)
        content = gzip.decompress(trace_gz.read_bytes()).decode()
        lines = content.splitlines()
        # Corrupt middle line (one of the requests) while keeping a valid
        # request after it so the corruption isn't the final line
        lines[2] = "not valid json"
        trace_gz.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode()))
        r = ReplayReaderV1(trace_gz)
        with pytest.raises(ValueError, match="not valid JSON"):
            list(r.read_records())

    def test_empty_lines_skipped(self, tmp_path):
        """Blank lines between records are OK."""
        trace = _make_trace(tmp_path, n_records=2, gzip_output=False, footer=False)
        content = trace.read_text()
        lines = content.splitlines()
        # Insert a blank line between header and first request
        lines.insert(1, "")
        trace.write_text("\n".join(lines) + "\n")
        r = ReplayReaderV1(trace)
        records = list(r.read_records())
        kinds = [k for k, _ in records]
        assert kinds == ["request", "request"]


# ---------------------------------------------------------------------------
# Header contract
# ---------------------------------------------------------------------------


class TestHeaderContract:
    def test_first_line_must_be_header(self, tmp_path):
        """First line with kind!=header is a malformed file."""
        bad = tmp_path / "noheader.jsonl"
        bad.write_text(
            json.dumps({"kind": "request", "schema_version": 1, "seq": 0}) + "\n"
        )
        r = ReplayReaderV1(bad)
        with pytest.raises(ValueError, match="expected 'header'"):
            r.read_header()

    def test_v1_reader_rejects_future_schema(self, tmp_path):
        """The v1 reader itself refuses v2 traces (load_reader dispatches,
        but direct construction still protects against drift)."""
        bad = tmp_path / "v2.jsonl"
        bad.write_text(json.dumps({"kind": "header", "schema_version": 2}) + "\n")
        r = ReplayReaderV1(bad)
        with pytest.raises(ValueError, match="Schema version mismatch"):
            r.read_header()

    def test_read_records_calls_read_header_implicitly(self, tmp_path):
        """Generator materializes header lazily; don't need to call
        read_header() first."""
        trace = _make_trace(tmp_path)
        r = ReplayReaderV1(trace)
        kinds = [k for k, _ in r.read_records()]
        assert kinds  # didn't crash
        assert r._header is not None  # got cached as a side effect
