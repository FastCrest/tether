"""Tests for the JSONL recorder (B.2 Day 1).

Covers schema v1 compliance (TECHNICAL_PLAN §D.1.3/4/6), image redaction
modes, gzip round-trip, disk-full degraded path, model/config hashing.

Pure stdlib — no model loads, no network.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tether.runtime.record import (
    SCHEMA_VERSION,
    RecordWriter,
    compute_config_hash,
    compute_model_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_writer(tmp_path: Path, **kwargs) -> RecordWriter:
    """Factory with sensible defaults for tests."""
    defaults = dict(
        model_hash="abc123def4567890",
        config_hash="0123456789abcdef",
        export_dir=str(tmp_path / "fake_export"),
        model_type="pi0.5",
        export_kind="monolithic",
        providers=["CUDAExecutionProvider"],
        gpu="test-gpu",
        cuda_version="12.6",
        ort_version="1.20.1",
        embodiment="franka",
        image_redaction="hash_only",
        tether_version="0.0.0-test",
    )
    defaults.update(kwargs)
    return RecordWriter(record_dir=tmp_path, **defaults)


def _read_all(path: Path) -> list[dict]:
    """Read a JSONL[.gz] file into a list of parsed records."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _dummy_request(rec: RecordWriter, i: int = 0, **overrides) -> int:
    kw = dict(
        chunk_id=i,
        image_b64="aGVsbG8gd29ybGQ=",
        instruction=f"test instruction {i}",
        state=[0.1, 0.2, 0.3],
        actions=[[0.0] * 7] * 50,
        action_dim=7,
        latency_total_ms=100.0 + i,
        mode="onnx_gpu",
    )
    kw.update(overrides)
    return rec.write_request(**kw)


# ---------------------------------------------------------------------------
# Schema compliance (D.1.3, D.1.4, D.1.6)
# ---------------------------------------------------------------------------


class TestHeaderFormat:
    def test_header_emitted_on_first_request(self, tmp_path):
        rec = _make_writer(tmp_path)
        _dummy_request(rec)
        rec.close()
        records = _read_all(rec.filepath)
        assert records[0]["kind"] == "header"
        assert records[0]["schema_version"] == SCHEMA_VERSION

    def test_header_has_required_fields(self, tmp_path):
        rec = _make_writer(tmp_path)
        _dummy_request(rec)
        rec.close()
        h = _read_all(rec.filepath)[0]
        for field in [
            "kind", "schema_version", "tether_version", "model_hash",
            "config_hash", "export_dir", "model_type", "export_kind",
            "hardware", "providers", "session_id", "started_at",
        ]:
            assert field in h, f"header missing required field '{field}'"

    def test_header_embodiment_propagates(self, tmp_path):
        rec = _make_writer(tmp_path, embodiment="so100")
        _dummy_request(rec)
        rec.close()
        assert _read_all(rec.filepath)[0]["embodiment"] == "so100"

    def test_header_tether_version_propagates(self, tmp_path):
        rec = _make_writer(tmp_path, tether_version="9.9.9-test")
        _dummy_request(rec)
        rec.close()
        assert _read_all(rec.filepath)[0]["tether_version"] == "9.9.9-test"

    def test_no_header_if_no_requests(self, tmp_path):
        """Empty session → no file written (lazy open)."""
        rec = _make_writer(tmp_path)
        rec.close()
        assert not rec.filepath.exists()


class TestRequestFormat:
    def test_request_has_required_fields(self, tmp_path):
        rec = _make_writer(tmp_path)
        _dummy_request(rec)
        rec.close()
        req_rec = _read_all(rec.filepath)[1]
        for field in [
            "kind", "schema_version", "seq", "chunk_id", "timestamp",
            "request", "response", "latency", "denoise", "mode",
        ]:
            assert field in req_rec, f"request missing '{field}'"
        assert req_rec["kind"] == "request"

    def test_seq_monotonic(self, tmp_path):
        rec = _make_writer(tmp_path)
        for i in range(5):
            _dummy_request(rec, i=i)
        rec.close()
        records = [r for r in _read_all(rec.filepath) if r["kind"] == "request"]
        assert [r["seq"] for r in records] == list(range(5))

    def test_response_action_shape(self, tmp_path):
        rec = _make_writer(tmp_path)
        _dummy_request(rec, actions=[[0.0] * 6] * 40, action_dim=6)
        rec.close()
        r = _read_all(rec.filepath)[1]
        assert r["response"]["num_actions"] == 40
        assert r["response"]["action_dim"] == 6

    def test_latency_stages_optional(self, tmp_path):
        """No stages provided → latency.stages key absent."""
        rec = _make_writer(tmp_path)
        _dummy_request(rec)  # no latency_stages
        rec.close()
        r = _read_all(rec.filepath)[1]
        assert "stages" not in r["latency"]
        assert r["latency"]["total_ms"] == 100.0

    def test_latency_stages_when_provided(self, tmp_path):
        rec = _make_writer(tmp_path)
        _dummy_request(
            rec,
            latency_total_ms=100.0,
        )
        # Second call with stages
        rec.write_request(
            chunk_id=99,
            image_b64="x",
            instruction="",
            state=None,
            actions=[[0.0]],
            action_dim=1,
            latency_total_ms=50.0,
            latency_stages={"vlm_prefix_ms": 40.0, "expert_denoise_ms": 10.0},
            mode="onnx_cpu",
        )
        rec.close()
        records = [r for r in _read_all(rec.filepath) if r["kind"] == "request"]
        assert records[1]["latency"]["stages"]["vlm_prefix_ms"] == 40.0

    def test_optional_fields_dropped_when_none(self, tmp_path):
        """cache / guard / deadline / rtc / error omitted when None."""
        rec = _make_writer(tmp_path)
        _dummy_request(rec)  # none of the optional fields passed
        rec.close()
        r = _read_all(rec.filepath)[1]
        for k in ["cache", "guard", "deadline", "rtc", "error"]:
            assert k not in r, f"optional field '{k}' should be absent when None"

    def test_rtc_field_preserved_when_provided(self, tmp_path):
        rec = _make_writer(tmp_path)
        _dummy_request(
            rec,
            rtc={
                "adaptive_chunking": {
                    "horizon": 4,
                    "reason": "guard_margin",
                    "risk_score": 0.75,
                    "replan_threshold_ratio": 0.6,
                },
                "adaptive_signal": {
                    "guard_margin": 0.03,
                    "correction_magnitude": 0.2,
                    "uncertainty": 0.4,
                },
                "last_action_delta": 0.12,
            },
        )
        rec.close()
        r = _read_all(rec.filepath)[1]
        assert r["rtc"]["adaptive_chunking"]["horizon"] == 4
        assert r["rtc"]["adaptive_signal"]["guard_margin"] == pytest.approx(0.03)
        assert r["rtc"]["last_action_delta"] == pytest.approx(0.12)


class TestFooterFormat:
    def test_footer_on_close(self, tmp_path):
        rec = _make_writer(tmp_path)
        _dummy_request(rec)
        rec.write_footer({"total_requests": 1})
        rec.close()
        records = _read_all(rec.filepath)
        assert records[-1]["kind"] == "footer"
        assert records[-1]["total_requests"] == 1
        assert records[-1]["schema_version"] == SCHEMA_VERSION
        assert "ended_at" in records[-1]

    def test_no_footer_if_no_header(self, tmp_path):
        """Footer without preceding header is a malformed file — skip it."""
        rec = _make_writer(tmp_path)
        rec.write_footer({"total_requests": 0})
        rec.close()
        # No file should have been opened at all
        assert not rec.filepath.exists()


# ---------------------------------------------------------------------------
# Image redaction (D.1.8)
# ---------------------------------------------------------------------------


class TestImageRedaction:
    @pytest.fixture
    def image_b64(self):
        # Base64 of "hello world" — stable SHA for regression-proofing
        return "aGVsbG8gd29ybGQ="

    def test_hash_only_default_drops_base64(self, tmp_path, image_b64):
        rec = _make_writer(tmp_path, image_redaction="hash_only")
        _dummy_request(rec, image_b64=image_b64)
        rec.close()
        req = _read_all(rec.filepath)[1]["request"]
        assert "image_b64" not in req
        assert "image_sha256" in req
        # Stable check
        expected = hashlib.sha256(image_b64.encode()).hexdigest()[:16]
        assert req["image_sha256"] == expected

    def test_full_keeps_base64_and_hash(self, tmp_path, image_b64):
        rec = _make_writer(tmp_path, image_redaction="full")
        _dummy_request(rec, image_b64=image_b64)
        rec.close()
        req = _read_all(rec.filepath)[1]["request"]
        assert req["image_b64"] == image_b64
        assert "image_sha256" in req

    def test_none_drops_both(self, tmp_path, image_b64):
        rec = _make_writer(tmp_path, image_redaction="none")
        _dummy_request(rec, image_b64=image_b64)
        rec.close()
        req = _read_all(rec.filepath)[1]["request"]
        assert "image_b64" not in req
        assert "image_sha256" not in req

    def test_none_image_input(self, tmp_path):
        """If no image given, no image fields regardless of mode."""
        rec = _make_writer(tmp_path, image_redaction="full")
        _dummy_request(rec, image_b64=None)
        rec.close()
        req = _read_all(rec.filepath)[1]["request"]
        assert "image_b64" not in req
        assert "image_sha256" not in req


# ---------------------------------------------------------------------------
# Gzip round-trip
# ---------------------------------------------------------------------------


class TestGzipRoundTrip:
    def test_gzip_default_produces_gz(self, tmp_path):
        rec = _make_writer(tmp_path, gzip_output=True)
        _dummy_request(rec)
        rec.close()
        assert rec.filepath.suffix == ".gz"

    def test_no_gzip_plain_jsonl(self, tmp_path):
        rec = _make_writer(tmp_path, gzip_output=False)
        _dummy_request(rec)
        rec.close()
        assert rec.filepath.suffix == ".jsonl"

    def test_gzip_readable(self, tmp_path):
        rec = _make_writer(tmp_path, gzip_output=True)
        for i in range(5):
            _dummy_request(rec, i=i)
        rec.close()
        records = _read_all(rec.filepath)
        assert len(records) == 6  # header + 5 requests

    def test_gzip_size_smaller_for_big_payload(self, tmp_path):
        """Sanity: gzip should be at least somewhat smaller for 10 records."""
        big_actions = [[0.1] * 7] * 50
        rec_gz = _make_writer(tmp_path / "gz", gzip_output=True)
        rec_pl = _make_writer(tmp_path / "plain", gzip_output=False)
        for i in range(10):
            _dummy_request(rec_gz, i=i, actions=big_actions)
            _dummy_request(rec_pl, i=i, actions=big_actions)
        rec_gz.close()
        rec_pl.close()
        assert rec_gz.filepath.stat().st_size < rec_pl.filepath.stat().st_size


# ---------------------------------------------------------------------------
# Disk-full degraded path (D.1.11)
# ---------------------------------------------------------------------------


class TestDegradedPath:
    def test_degraded_on_write_failure(self, tmp_path, monkeypatch):
        """When file.write() raises OSError, recorder marks degraded + stops.
        Seq assignment happens BEFORE the failed emit, so the degrading call
        still returns a valid seq — subsequent calls return -1."""
        rec = _make_writer(tmp_path)
        _dummy_request(rec)  # opens file + writes header + record
        assert not rec.degraded

        # Make every subsequent write raise
        def always_fail(_data):
            raise OSError("simulated disk full")

        rec._fh.write = always_fail  # type: ignore[assignment]

        # The call that triggers degradation still returns a valid seq
        _dummy_request(rec, i=1)
        assert rec.degraded is True

        # Subsequent calls short-circuit and return -1
        seq_after = _dummy_request(rec, i=2)
        assert seq_after == -1

    def test_degraded_recorder_skips_subsequent_writes(self, tmp_path):
        """Once degraded, write_request returns -1 and doesn't touch the file."""
        rec = _make_writer(tmp_path)
        _dummy_request(rec)
        rec.degraded = True
        size_before = rec.filepath.stat().st_size
        seq = _dummy_request(rec, i=99)
        assert seq == -1
        assert rec.filepath.stat().st_size == size_before

    def test_degraded_skips_footer(self, tmp_path):
        """write_footer is a no-op when degraded."""
        rec = _make_writer(tmp_path)
        _dummy_request(rec)
        rec.degraded = True
        rec.write_footer({"total_requests": 99})
        rec.close()
        records = _read_all(rec.filepath)
        kinds = [r["kind"] for r in records]
        assert "footer" not in kinds


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


class TestHashHelpers:
    def test_compute_model_hash_missing_dir(self, tmp_path):
        assert compute_model_hash(tmp_path / "does_not_exist") == ""

    def test_compute_model_hash_empty_dir(self, tmp_path):
        assert compute_model_hash(tmp_path) == ""

    def test_compute_model_hash_stable(self, tmp_path):
        (tmp_path / "model.onnx").write_bytes(b"some-onnx-bytes")
        h1 = compute_model_hash(tmp_path)
        h2 = compute_model_hash(tmp_path)
        assert h1 == h2
        assert len(h1) == 16

    def test_compute_model_hash_changes_on_edit(self, tmp_path):
        (tmp_path / "model.onnx").write_bytes(b"v1")
        h1 = compute_model_hash(tmp_path)
        (tmp_path / "model.onnx").write_bytes(b"v2")
        h2 = compute_model_hash(tmp_path)
        assert h1 != h2

    def test_compute_model_hash_includes_bin(self, tmp_path):
        """External weights files in .bin also contribute."""
        (tmp_path / "model.onnx").write_bytes(b"graph")
        h_onnx_only = compute_model_hash(tmp_path)
        (tmp_path / "model.onnx.data.bin").write_bytes(b"weights")
        h_both = compute_model_hash(tmp_path)
        assert h_onnx_only != h_both

    def test_compute_config_hash_missing(self, tmp_path):
        assert compute_config_hash(tmp_path) == ""

    def test_compute_config_hash_canonical(self, tmp_path):
        """Key order doesn't matter — canonicalization sorts."""
        (tmp_path / "tether_config.json").write_text('{"b": 2, "a": 1}')
        h1 = compute_config_hash(tmp_path)
        (tmp_path / "tether_config.json").write_text('{"a": 1, "b": 2}')
        h2 = compute_config_hash(tmp_path)
        assert h1 == h2

    def test_compute_config_hash_invalid_json(self, tmp_path):
        (tmp_path / "tether_config.json").write_text("not json {{{")
        assert compute_config_hash(tmp_path) == ""
