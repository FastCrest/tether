"""Unit tests for the latency-histograms + determinism-version-hash goals.

The server adds two field groups to every /act response:

* latency percentiles (p50/p95/p99 + jitter_ms) computed over a rolling
  1024-sample window;
* deployment fingerprint (model_hash, config_hash, tether_version) so
  callers can assert they're hitting the expected artifact.

These tests exercise the two pure helpers on a minimally-instantiated
TetherServer so we don't need ONNX runtime or a real export.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def stub_export_dir(tmp_path):
    """Minimal export dir — tether_config.json + stub onnx file."""
    cfg = {
        "model_id": "lerobot/smolvla_base",
        "model_type": "smolvla",
        "target": "desktop",
        "action_chunk_size": 50,
        "action_dim": 32,
        "expert": {"expert_hidden": 720, "action_dim": 32, "num_layers": 16},
    }
    (tmp_path / "tether_config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.onnx").write_bytes(b"\x00\x01\x02\x03")
    return tmp_path


@pytest.fixture
def stub_server(stub_export_dir):
    """Instantiate TetherServer without calling load() (no ONNX runtime)."""
    from tether.runtime.server import TetherServer
    return TetherServer(stub_export_dir, device="cpu")


class TestLatencyHistograms:
    def test_empty_history_returns_zeros(self, stub_server):
        pcts = stub_server._latency_percentiles()
        assert pcts == {
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
            "jitter_ms": 0.0,
        }

    def test_single_sample_all_percentiles_equal(self, stub_server):
        stub_server._latency_history.append(12.5)
        pcts = stub_server._latency_percentiles()
        assert pcts["latency_p50_ms"] == 12.5
        assert pcts["latency_p95_ms"] == 12.5
        assert pcts["latency_p99_ms"] == 12.5
        assert pcts["jitter_ms"] == 0.0

    def test_monotonic_samples(self, stub_server):
        for ms in range(1, 101):  # 1..100
            stub_server._latency_history.append(float(ms))
        pcts = stub_server._latency_percentiles()
        # 100 samples sorted, nearest-rank on idx = round(p * (n-1)):
        #   p50 → idx round(49.5) = 50 (banker) → 51.0
        #   p95 → idx round(94.05) = 94 → 95.0
        #   p99 → idx round(98.01) = 98 → 99.0
        assert pcts["latency_p50_ms"] == 51.0
        assert pcts["latency_p95_ms"] == 95.0
        assert pcts["latency_p99_ms"] == 99.0
        assert pcts["jitter_ms"] == 48.0  # p99 - p50

    def test_history_is_capped(self, stub_server):
        """Rolling window should be <=1024."""
        assert stub_server._latency_history.maxlen == 1024
        for ms in range(2000):
            stub_server._latency_history.append(float(ms))
        assert len(stub_server._latency_history) == 1024


class TestDeterminismHash:
    def test_fields_present(self, stub_server):
        fields = stub_server._determinism_fields()
        assert set(fields.keys()) == {"model_hash", "config_hash", "tether_version"}

    def test_model_hash_is_stable(self, stub_server):
        """Same bytes on disk → same hash across calls."""
        a = stub_server._determinism_fields()["model_hash"]
        b = stub_server._determinism_fields()["model_hash"]
        assert a == b
        assert len(a) == 16  # sha256 truncated

    def test_model_hash_changes_with_file_content(
        self, stub_export_dir, stub_server
    ):
        """Changing the onnx file's bytes changes the hash."""
        before = stub_server._determinism_fields()["model_hash"]
        (stub_export_dir / "model.onnx").write_bytes(b"\x99\x88\x77\x66")
        # Reset the cache; re-compute on next call.
        stub_server._model_hash = None
        after = stub_server._determinism_fields()["model_hash"]
        assert before != after

    def test_config_hash_is_stable(self, stub_server):
        a = stub_server._determinism_fields()["config_hash"]
        b = stub_server._determinism_fields()["config_hash"]
        assert a == b
        assert len(a) == 16

    def test_tether_version_is_string(self, stub_server):
        v = stub_server._determinism_fields()["tether_version"]
        assert isinstance(v, str)
        # "unknown" if the package metadata can't be read — still a string.
        assert len(v) > 0
