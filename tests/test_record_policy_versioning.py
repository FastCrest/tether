"""Day 7 record-replay schema additions for policy-versioning.

Per ADR 2026-04-25-policy-versioning-architecture: header gets an
optional `policies` block; per-request gets an optional `routing`
block. v1 readers ignore unknown fields -> no schema_version bump,
no back-compat break.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from tether.runtime.record import RecordWriter


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL.gz file produced by RecordWriter into a list of dicts."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _make_writer(tmp_path: Path, **overrides) -> RecordWriter:
    base = dict(
        record_dir=tmp_path,
        model_hash="abc123",
        config_hash="def456",
        export_dir=tmp_path,
        model_type="pi05_decomposed",
        export_kind="decomposed",
        providers=["CUDAExecutionProvider"],
        gzip_output=True,
    )
    base.update(overrides)
    return RecordWriter(**base)


# ---------------------------------------------------------------------------
# Header `policies` block
# ---------------------------------------------------------------------------


def test_header_omits_policies_block_when_not_passed(tmp_path):
    """Single-policy mode: no `policies` block in header (back-compat)."""
    writer = _make_writer(tmp_path)
    writer.write_request(
        chunk_id=0, image_b64=None, instruction="x", state=None,
        actions=[[0.0]], action_dim=1, latency_total_ms=1.0,
    )
    writer.close()
    records = _read_jsonl(writer.filepath)
    header = records[0]
    assert header["kind"] == "header"
    assert "policies" not in header


def test_header_emits_policies_block_when_passed(tmp_path):
    """2-policy mode: header contains `policies` list with per-policy meta."""
    policies_meta = [
        {"slot": "a", "model_id": "pi0-libero-v1", "model_hash": "aaaaa1"},
        {"slot": "b", "model_id": "pi0-libero-v2", "model_hash": "bbbbb2"},
    ]
    writer = _make_writer(tmp_path, policies=policies_meta)
    writer.write_request(
        chunk_id=0, image_b64=None, instruction="x", state=None,
        actions=[[0.0]], action_dim=1, latency_total_ms=1.0,
    )
    writer.close()
    records = _read_jsonl(writer.filepath)
    header = records[0]
    assert header["kind"] == "header"
    assert "policies" in header
    assert header["policies"] == policies_meta


def test_header_policies_block_back_compat_v1_readers_ignore(tmp_path):
    """schema_version stays at the existing value -- additive only.
    Verifies that adding `policies` doesn't bump schema_version."""
    writer1 = _make_writer(tmp_path / "no-policies")
    writer1.write_request(
        chunk_id=0, image_b64=None, instruction="x", state=None,
        actions=[[0.0]], action_dim=1, latency_total_ms=1.0,
    )
    writer1.close()
    h1 = _read_jsonl(writer1.filepath)[0]

    writer2 = _make_writer(
        tmp_path / "with-policies",
        policies=[{"slot": "a", "model_id": "x", "model_hash": "y"}],
    )
    writer2.write_request(
        chunk_id=0, image_b64=None, instruction="x", state=None,
        actions=[[0.0]], action_dim=1, latency_total_ms=1.0,
    )
    writer2.close()
    h2 = _read_jsonl(writer2.filepath)[0]
    # Same schema_version on both -- additive evolution
    assert h1["schema_version"] == h2["schema_version"]


# ---------------------------------------------------------------------------
# Per-request `routing` block
# ---------------------------------------------------------------------------


def test_request_omits_routing_when_not_passed(tmp_path):
    writer = _make_writer(tmp_path)
    writer.write_request(
        chunk_id=0, image_b64=None, instruction="x", state=None,
        actions=[[0.0]], action_dim=1, latency_total_ms=1.0,
    )
    writer.close()
    records = _read_jsonl(writer.filepath)
    request = next(r for r in records if r["kind"] == "request")
    assert "routing" not in request


def test_request_emits_routing_when_passed(tmp_path):
    writer = _make_writer(tmp_path)
    routing = {
        "slot": "a",
        "bucket_decision": 42,
        "routing_key": "ep_xyz",
        "degraded": False,
        "cached": False,
    }
    writer.write_request(
        chunk_id=0, image_b64=None, instruction="x", state=None,
        actions=[[0.1, 0.2]], action_dim=2, latency_total_ms=1.0,
        routing=routing,
    )
    writer.close()
    records = _read_jsonl(writer.filepath)
    request = next(r for r in records if r["kind"] == "request")
    assert "routing" in request
    assert request["routing"] == routing


def test_routing_block_supports_shadow_actions(tmp_path):
    """Legacy inline shadow inference can still carry routing.shadow_actions."""
    writer = _make_writer(tmp_path)
    routing = {
        "slot": "prod",
        "shadow_actions": [[0.5, -0.3], [0.4, -0.2]],
        "shadow_model_id": "candidate-v3",
    }
    writer.write_request(
        chunk_id=0, image_b64=None, instruction="x", state=None,
        actions=[[0.1, 0.2]], action_dim=2, latency_total_ms=1.0,
        routing=routing,
    )
    writer.close()
    records = _read_jsonl(writer.filepath)
    request = next(r for r in records if r["kind"] == "request")
    assert "shadow_actions" in request["routing"]
    assert request["routing"]["shadow_actions"] == routing["shadow_actions"]


def test_request_records_carry_routing_per_call(tmp_path):
    """Each request can have its own routing decision (sticky-per-episode
    so usually constant within a session, but per-record for audit)."""
    writer = _make_writer(tmp_path)
    for i, slot in enumerate(["a", "a", "b", "b", "a"]):
        writer.write_request(
            chunk_id=i, image_b64=None, instruction="x", state=None,
            actions=[[0.0]], action_dim=1, latency_total_ms=1.0,
            routing={"slot": slot, "routing_key": f"ep_{i}"},
        )
    writer.close()
    records = _read_jsonl(writer.filepath)
    requests = [r for r in records if r["kind"] == "request"]
    assert len(requests) == 5
    slots = [r["routing"]["slot"] for r in requests]
    assert slots == ["a", "a", "b", "b", "a"]
