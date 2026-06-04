"""Tests for the tamper-evident hash chain in the runtime trace recorder.

Validates the chain functions directly, and that ``RecordWriter._emit`` writes a
trace that ``verify_record_chain`` accepts and that any tamper breaks.
"""
from __future__ import annotations

import json

from tether.runtime.record import RecordWriter, _chain_hash, verify_record_chain


def _build_chain(payloads):
    """Mirror RecordWriter._emit's chaining over a list of record payloads."""
    prev = "0" * 64
    out = []
    for p in payloads:
        rec = dict(p)
        rec["prev_record_hash"] = prev
        rec["record_hash"] = _chain_hash(prev, rec)
        prev = rec["record_hash"]
        out.append(rec)
    return out


def test_intact_chain_verifies():
    recs = _build_chain([{"kind": "header"}, {"kind": "request", "seq": 1}, {"kind": "footer"}])
    ok, idx = verify_record_chain(recs)
    assert ok is True and idx is None


def test_content_edit_is_detected():
    recs = _build_chain([{"kind": "header"}, {"kind": "request", "seq": 1, "action_sha256": "a"}])
    recs[1]["action_sha256"] = "TAMPERED"  # edit a request's content
    ok, idx = verify_record_chain(recs)
    assert ok is False and idx == 1


def test_record_insertion_is_detected():
    recs = _build_chain([{"kind": "header"}, {"kind": "request", "seq": 1}])
    forged = {"kind": "request", "seq": 99, "prev_record_hash": recs[-1]["record_hash"]}
    forged["record_hash"] = _chain_hash(forged["prev_record_hash"], forged)
    recs.insert(1, forged)  # splice a record into the middle
    ok, idx = verify_record_chain(recs)
    assert ok is False and idx == 1


def test_reorder_is_detected():
    recs = _build_chain([{"kind": "header"}, {"kind": "request", "seq": 1}, {"kind": "request", "seq": 2}])
    recs[1], recs[2] = recs[2], recs[1]
    ok, idx = verify_record_chain(recs)
    assert ok is False


def test_recordwriter_emits_a_verifiable_chain(tmp_path):
    w = RecordWriter(
        tmp_path,
        model_hash="m0",
        config_hash="c0",
        export_dir=str(tmp_path),
        model_type="pi05",
        export_kind="decomposed",
        providers=["CPUExecutionProvider"],
        gzip_output=False,
    )
    w._emit({"kind": "header", "schema_version": 1})
    w._emit({"kind": "request", "seq": 0})
    w._emit({"kind": "footer"})
    lines = [json.loads(line) for line in open(w.filepath) if line.strip()]
    assert len(lines) == 3
    ok, idx = verify_record_chain(lines)
    assert ok is True and idx is None
    # And tampering the persisted trace is caught.
    lines[1]["seq"] = 7
    ok2, idx2 = verify_record_chain(lines)
    assert ok2 is False and idx2 == 1
