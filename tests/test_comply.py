"""Tests for reflex.comply — the EU conformity evidence pack.

Synthetic export dir + audit log; no GPU/robot. Validates the SBOM, the reg-checklist
mapping (met / gap / customer-responsibility), the audit-log summary, and the
end-to-end conformity bundle (artifacts detected + hashed, files written, serializable).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from reflex.comply import (
    build_conformity_bundle,
    evaluate,
    generate_sbom,
    render_markdown,
    sbom_json,
    summarize_audit_log,
)
from reflex.comply.checklist import ARTIFACTS

ALL_ARTIFACTS = set(ARTIFACTS)


# --------------------------------------------------------------------------- #
# SBOM (CRA)                                                                   #
# --------------------------------------------------------------------------- #


def test_sbom_is_valid_cyclonedx():
    s = generate_sbom()
    assert s["bomFormat"] == "CycloneDX"
    assert s["specVersion"] == "1.5"
    assert s["metadata"]["component"]["type"] == "application"
    assert len(s["components"]) > 0  # the test env has installed packages
    comp = s["components"][0]
    assert {"type", "name", "version", "purl"} <= set(comp)
    json.loads(sbom_json())  # parses


# --------------------------------------------------------------------------- #
# checklist mapping                                                           #
# --------------------------------------------------------------------------- #


def test_checklist_met_gap_and_customer():
    by_ref = {s.requirement.ref: s for s in evaluate(ALL_ARTIFACTS) if s.requirement.framework == "EU AI Act"}
    assert by_ref["Art 12"].status == "met"          # logging satisfied (audit_log present)
    assert by_ref["Art 13"].status == "customer-responsibility"
    assert by_ref["Art 9"].status == "partial"       # partial role, evidence present

    no_audit = ALL_ARTIFACTS - {"audit_log"}
    g = {s.requirement.ref: s for s in evaluate(no_audit) if s.requirement.framework == "EU AI Act"}
    assert g["Art 12"].status == "gap"
    assert "audit_log" in g["Art 12"].missing


def test_checklist_framework_filter():
    cra = evaluate(ALL_ARTIFACTS, frameworks=("cra",))
    assert cra
    assert all(s.requirement.framework == "CRA" for s in cra)


# --------------------------------------------------------------------------- #
# audit-log summary                                                           #
# --------------------------------------------------------------------------- #


def _write_audit_log(p: Path, gz: bool = False) -> None:
    recs = [
        {"meta": {"model_hash": "abc123"}, "session": "s1"},  # header
        {"timestamp": "2026-05-31T10:00:00Z", "model_hash": "abc123", "action": [0.1] * 7},
        {"timestamp": "2026-05-31T10:00:01Z", "model_hash": "abc123", "guard_clamped": True},
    ]
    data = "\n".join(json.dumps(r) for r in recs) + "\n"
    if gz:
        with gzip.open(p, "wt", encoding="utf-8") as f:
            f.write(data)
    else:
        p.write_text(data)


def test_summarize_audit_log(tmp_path):
    log = tmp_path / "audit.jsonl"
    _write_audit_log(log)
    s = summarize_audit_log(log)
    assert s["records"] == 3
    assert s["model_hashes"] == ["abc123"]
    assert s["model_hash_consistent"] is True
    assert s["safety_events"] == 1
    assert s["time_range"] == ["2026-05-31T10:00:00Z", "2026-05-31T10:00:01Z"]
    assert len(s["file_sha256"]) == 64


def test_summarize_audit_log_gzip(tmp_path):
    log = tmp_path / "audit.jsonl.gz"
    _write_audit_log(log, gz=True)
    s = summarize_audit_log(log)
    assert s["records"] == 3 and s["safety_events"] == 1


# --------------------------------------------------------------------------- #
# end-to-end bundle                                                           #
# --------------------------------------------------------------------------- #


def test_build_bundle_full(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    (export / "VERIFICATION.md").write_text("# Reflex Export Verification\n\ncos=1.0\n")
    (export / "reflex_config.json").write_text(json.dumps({"model_id": "pi05", "target": "orin"}))
    log = tmp_path / "audit.jsonl"
    _write_audit_log(log)
    out = tmp_path / "bundle"

    report = build_conformity_bundle(export, log, out_dir=out)

    assert report.artifacts["parity_cert"]["present"] is True
    assert len(report.artifacts["parity_cert"]["sha256"]) == 64
    assert report.artifacts["model_metadata"]["present"] is True
    assert report.artifacts["audit_log"]["present"] is True
    assert report.artifacts["sbom"]["present"] is True
    assert report.audit_summary["records"] == 3

    assert report.counts["met"] >= 1
    assert report.counts["customer-responsibility"] >= 1
    assert sum(report.counts.values()) == len(report.checklist)

    assert (out / "CONFORMITY.md").exists()
    assert (out / "conformity.json").exists()
    assert (out / "sbom.json").exists()
    json.loads((out / "conformity.json").read_text())
    md = (out / "CONFORMITY.md").read_text()
    assert "Conformity Evidence Bundle" in md
    assert "Disclaimer" in md


def test_build_bundle_flags_gap_without_audit_log(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    (export / "VERIFICATION.md").write_text("# cert")
    report = build_conformity_bundle(export, audit_log_path=None)
    art12 = next(r for r in report.checklist if r["ref"] == "Art 12")
    assert art12["status"] == "gap"
    assert "audit_log" in art12["missing"]
    assert report.artifacts["audit_log"]["present"] is False


def test_report_serializable_and_renderable(tmp_path):
    export = tmp_path / "e"
    export.mkdir()
    report = build_conformity_bundle(export)
    d = report.to_dict()
    json.dumps(d)  # must not raise
    assert "disclaimer" in d
    assert isinstance(render_markdown(report), str)
