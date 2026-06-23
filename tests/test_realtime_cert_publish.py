"""Tests for the realtime-certificate latency-table publisher."""

from __future__ import annotations

import json

import pytest

from tether.realtime_cert import (
    build_realtime_certificate,
    format_realtime_certificates_markdown_table,
    write_realtime_certificate,
)
from tether.realtime_cert_publish import (
    CertificateLoadError,
    inject_table_into_readme,
    load_certificate,
    publish,
)


def _receipt(export_dir: str, *, p95: float, hz: int = 20) -> dict:
    """A deployment-proof receipt with controllable p95 + export dir."""

    return {
        "schema_version": 1,
        "kind": "tether.deployment_proof",
        "passed": True,
        "export_dir": export_dir,
        "profile": {
            "name": "warehouse-safe",
            "thresholds": {"control_hz": hz, "max_jitter_p95_minus_p50_ms": 50},
        },
        "act_samples": [
            {"roundtrip_ms": 10.0},
            {"roundtrip_ms": 20.0},
            {"roundtrip_ms": p95},
        ],
        "latency": {
            "samples": 3,
            "roundtrip_ms": {"p50_ms": 20.0, "p95_ms": p95, "p99_ms": p95, "max_ms": p95},
            "jitter": {"p95_minus_p50_ms": 5.0},
            "control_budget": {
                "control_hz": float(hz),
                "period_ms": 1000.0 / hz,
                "missed_samples": 0,
            },
            "deadline_misses": 0,
            "act_errors": 0,
        },
    }


def _write_cert(tmp_path, name, export_dir, p95, target="orin-nano"):
    report = build_realtime_certificate(_receipt(export_dir, p95=p95), target=target)
    out = tmp_path / name
    write_realtime_certificate(report, out)
    return out


# ── table formatter ──────────────────────────────────────────────────────────


def test_table_has_one_row_per_certificate():
    fast = build_realtime_certificate(
        _receipt("/c/exports/smolvla-base", p95=25.0), target="orin-nano"
    )
    slow = build_realtime_certificate(
        _receipt("/c/exports/smolvla-libero", p95=80.0), target="orin-nano"
    )
    table = format_realtime_certificates_markdown_table([fast, slow])

    assert "## Realtime serving latency" in table
    assert "`smolvla-base`" in table
    assert "`smolvla-libero`" in table
    assert table.count("`orin-nano`") == 2
    assert "25.0" in table
    assert "80.0" in table


def test_decision_column_reflects_control_budget():
    # 25ms p95 under the 50ms (20Hz) budget -> PASS; 999ms over -> FAIL.
    fast = build_realtime_certificate(
        _receipt("/c/exports/smolvla-base", p95=25.0), target="orin-nano"
    )
    slow = build_realtime_certificate(
        _receipt("/c/exports/pi05-student", p95=999.0), target="orin-nano"
    )
    table = format_realtime_certificates_markdown_table([fast, slow])
    rows = {
        line.split("|")[1].strip(): line
        for line in table.splitlines()
        if line.startswith("| `")
    }
    assert "**PASS**" in rows["`smolvla-base`"]
    assert "**FAIL**" in rows["`pi05-student`"]


def test_missing_metrics_render_placeholders():
    report = {
        "kind": "tether.realtime_serving_certificate",
        "decision": "FAIL",
        "target": "",
        "source": {},
        "control_budget": {},
        "latency": {},
    }
    table = format_realtime_certificates_markdown_table([report])
    assert "—" in table  # missing latency metrics -> em dash
    assert "unknown" in table  # model label falls back to "unknown"


# ── load_certificate ─────────────────────────────────────────────────────────


def test_load_certificate_from_dir_or_file(tmp_path):
    cert_dir = _write_cert(tmp_path, "certA", "/c/exports/smolvla-base", 12.0)

    _, from_dir = load_certificate(cert_dir)
    assert from_dir["kind"] == "tether.realtime_serving_certificate"

    _, from_file = load_certificate(cert_dir / "realtime-serving-cert.json")
    assert from_file == from_dir


def test_load_certificate_rejects_non_certificate(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"kind": "tether.deployment_proof"}))
    with pytest.raises(CertificateLoadError):
        load_certificate(bad)

    with pytest.raises(CertificateLoadError):
        load_certificate(tmp_path / "does-not-exist")


# ── publish orchestrator ─────────────────────────────────────────────────────


def test_publish_writes_doc_and_injects_readme(tmp_path):
    a = _write_cert(tmp_path, "certA", "/c/exports/smolvla-base", 12.0)
    b = _write_cert(tmp_path, "certB", "/c/exports/smolvla-libero", 14.0)
    readme = tmp_path / "README.md"
    readme.write_text(
        "intro\n\n<!-- BEGIN:jetson-latency-table -->\n"
        "<!-- END:jetson-latency-table -->\n\noutro\n"
    )
    out = tmp_path / "results.md"

    result = publish([a, b], out=out, readme=readme)

    assert result["count"] == 2
    assert result["readme_updated"] is True
    doc = out.read_text()
    assert "## Realtime serving latency" in doc
    assert "`smolvla-base`" in doc and "`smolvla-libero`" in doc
    assert "## Per-certificate detail" in doc

    injected = readme.read_text()
    assert "`smolvla-base`" in injected
    assert injected.startswith("intro") and injected.rstrip().endswith("outro")

    # idempotent: a second run leaves the README byte-identical
    publish([a, b], out=out, readme=readme)
    assert readme.read_text() == injected


def test_publish_without_readme(tmp_path):
    a = _write_cert(tmp_path, "certA", "/c/exports/smolvla-base", 12.0)
    out = tmp_path / "results.md"

    result = publish([a], out=out, readme=None)

    assert result["readme_updated"] is False
    assert out.exists()


def test_inject_skips_malformed_markers(tmp_path):
    # END marker before BEGIN (out of order) must return False, not raise.
    readme = tmp_path / "README.md"
    readme.write_text(
        "x\n<!-- END:jetson-latency-table -->\n<!-- BEGIN:jetson-latency-table -->\ny\n"
    )
    before = readme.read_text()
    assert inject_table_into_readme(readme, "| t |") is False
    assert readme.read_text() == before  # left untouched
