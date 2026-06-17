"""Tests for the first-class tether smoke receipt."""

from __future__ import annotations

import json

import pytest

from tether.runtime.tokenizers import find_bundled_tokenizer_path
from tether.smoke import (
    create_smoke_export,
    format_smoke_human,
    format_smoke_markdown,
    summarize_latency_samples,
)


def test_create_smoke_export_has_model_config_and_tokenizer(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")

    export_dir = create_smoke_export(tmp_path / "export")

    assert (export_dir / "model.onnx").exists()
    assert (export_dir / "tokenizer" / "tokenizer.json").exists()

    config = json.loads((export_dir / "tether_config.json").read_text())
    assert config["model_type"] == "smolvla"
    assert config["smoke_export"] is True
    assert config["chunk_size"] == 50
    assert config["action_dim"] == 32
    assert find_bundled_tokenizer_path(export_dir, config) == export_dir / "tokenizer"


def test_smoke_formatters_report_key_receipt_fields():
    receipt = {
        "passed": True,
        "tether_version": "0.0.test",
        "python": "3.12.0",
        "offline": True,
        "export_dir": "/tmp/tether-smoke/export",
        "duration_ms": 123.4,
        "server": {"url": "http://127.0.0.1:18080"},
        "doctor": {"summary": {"pass": 4, "fail": 0, "warn": 1, "skip": 2}},
        "latency": {
            "samples": 3,
            "first_sample": {"inference_ms": 4.0, "roundtrip_ms": 10.0},
            "inference_ms": {"p50_ms": 2.5, "p95_ms": 3.9, "max_ms": 4.0},
            "roundtrip_ms": {"p50_ms": 8.1, "p95_ms": 9.9, "max_ms": 10.0},
            "warm_inference_ms": {"p50_ms": 2.0, "p95_ms": 2.5, "max_ms": 2.5},
            "warm_roundtrip_ms": {"p50_ms": 7.0, "p95_ms": 8.1, "max_ms": 8.1},
        },
        "act": {
            "num_actions": 50,
            "action_dim": 32,
            "provider_mode": "onnx_cpu",
            "active_providers": ["CPUExecutionProvider"],
            "latency_ms": 2.5,
            "roundtrip_ms": 8.1,
        },
    }

    human = format_smoke_human(receipt)
    markdown = format_smoke_markdown(receipt)

    assert "tether smoke - PASS" in human
    assert "doctor:  4 pass, 0 fail, 1 warn, 2 skip" in human
    assert "roundtrip p50/p95=8.1/9.9ms" in human
    assert "warm roundtrip p50/p95=7.0/8.1ms" in human
    assert "- Status: PASS" in markdown
    assert "- Shape: 50 x 32" in markdown
    assert "- Roundtrip p95: 9.9 ms" in markdown
    assert "- Warm roundtrip p95: 8.1 ms" in markdown


def test_summarize_latency_samples_computes_p50_p95():
    summary = summarize_latency_samples([
        {"latency_ms": 10.0, "roundtrip_ms": 15.0},
        {"latency_ms": 20.0, "roundtrip_ms": 25.0},
        {"latency_ms": 100.0, "roundtrip_ms": 110.0},
    ])

    assert summary["samples"] == 3
    assert summary["first_sample"]["inference_ms"] == 10.0
    assert summary["first_sample"]["roundtrip_ms"] == 15.0
    assert summary["inference_ms"]["p50_ms"] == 20.0
    assert summary["inference_ms"]["p95_ms"] == 92.0
    assert summary["roundtrip_ms"]["p50_ms"] == 25.0
    assert summary["roundtrip_ms"]["p95_ms"] == 101.5
    assert summary["warm_inference_ms"]["p50_ms"] == 60.0
    assert summary["warm_inference_ms"]["p95_ms"] == 96.0
    assert summary["warm_roundtrip_ms"]["p50_ms"] == 67.5
    assert summary["warm_roundtrip_ms"]["p95_ms"] == 105.8
