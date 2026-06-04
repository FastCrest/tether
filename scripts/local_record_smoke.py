"""Smoke test: instantiate RecordWriter, emit 3 fake /act records, read back.

Doesn't need a model — exercises the writer module in isolation. Validates
that the JSONL is schema-conformant per TECHNICAL_PLAN §D.1.

Run:
    PYTHONPATH=src .venv/bin/python scripts/local_record_smoke.py
    # writes to /tmp/tether-record-smoke/
"""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tether.runtime.record import (  # noqa: E402
    SCHEMA_VERSION,
    RecordWriter,
    compute_config_hash,
    compute_model_hash,
)


def main() -> int:
    out_dir = Path(tempfile.mkdtemp(prefix="tether-record-smoke-"))
    print(f"output dir: {out_dir}")

    rec = RecordWriter(
        record_dir=out_dir,
        model_hash="abc123def4567890",
        config_hash="0123456789abcdef",
        export_dir="/fake/export",
        model_type="pi0.5",
        export_kind="monolithic",
        providers=["CUDAExecutionProvider"],
        gpu="NVIDIA A10G",
        cuda_version="12.6",
        ort_version="1.20.1",
        embodiment="franka",
        image_redaction="hash_only",
        tether_version="0.1.0-smoketest",
    )
    print(f"file: {rec.filepath}")

    for i in range(3):
        seq = rec.write_request(
            chunk_id=i,
            image_b64="aGVsbG8gd29ybGQ=",  # base64("hello world")
            instruction=f"pick up object {i}",
            state=[0.1 * i, 0.2 * i, 0.3 * i],
            actions=[[0.0] * 7] * 50,
            action_dim=7,
            latency_total_ms=100.0 + i,
            latency_stages={
                "preprocess_ms": 1.0,
                "vlm_prefix_ms": 80.0,
                "expert_denoise_ms": 18.0 + i,
            },
            mode="onnx_gpu",
        )
        print(f"  emitted seq={seq}")

    rec.write_footer({"total_requests": rec.seq, "total_errors": 0})
    rec.close()

    # Read back
    print(f"\nReading back {rec.filepath}...")
    opener = gzip.open if rec.filepath.suffix == ".gz" else open
    with opener(rec.filepath, "rt", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    print(f"  total lines: {len(lines)}")
    assert len(lines) == 1 + 3 + 1, f"expected 5 lines (header+3+footer), got {len(lines)}"
    assert lines[0]["kind"] == "header"
    assert lines[0]["schema_version"] == SCHEMA_VERSION
    assert lines[0]["embodiment"] == "franka"
    assert lines[0]["redaction"]["image"] == "hash_only"
    print("  header OK")

    for i, rec_line in enumerate(lines[1:4]):
        assert rec_line["kind"] == "request", f"line {i+1}: kind={rec_line['kind']}"
        assert rec_line["seq"] == i
        assert rec_line["chunk_id"] == i
        assert "image_sha256" in rec_line["request"]
        assert "image_b64" not in rec_line["request"], (
            "hash_only redaction should drop image_b64"
        )
        assert rec_line["request"]["instruction"] == f"pick up object {i}"
        assert rec_line["response"]["num_actions"] == 50
        assert rec_line["response"]["action_dim"] == 7
        assert rec_line["latency"]["total_ms"] == 100.0 + i
        assert rec_line["latency"]["stages"]["vlm_prefix_ms"] == 80.0
    print("  3 request records OK (correct schema, redaction applied)")

    assert lines[-1]["kind"] == "footer"
    assert lines[-1]["total_requests"] == 3
    print("  footer OK")

    print(f"\nAll smoke checks passed.")
    print(f"  file: {rec.filepath}")
    print(f"  size: {rec.filepath.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
