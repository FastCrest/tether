"""Regression tests for VERIFICATION.md auto-generation.

Covers:
  - Skeleton (no parity) has metadata, file manifest, sha256, "not yet verified"
  - With parity: PASS/FAIL shown, max_abs_diff tabulated, fixtures listed
  - Missing tether_config.json: falls back to "unknown" without crashing
  - Parity section stays consistent when overwriting a prior report
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tether.verification_report import REPORT_FILENAME, write_verification_report


def _make_export_dir(tmp_path: Path, with_config: bool = True) -> Path:
    d = tmp_path / "export"
    d.mkdir()
    if with_config:
        (d / "tether_config.json").write_text(json.dumps({
            "model_id": "lerobot/smolvla_base",
            "model_type": "smolvla",
            "target": "orin-nano",
            "opset": 19,
            "num_denoising_steps": 1,
            "chunk_size": 50,
        }))
    (d / "vision_encoder.onnx").write_bytes(b"fake onnx" * 100)
    return d


def test_skeleton_without_parity(tmp_path):
    export_dir = _make_export_dir(tmp_path)
    path = write_verification_report(export_dir, parity=None)

    assert path.name == REPORT_FILENAME
    text = path.read_text()
    assert "# Tether Export Verification" in text
    assert "`lerobot/smolvla_base`" in text
    assert "ONNX opset:** 19" in text
    assert "Not yet verified" in text
    # sha256 should appear for the fake ONNX
    assert "vision_encoder.onnx" in text
    assert "`" in text  # hash wrapped in backticks


def test_with_parity_passing(tmp_path):
    export_dir = _make_export_dir(tmp_path)
    parity = {
        "model_type": "smolvla",
        "threshold": 1e-4,
        "num_test_cases": 3,
        "seed": 0,
        "results": [
            {"fixture_idx": 0, "max_abs_diff": 1.2e-6, "mean_abs_diff": 3.4e-7, "passed": True},
            {"fixture_idx": 1, "max_abs_diff": 2.1e-6, "mean_abs_diff": 5.6e-7, "passed": True},
            {"fixture_idx": 2, "max_abs_diff": 1.8e-6, "mean_abs_diff": 4.2e-7, "passed": True},
        ],
        "summary": {"max_abs_diff_across_all": 2.1e-6, "passed": True},
    }
    path = write_verification_report(export_dir, parity=parity)
    text = path.read_text()
    assert "Verdict:** PASS" in text
    assert "2.100e-06" in text
    # All three fixtures listed
    assert "| 0 |" in text
    assert "| 1 |" in text
    assert "| 2 |" in text
    assert "Not yet verified" not in text


def test_with_parity_failing(tmp_path):
    export_dir = _make_export_dir(tmp_path)
    parity = {
        "threshold": 1e-4,
        "num_test_cases": 1,
        "seed": 0,
        "results": [
            {"fixture_idx": 0, "max_abs_diff": 1e-3, "mean_abs_diff": 5e-4, "passed": False},
        ],
        "summary": {"max_abs_diff_across_all": 1e-3, "passed": False},
    }
    path = write_verification_report(export_dir, parity=parity)
    text = path.read_text()
    assert "Verdict:** FAIL" in text
    assert "FAIL" in text


def test_missing_config_is_tolerated(tmp_path):
    # No tether_config.json — values fall back to "unknown" without crashing
    export_dir = _make_export_dir(tmp_path, with_config=False)
    path = write_verification_report(export_dir, parity=None)
    text = path.read_text()
    assert "unknown" in text
    assert "# Tether Export Verification" in text


def test_overwrites_prior_report(tmp_path):
    export_dir = _make_export_dir(tmp_path)
    # First: skeleton
    write_verification_report(export_dir, parity=None)
    text_before = (export_dir / REPORT_FILENAME).read_text()
    assert "Not yet verified" in text_before
    # Second: with parity — should overwrite the "Not yet verified" line
    parity = {
        "threshold": 1e-4,
        "num_test_cases": 1,
        "seed": 0,
        "results": [{"fixture_idx": 0, "max_abs_diff": 1e-6, "mean_abs_diff": 1e-7, "passed": True}],
        "summary": {"max_abs_diff_across_all": 1e-6, "passed": True},
    }
    write_verification_report(export_dir, parity=parity)
    text_after = (export_dir / REPORT_FILENAME).read_text()
    assert "Not yet verified" not in text_after
    assert "Verdict:** PASS" in text_after


def test_missing_export_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_verification_report(tmp_path / "does_not_exist")


def _extract_sha256_for(report_text: str, filename: str) -> str:
    """Pull the sha256 hex string from the VERIFICATION.md table row for *filename*."""
    for line in report_text.splitlines():
        if filename in line:
            # Table row format: | `filename` | <size> | `<sha256>` |
            parts = line.split("|")
            for part in parts:
                part = part.strip().strip("`")
                # sha256 hex is exactly 64 hex chars
                if len(part) == 64 and all(c in "0123456789abcdef" for c in part):
                    return part
    raise AssertionError(f"Could not find sha256 for {filename!r} in report")


def test_hash_freshness_after_file_mutation(tmp_path):
    """A second write_verification_report call after mutating a file records
    the NEW hash, not the stale pre-mutation one.

    This directly validates that the post-fusion VERIFICATION.md refresh in
    export_monolithic will produce correct hashes when model.onnx bytes change.
    """
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    (export_dir / "tether_config.json").write_text(json.dumps({
        "model_id": "lerobot/smolvla_base",
        "model_type": "smolvla",
        "target": "desktop",
        "opset": 19,
        "num_denoising_steps": 1,
        "chunk_size": 50,
    }))
    onnx_file = export_dir / "model.onnx"
    onnx_file.write_bytes(b"pre-fusion onnx bytes -- version 1")

    # First write: hashes the pre-mutation file
    write_verification_report(export_dir, parity=None)
    text_before = (export_dir / REPORT_FILENAME).read_text()
    hash_before = _extract_sha256_for(text_before, "model.onnx")

    # Simulate weight fusion: atomically replace model.onnx with different bytes
    onnx_file.write_bytes(b"post-fusion onnx bytes -- version 2 structurally different")

    # Second write: should recompute and record the new hash
    write_verification_report(export_dir, parity=None)
    text_after = (export_dir / REPORT_FILENAME).read_text()
    hash_after = _extract_sha256_for(text_after, "model.onnx")

    assert hash_before != hash_after, (
        "Expected hash to change after mutating model.onnx, "
        f"but both writes recorded: {hash_before}"
    )
    # Confirm the second hash matches what we compute independently
    import hashlib
    expected = hashlib.sha256(
        b"post-fusion onnx bytes -- version 2 structurally different"
    ).hexdigest()
    assert hash_after == expected, (
        f"Post-mutation hash in report ({hash_after!r}) does not match "
        f"expected sha256 ({expected!r})"
    )
