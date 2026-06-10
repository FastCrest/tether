"""Regression tests for fp16_convert external-data cleanup scoping.

Guards the data-loss bug where the pre-save cleanup globbed every *.bin/*.data
in the destination's parent directory and unlinked them — destroying OTHER
models' weight files when a user converted into a shared export dir. The fix
scopes deletion strictly to the model being converted (dst.stem).
"""
from __future__ import annotations

from pathlib import Path

from tether.exporters.fp16_convert import _remove_stale_external_data


def _touch(p: Path) -> Path:
    p.write_bytes(b"x")
    return p


def test_removes_this_models_stale_bin(tmp_path: Path) -> None:
    dst = tmp_path / "model_fp16.onnx"
    stale = _touch(tmp_path / "model_fp16.bin")
    removed = _remove_stale_external_data(dst)
    assert stale in removed
    assert not stale.exists()


def test_removes_legacy_data_variants_for_this_model(tmp_path: Path) -> None:
    dst = tmp_path / "model_fp16.onnx"
    legacy1 = _touch(tmp_path / "model_fp16.data")
    legacy2 = _touch(tmp_path / "model_fp16.onnx.data")
    removed = _remove_stale_external_data(dst)
    assert legacy1 in removed and legacy2 in removed
    assert not legacy1.exists() and not legacy2.exists()


def test_preserves_other_models_weights(tmp_path: Path) -> None:
    """The core data-loss guard: a sibling model's external data must survive."""
    dst = tmp_path / "model_fp16.onnx"
    _touch(tmp_path / "model_fp16.bin")  # ours — will be removed
    other_bin = _touch(tmp_path / "other_model.bin")
    other_data = _touch(tmp_path / "unrelated.onnx.data")
    other_onnx = _touch(tmp_path / "other_model.onnx")

    removed = _remove_stale_external_data(dst)

    assert other_bin.exists(), "sibling model .bin was destroyed"
    assert other_data.exists(), "unrelated .onnx.data was destroyed"
    assert other_onnx.exists()
    assert other_bin not in removed and other_data not in removed


def test_never_removes_dst_itself(tmp_path: Path) -> None:
    dst = _touch(tmp_path / "model_fp16.onnx")
    _remove_stale_external_data(dst)
    assert dst.exists(), "dst .onnx must never be unlinked"


def test_no_leftovers_is_noop(tmp_path: Path) -> None:
    dst = tmp_path / "model_fp16.onnx"
    assert _remove_stale_external_data(dst) == []
