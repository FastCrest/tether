"""Tests for the model_type fallback in src/tether/exporters/monolithic.py.

Caught by 2026-04-25 self-distilling-serve distill smoke (reflex_context
experiment). When export_monolithic is called with a local checkpoint
path that doesn't carry the family name (e.g., distill output dirs),
the substring match fails. Fallback reads config.json's model_type or
policy_type field.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tether.exporters.monolithic import (
    _model_type_from_local_config,
    export_monolithic,
)


# ---------------------------------------------------------------------------
# _model_type_from_local_config helper
# ---------------------------------------------------------------------------


def test_helper_returns_none_for_nonexistent_path():
    assert _model_type_from_local_config("/does/not/exist") is None


def test_helper_returns_none_when_config_absent(tmp_path):
    # Empty dir, no config.json
    assert _model_type_from_local_config(str(tmp_path)) is None


def test_helper_returns_none_for_unparseable_config(tmp_path):
    (tmp_path / "config.json").write_text("not valid json {{{ }}}")
    assert _model_type_from_local_config(str(tmp_path)) is None


def test_helper_extracts_pi05_from_model_type_field(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "pi05",
        "other_field": "irrelevant",
    }))
    assert _model_type_from_local_config(str(tmp_path)) == "pi05"


def test_helper_extracts_pi05_from_policy_type_field(tmp_path):
    """lerobot convention uses `policy_type`, not `model_type`."""
    (tmp_path / "config.json").write_text(json.dumps({
        "policy_type": "pi05",
    }))
    assert _model_type_from_local_config(str(tmp_path)) == "pi05"


def test_helper_extracts_pi05_from_type_field(tmp_path):
    """SnapFlow checkpoint convention uses `type`, not `model_type` or
    `policy_type`. Caught by 2026-04-25 distill smoke v2."""
    (tmp_path / "config.json").write_text(json.dumps({
        "type": "pi05",
    }))
    assert _model_type_from_local_config(str(tmp_path)) == "pi05"


def test_helper_prefers_model_type_over_type_field(tmp_path):
    """When both fields present, model_type wins (HF convention is
    canonical when unambiguous)."""
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "pi05",
        "type": "should-not-be-used",
    }))
    assert _model_type_from_local_config(str(tmp_path)) == "pi05"


def test_helper_extracts_smolvla(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "policy_type": "smolvla",
    }))
    assert _model_type_from_local_config(str(tmp_path)) == "smolvla"


def test_helper_extracts_pi0(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "policy_type": "pi0",
    }))
    assert _model_type_from_local_config(str(tmp_path)) == "pi0"


def test_helper_extracts_gr00t(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "policy_type": "gr00t",
    }))
    assert _model_type_from_local_config(str(tmp_path)) == "gr00t"


def test_helper_handles_underscore_variants(tmp_path):
    """pi_05 / pi0_5 / pi_0 underscore variants in config map to the
    bounded enum."""
    (tmp_path / "config.json").write_text(json.dumps({
        "policy_type": "pi_05_some_variant",
    }))
    assert _model_type_from_local_config(str(tmp_path)) == "pi05"


def test_helper_returns_none_for_unknown_model_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "policy_type": "not-a-known-family",
    }))
    assert _model_type_from_local_config(str(tmp_path)) is None


def test_helper_returns_none_when_field_not_a_string(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "policy_type": 42,  # invalid type
    }))
    assert _model_type_from_local_config(str(tmp_path)) is None


def test_helper_returns_none_when_neither_field_present(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "name": "some-model",
    }))
    assert _model_type_from_local_config(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# export_monolithic uses the fallback (without invoking real export logic)
# ---------------------------------------------------------------------------


def test_export_monolithic_fallback_to_config_for_unknown_path(tmp_path, monkeypatch):
    """The original distill bug: model_id is a path that doesn't carry
    the family name. Fallback reads config.json -> dispatches correctly."""
    distill_out = tmp_path / "distill_validation_smoke_2026-04-25" / "checkpoints" / "00000100" / "pretrained_model"
    distill_out.mkdir(parents=True)
    (distill_out / "config.json").write_text(json.dumps({
        "policy_type": "pi05",
    }))

    # Stub the per-family exporter so we don't actually run pytorch->onnx.
    captured = {}

    def _fake_export_pi05(model_id, output_dir, *, num_steps=10, target="desktop"):
        captured["called"] = True
        captured["model_id"] = model_id
        captured["model_type_inferred"] = "pi05"
        return {"status": "ok", "onnx_path": str(Path(output_dir) / "model.onnx")}

    monkeypatch.setattr(
        "tether.exporters.monolithic.export_pi05_monolithic", _fake_export_pi05,
    )

    output = tmp_path / "out"
    output.mkdir()
    result = export_monolithic(model_id=str(distill_out), output_dir=output)

    # Fallback resolved pi05 from config.json -> dispatched to pi05 exporter
    assert captured.get("called")
    assert captured.get("model_type_inferred") == "pi05"
    assert result["status"] == "ok"


def test_export_monolithic_still_raises_when_no_fallback_available(tmp_path):
    """When neither substring match nor config.json fallback resolves,
    raise the documented error."""
    bare = tmp_path / "no-config-here"
    bare.mkdir()
    with pytest.raises(ValueError, match="Cannot infer model_type"):
        export_monolithic(model_id=str(bare), output_dir=tmp_path / "out")


def test_export_monolithic_substring_match_takes_precedence(tmp_path, monkeypatch):
    """Substring match still works first (cheaper than reading config.json)."""
    captured = {}

    def _fake_export_smolvla(model_id, output_dir, *, num_steps=10, target="desktop"):
        captured["smolvla_called"] = True
        return {"status": "ok"}

    monkeypatch.setattr(
        "tether.exporters.monolithic.export_smolvla_monolithic", _fake_export_smolvla,
    )

    # HF id substring match -> dispatches without touching the filesystem
    export_monolithic(
        model_id="HuggingFaceVLA/smolvla_libero",
        output_dir=tmp_path / "out",
    )
    assert captured.get("smolvla_called")
