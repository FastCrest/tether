"""Family detection must use exact metadata before repository tokens."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from tether.exporters.monolithic import (
    _FAMILY_ALIASES,
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


def test_helper_rejects_unparseable_config(tmp_path):
    (tmp_path / "config.json").write_text("not valid json {{{ }}}")
    with pytest.raises(ValueError, match="Cannot read model family"):
        _model_type_from_local_config(str(tmp_path))


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


def test_helper_prefers_type_over_model_type_field(tmp_path):
    """LeRobot policy type wins over backbone model_type."""
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "paligemma",
        "type": "pi05",
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
        "policy_type": "pi_05",
    }))
    assert _model_type_from_local_config(str(tmp_path)) == "pi05"


def test_helper_rejects_unknown_model_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "policy_type": "not-a-known-family",
    }))
    with pytest.raises(ValueError, match="model family"):
        _model_type_from_local_config(str(tmp_path))


def test_helper_rejects_non_string_field(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "policy_type": 42,  # invalid type
    }))
    with pytest.raises(ValueError, match="model family"):
        _model_type_from_local_config(str(tmp_path))


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
    """When neither repository token nor config metadata resolves,
    raise the documented error."""
    bare = tmp_path / "no-config-here"
    bare.mkdir()
    with pytest.raises(ValueError, match="Cannot infer model_type"):
        export_monolithic(model_id=str(bare), output_dir=tmp_path / "out")


def test_export_monolithic_id_when_config_absent(tmp_path, monkeypatch):
    """Without metadata, an exact repository token selects the family."""
    monkeypatch.setattr("tether.exporters.monolithic._model_type_from_hub_config", lambda _: None)
    captured = {}

    def _fake_export_smolvla(model_id, output_dir, *, num_steps=10, target="desktop"):
        captured["smolvla_called"] = True
        return {"status": "ok"}

    monkeypatch.setattr(
        "tether.exporters.monolithic.export_smolvla_monolithic", _fake_export_smolvla,
    )

    # Missing Hub metadata permits exact repository-token dispatch
    export_monolithic(
        model_id="HuggingFaceVLA/smolvla_libero",
        output_dir=tmp_path / "out",
    )
    assert captured.get("smolvla_called")


@pytest.mark.parametrize("alias,family", list(_FAMILY_ALIASES.items()))
@pytest.mark.parametrize("field", ["type", "policy_type", "model_type"])
def test_every_config_family_alias(tmp_path, monkeypatch, alias, family, field):
    from tether.exporters import monolithic as m
    checkpoint = tmp_path / "pi0-misleading-name"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({field: alias}))
    if family == "pi0_fast":
        with pytest.raises(ValueError, match="pi0-FAST.*no monolithic exporter"):
            export_monolithic(str(checkpoint), tmp_path / "out")
    else:
        monkeypatch.setattr(m, f"export_{family}_monolithic", lambda *a, **k: {"family": family})
        assert export_monolithic(str(checkpoint), tmp_path / "out")["family"] == family


@pytest.mark.parametrize("alias,family", [
    (alias, family) for alias, family in _FAMILY_ALIASES.items() if alias != "gr00tn1d6"
])
@pytest.mark.parametrize("suffix", ["", "_base", "-custom"])
def test_every_id_family_alias(tmp_path, monkeypatch, alias, family, suffix):
    from tether.exporters import monolithic as m
    monkeypatch.setattr(m, "_model_type_from_hub_config", lambda _: None)
    model_id = f"org/{alias}{suffix}"
    if family == "pi0_fast":
        with pytest.raises(ValueError, match="pi0-FAST.*no monolithic exporter"):
            export_monolithic(model_id, tmp_path / "out")
    else:
        monkeypatch.setattr(m, f"export_{family}_monolithic", lambda *a, **k: {"family": family})
        assert export_monolithic(model_id, tmp_path / "out")["family"] == family


@pytest.mark.parametrize("model_id", ["pi0-org/unrelated", "org/notpi0", "org/pi0fastish", "org/pi0.50", "org/x-smolvla"])
def test_id_substrings_are_not_families(tmp_path, monkeypatch, model_id):
    monkeypatch.setattr("tether.exporters.monolithic._model_type_from_hub_config", lambda _: None)
    with pytest.raises(ValueError, match="Cannot infer model_type"):
        export_monolithic(model_id, tmp_path / "out")


@pytest.mark.parametrize("config", [{"type": "unknown"}, {"type": "pi0fastish"}, {"type": None}, []])
def test_bad_metadata_cannot_fall_back_to_pi0_id(tmp_path, config):
    checkpoint = tmp_path / "pi0-base"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        export_monolithic(str(checkpoint), tmp_path / "out")


def test_hub_metadata_precedes_id(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"type": "pi0fast", "model_type": "pi0"}))
    download = Mock(return_value=str(cfg))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    with pytest.raises(ValueError, match="pi0-FAST.*no monolithic exporter"):
        export_monolithic("org/pi0_base", tmp_path / "out")
    download.assert_called_once_with(repo_id="org/pi0_base", filename="config.json")


def test_hub_read_failure_cannot_fall_back_to_id(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise OSError("offline")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)
    with pytest.raises(ValueError, match="Cannot read config.json.*offline"):
        export_monolithic("org/pi0_base", tmp_path / "out")


@pytest.mark.parametrize("alias", ["pi0fast", "pi0_fast", "pi0-fast"])
def test_explicit_fast_family_is_refused(tmp_path, alias):
    with pytest.raises(ValueError, match="pi0-FAST.*no monolithic exporter"):
        export_monolithic("unrelated", tmp_path / "out", model_type=alias)


@pytest.mark.parametrize("alias,family", list(_FAMILY_ALIASES.items()))
def test_every_explicit_family_alias(tmp_path, monkeypatch, alias, family):
    from tether.exporters import monolithic as m

    if family == "pi0_fast":
        with pytest.raises(ValueError, match="pi0-FAST.*no monolithic exporter"):
            export_monolithic("unrelated", tmp_path / "out", model_type=alias)
    else:
        monkeypatch.setattr(m, f"export_{family}_monolithic", lambda *a, **k: {"family": family})
        assert export_monolithic("unrelated", tmp_path / "out", model_type=alias)["family"] == family


def test_missing_hub_config_permits_id_detection(tmp_path, monkeypatch):
    import requests
    from huggingface_hub.utils import EntryNotFoundError

    from tether.exporters import monolithic as m

    response = requests.Response()
    response.status_code = 404
    missing = EntryNotFoundError("config missing")
    missing.response = response
    download = Mock(side_effect=missing)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(m, "export_pi05_monolithic", lambda *a, **k: {"family": "pi05"})
    assert export_monolithic("org/pi0.5_custom", tmp_path / "out")["family"] == "pi05"


def test_offline_hub_cache_miss_cannot_fall_back(tmp_path, monkeypatch):
    from huggingface_hub.utils import LocalEntryNotFoundError

    download = Mock(side_effect=LocalEntryNotFoundError("offline cache miss"))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    with pytest.raises(ValueError, match="Cannot read config.json"):
        export_monolithic("org/pi0_base", tmp_path / "out")
