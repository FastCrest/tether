"""Weight licences and commercial-use metadata must survive CLI rendering."""

import json

import pytest
from typer.testing import CliRunner

from tether.cli import app
from tether.registry import ModelEntry, by_id

EXPECTED = [
    ("pi0-base", "gemma", "conditional"),
    ("pi05-base", "gemma", "conditional"),
    ("pi05-libero", "gemma", "conditional"),
    ("pi05-libero10-fluxvla", "gemma", "conditional"),
    ("gr00t-n1.6", "nvidia-oneway-noncommercial", "prohibited"),
    ("dreamzero-libero10", "unknown (derived; upstream cc-by-nc-4.0)", "prohibited"),
    ("molmoact2-base", "unknown", "unknown"),
    ("molmoact2-libero", "unknown", "unknown"),
    ("molmoact2-droid", "unknown", "unknown"),
]


@pytest.mark.parametrize("model_id,licence,commercial", EXPECTED)
def test_registry_weight_licences(model_id, licence, commercial):
    entry = by_id(model_id)
    assert entry.license == licence
    assert entry.commercial_use == commercial
    assert "Apache" not in entry.description


@pytest.mark.parametrize("model_id,licence,commercial", EXPECTED)
def test_models_info_json_licences(model_id, licence, commercial):
    result = CliRunner().invoke(app, ["models", "info", model_id, "--json"])
    assert result.exit_code == 0, result.output
    row = json.loads(result.output)
    assert row["license"] == licence
    assert row["commercial_use"] == commercial


def test_models_list_json_licences():
    result = CliRunner().invoke(app, ["models", "list", "--json"])
    assert result.exit_code == 0, result.output
    rows = {row["model_id"]: row for row in json.loads(result.output)["models"]}
    for model_id, licence, commercial in EXPECTED:
        assert rows[model_id]["license"] == licence
        assert rows[model_id]["commercial_use"] == commercial


def test_models_info_human_shows_restrictions():
    result = CliRunner().invoke(app, ["models", "info", "gr00t-n1.6"])
    assert result.exit_code == 0, result.output
    assert "nvidia-oneway-noncommercial" in result.output
    assert "commercial use: prohibited" in result.output
    assert "research/evaluation only" in result.output


def test_commercial_use_defaults_to_unknown_and_validates():
    kwargs = {
        "model_id": "custom",
        "hf_repo": "org/custom",
        "family": "pi0",
        "action_dim": 7,
        "size_mb": 1,
    }
    assert ModelEntry(**kwargs).commercial_use == "unknown"
    with pytest.raises(ValueError, match="commercial_use"):
        ModelEntry(**kwargs, commercial_use="yes")


GROOT_REVISION = "d0814e7ecb19202e7c8468b46098b0b7ef3a6d61"


@pytest.mark.parametrize("model_ref", ["gr00t-n1.6", "nvidia/GR00T-N1.6-3B"])
def test_groot_pull_uses_licence_checked_revision(tmp_path, monkeypatch, model_ref):
    from unittest.mock import Mock

    download = Mock()
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    entry = by_id("gr00t-n1.6")
    assert entry.hf_revision == GROOT_REVISION
    result = CliRunner().invoke(
        app,
        ["models", "pull", model_ref, "--target-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert download.call_args.kwargs["repo_id"] == entry.hf_repo
    assert download.call_args.kwargs["revision"] == GROOT_REVISION


@pytest.mark.parametrize("model_ref", ["gr00t-n1.6", "nvidia/GR00T-N1.6-3B"])
def test_groot_checkpoint_download_uses_licence_checked_revision(tmp_path, monkeypatch, model_ref):
    from unittest.mock import Mock
    import torch
    from safetensors.torch import save_file
    from tether.checkpoint import load_checkpoint

    save_file({"weight": torch.zeros(1)}, str(tmp_path / "model.safetensors"))
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    weights, _ = load_checkpoint(model_ref)
    assert "weight" in weights
    assert download.call_args.kwargs["revision"] == GROOT_REVISION
    assert download.call_args.args == ("nvidia/GR00T-N1.6-3B",)


def test_groot_family_config_download_uses_licence_checked_revision(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from tether.exporters.monolithic import _model_type_from_hub_config

    config = tmp_path / "config.json"
    config.write_text('{"model_type": "gr00t"}')
    download = Mock(return_value=str(config))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    assert _model_type_from_hub_config("nvidia/GR00T-N1.6-3B") == "gr00t"
    assert download.call_args.kwargs["revision"] == GROOT_REVISION
