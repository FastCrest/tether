"""Tests for the model registry + `reflex models {list,pull,info}` CLI.

Covers:
- Registry data invariants (every entry parses; no duplicate model_ids;
  hf_repo well-formed; family in canonical set; action_dim positive)
- Filter helpers (by_id, filter_models, list_families, list_devices)
- ModelEntry frozen-dataclass validation guards (post_init raises on bad inputs)
- ModelBenchmark per-device lookup
- CLI subcommands list/info/pull via typer.testing.CliRunner with the HF
  download mocked out (no real network calls)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from reflex.registry import (
    REGISTRY,
    ModelEntry,
    ModelBenchmark,
    by_id,
    filter_models,
    list_devices,
    list_families,
)


# ---------- registry data invariants ----------

class TestRegistryInvariants:
    def test_registry_non_empty(self):
        assert len(REGISTRY) >= 4, "ship at least 4 curated entries to start"

    def test_no_duplicate_model_ids(self):
        ids = [e.model_id for e in REGISTRY]
        assert len(ids) == len(set(ids)), f"duplicate model_id: {ids}"

    def test_every_hf_repo_has_org_slash_name(self):
        for e in REGISTRY:
            assert "/" in e.hf_repo, f"{e.model_id}: hf_repo missing org/: {e.hf_repo!r}"

    def test_every_family_is_canonical(self):
        canonical = {"pi0", "pi05", "smolvla", "openvla", "groot"}
        for e in REGISTRY:
            assert e.family in canonical, f"{e.model_id}: bad family {e.family!r}"

    def test_every_action_dim_positive(self):
        for e in REGISTRY:
            assert e.action_dim > 0, f"{e.model_id}: action_dim must be > 0"

    def test_every_entry_has_description(self):
        for e in REGISTRY:
            assert e.description, f"{e.model_id}: description empty"

    def test_supported_devices_in_known_set(self):
        known = {"orin_nano", "agx_orin", "thor", "a10g", "a100", "h100", "h200"}
        for e in REGISTRY:
            for d in e.supported_devices:
                assert d in known, f"{e.model_id}: unknown device {d!r}"


# ---------- filter helpers ----------

class TestFilters:
    def test_by_id_returns_match(self):
        assert by_id("pi05-base") is not None
        assert by_id("pi05-base").family == "pi05"

    def test_by_id_returns_none_for_miss(self):
        assert by_id("nope-doesnt-exist") is None

    def test_filter_by_family(self):
        results = filter_models(family="pi05")
        assert len(results) >= 1
        assert all(e.family == "pi05" for e in results)

    def test_filter_by_device(self):
        # SmolVLA entries support orin_nano in the seed; pi0/pi0.5 do not
        nano = filter_models(device="orin_nano")
        assert len(nano) >= 1
        assert all("orin_nano" in e.supported_devices for e in nano)

    def test_filter_by_embodiment(self):
        franka = filter_models(embodiment="franka")
        assert len(franka) >= 1
        assert all("franka" in e.supported_embodiments for e in franka)

    def test_filter_compose(self):
        # SmolVLA + orin_nano + franka should yield at least smolvla-base
        results = filter_models(family="smolvla", device="orin_nano", embodiment="franka")
        ids = {e.model_id for e in results}
        assert "smolvla-base" in ids

    def test_filter_returns_empty_for_impossible_combo(self):
        results = filter_models(family="pi0", device="orin_nano")
        # pi0 base is NOT marked orin_nano-compatible in seed registry
        assert results == []

    def test_list_families_returns_distinct(self):
        fams = list_families()
        assert "pi0" in fams
        assert "pi05" in fams
        assert "smolvla" in fams
        assert len(fams) == len(set(fams))

    def test_list_devices_returns_sorted_union(self):
        devs = list_devices()
        assert devs == sorted(devs)
        assert "a10g" in devs


# ---------- ModelEntry validation ----------

class TestModelEntryValidation:
    def test_empty_model_id_raises(self):
        with pytest.raises(ValueError, match="model_id required"):
            ModelEntry(model_id="", hf_repo="org/repo", family="pi0", action_dim=7, size_mb=10)

    def test_slash_in_model_id_raises(self):
        with pytest.raises(ValueError, match="kebab-case"):
            ModelEntry(model_id="bad/id", hf_repo="org/repo", family="pi0", action_dim=7, size_mb=10)

    def test_hf_repo_without_org_slash_raises(self):
        with pytest.raises(ValueError, match="org/name"):
            ModelEntry(model_id="x", hf_repo="no_slash", family="pi0", action_dim=7, size_mb=10)

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="family must be one of"):
            ModelEntry(model_id="x", hf_repo="org/repo", family="bogus", action_dim=7, size_mb=10)

    def test_zero_action_dim_raises(self):
        with pytest.raises(ValueError, match="action_dim"):
            ModelEntry(model_id="x", hf_repo="org/repo", family="pi0", action_dim=0, size_mb=10)


class TestModelBenchmark:
    def test_benchmark_for_returns_match(self):
        e = by_id("pi05-base")
        assert e is not None
        b = e.benchmark_for("a10g")
        assert b is not None
        assert b.device == "a10g"
        assert b.p50_ms > 0

    def test_benchmark_for_returns_none_for_miss(self):
        e = by_id("pi05-base")
        assert e.benchmark_for("orin_nano_pro_max") is None


# ---------- CLI integration ----------

@pytest.fixture
def runner():
    typer_testing = pytest.importorskip("typer.testing")
    return typer_testing.CliRunner()


@pytest.fixture
def cli_app():
    from reflex.cli import app
    return app


class TestCliList:
    def test_models_list_table(self, runner, cli_app):
        result = runner.invoke(cli_app, ["models", "list"])
        assert result.exit_code == 0
        # Table should mention each model_id from the registry
        for e in REGISTRY:
            assert e.model_id in result.stdout

    def test_models_list_filter_by_family(self, runner, cli_app):
        result = runner.invoke(cli_app, ["models", "list", "--family", "smolvla"])
        assert result.exit_code == 0
        # smolvla entries should show, pi0/pi05 should not
        assert "smolvla-base" in result.stdout
        # pi05-base is in pi05 family, should be filtered out
        assert "pi05-base" not in result.stdout or "smolvla" in result.stdout.split("pi05-base")[0]

    def test_models_list_json(self, runner, cli_app):
        result = runner.invoke(cli_app, ["models", "list", "--format", "json"])
        assert result.exit_code == 0
        # Parse the JSON portion (rich may add trailing newlines but shouldn't change JSON)
        # Find the JSON block
        body = json.loads(result.stdout)
        assert body["n"] == len(REGISTRY)
        assert len(body["models"]) == len(REGISTRY)
        ids = {m["model_id"] for m in body["models"]}
        assert "pi05-base" in ids

    def test_models_list_no_match_message(self, runner, cli_app):
        # Use a family that's genuinely not in the registry. openvla was
        # added in v0.9.6 so it now matches; pick a sentinel name that
        # won't accidentally match a future addition either.
        result = runner.invoke(cli_app, ["models", "list", "--family", "nonexistent_family_xyz"])
        assert result.exit_code == 0
        assert "No models match" in result.stdout

    def test_models_list_invalid_format(self, runner, cli_app):
        result = runner.invoke(cli_app, ["models", "list", "--format", "xml"])
        assert result.exit_code == 2


class TestCliInfo:
    def test_info_human(self, runner, cli_app):
        result = runner.invoke(cli_app, ["models", "info", "pi05-base"])
        assert result.exit_code == 0
        assert "pi05-base" in result.stdout
        assert "lerobot/pi05_base" in result.stdout

    def test_info_json(self, runner, cli_app):
        result = runner.invoke(cli_app, ["models", "info", "smolvla-base", "--format", "json"])
        assert result.exit_code == 0
        body = json.loads(result.stdout)
        assert body["model_id"] == "smolvla-base"
        assert body["family"] == "smolvla"
        assert body["action_dim"] == 7

    def test_info_unknown_id_exit_2(self, runner, cli_app):
        result = runner.invoke(cli_app, ["models", "info", "nope-xyz"])
        assert result.exit_code == 2
        assert "Unknown model_id" in result.stdout


class TestCliPull:
    def test_pull_unknown_id_lists_available(self, runner, cli_app):
        result = runner.invoke(cli_app, ["models", "pull", "nope-xyz"])
        assert result.exit_code == 2
        assert "Unknown model_id" in result.stdout
        assert "Available registry ids:" in result.stdout
        # Should list every registered model
        for e in REGISTRY:
            assert e.model_id in result.stdout

    def test_pull_calls_snapshot_download_with_repo_id(self, runner, cli_app, tmp_path):
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = str(tmp_path / "fake")
            (tmp_path / "fake").mkdir()
            (tmp_path / "fake" / "config.json").write_text("{}")
            result = runner.invoke(cli_app, ["models", "pull", "smolvla-base", "--target-dir", str(tmp_path / "fake")])
        assert result.exit_code == 0, result.stdout
        mock_dl.assert_called_once()
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["repo_id"] == "lerobot/smolvla_base"
        assert kwargs["local_dir"] == str(tmp_path / "fake")

    def test_pull_handles_download_failure_with_exit_1(self, runner, cli_app, tmp_path):
        with patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("403 forbidden")):
            result = runner.invoke(cli_app, ["models", "pull", "smolvla-base", "--target-dir", str(tmp_path / "out")])
        assert result.exit_code == 1
        assert "Download failed" in result.stdout

    def test_pull_passes_revision_override(self, runner, cli_app, tmp_path):
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = str(tmp_path / "out")
            (tmp_path / "out").mkdir()
            runner.invoke(
                cli_app,
                ["models", "pull", "smolvla-base", "--target-dir", str(tmp_path / "out"),
                 "--revision", "abc1234"],
            )
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["revision"] == "abc1234"

    def test_pull_warns_when_requires_export(self, runner, cli_app, tmp_path):
        # pi05-base requires_export=True
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = str(tmp_path / "out")
            (tmp_path / "out").mkdir()
            (tmp_path / "out" / "config.json").write_text("{}")
            result = runner.invoke(
                cli_app,
                ["models", "pull", "pi05-base", "--target-dir", str(tmp_path / "out")],
            )
        assert result.exit_code == 0, result.stdout
        assert "reflex export" in result.stdout
