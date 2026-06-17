"""Verify the verb-noun CLI structure: top-level surface + new subgroup paths.

The 2026-04-24 verb-noun refactor consolidated 12+ flat commands into 6 visible
top-level commands (`serve`, `doctor`, `models`, `train`, `validate`, `inspect`)
plus subcommands. Old top-level names stay registered as `hidden=True` aliases
so existing scripts keep working through one release; they're removed in v0.2.

These tests lock in:
- The 6 visible top-level commands
- The 4 new subgroups + their commands
- Every old top-level command still exists (hidden) for back-compat
"""
from __future__ import annotations

import pytest


@pytest.fixture
def runner():
    typer_testing = pytest.importorskip("typer.testing")
    return typer_testing.CliRunner()


@pytest.fixture
def cli_app():
    from tether.cli import app
    return app


class TestVisibleTopLevel:
    """The 6 canonical top-level commands."""

    @pytest.mark.parametrize("cmd", ["serve", "doctor", "models", "train", "validate", "inspect"])
    def test_visible_top_level_command_exists(self, runner, cli_app, cmd):
        result = runner.invoke(cli_app, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"

    def test_help_lists_six_visible_commands(self, runner, cli_app):
        result = runner.invoke(cli_app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("serve", "doctor", "models", "train", "validate", "inspect"):
            assert cmd in result.output

    @pytest.mark.parametrize("hidden_cmd", [
        "export", "validate-legacy", "bench", "guard", "ros2-serve",
        "replay", "targets", "validate-dataset", "finetune", "distill",
    ])
    def test_hidden_command_not_in_help_listing(self, runner, cli_app, hidden_cmd):
        # Match the typer command-row pattern: "│ <cmd>" followed by whitespace.
        # This avoids false positives where "export" / "bench" / "guard" etc.
        # appear as natural-language words inside subgroup descriptions.
        import re
        result = runner.invoke(cli_app, ["--help"])
        if "Commands" in result.output:
            commands_section = result.output.split("Commands")[1].split("╰")[0]
            row_pattern = re.compile(rf"│\s+{re.escape(hidden_cmd)}\s")
            assert not row_pattern.search(commands_section), (
                f"{hidden_cmd!r} should be hidden but appears as a command row in --help"
            )


class TestModelsSubgroup:
    @pytest.mark.parametrize("sub", ["list", "pull", "info", "export"])
    def test_models_subcommand_exists(self, runner, cli_app, sub):
        result = runner.invoke(cli_app, ["models", sub, "--help"])
        assert result.exit_code == 0, result.output


class TestTrainSubgroup:
    @pytest.mark.parametrize("sub", ["finetune", "distill"])
    def test_train_subcommand_exists(self, runner, cli_app, sub):
        result = runner.invoke(cli_app, ["train", sub, "--help"])
        assert result.exit_code == 0, result.output


class TestValidateSubgroup:
    @pytest.mark.parametrize("sub", ["dataset", "export"])
    def test_validate_subcommand_exists(self, runner, cli_app, sub):
        result = runner.invoke(cli_app, ["validate", sub, "--help"])
        assert result.exit_code == 0, result.output


class TestInspectSubgroup:
    @pytest.mark.parametrize("sub", ["bench", "replay", "targets", "guard", "doctor"])
    def test_inspect_subcommand_exists(self, runner, cli_app, sub):
        result = runner.invoke(cli_app, ["inspect", sub, "--help"])
        assert result.exit_code == 0, result.output


class TestBackwardCompatAliases:
    """Old top-level commands stay hidden but still callable through one release."""

    def test_old_validate_dataset_still_callable(self, runner, cli_app):
        # Just `--help` to confirm the alias still routes correctly
        result = runner.invoke(cli_app, ["validate-dataset", "--help"])
        assert result.exit_code == 0, result.output

    def test_old_bench_still_callable(self, runner, cli_app):
        result = runner.invoke(cli_app, ["bench", "--help"])
        assert result.exit_code == 0

    def test_old_replay_still_callable(self, runner, cli_app):
        result = runner.invoke(cli_app, ["replay", "--help"])
        assert result.exit_code == 0

    def test_old_targets_still_callable(self, runner, cli_app):
        result = runner.invoke(cli_app, ["targets"])
        # `targets` takes no args — direct invocation should work
        assert result.exit_code == 0

    def test_old_export_still_callable(self, runner, cli_app):
        result = runner.invoke(cli_app, ["export", "--help"])
        assert result.exit_code == 0

    def test_old_validate_legacy_still_callable(self, runner, cli_app):
        # Renamed from `validate` to avoid collision with the new validate subgroup
        result = runner.invoke(cli_app, ["validate-legacy", "--help"])
        assert result.exit_code == 0
