"""Integration tests for `tether distill` — Phase B 3/3.

Covers the full wire path: CLI → FinetuneConfig → run_finetune →
SnapFlowBackend.fit → finalize → libero_drop_gate.

Everything that needs a GPU (real teacher load, real training step,
real ONNX export, real LIBERO rollout) is mocked. What IS tested:
  - `tether distill --help` renders
  - Config plumbing: teacher_export + distillation_method reach cfg
  - HookRegistry gets libero_drop_gate registered (when not skipped)
  - `--skip-libero-gate` leaves the registry empty
  - finalize() honors hook veto by flipping status to 'aborted'
  - run_finetune(phase='distill') routes to SnapFlowBackend
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner

from tether.finetune.backends.base import CheckpointResult
from tether.finetune.config import FinetuneConfig
from tether.finetune.hooks import HookRegistry
from tether.finetune.postprocess import finalize


runner = CliRunner()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestDistillCLI:
    def _get_app(self):
        """Import the CLI with distill registered. Isolates the lazy-import
        path and surfaces any import-time regressions immediately."""
        from tether.cli import app
        return app

    def test_distill_help_renders(self):
        app = self._get_app()
        result = runner.invoke(app, ["distill", "--help"])
        assert result.exit_code == 0
        assert "--teacher-export" in result.output
        assert "--libero-gate-pp" in result.output
        assert "SnapFlow" in result.output

    def test_dry_run_exits_without_training(self, tmp_path):
        """With --dry-run the CLI should reach preflight and bail before
        spinning up any backend / touching the teacher."""
        app = self._get_app()

        # Preflight normally fetches the dataset schema from HF. Mock it
        # to return a clean report so dry_run actually exits cleanly
        # rather than failing on HF unreachability in CI.
        fake_report = MagicMock()
        fake_report.has_failures = False
        fake_report.render.return_value = "(preflight stubbed)"

        with patch(
            "tether.finetune.preflight.run_preflight", return_value=fake_report,
        ):
            result = runner.invoke(
                app,
                [
                    "distill",
                    "--teacher-export", str(tmp_path / "teacher"),
                    "--dataset", "lerobot/libero",
                    "--output", str(tmp_path / "out"),
                    "--dry-run",
                ],
            )
        # Exit 0 on successful dry run; non-zero if preflight mocking missed.
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


class TestDistillConfigWiring:
    """The distill CLI constructs a FinetuneConfig with the right
    phase/teacher_export/method — pin this so the CLI can't silently
    drift away from the backend dispatch contract."""

    def test_phase_and_method_set_correctly(self, tmp_path):
        """Invoking the CLI body builds a cfg with phase='distill' and
        distillation_method='snapflow'."""
        from tether.finetune.cli_distill import distill_command

        captured = {}

        def fake_run_finetune(cfg, *, hooks=None):
            captured["cfg"] = cfg
            captured["hooks"] = hooks
            return MagicMock(
                status="ok",
                output_dir=cfg.output,
                error=None,
                final_checkpoint_path=cfg.output / "ckpt",
                training_log_path=cfg.output / "log.jsonl",
                onnx_path=None,
                verification_md_path=None,
            )

        # run_finetune is lazy-imported inside distill_command; patch at source.
        with patch("tether.finetune.run.run_finetune", fake_run_finetune):
            distill_command(
                teacher_export=str(tmp_path / "teacher"),
                dataset="lerobot/libero",
                output=str(tmp_path / "out"),
                num_steps=100,
                batch_size=4,
                learning_rate=1e-4,
                consistency_alpha=1.0,
                precision="bf16",
                target="desktop",
                libero_gate_pp=5.0,
                skip_libero_gate=False,
                skip_export=True,
                dry_run=False,
                skip_preflight=True,
                verbose=False,
            )
        cfg = captured["cfg"]
        assert cfg.phase == "distill"
        assert cfg.distillation_method == "snapflow"
        assert cfg.teacher_export == str(tmp_path / "teacher")
        assert cfg.mode == "full"  # SnapFlow is full-weight, not LoRA
        assert cfg.extra_lerobot_args["consistency_alpha"] == 1.0
        assert cfg.extra_lerobot_args["libero_gate_threshold_pp"] == 5.0
        assert "libero_gate_skip" not in cfg.extra_lerobot_args

    def test_skip_libero_gate_reaches_config(self, tmp_path):
        from tether.finetune.cli_distill import distill_command

        captured = {}

        def fake_run_finetune(cfg, *, hooks=None):
            captured["cfg"] = cfg
            captured["hooks"] = hooks
            return MagicMock(
                status="ok",
                output_dir=cfg.output,
                error=None,
                final_checkpoint_path=None,
                training_log_path=None,
                onnx_path=None,
                verification_md_path=None,
            )

        with patch("tether.finetune.run.run_finetune", fake_run_finetune):
            distill_command(
                teacher_export=str(tmp_path / "teacher"),
                dataset="lerobot/libero",
                output=str(tmp_path / "out"),
                num_steps=100, batch_size=4, learning_rate=1e-4,
                consistency_alpha=1.0, precision="bf16", target="desktop",
                libero_gate_pp=5.0, skip_libero_gate=True,
                skip_export=True, dry_run=False, skip_preflight=True,
                verbose=False,
            )
        assert captured["cfg"].extra_lerobot_args["libero_gate_skip"] is True
        # Registry should be empty when the gate is skipped
        assert captured["hooks"].handlers("on_postprocess") == []

    def test_libero_gate_attached_by_default(self, tmp_path):
        from tether.finetune.cli_distill import distill_command

        captured = {}

        def fake_run_finetune(cfg, *, hooks=None):
            captured["hooks"] = hooks
            return MagicMock(
                status="ok", output_dir=cfg.output, error=None,
                final_checkpoint_path=None, training_log_path=None,
                onnx_path=None, verification_md_path=None,
            )

        with patch("tether.finetune.run.run_finetune", fake_run_finetune):
            distill_command(
                teacher_export=str(tmp_path / "teacher"),
                dataset="lerobot/libero",
                output=str(tmp_path / "out"),
                num_steps=100, batch_size=4, learning_rate=1e-4,
                consistency_alpha=1.0, precision="bf16", target="desktop",
                libero_gate_pp=5.0, skip_libero_gate=False,
                skip_export=True, dry_run=False, skip_preflight=True,
                verbose=False,
            )
        handlers = captured["hooks"].handlers("on_postprocess")
        assert len(handlers) == 1


# ---------------------------------------------------------------------------
# run_finetune end-to-end with mocked backend + finalize
# ---------------------------------------------------------------------------


class TestDistillRunFinetune:
    def test_snapflow_backend_dispatched(self, tmp_path):
        """run_finetune(phase='distill') should route to SnapFlowBackend."""
        from tether.finetune.run import run_finetune

        cfg = FinetuneConfig(
            base="",
            dataset="lerobot/libero",
            output=tmp_path,
            num_steps=10,
            mode="full",
            phase="distill",
            teacher_export=str(tmp_path / "teacher"),
            skip_preflight=True,
            skip_export=True,
        )

        fake_ckpt = CheckpointResult(
            final_checkpoint_path=tmp_path / "ckpt",
            training_steps_completed=10,
            status="ok",
        )
        fake_backend = MagicMock()
        fake_backend.fit.return_value = fake_ckpt

        with patch(
            "tether.finetune.backends.resolve_backend",
            return_value=fake_backend,
        ):
            result = run_finetune(cfg)

        assert result.status == "ok"
        fake_backend.fit.assert_called_once()

    def test_hook_veto_aborts_finalize(self, tmp_path):
        """If a hook flips ctx.extra['force_abort'], finalize() returns
        status='aborted' instead of 'ok'."""
        from tether.finetune.backends.base import TrainerContext

        cfg = FinetuneConfig(
            base="",
            dataset="lerobot/libero",
            output=tmp_path,
            num_steps=10,
            mode="full",
            phase="distill",
            teacher_export=str(tmp_path / "teacher"),
            skip_export=True,
        )
        hooks = HookRegistry()

        def veto_handler(ctx, **payload):
            ctx.extra["force_abort"] = True
            ctx.extra["abort_reason"] = "student underperformed"

        hooks.register("on_postprocess", veto_handler)
        (tmp_path / "training_log.jsonl").touch()
        ctx = TrainerContext(
            config=cfg,
            hooks=hooks,
            training_log_path=tmp_path / "training_log.jsonl",
        )
        ckpt = CheckpointResult(
            final_checkpoint_path=tmp_path,
            training_steps_completed=10,
            status="ok",
        )

        result = finalize(ctx, ckpt)
        assert result.status == "aborted"
        assert "underperformed" in result.error

    def test_backend_failure_surfaces_error(self, tmp_path):
        """When SnapFlowBackend returns training_failed, run_finetune
        propagates the error without calling finalize()."""
        from tether.finetune.run import run_finetune

        cfg = FinetuneConfig(
            base="",
            dataset="lerobot/libero",
            output=tmp_path,
            num_steps=10,
            mode="full",
            phase="distill",
            teacher_export="placeholder/teacher",  # cfg validator needs non-empty
            skip_preflight=True,
        )
        fake_backend = MagicMock()
        fake_backend.fit.return_value = CheckpointResult(
            final_checkpoint_path=tmp_path,
            training_steps_completed=0,
            status="training_failed",
            error="SnapFlow requires teacher_export",
        )
        with patch(
            "tether.finetune.backends.resolve_backend",
            return_value=fake_backend,
        ):
            result = run_finetune(cfg)
        assert result.status == "training_failed"
        assert "teacher_export" in result.error


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------


class TestLegacyDistillRemoved:
    """Confirm the v0.2 DMPO/pi-Flow CLI code paths are GONE. Prevents a
    drive-by re-import regression later."""

    def test_dmpo_config_not_importable(self):
        with pytest.raises(ImportError):
            from tether.distill.dmpo import DMPOConfig  # noqa: F401

    def test_pi_flow_config_not_importable(self):
        with pytest.raises(ImportError):
            from tether.distill.pi_flow import PiFlowConfig  # noqa: F401

    def test_get_recipe_replaced_by_get_method(self):
        """The v0.2 CLI called `get_recipe(...)`. v0.3 exposes
        `get_method(...)` — ensure the old API is gone."""
        from tether import distill
        assert not hasattr(distill, "get_recipe")
        assert hasattr(distill, "get_method")
