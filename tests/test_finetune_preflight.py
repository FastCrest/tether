"""Tests for tether.finetune.preflight — the v0.5 validator."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from tether.finetune.config import FinetuneConfig
from tether.finetune.preflight import PreflightCheck, PreflightReport, run_preflight
from tether.finetune.preflight.dataset_size import (
    EPISODE_FLOORS,
    _infer_policy_type,
    check_dataset_size,
)
from tether.finetune.preflight.schema import (
    _extract_base_action_dim,
    _extract_dataset_action_dim,
    check_schema,
)


class TestReport:
    def test_empty_report_is_clean(self):
        r = PreflightReport()
        assert not r.has_failures
        assert not r.has_warnings

    def test_warn_doesnt_block(self):
        r = PreflightReport()
        r.add(PreflightCheck("x", "warn", "small dataset"))
        assert r.has_warnings
        assert not r.has_failures

    def test_fail_blocks(self):
        r = PreflightReport()
        r.add(PreflightCheck("x", "fail", "dim mismatch"))
        assert r.has_failures

    def test_render_has_badge(self):
        r = PreflightReport()
        r.add(PreflightCheck("schema", "ok", "all good"))
        r.add(PreflightCheck("dataset_size", "warn", "small"))
        r.add(PreflightCheck("mem", "fail", "OOM"))
        out = r.render()
        assert "✓ [schema]" in out
        assert "⚠ [dataset_size]" in out
        assert "✗ [mem]" in out

    def test_error_message_lists_failures_only(self):
        r = PreflightReport()
        r.add(PreflightCheck("x", "ok", "fine"))
        r.add(PreflightCheck("y", "warn", "eh"))
        r.add(PreflightCheck("z", "fail", "broken"))
        msg = r.error_message()
        assert "[z] broken" in msg
        assert "[x]" not in msg
        assert "[y]" not in msg


class TestSchemaExtraction:
    def test_extract_dataset_action_dim(self):
        assert _extract_dataset_action_dim(
            {"action": {"shape": [7]}}
        ) == 7
        assert _extract_dataset_action_dim(
            {"action": {"shape": [1, 8]}}
        ) == 8  # last dim
        assert _extract_dataset_action_dim({"action": {}}) is None
        assert _extract_dataset_action_dim({}) is None

    def test_extract_base_action_dim_prefers_output_features(self):
        cfg = {
            "output_features": {"action": {"shape": [7]}},
            "max_action_dim": 32,
        }
        # Should prefer the real action shape, not the padded max
        assert _extract_base_action_dim(cfg) == 7

    def test_extract_base_action_dim_falls_back_to_max(self):
        cfg = {"max_action_dim": 32}
        assert _extract_base_action_dim(cfg) == 32

    def test_extract_base_action_dim_returns_none_if_missing(self):
        assert _extract_base_action_dim({}) is None


class TestSchemaCheck:
    def _run_with_mocks(self, cfg_dataset_features, cfg_base_config):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output="/tmp/x",
        )
        with patch(
            "tether.finetune.preflight.schema._fetch_dataset_features",
            return_value=cfg_dataset_features,
        ), patch(
            "tether.finetune.preflight.schema._fetch_base_config",
            return_value=cfg_base_config,
        ):
            return check_schema(cfg)

    def test_matching_dims_pass(self):
        result = self._run_with_mocks(
            {"action": {"shape": [7]}},
            {"output_features": {"action": {"shape": [7]}}},
        )
        assert result.severity == "ok"
        assert "7-D" in result.summary

    def test_mismatched_dims_fail(self):
        result = self._run_with_mocks(
            {"action": {"shape": [6]}},
            {"output_features": {"action": {"shape": [7]}}},
        )
        assert result.severity == "fail"
        assert "mismatch" in result.summary.lower()
        assert result.detail["dataset_action_dim"] == 6
        assert result.detail["base_action_dim"] == 7

    def test_unresolvable_dataset_warns(self):
        """Network issue or local dataset → warn, don't fail."""
        result = self._run_with_mocks(
            None,  # couldn't fetch
            {"output_features": {"action": {"shape": [7]}}},
        )
        assert result.severity == "warn"

    def test_unresolvable_base_warns(self):
        result = self._run_with_mocks(
            {"action": {"shape": [7]}},
            None,
        )
        assert result.severity == "warn"


class TestDatasetSizeCheck:
    def test_policy_type_inference(self):
        assert _infer_policy_type("lerobot/smolvla_base") == "smolvla"
        assert _infer_policy_type("lerobot/pi0_base") == "pi0"
        assert _infer_policy_type("lerobot/pi05_base") == "pi05"
        assert _infer_policy_type("nvidia/GR00T-N1.6-3B") == "gr00t_n1_5"
        assert _infer_policy_type("unknown/model") is None

    def test_floors_are_reasonable(self):
        # Sanity: pi0.5 floor > pi0 floor (quantile norm needs more data)
        assert EPISODE_FLOORS["pi05"] > EPISODE_FLOORS["pi0"]
        # SmolVLA is smallest, tolerates least data
        assert EPISODE_FLOORS["smolvla"] < EPISODE_FLOORS["pi0"]

    def test_below_floor_warns(self):
        cfg = FinetuneConfig(
            base="lerobot/pi05_base",
            dataset="lerobot/tiny",
            output="/tmp/x",
        )
        with patch(
            "tether.finetune.preflight.dataset_size._fetch_dataset_info",
            return_value={"total_episodes": 100},
        ):
            result = check_dataset_size(cfg)
        assert result.severity == "warn"
        assert "100 episodes" in result.summary
        assert "≥1000" in result.summary or "1000" in result.summary

    def test_at_floor_passes(self):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/big",
            output="/tmp/x",
        )
        with patch(
            "tether.finetune.preflight.dataset_size._fetch_dataset_info",
            return_value={"total_episodes": 500},
        ):
            result = check_dataset_size(cfg)
        assert result.severity == "ok"

    def test_unknown_base_warns_not_fails(self):
        cfg = FinetuneConfig(
            base="random/model",
            dataset="lerobot/x",
            output="/tmp/x",
        )
        result = check_dataset_size(cfg)
        assert result.severity == "warn"


class TestRunPreflight:
    def test_all_ok_passes(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
        )
        with patch(
            "tether.finetune.preflight.runner.check_schema",
            return_value=PreflightCheck("schema", "ok", "fine"),
        ), patch(
            "tether.finetune.preflight.runner.check_dataset_size",
            return_value=PreflightCheck("dataset_size", "ok", "fine"),
        ):
            report = run_preflight(cfg)
        assert not report.has_failures
        assert not report.has_warnings

    def test_one_fail_surfaces(self, tmp_path):
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
        )
        with patch(
            "tether.finetune.preflight.runner.check_schema",
            return_value=PreflightCheck("schema", "fail", "dim mismatch"),
        ), patch(
            "tether.finetune.preflight.runner.check_dataset_size",
            return_value=PreflightCheck("dataset_size", "ok", "fine"),
        ):
            report = run_preflight(cfg)
        assert report.has_failures

    def test_crashing_check_becomes_warn(self, tmp_path):
        """A check that raises shouldn't take down the whole preflight."""
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
        )
        def _boom(_cfg):
            raise RuntimeError("unexpected")
        # Wrap _boom so it has __name__ (mocks lacking __name__ crash
        # the runner's logger format string).
        _boom.__name__ = "check_schema"
        with patch(
            "tether.finetune.preflight.runner.check_schema",
            new=_boom,
        ), patch(
            "tether.finetune.preflight.runner.check_dataset_size",
            return_value=PreflightCheck("dataset_size", "ok", "fine"),
        ):
            report = run_preflight(cfg)
        # Crash → warn, not fail; run continues
        assert not report.has_failures
        assert report.has_warnings


class TestIntegrationWithRunFinetune:
    def test_preflight_failure_aborts_run(self, tmp_path):
        """A blocking preflight failure should abort run_finetune before
        the subprocess launches."""
        from tether.finetune.run import run_finetune
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
        )
        # run_preflight is imported at call time inside run_finetune, so
        # patch at its source module, not at tether.finetune.run.
        with patch(
            "tether.finetune.preflight.run_preflight",
            return_value=PreflightReport(
                checks=[PreflightCheck("schema", "fail", "dim mismatch")],
            ),
        ), patch("tether.finetune.run._run_lerobot_training") as mock_train:
            result = run_finetune(cfg)
        assert result.status == "aborted"
        mock_train.assert_not_called()
        assert "[schema] dim mismatch" in (result.error or "")

    def test_skip_preflight_bypasses(self, tmp_path):
        """--skip-preflight should route around the validator entirely."""
        from tether.finetune.run import run_finetune
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
            skip_preflight=True,
        )
        with patch("tether.finetune.preflight.run_preflight") as mock_pf, \
             patch("tether.finetune.run._run_lerobot_training", return_value=42):
            result = run_finetune(cfg)
        mock_pf.assert_not_called()
        # Training failed (code 42) because we mocked it — but preflight
        # was skipped as requested.
        assert result.status == "training_failed"

    def test_dry_run_skips_training(self, tmp_path):
        from tether.finetune.run import run_finetune
        cfg = FinetuneConfig(
            base="lerobot/smolvla_base",
            dataset="lerobot/libero",
            output=tmp_path,
            dry_run=True,
        )
        with patch(
            "tether.finetune.preflight.run_preflight",
            return_value=PreflightReport(
                checks=[PreflightCheck("schema", "ok", "fine")],
            ),
        ), patch("tether.finetune.run._run_lerobot_training") as mock_train:
            result = run_finetune(cfg)
        mock_train.assert_not_called()
        assert result.status == "ok"
        assert result.error is None
