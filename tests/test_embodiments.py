"""Tests for per-embodiment configs (B.1).

Three test classes:
- TestEmbodimentLoading — all 3 shipped presets load via load_preset()
- TestSchemaValidation — JSON-schema layer rejects malformed configs
- TestCrossValidation — Python-side cross-field rules catch mismatches

Style mirrors tests/test_config.py.
"""
from __future__ import annotations

import copy

import pytest

from reflex.embodiments import EmbodimentConfig, get_schema_path, list_presets
from reflex.embodiments.validate import (
    validate_against_schema,
    validate_cross_field,
    validate_embodiment_config,
)

ALL_PRESETS = ["franka", "quadcopter", "so100", "ur5"]


# ---------------------------------------------------------------------------
# TestEmbodimentLoading — the 3 shipped presets must load + validate
# ---------------------------------------------------------------------------


class TestEmbodimentLoading:
    def test_list_presets_finds_all_three(self):
        # Order is alphabetical from list_presets()
        assert list_presets() == ALL_PRESETS

    @pytest.mark.parametrize("name", ALL_PRESETS)
    def test_preset_loads(self, name):
        cfg = EmbodimentConfig.load_preset(name)
        assert cfg.embodiment == name
        assert cfg.schema_version == 1
        assert cfg._source_path.endswith(f"{name}.json")

    @pytest.mark.parametrize("name", ALL_PRESETS)
    def test_preset_validates_clean(self, name):
        cfg = EmbodimentConfig.load_preset(name)
        ok, errors = validate_embodiment_config(cfg)
        # Allow warnings (e.g. RTC horizon), reject errors
        blocking = [e for e in errors if e["severity"] == "error"]
        assert ok and not blocking, f"{name} validation failed: {blocking}"

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown embodiment preset"):
            EmbodimentConfig.load_preset("nonexistent")

    def test_load_custom_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            EmbodimentConfig.load_custom("/tmp/does_not_exist_42.json")

    def test_franka_action_dim_seven(self):
        cfg = EmbodimentConfig.load_preset("franka")
        assert cfg.action_dim == 7
        assert cfg.gripper_idx == 6

    def test_so100_lower_frequency(self):
        cfg = EmbodimentConfig.load_preset("so100")
        # SO-100 deliberately runs slower for compute-constrained Orin Nano
        assert cfg.control["frequency_hz"] < 20.0

    def test_ur5_action_dim_seven(self):
        cfg = EmbodimentConfig.load_preset("ur5")
        assert cfg.action_dim == 7

    def test_quadcopter_action_dim_five(self):
        cfg = EmbodimentConfig.load_preset("quadcopter")
        assert cfg.action_dim == 5
        assert cfg.control["frequency_hz"] == 50.0

    def test_to_dict_round_trip(self):
        original = EmbodimentConfig.load_preset("franka")
        d = original.to_dict()
        restored = EmbodimentConfig.from_dict(d)
        assert restored.embodiment == original.embodiment
        assert restored.action_space == original.action_space
        assert restored.normalization == original.normalization
        # _source_path is loader metadata; not part of to_dict
        assert restored._source_path == ""


# ---------------------------------------------------------------------------
# TestSchemaValidation — the JSON schema layer must reject malformed configs
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_schema_file_exists(self):
        path = get_schema_path()
        assert path.exists()

    def test_valid_franka_passes_schema(self):
        cfg = EmbodimentConfig.load_preset("franka")
        errors = validate_against_schema(cfg.to_dict())
        assert errors == []

    def test_schema_rejects_missing_required_field(self):
        cfg = EmbodimentConfig.load_preset("franka").to_dict()
        del cfg["action_space"]
        errors = validate_against_schema(cfg)
        assert any("action_space" in e["message"] or e["field"] == "action_space" for e in errors)

    def test_schema_rejects_unknown_embodiment_enum(self):
        cfg = EmbodimentConfig.load_preset("franka").to_dict()
        cfg["embodiment"] = "tesla-bot"  # not in enum
        errors = validate_against_schema(cfg)
        assert any(e["field"] == "embodiment" for e in errors)

    def test_schema_rejects_negative_action_dim(self):
        cfg = EmbodimentConfig.load_preset("franka").to_dict()
        cfg["action_space"]["dim"] = -3
        errors = validate_against_schema(cfg)
        assert any("dim" in e["field"] for e in errors)

    def test_schema_rejects_zero_std_action(self):
        cfg = EmbodimentConfig.load_preset("franka").to_dict()
        cfg["normalization"]["std_action"][0] = 0.0  # exclusiveMinimum=0
        errors = validate_against_schema(cfg)
        assert any("std_action" in e["field"] for e in errors)

    def test_schema_rejects_extra_top_level_field(self):
        # additionalProperties: false should reject unknowns
        cfg = EmbodimentConfig.load_preset("franka").to_dict()
        cfg["nonsense_key"] = "ignored"
        errors = validate_against_schema(cfg)
        assert any("nonsense_key" in e["message"] for e in errors)

    def test_schema_rejects_bad_color_space(self):
        cfg = EmbodimentConfig.load_preset("franka").to_dict()
        cfg["cameras"][0]["color_space"] = "yuv422"  # not in enum
        errors = validate_against_schema(cfg)
        assert any("color_space" in e["field"] for e in errors)


# ---------------------------------------------------------------------------
# TestCrossValidation — Python-side rules that JSON schema can't express
# ---------------------------------------------------------------------------


class TestCrossValidation:
    def test_action_ranges_length_must_match_dim(self):
        d = EmbodimentConfig.load_preset("franka").to_dict()
        d["action_space"]["ranges"] = d["action_space"]["ranges"][:-1]  # drop last
        cfg = EmbodimentConfig.from_dict(d)
        errors = validate_cross_field(cfg)
        slugs = {e["slug"] for e in errors}
        assert "action-ranges-length-mismatch" in slugs

    def test_inverted_range_caught(self):
        d = EmbodimentConfig.load_preset("franka").to_dict()
        d["action_space"]["ranges"][0] = [1.0, -1.0]  # lo > hi
        cfg = EmbodimentConfig.from_dict(d)
        errors = validate_cross_field(cfg)
        assert any(e["slug"] == "action-range-inverted" for e in errors)

    def test_norm_action_length_mismatch(self):
        d = EmbodimentConfig.load_preset("franka").to_dict()
        d["normalization"]["mean_action"] = [0.0, 0.0]  # length 2 vs dim 7
        cfg = EmbodimentConfig.from_dict(d)
        errors = validate_cross_field(cfg)
        assert any(e["slug"] == "norm-mean-action-length-mismatch" for e in errors)

    def test_norm_state_length_mismatch_caught(self):
        d = EmbodimentConfig.load_preset("franka").to_dict()
        d["normalization"]["mean_state"] = [0.0]  # length 1 vs std_state length 2
        cfg = EmbodimentConfig.from_dict(d)
        errors = validate_cross_field(cfg)
        assert any(e["slug"] == "norm-state-length-mismatch" for e in errors)

    def test_gripper_idx_out_of_range_caught(self):
        d = EmbodimentConfig.load_preset("franka").to_dict()
        d["gripper"]["component_idx"] = 99  # action_dim=7, so 99 is out
        cfg = EmbodimentConfig.from_dict(d)
        errors = validate_cross_field(cfg)
        assert any(e["slug"] == "gripper-idx-out-of-range" for e in errors)

    def test_rtc_horizon_too_short_warns(self):
        """Per ADR 2026-04-25 decision #8, rtc_execution_horizon is now an
        INTEGER COUNT of actions. Below 1 = degenerate RTC."""
        d = EmbodimentConfig.load_preset("franka").to_dict()
        d["control"]["rtc_execution_horizon"] = 0  # 0 actions = degenerate
        cfg = EmbodimentConfig.from_dict(d)
        errors = validate_cross_field(cfg)
        warnings = [e for e in errors if e["slug"] == "rtc-horizon-too-short"]
        assert len(warnings) == 1
        assert warnings[0]["severity"] == "warn"

    def test_rtc_horizon_exceeds_chunk_warns(self):
        """horizon > chunk_size makes no sense (can't lock more actions than
        the chunk holds)."""
        d = EmbodimentConfig.load_preset("franka").to_dict()
        d["control"]["chunk_size"] = 50
        d["control"]["rtc_execution_horizon"] = 100  # > chunk_size
        cfg = EmbodimentConfig.from_dict(d)
        errors = validate_cross_field(cfg)
        warnings = [e for e in errors if e["slug"] == "rtc-horizon-exceeds-chunk"]
        assert len(warnings) == 1

    def test_rtc_horizon_fractional_value_auto_migrates_to_integer(self):
        """Per ADR 2026-04-25 decision #8: legacy fractional values
        (0 < value < 1) auto-convert to int(value * chunk_size) at load
        time + emit a one-time deprecation warning. Integer-only after
        migration."""
        d = EmbodimentConfig.load_preset("franka").to_dict()
        d["control"]["chunk_size"] = 50
        d["control"]["rtc_execution_horizon"] = 0.5  # legacy fraction
        cfg = EmbodimentConfig.from_dict(d)
        # 0.5 * 50 = 25 (integer count after migration)
        assert cfg.control["rtc_execution_horizon"] == 25
        # And it should pass validation cleanly (no degenerate warning)
        errors = validate_cross_field(cfg)
        assert not any(
            e["slug"] in ("rtc-horizon-too-short", "rtc-horizon-exceeds-chunk")
            for e in errors
        )

    def test_rtc_horizon_integer_passthrough_no_migration(self):
        """Integer values pass through unchanged."""
        d = EmbodimentConfig.load_preset("franka").to_dict()
        d["control"]["chunk_size"] = 50
        d["control"]["rtc_execution_horizon"] = 25
        cfg = EmbodimentConfig.from_dict(d)
        assert cfg.control["rtc_execution_horizon"] == 25

    def test_rtc_horizon_migration_emits_deprecation_warning(self, caplog):
        """First fractional load logs a warning; second load of same config
        is silenced (one-time-per-(source, embodiment) dedup)."""
        import logging
        from reflex.embodiments import _RTC_HORIZON_MIGRATION_WARNED
        # Reset the dedup set for the test
        _RTC_HORIZON_MIGRATION_WARNED.clear()

        caplog.set_level(logging.WARNING)
        d = EmbodimentConfig.load_preset("franka").to_dict()
        d["control"]["chunk_size"] = 50
        d["control"]["rtc_execution_horizon"] = 0.5
        # First load emits warning
        EmbodimentConfig.from_dict(d, source_path="/test/franka.json")
        first_warnings = [r for r in caplog.records if "fraction" in r.message]
        assert len(first_warnings) >= 1
        # Second load with same source + embodiment: silenced
        caplog.clear()
        EmbodimentConfig.from_dict(d, source_path="/test/franka.json")
        second_warnings = [r for r in caplog.records if "fraction" in r.message]
        assert len(second_warnings) == 0

    def test_duplicate_camera_name_caught(self):
        d = EmbodimentConfig.load_preset("franka").to_dict()
        d["cameras"].append(copy.deepcopy(d["cameras"][0]))  # same name twice
        cfg = EmbodimentConfig.from_dict(d)
        errors = validate_cross_field(cfg)
        assert any(e["slug"] == "duplicate-camera-name" for e in errors)


# ---------------------------------------------------------------------------
# TestPresetSemantics — embodiment-specific invariants worth pinning
# ---------------------------------------------------------------------------


class TestPresetSemantics:
    @pytest.mark.parametrize("name", ALL_PRESETS)
    def test_collision_check_enabled(self, name):
        """Safety-first default: presets ship with collision_check on."""
        cfg = EmbodimentConfig.load_preset(name)
        assert cfg.constraints["collision_check"] is True

    @pytest.mark.parametrize("name", ALL_PRESETS)
    def test_max_ee_velocity_capped(self, name):
        """Sanity: presets must keep ee velocity under 2 m/s (or 6 m/s for flight)."""
        cfg = EmbodimentConfig.load_preset(name)
        cap = 6.0 if name == "quadcopter" else 2.0
        assert 0 < cfg.constraints["max_ee_velocity"] <= cap

    @pytest.mark.parametrize("name", ALL_PRESETS)
    def test_chunk_size_reasonable(self, name):
        """Sanity: chunk_size between 10 and 100 (matches model output range)."""
        cfg = EmbodimentConfig.load_preset(name)
        assert 10 <= cfg.control["chunk_size"] <= 100

    @pytest.mark.parametrize("name", ALL_PRESETS)
    def test_state_dim_matches_norm_arrays(self, name):
        """Cross-check: state_dim accessor uses normalization arrays."""
        cfg = EmbodimentConfig.load_preset(name)
        assert cfg.state_dim == len(cfg.normalization["std_state"])
