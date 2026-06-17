"""Tests for hardware profiles and configuration."""

import pytest

from tether.config import get_hardware_profile, HARDWARE_PROFILES, ExportConfig


class TestHardwareProfiles:
    def test_all_profiles_exist(self):
        for target in ["orin-nano", "orin", "orin-64", "thor", "desktop"]:
            profile = get_hardware_profile(target)
            assert profile.name
            assert profile.memory_gb > 0

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown target"):
            get_hardware_profile("nonexistent")

    def test_orin_no_fp8(self):
        profile = get_hardware_profile("orin-nano")
        assert not profile.supports_fp8

    def test_thor_has_fp8(self):
        profile = get_hardware_profile("thor")
        assert profile.supports_fp8
        assert profile.supports_fp4


class TestExportConfig:
    def test_defaults(self):
        config = ExportConfig(model_id="test", target="desktop", output_dir="./out")
        assert config.precision == "fp16"
        assert config.opset == 19
        assert config.num_denoising_steps == 10
