"""Unit tests for the LeRobot ↔ SO-ARM100 bridge.

Focused on the conversion math + JSON round-trip; the higher-level adapter
plumbing is covered in tests/test_so_arm100_adapter.py.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tether.embodiments.so_arm100 import (
    SO_ARM100_JOINT_NAMES,
    SO_ARM100_MOTOR_IDS,
    JointConfig,
    SOARM100Adapter,
    SOARM100Config,
)
from tether.embodiments.so_arm100.lerobot_bridge import (
    RADIANS_PER_TICK,
    TICKS_PER_REV,
    chunk_to_servo_command_stream,
    lerobot_action_to_servo_commands,
    normalized_gripper_to_servo_units,
    radians_to_servo_units,
    servo_state_to_lerobot_state,
    servo_units_to_normalized_gripper,
    servo_units_to_radians,
)


# ─── TestUnits ──────────────────────────────────────────────────────────────


class TestUnits:
    def test_ticks_per_revolution(self):
        assert TICKS_PER_REV == 4096
        # 4096 ticks per 2pi rad.
        assert RADIANS_PER_TICK == pytest.approx(2 * math.pi / 4096, rel=1e-12)

    def test_full_revolution_in_radians(self):
        cfg = SOARM100Config.default()
        joint = cfg.joint("shoulder_pan")
        # 2pi rad command should produce ~4096 ticks above mid (clamped at max).
        raw = radians_to_servo_units(2 * math.pi, joint)
        # With range 0..4095 + mid=2047, 2047 + 4096 wraps far beyond 4095, so
        # we clamp at range_max=4095. This confirms the clamp fires.
        assert raw == joint.range_max


# ─── TestRoundTrip ──────────────────────────────────────────────────────────


class TestRoundTrip:
    @pytest.mark.parametrize("rad_in", [-1.5, -0.7, -0.1, 0.0, 0.1, 0.7, 1.5])
    def test_radian_to_servo_to_radian(self, rad_in):
        cfg = SOARM100Config.default()
        joint = cfg.joint("shoulder_pan")
        raw = radians_to_servo_units(rad_in, joint)
        rad_out = servo_units_to_radians(raw, joint)
        # Within one tick.
        assert abs(rad_out - rad_in) < RADIANS_PER_TICK + 1e-9

    @pytest.mark.parametrize("gripper_in", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_gripper_round_trip(self, gripper_in):
        cfg = SOARM100Config.default()
        raw = normalized_gripper_to_servo_units(gripper_in, cfg)
        gripper_out = servo_units_to_normalized_gripper(raw, cfg)
        # Quantization within 1 / span.
        span = abs(
            cfg.gripper_closed_servo_units - cfg.gripper_open_servo_units
        )
        assert abs(gripper_out - gripper_in) <= 1 / span + 1e-9

    def test_drive_mode_inverts_round_trip(self):
        """drive_mode=1 inverts BOTH directions of the conversion; a value
        encoded with drive_mode=1 must decode back to the same value."""
        cfg = SOARM100Config.default()
        joint = replace(cfg.joint("wrist_flex"), drive_mode=1)
        for rad_in in (-1.0, -0.3, 0.0, 0.3, 1.0):
            raw = radians_to_servo_units(rad_in, joint)
            rad_out = servo_units_to_radians(raw, joint)
            assert abs(rad_out - rad_in) < RADIANS_PER_TICK + 1e-9


# ─── TestCalibrationJsonRoundTrip ───────────────────────────────────────────


class TestCalibrationJsonRoundTrip:
    """A LeRobot calibration JSON is the durable substrate. Reflex must
    preserve it lossless on the LeRobot fields so users with existing
    calibrations can import / re-export without losing data."""

    @pytest.fixture
    def cal_dict(self) -> dict[str, dict]:
        return {
            "shoulder_pan":  {"id": 1, "drive_mode": 0, "homing_offset":  42, "range_min": 1024, "range_max": 3072},
            "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset":   0, "range_min": 1200, "range_max": 2800},
            "elbow_flex":    {"id": 3, "drive_mode": 0, "homing_offset": -10, "range_min": 1100, "range_max": 3000},
            "wrist_flex":    {"id": 4, "drive_mode": 1, "homing_offset":   5, "range_min": 1024, "range_max": 3072},
            "wrist_roll":    {"id": 5, "drive_mode": 0, "homing_offset":   0, "range_min": 1024, "range_max": 3072},
            "gripper":       {"id": 6, "drive_mode": 0, "homing_offset":   0, "range_min": 1024, "range_max": 3000},
        }

    def test_load_emit_load_is_bit_identical_on_lerobot_fields(
        self,
        cal_dict: dict,
        tmp_path: Path,
    ):
        p1 = tmp_path / "in.json"
        p1.write_text(json.dumps(cal_dict))

        adapter = SOARM100Adapter.from_calibration(p1)
        re_emitted = adapter.config.to_lerobot_calibration()

        # All 5 LeRobot-native fields per joint should match exactly.
        for name in SO_ARM100_JOINT_NAMES:
            for field in ("id", "drive_mode", "homing_offset", "range_min", "range_max"):
                assert re_emitted[name][field] == cal_dict[name][field], (
                    f"mismatch on {name}.{field}: "
                    f"{re_emitted[name][field]} != {cal_dict[name][field]}"
                )

    def test_motor_ids_must_match_canonical(self, tmp_path: Path):
        cal = {
            "shoulder_pan":  {"id": 99, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
            "shoulder_lift": {"id":  2, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
            "elbow_flex":    {"id":  3, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
            "wrist_flex":    {"id":  4, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
            "wrist_roll":    {"id":  5, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
            "gripper":       {"id":  6, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
        }
        p = tmp_path / "bad_id.json"
        p.write_text(json.dumps(cal))
        with pytest.raises(ValueError, match=r"motor_id=99"):
            SOARM100Adapter.from_calibration(p)


# ─── TestActionFormatConversion ─────────────────────────────────────────────


class TestActionFormatConversion:
    def test_full_action_to_servo_commands_signature(self):
        cfg = SOARM100Config.default()
        cmds = lerobot_action_to_servo_commands([0.0] * 6, cfg)
        assert isinstance(cmds, list)
        assert len(cmds) == 6
        assert all(isinstance(c, tuple) and len(c) == 2 for c in cmds)
        # Motor IDs preserved in canonical order.
        assert [c[0] for c in cmds] == list(SO_ARM100_MOTOR_IDS)

    def test_servo_state_dict_input(self):
        cfg = SOARM100Config.default()
        pose = {name: 2047 for name in SO_ARM100_JOINT_NAMES}
        state = servo_state_to_lerobot_state(pose, cfg)
        assert state.shape == (6,)

    def test_servo_state_seq_input(self):
        cfg = SOARM100Config.default()
        state = servo_state_to_lerobot_state([2047] * 6, cfg)
        assert state.shape == (6,)

    def test_chunk_streams_t_steps(self):
        cfg = SOARM100Config.default()
        T = 12
        chunk = np.zeros((T, 6))
        stream = chunk_to_servo_command_stream(chunk, cfg)
        assert len(stream) == T

    def test_action_handles_numpy_input(self):
        cfg = SOARM100Config.default()
        action = np.zeros(6)
        cmds = lerobot_action_to_servo_commands(action, cfg)
        assert len(cmds) == 6

    def test_action_handles_2d_input_one_row(self):
        """Some upstream loops produce a (1, 6) batched action. Our impl
        reshapes to flat 6 so we don't crash on this common shape."""
        cfg = SOARM100Config.default()
        action = np.zeros((1, 6))
        cmds = lerobot_action_to_servo_commands(action, cfg)
        assert len(cmds) == 6


# ─── TestLeRobotSchemaCompat ────────────────────────────────────────────────


class TestLeRobotSchemaCompat:
    """If LeRobot is installed locally, verify our calibration JSON loads
    cleanly with LeRobot's own draccus-backed loader. Skipped otherwise."""

    def test_lerobot_can_load_our_calibration(self, tmp_path: Path):
        pytest.importorskip("lerobot")
        try:
            from lerobot.motors import MotorCalibration
        except ImportError:
            pytest.skip("lerobot.motors.MotorCalibration unavailable")

        cfg = SOARM100Config.default()
        out = tmp_path / "cal.json"
        cfg.write_lerobot_calibration(out)

        # Hand-load + construct MotorCalibration; this is what
        # `LeRobot.Robot._load_calibration` does internally.
        with out.open() as f:
            data = json.load(f)
        for name in SO_ARM100_JOINT_NAMES:
            # Should not raise — all required fields present.
            MotorCalibration(**data[name])
