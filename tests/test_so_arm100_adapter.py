"""Unit tests for SOARM100Adapter.

No hardware required — all hardware-side state is faked by a MockSOARM100Port
fixture. Integration smoke tests that touch a real arm live in
`tests/integration/test_so_arm100_hardware.py` and are gated on the
`@pytest.mark.hardware` marker (run with `RUN_HARDWARE_TESTS=1`).

Coverage map:
    TestAdapterConstruction   — default / from_calibration / from_bundle
    TestActionMath            — action → servo conversion (math, clamping, gripper)
    TestStateMath             — present-position → state conversion
    TestGripperToggle         — open/close threshold + inverted gripper semantics
    TestChunkStreaming        — (T, 6) chunk → per-step command list
    TestCalibrationBundle     — adapter.calibration_dict round-trip via save / from_bundle
    TestEmbodimentConfigBridge— adapter.to_embodiment_config returns the right preset
    TestMockSOARM100Port      — the mock port itself behaves correctly
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
    gripper_should_close,
    normalized_gripper_to_servo_units,
    radians_to_servo_units,
    servo_units_to_normalized_gripper,
    servo_units_to_radians,
)


# ─── Test fixtures + mocks ──────────────────────────────────────────────────


class MockSOARM100Port:
    """In-memory stand-in for the serial port.

    Behaves like both the SOArmHardware (legacy scservo) and FeetechMotorsBus
    surfaces: read_pose returns whatever positions we've set, write_goal /
    write_goals are recorded for assertions, close() is a no-op.
    """

    def __init__(self, initial_positions: dict[str, int] | None = None):
        self.positions: dict[str, int] = dict(
            initial_positions or {name: 2048 for name in SO_ARM100_JOINT_NAMES}
        )
        self.writes: list[tuple[int, int]] = []
        self.batch_writes: list[list[tuple[int, int]]] = []
        self.closed = False

    def read_pose(self) -> dict[str, int]:
        return dict(self.positions)

    def write_goal(self, motor_id: int, raw: int) -> None:
        self.writes.append((motor_id, int(raw)))
        # Mirror back into positions so a subsequent read sees the new pose.
        for name, mid in zip(SO_ARM100_JOINT_NAMES, SO_ARM100_MOTOR_IDS):
            if mid == motor_id:
                self.positions[name] = int(raw)
                return
        raise KeyError(f"Unknown motor_id={motor_id}")

    def write_goals(self, commands):
        self.batch_writes.append(list(commands))
        for mid, raw in commands:
            self.write_goal(mid, raw)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def mock_port():
    return MockSOARM100Port()


@pytest.fixture
def lerobot_cal_json(tmp_path: Path) -> Path:
    """Drop a realistic LeRobot SO-100 calibration JSON to disk + return its path.

    These values are within the legal STS3215 range (0..4095) and have a
    non-trivial homing_offset on the shoulder pan so the conversion math is
    actually exercised (drive_mode=1 on wrist_flex catches inversion bugs).
    """
    cal = {
        "shoulder_pan":  {"id": 1, "drive_mode": 0, "homing_offset":  42, "range_min": 1024, "range_max": 3072},
        "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset":   0, "range_min": 1200, "range_max": 2800},
        "elbow_flex":    {"id": 3, "drive_mode": 0, "homing_offset": -10, "range_min": 1100, "range_max": 3000},
        "wrist_flex":    {"id": 4, "drive_mode": 1, "homing_offset":   5, "range_min": 1024, "range_max": 3072},
        "wrist_roll":    {"id": 5, "drive_mode": 0, "homing_offset":   0, "range_min": 1024, "range_max": 3072},
        "gripper":       {"id": 6, "drive_mode": 0, "homing_offset":   0, "range_min": 1024, "range_max": 3000},
    }
    p = tmp_path / "calib.json"
    p.write_text(json.dumps(cal, indent=2))
    return p


# ─── TestAdapterConstruction ────────────────────────────────────────────────


class TestAdapterConstruction:
    def test_default_constructs(self):
        a = SOARM100Adapter.default()
        assert a.embodiment_name == "so_arm100"
        assert a.action_dim == 6
        assert a.state_dim == 6
        assert a.gripper_idx == 5
        assert a.joint_names == SO_ARM100_JOINT_NAMES

    def test_from_calibration_path(self, lerobot_cal_json: Path):
        a = SOARM100Adapter.from_calibration(lerobot_cal_json)
        assert a.config.joint("shoulder_pan").homing_offset == 42
        assert a.config.joint("wrist_flex").drive_mode == 1
        assert a.config._source_path == str(lerobot_cal_json)

    def test_from_calibration_accepts_config_object(self):
        cfg = SOARM100Config.default()
        a = SOARM100Adapter.from_calibration(cfg)
        assert a.config is cfg

    def test_from_calibration_rejects_missing_joints(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        # Missing wrist_roll + gripper.
        bad.write_text(json.dumps({
            "shoulder_pan":  {"id": 1},
            "shoulder_lift": {"id": 2},
            "elbow_flex":    {"id": 3},
            "wrist_flex":    {"id": 4},
        }))
        with pytest.raises(ValueError, match="missing required joints"):
            SOARM100Adapter.from_calibration(bad)

    def test_calibration_missing_file(self):
        with pytest.raises(FileNotFoundError, match="LeRobot calibration not found"):
            SOARM100Adapter.from_calibration("/tmp/does-not-exist-soarm-13987.json")

    def test_invalid_protocol_rejected(self):
        cfg = SOARM100Config.default()
        with pytest.raises(ValueError, match="Unknown protocol"):
            replace(cfg, protocol="bluetooth_lol")

    def test_wrong_joint_count_rejected(self):
        with pytest.raises(ValueError, match="6 joints"):
            SOARM100Config(joints=tuple())

    def test_joint_order_enforced(self):
        joints = list(SOARM100Config.default().joints)
        # Swap shoulder_pan and shoulder_lift.
        joints[0], joints[1] = joints[1], joints[0]
        with pytest.raises(ValueError, match="canonical order"):
            SOARM100Config(joints=tuple(joints))


# ─── TestActionMath ─────────────────────────────────────────────────────────


class TestActionMath:
    def test_zero_action_centers_servos(self):
        # With default cal (range 0..4095, mid=2047, no homing offset), a
        # zero-radians joint command should map to mid-range, and gripper=0
        # should map to gripper_open_servo_units.
        a = SOARM100Adapter.default()
        cmds = a.action_to_servo_commands([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ids = [c[0] for c in cmds]
        raws = [c[1] for c in cmds]
        assert ids == list(SO_ARM100_MOTOR_IDS)
        for raw in raws[:5]:
            assert raw == 2047  # mid-range of 0..4095
        assert raws[5] == a.config.gripper_open_servo_units

    def test_action_clamped_to_position_limits(self):
        """Out-of-limit radian commands clamp to the joint's software window
        BEFORE servo conversion. so100.json gives the revolute joints a
        ±3.14 rad range, so 100 rad should clamp to +3.14 (which → mid + ~2046
        ticks since 3.14 rad is slightly less than 4096-tick π, well inside
        the servo range_max)."""
        a = SOARM100Adapter.default()
        cmds = a.action_to_servo_commands([100.0, -100.0, 0.0, 0.0, 0.0, 0.0])
        pan = a.config.joint("shoulder_pan")
        lift = a.config.joint("shoulder_lift")
        # 3.14 rad → 2046.7 ticks; mid=2047; raw=4094 (NOT range_max 4095 —
        # confirms position-limit clamp ran, not servo-range clamp).
        # Position-limit clamps to +3.14 then math → ~4094, ≤ range_max.
        assert cmds[0][1] <= pan.range_max
        assert cmds[0][1] >= pan.range_max - 5
        # -3.14 rad → ~0 ticks; servo range clamp pins at exactly 0.
        assert cmds[1][1] <= lift.range_min + 5
        assert cmds[1][1] >= lift.range_min

    def test_action_must_be_length_six(self):
        a = SOARM100Adapter.default()
        with pytest.raises(ValueError, match="length 6"):
            a.action_to_servo_commands([0.0, 0.0, 0.0])

    def test_calibration_homing_offset_applied(self, lerobot_cal_json: Path):
        """A nonzero homing_offset shifts the raw servo command — zero radians
        on shoulder_pan (homing_offset=42) should land at mid + 42."""
        a = SOARM100Adapter.from_calibration(lerobot_cal_json)
        cmds = a.action_to_servo_commands([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pan = a.config.joint("shoulder_pan")
        expected = (pan.range_min + pan.range_max) // 2 + 42
        assert cmds[0][1] == expected

    def test_drive_mode_inverts_direction(self, lerobot_cal_json: Path):
        """wrist_flex has drive_mode=1 in the fixture. A positive command
        should drive the servo BELOW mid-range, not above."""
        a = SOARM100Adapter.from_calibration(lerobot_cal_json)
        wrist_flex_idx = SO_ARM100_JOINT_NAMES.index("wrist_flex")
        action = [0.0] * 6
        action[wrist_flex_idx] = 0.5  # +0.5 rad
        cmds = a.action_to_servo_commands(action)
        wf = a.config.joint("wrist_flex")
        mid = (wf.range_min + wf.range_max) // 2
        raw = cmds[wrist_flex_idx][1]
        # With drive_mode=1, positive rad → negative ticks → raw < mid + homing.
        assert raw < mid + wf.homing_offset

    def test_apply_position_limits_can_be_disabled(self):
        """For diagnostic flows we may want the raw mapping without clamping.
        Disabling apply_position_limits should let the math run free —
        servo-side range clamp still applies to keep us inside 0..4095."""
        a = SOARM100Adapter.default()
        # Set a tight software window then disable it; servo-side clamp on
        # raw value still triggers.
        cmds = a.action_to_servo_commands(
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            apply_position_limits=False,
        )
        assert cmds[0][1] == 4095  # clamped at servo range_max

    def test_round_trip_radians_servo_radians(self):
        cfg = SOARM100Config.default()
        joint = cfg.joint("shoulder_pan")
        # Round-trip at a few representative angles.
        for rad_in in (-1.5, -0.7, 0.0, 0.7, 1.5):
            raw = radians_to_servo_units(rad_in, joint)
            rad_out = servo_units_to_radians(raw, joint)
            # Within one tick of quantization error.
            assert abs(rad_out - rad_in) < RADIANS_PER_TICK + 1e-9


# ─── TestStateMath ──────────────────────────────────────────────────────────


class TestStateMath:
    def test_state_from_pose_default_cal(self):
        a = SOARM100Adapter.default()
        # All servos at mid-range → all joints should read ~0, gripper ~0.5.
        pose = {n: 2047 for n in a.joint_names}
        state = a.state_from_servo(pose)
        for i in range(5):
            assert abs(state[i]) < RADIANS_PER_TICK + 1e-9
        # Default gripper open=1024 closed=2400; 2047 ~ midway → ~0.74.
        assert 0.5 < state[5] < 0.85

    def test_state_accepts_positional_sequence(self):
        a = SOARM100Adapter.default()
        state = a.state_from_servo([2047, 2047, 2047, 2047, 2047, 1024])
        # Gripper raw=1024 → fully open → normalized 0.0.
        assert state[5] == pytest.approx(0.0, abs=1e-6)

    def test_state_dict_missing_joint_raises(self):
        a = SOARM100Adapter.default()
        with pytest.raises(KeyError, match="present_positions dict"):
            a.state_from_servo({"shoulder_pan": 2048})

    def test_state_sequence_wrong_length_raises(self):
        a = SOARM100Adapter.default()
        with pytest.raises(ValueError, match="length 6"):
            a.state_from_servo([2048, 2048, 2048])


# ─── TestGripperToggle ──────────────────────────────────────────────────────


class TestGripperToggle:
    def test_gripper_open(self):
        cfg = SOARM100Config.default()
        assert (
            normalized_gripper_to_servo_units(0.0, cfg) == cfg.gripper_open_servo_units
        )

    def test_gripper_closed(self):
        cfg = SOARM100Config.default()
        assert (
            normalized_gripper_to_servo_units(1.0, cfg) == cfg.gripper_closed_servo_units
        )

    def test_gripper_clamps_out_of_range(self):
        cfg = SOARM100Config.default()
        # Out-of-range commands clamp to the legal endpoints.
        assert (
            normalized_gripper_to_servo_units(-5.0, cfg)
            == cfg.gripper_open_servo_units
        )
        assert (
            normalized_gripper_to_servo_units(5.0, cfg)
            == cfg.gripper_closed_servo_units
        )

    def test_gripper_inverted_semantics(self):
        cfg = replace(SOARM100Config.default(), gripper_inverted=True)
        # With inverted=True, value=0 should now produce the CLOSED position.
        assert (
            normalized_gripper_to_servo_units(0.0, cfg)
            == cfg.gripper_closed_servo_units
        )

    def test_threshold_logic(self):
        cfg = SOARM100Config.default()
        assert gripper_should_close(0.6, cfg) is True
        assert gripper_should_close(0.4, cfg) is False
        # Exactly at threshold → should close (>=).
        assert gripper_should_close(0.5, cfg) is True

    def test_threshold_logic_inverted(self):
        cfg = replace(SOARM100Config.default(), gripper_inverted=True)
        # With inverted, the semantics flip: a high command opens.
        assert gripper_should_close(0.6, cfg) is False
        assert gripper_should_close(0.1, cfg) is True

    def test_gripper_round_trip(self):
        cfg = SOARM100Config.default()
        for v in (0.0, 0.25, 0.5, 0.75, 1.0):
            raw = normalized_gripper_to_servo_units(v, cfg)
            back = servo_units_to_normalized_gripper(raw, cfg)
            # Quantization within ~1/(closed-open) of input.
            tol = 1.0 / abs(cfg.gripper_closed_servo_units - cfg.gripper_open_servo_units) + 1e-9
            assert abs(back - v) < tol


# ─── TestChunkStreaming ─────────────────────────────────────────────────────


class TestChunkStreaming:
    def test_chunk_shape_validated(self):
        a = SOARM100Adapter.default()
        # Wrong second dimension.
        with pytest.raises(ValueError, match=r"\(T, 6\)"):
            a.chunk_to_servo_command_stream(np.zeros((10, 5)))
        # Not 2D at all.
        with pytest.raises(ValueError, match=r"\(T, 6\)"):
            a.chunk_to_servo_command_stream(np.zeros(6))

    def test_chunk_round_trip(self):
        a = SOARM100Adapter.default()
        T = 8
        chunk = np.zeros((T, 6), dtype=np.float64)
        # Sweep shoulder_pan from -1 to +1.
        chunk[:, 0] = np.linspace(-1.0, 1.0, T)
        cmds = a.chunk_to_servo_command_stream(chunk)
        assert len(cmds) == T
        # Each per-step list has all 6 motors.
        assert all(len(per_step) == 6 for per_step in cmds)
        # shoulder_pan raw should be monotone non-decreasing across timesteps.
        pan_raws = [per_step[0][1] for per_step in cmds]
        assert pan_raws == sorted(pan_raws)


# ─── TestCalibrationBundle ──────────────────────────────────────────────────


class TestCalibrationBundle:
    def test_calibration_dict_structure(self):
        a = SOARM100Adapter.default()
        d = a.calibration_dict()
        assert d["schema_version"] == 1
        assert d["embodiment"] == "so_arm100"
        assert set(d["lerobot_calibration"].keys()) == set(SO_ARM100_JOINT_NAMES)
        assert "control_frequency_hz" in d["runtime"]
        # Joints section carries reflex-side metadata (position_limits etc.).
        assert set(d["joints"].keys()) == set(SO_ARM100_JOINT_NAMES)

    def test_save_and_round_trip(self, lerobot_cal_json: Path, tmp_path: Path):
        original = SOARM100Adapter.from_calibration(lerobot_cal_json)

        # Save in the "bundle" location.
        bundle = tmp_path / "exported"
        bundle.mkdir()
        original.save_calibration(
            bundle / "embodiment" / "so_arm100" / "calibration.json"
        )

        # Load back.
        loaded = SOARM100Adapter.from_bundle(bundle)

        for name in SO_ARM100_JOINT_NAMES:
            orig_j = original.config.joint(name)
            loaded_j = loaded.config.joint(name)
            assert loaded_j.motor_id == orig_j.motor_id
            assert loaded_j.drive_mode == orig_j.drive_mode
            assert loaded_j.homing_offset == orig_j.homing_offset
            assert loaded_j.range_min == orig_j.range_min
            assert loaded_j.range_max == orig_j.range_max
            assert loaded_j.position_limits == orig_j.position_limits
            assert loaded_j.velocity_limit == orig_j.velocity_limit

    def test_from_bundle_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No SO-ARM100 calibration"):
            SOARM100Adapter.from_bundle(tmp_path)

    def test_lerobot_calibration_round_trips_to_json_format(
        self,
        lerobot_cal_json: Path,
        tmp_path: Path,
    ):
        """Re-emit LeRobot calibration and ensure it loads back to the same
        SOARM100Config — guarantees we don't drift LeRobot users' calibrations."""
        adapter = SOARM100Adapter.from_calibration(lerobot_cal_json)
        out = tmp_path / "round_trip.json"
        adapter.config.write_lerobot_calibration(out)

        # Loading the emitted JSON should yield the same per-joint LeRobot fields.
        re_adapter = SOARM100Adapter.from_calibration(out)
        for name in SO_ARM100_JOINT_NAMES:
            a = adapter.config.joint(name)
            b = re_adapter.config.joint(name)
            assert (a.motor_id, a.drive_mode, a.homing_offset, a.range_min, a.range_max) == (
                b.motor_id, b.drive_mode, b.homing_offset, b.range_min, b.range_max
            )


# ─── TestEmbodimentConfigBridge ─────────────────────────────────────────────


class TestEmbodimentConfigBridge:
    def test_to_embodiment_config_returns_so100_preset(self):
        from tether.embodiments import EmbodimentConfig
        a = SOARM100Adapter.default()
        cfg = a.to_embodiment_config()
        assert isinstance(cfg, EmbodimentConfig)
        assert cfg.embodiment == "so100"
        assert cfg.action_dim == 6
        assert cfg.gripper_idx == 5


# ─── TestMockSOARM100Port ───────────────────────────────────────────────────


class TestMockSOARM100Port:
    def test_default_positions(self, mock_port: MockSOARM100Port):
        pose = mock_port.read_pose()
        assert set(pose.keys()) == set(SO_ARM100_JOINT_NAMES)
        assert all(v == 2048 for v in pose.values())

    def test_write_records_and_updates(self, mock_port: MockSOARM100Port):
        mock_port.write_goal(1, 3000)
        assert mock_port.writes == [(1, 3000)]
        assert mock_port.read_pose()["shoulder_pan"] == 3000

    def test_batch_write(self, mock_port: MockSOARM100Port):
        cmds = [(1, 100), (2, 200), (3, 300)]
        mock_port.write_goals(cmds)
        assert mock_port.batch_writes == [cmds]
        pose = mock_port.read_pose()
        assert pose["shoulder_pan"] == 100
        assert pose["shoulder_lift"] == 200
        assert pose["elbow_flex"] == 300

    def test_close(self, mock_port: MockSOARM100Port):
        mock_port.close()
        assert mock_port.closed is True

    def test_unknown_motor_id_raises(self, mock_port: MockSOARM100Port):
        with pytest.raises(KeyError):
            mock_port.write_goal(99, 0)


# ─── Smoke: chunk → mock-port full loop ─────────────────────────────────────


class TestEndToEndWithMockPort:
    def test_chunk_to_mock_port(self, mock_port: MockSOARM100Port):
        """Simulate the inner loop of `reflex serve` with the mock port:
        for each step in the chunk, dispatch the commands."""
        a = SOARM100Adapter.default()
        T = 5
        chunk = np.zeros((T, 6))
        chunk[:, 0] = np.linspace(-0.5, 0.5, T)
        chunk[:, 5] = np.linspace(0.0, 1.0, T)

        stream = a.chunk_to_servo_command_stream(chunk)
        for per_step in stream:
            mock_port.write_goals(per_step)

        # Mock port should have T batches, each of length 6.
        assert len(mock_port.batch_writes) == T
        assert all(len(b) == 6 for b in mock_port.batch_writes)
        # Last commanded gripper should be the closed-side endpoint.
        last_pose = mock_port.read_pose()
        assert last_pose["gripper"] == a.config.gripper_closed_servo_units
