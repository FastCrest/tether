"""SO-ARM100 configuration dataclasses.

The SO-ARM100 (TheRobotStudio + HuggingFace LeRobot) is a 6-DOF, ~$150-250 BOM,
3D-printable arm. This module defines the immutable physical + runtime config
that the adapter and bridge consume.

Two layers:
    JointConfig          per-joint limits + calibration offsets + drive mode
    SOARM100Config       arm-level config: joint table, gripper handling,
                         serial protocol, control rate

Design choices:
    - Frozen dataclasses (matches tether.embodiments.EmbodimentConfig style):
      load once at startup, pass around safely.
    - LeRobot-compatible field names (shoulder_pan / shoulder_lift / elbow_flex
      / wrist_flex / wrist_roll / gripper) so calibrations round-trip cleanly.
    - Action space + normalization stats reuse the bundled `so100.json` preset
      under `tether.embodiments.presets/`, so the same arm has ONE source of
      truth for both the `--embodiment so_arm100` flag (preset lookup) and the
      programmatic `SOARM100Adapter` API (this module).

See docs/embodiments/so_arm100.md for hardware + calibration walkthrough.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# Canonical LeRobot joint order for SO-ARM100. The order matters: the policy
# emits a 6-vector and SO-ARM100 servos are addressed by ID 1..6 in this order.
# Source: reference/lerobot/src/lerobot/robots/so_follower/so_follower.py
SO_ARM100_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Motor IDs match the LeRobot/Feetech wiring convention (id == 1-based joint
# index). Hard-coded constants instead of "1..6" magic numbers so misordering
# is loud + reviewable.
SO_ARM100_MOTOR_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

# Index of the gripper in the 6-vector. Mirrors so100.json's gripper.component_idx.
GRIPPER_INDEX: int = 5


@dataclass(frozen=True)
class JointConfig:
    """Per-joint physical + calibration config.

    The four LeRobot-derived calibration fields (drive_mode / homing_offset /
    range_min / range_max) are encoded the same way as
    `lerobot.motors.MotorCalibration` so JSON round-trips lossless.

    `position_limits` is the SAFE software-side joint range in radians (for the
    arm) or [0, 1] (for the gripper). The runtime clamps every commanded joint
    to this window BEFORE writing to the servo so the policy can never drive
    the arm into a hard stop.
    """

    name: str
    motor_id: int

    # LeRobot-format calibration. drive_mode=0 (factory default) is "positive
    # input -> positive servo motion"; drive_mode=1 inverts. homing_offset is
    # in servo units (signed int16); range_{min,max} is also in servo units
    # (0..4095 for the Feetech STS3215).
    drive_mode: int = 0
    homing_offset: int = 0
    range_min: int = 0
    range_max: int = 4095

    # Software-side limits in PHYSICAL units (radians for revolute joints; a
    # normalized [0,1] for the gripper). These are what the adapter clamps to
    # AFTER applying calibration. They're narrower than the servo's hard
    # range_min/range_max by intent — leaves a guard band.
    position_limits: tuple[float, float] = (-3.14, 3.14)
    velocity_limit: float = 3.14  # rad/s; per-step deltas clamped to this.

    @classmethod
    def from_lerobot_calibration(
        cls,
        name: str,
        cal: dict,
        *,
        position_limits: tuple[float, float] = (-3.14, 3.14),
        velocity_limit: float = 3.14,
    ) -> JointConfig:
        """Build from a parsed LeRobot calibration entry.

        `cal` shape (per `lerobot.motors.MotorCalibration`):
            {"id": 1, "drive_mode": 0, "homing_offset": 0,
             "range_min": 1024, "range_max": 3072}

        Position limits + velocity limit aren't in LeRobot's per-motor cal
        (LeRobot keeps them at the policy/normalization layer); we expose them
        as keyword args so the caller (typically `SOARM100Adapter.from_lerobot_
        calibration`) can pin them from a sister source.
        """
        return cls(
            name=name,
            motor_id=int(cal["id"]),
            drive_mode=int(cal.get("drive_mode", 0)),
            homing_offset=int(cal.get("homing_offset", 0)),
            range_min=int(cal.get("range_min", 0)),
            range_max=int(cal.get("range_max", 4095)),
            position_limits=position_limits,
            velocity_limit=velocity_limit,
        )

    def to_lerobot_calibration(self) -> dict:
        """Serialize the four LeRobot fields. Lossless with
        `from_lerobot_calibration` (position_limits + velocity_limit are
        reflex-side metadata that LeRobot's MotorCalibration doesn't carry —
        round-trip through reflex preserves them via SOARM100Config.to_dict)."""
        return {
            "id": self.motor_id,
            "drive_mode": self.drive_mode,
            "homing_offset": self.homing_offset,
            "range_min": self.range_min,
            "range_max": self.range_max,
        }


@dataclass(frozen=True)
class SOARM100Config:
    """Top-level SO-ARM100 config consumed by the adapter + serve runtime.

    Construction paths (most→least convenient):
        1. SOARM100Config.default()                 — factory defaults
        2. SOARM100Config.from_lerobot_calibration(json_path)
                                                    — import an existing
                                                      LeRobot calibration
        3. SOARM100Config(joints=[...], ...)        — explicit
    """

    # Per-joint table. MUST be length 6, MUST match SO_ARM100_JOINT_NAMES order.
    joints: tuple[JointConfig, ...]

    # Serial communication.
    port: str = "/dev/ttyUSB0"
    baud: int = 1_000_000
    # SDK selector. "feetech" = LeRobot's lerobot.motors.feetech bus (preferred
    # — matches the same code path SO-ARM100 datasets are recorded on);
    # "scservo" = the legacy auto_soarm wiring via `scservo_sdk`. The adapter
    # picks the right driver class at construction time.
    protocol: str = "feetech"

    # Gripper handling.
    gripper_open_servo_units: int = 1024   # raw servo position for "fully open"
    gripper_closed_servo_units: int = 2400  # raw servo position for "fully closed"
    gripper_close_threshold: float = 0.5    # normalized [0,1]; >= threshold → close
    gripper_inverted: bool = False          # if true, semantics flipped

    # Control loop.
    control_frequency_hz: float = 15.0
    chunk_size: int = 30
    rtc_execution_horizon: int = 12

    # Safety constraints (mirror EmbodimentConfig.constraints).
    max_ee_velocity: float = 0.5
    max_gripper_velocity: float = 1.0

    # Optional calibration source path — populated by from_lerobot_calibration
    # for the audit trail in the parity cert. Empty string for in-memory configs.
    _source_path: str = ""

    # ─── validation ──────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if len(self.joints) != 6:
            raise ValueError(
                f"SO-ARM100 has 6 joints; got {len(self.joints)}. "
                f"Expected order: {SO_ARM100_JOINT_NAMES}"
            )
        names = tuple(j.name for j in self.joints)
        if names != SO_ARM100_JOINT_NAMES:
            raise ValueError(
                f"Joint names must match SO-ARM100 canonical order. "
                f"Expected {SO_ARM100_JOINT_NAMES!r}, got {names!r}. "
                f"Reorder so dataset/calibration alignment is unambiguous."
            )
        for j, expected_id in zip(self.joints, SO_ARM100_MOTOR_IDS):
            if j.motor_id != expected_id:
                raise ValueError(
                    f"Joint {j.name!r} has motor_id={j.motor_id}; "
                    f"SO-ARM100 wiring expects {expected_id}. If your arm is "
                    f"wired differently, rewire to match the reference build "
                    f"(see docs/embodiments/so_arm100.md) — out-of-order motor "
                    f"IDs make datasets cross-incompatible."
                )
        if self.gripper_open_servo_units == self.gripper_closed_servo_units:
            raise ValueError(
                "gripper open/closed servo units must differ — equal values "
                "make the gripper-toggle math degenerate."
            )
        if self.protocol not in ("feetech", "scservo"):
            raise ValueError(
                f"Unknown protocol {self.protocol!r}; supported: feetech, scservo"
            )

    # ─── accessors ───────────────────────────────────────────────────────────

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def gripper_idx(self) -> int:
        return GRIPPER_INDEX

    def joint(self, name: str) -> JointConfig:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(
            f"No joint named {name!r}; known: "
            f"{[j.name for j in self.joints]}"
        )

    # ─── constructors ────────────────────────────────────────────────────────

    @classmethod
    def default(cls, *, port: str = "/dev/ttyUSB0") -> SOARM100Config:
        """Factory-default config — usable on a freshly-assembled SO-ARM100
        without any calibration imported. Joint software-limits are the
        full hardware range; you SHOULD calibrate before running real policies.
        """
        joints = tuple(
            JointConfig(
                name=name,
                motor_id=mid,
                # No calibration offsets — assume servos are at factory mid-pose.
                drive_mode=0,
                homing_offset=0,
                range_min=0,
                range_max=4095,
                # 5 revolute arm joints (radian limits matching so100.json
                # action_space.ranges) + the gripper in [0, 1].
                position_limits=(
                    (0.0, 1.0) if name == "gripper" else (-3.14, 3.14)
                ),
                velocity_limit=1.0 if name == "gripper" else 3.14,
            )
            for name, mid in zip(SO_ARM100_JOINT_NAMES, SO_ARM100_MOTOR_IDS)
        )
        return cls(joints=joints, port=port)

    @classmethod
    def from_lerobot_calibration(
        cls,
        cal_path: str,
        *,
        port: str = "/dev/ttyUSB0",
        protocol: str = "feetech",
    ) -> SOARM100Config:
        """Load a LeRobot calibration JSON and build an SOARM100Config.

        Expected JSON shape (matches LeRobot's so_follower calibration files
        at ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json):

            {
              "shoulder_pan":   {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 1024, "range_max": 3072},
              "shoulder_lift":  {"id": 2, ...},
              ...
              "gripper":        {"id": 6, ...}
            }
        """
        import json
        from pathlib import Path

        p = Path(cal_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(
                f"LeRobot calibration not found: {p}\n"
                f"  Run `lerobot-calibrate --robot so_follower` to produce one, "
                f"or `tether calibrate so_arm100 --port /dev/ttyUSB0`."
            )
        with p.open() as f:
            raw = json.load(f)

        missing = [n for n in SO_ARM100_JOINT_NAMES if n not in raw]
        if missing:
            raise ValueError(
                f"Calibration {p} is missing required joints: {missing}. "
                f"Expected the full LeRobot SO-100/101 joint set: "
                f"{SO_ARM100_JOINT_NAMES}"
            )

        joints = tuple(
            JointConfig.from_lerobot_calibration(
                name=name,
                cal=raw[name],
                position_limits=(
                    (0.0, 1.0) if name == "gripper" else (-3.14, 3.14)
                ),
                velocity_limit=1.0 if name == "gripper" else 3.14,
            )
            for name in SO_ARM100_JOINT_NAMES
        )
        return cls(
            joints=joints,
            port=port,
            protocol=protocol,
            _source_path=str(p),
        )

    # ─── serialization ───────────────────────────────────────────────────────

    def to_lerobot_calibration(self) -> dict[str, dict]:
        """Serialize back to LeRobot's calibration JSON shape. Lossless with
        `from_lerobot_calibration` for the 5 LeRobot-native fields."""
        return {j.name: j.to_lerobot_calibration() for j in self.joints}

    def write_lerobot_calibration(self, path: str) -> None:
        """Convenience: write the LeRobot calibration JSON to disk."""
        import json
        from pathlib import Path

        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(self.to_lerobot_calibration(), f, indent=4)


__all__ = [
    "GRIPPER_INDEX",
    "JointConfig",
    "SO_ARM100_JOINT_NAMES",
    "SO_ARM100_MOTOR_IDS",
    "SOARM100Config",
]
