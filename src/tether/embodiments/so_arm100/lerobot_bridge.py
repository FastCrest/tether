"""LeRobot ↔ SO-ARM100 servo bridge.

Converts the action representations LeRobot policies emit (joint deltas, or
absolute joint positions in radians + a normalized [0,1] gripper) into raw
Feetech servo commands (integer 0..4095 ticks), and vice versa.

Math is dead-deterministic and intentionally small. The bigger this file gets,
the more surface area for "looks right but drifts after 1000 episodes" bugs.

Reference: lerobot.motors.feetech.tables — the STS3215 used in the SO-ARM100
encodes 360° as 4096 ticks (4095 == one tick shy of a full rotation).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from tether.embodiments.so_arm100.config import (
    GRIPPER_INDEX,
    SO_ARM100_JOINT_NAMES,
    JointConfig,
    SOARM100Config,
)

# STS3215 ticks per full revolution. 4096 ticks span 360°, so 1 tick = 360/4096°.
TICKS_PER_REV: int = 4096
RADIANS_PER_TICK: float = (2.0 * math.pi) / TICKS_PER_REV


def radians_to_servo_units(
    radians: float,
    joint: JointConfig,
) -> int:
    """Convert a target joint angle (radians, centered at homing_offset) to
    raw servo units, applying drive_mode inversion.

    Math:
        ticks_from_home = round(radians / RADIANS_PER_TICK)
        if drive_mode == 1: ticks_from_home = -ticks_from_home
        raw = mid_range + homing_offset + ticks_from_home

    Where mid_range = (range_min + range_max) / 2 is the calibrated zero pose.

    Clamp to [range_min, range_max] AFTER the math so the servo never receives
    an out-of-band command (Feetech servos silently wrap or refuse).
    """
    ticks = int(round(radians / RADIANS_PER_TICK))
    if joint.drive_mode == 1:
        ticks = -ticks
    mid_range = (joint.range_min + joint.range_max) // 2
    raw = mid_range + joint.homing_offset + ticks
    return _clamp_int(raw, joint.range_min, joint.range_max)


def servo_units_to_radians(
    raw: int,
    joint: JointConfig,
) -> float:
    """Inverse of `radians_to_servo_units`. Used to read present-position back
    into the policy's state vector."""
    mid_range = (joint.range_min + joint.range_max) // 2
    ticks = int(raw) - mid_range - joint.homing_offset
    if joint.drive_mode == 1:
        ticks = -ticks
    return ticks * RADIANS_PER_TICK


def normalized_gripper_to_servo_units(
    normalized: float,
    cfg: SOARM100Config,
) -> int:
    """Map a [0, 1] gripper command (0=open, 1=closed unless inverted) to
    a servo units. Linearly interpolates between cfg.gripper_open_servo_units
    and cfg.gripper_closed_servo_units; clamps to that interval."""
    normalized = max(0.0, min(1.0, float(normalized)))
    if cfg.gripper_inverted:
        normalized = 1.0 - normalized
    raw = (
        cfg.gripper_open_servo_units
        + normalized * (cfg.gripper_closed_servo_units - cfg.gripper_open_servo_units)
    )
    lo = min(cfg.gripper_open_servo_units, cfg.gripper_closed_servo_units)
    hi = max(cfg.gripper_open_servo_units, cfg.gripper_closed_servo_units)
    return _clamp_int(int(round(raw)), lo, hi)


def servo_units_to_normalized_gripper(
    raw: int,
    cfg: SOARM100Config,
) -> float:
    """Inverse of `normalized_gripper_to_servo_units`. Returns a value in [0, 1]."""
    span = cfg.gripper_closed_servo_units - cfg.gripper_open_servo_units
    if span == 0:
        return 0.0
    normalized = (raw - cfg.gripper_open_servo_units) / span
    if cfg.gripper_inverted:
        normalized = 1.0 - normalized
    return max(0.0, min(1.0, float(normalized)))


def lerobot_action_to_servo_commands(
    action: Sequence[float] | np.ndarray,
    cfg: SOARM100Config,
    *,
    apply_position_limits: bool = True,
) -> list[tuple[int, int]]:
    """Translate a single 6-vector LeRobot action into a list of
    (motor_id, raw_servo_command) tuples ready to ship over the wire.

    Action vector layout (matches so100.json):
        [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        - indices 0..4 are joint positions in RADIANS
        - index 5 (= GRIPPER_INDEX) is the normalized [0, 1] gripper command

    Steps:
        1. Validate shape.
        2. (Optional) clamp each joint to its software-side position_limits —
           this is the LAST defence before the wire and is enabled by default.
        3. Per joint:
            - revolute → radians_to_servo_units
            - gripper  → normalized_gripper_to_servo_units
        4. Pair each raw value with its motor_id; return in canonical order.

    The action ordering is fixed by SO_ARM100_JOINT_NAMES; we do NOT support
    arbitrary reordering inside this function (caller-side responsibility).
    """
    arr = np.asarray(action, dtype=np.float64).reshape(-1)
    if arr.shape != (6,):
        raise ValueError(
            f"SO-ARM100 action vector must be length 6 "
            f"(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, "
            f"wrist_roll, gripper); got shape {arr.shape}"
        )

    commands: list[tuple[int, int]] = []
    for i, joint in enumerate(cfg.joints):
        value = float(arr[i])
        if apply_position_limits:
            lo, hi = joint.position_limits
            value = max(lo, min(hi, value))
        if i == GRIPPER_INDEX:
            raw = normalized_gripper_to_servo_units(value, cfg)
        else:
            raw = radians_to_servo_units(value, joint)
        commands.append((joint.motor_id, raw))
    return commands


def servo_state_to_lerobot_state(
    present_positions: dict[str, int] | Sequence[int],
    cfg: SOARM100Config,
) -> np.ndarray:
    """Build a LeRobot-compatible 6-vector state from the arm's present-position
    readings.

    Accepts either a name → raw dict (as `SOArmHardware.read_pose()` returns)
    or a positional sequence in canonical order.
    """
    if isinstance(present_positions, dict):
        try:
            ordered = [present_positions[name] for name in SO_ARM100_JOINT_NAMES]
        except KeyError as e:
            raise KeyError(
                f"present_positions dict missing joint {e.args[0]!r}; "
                f"need all of {SO_ARM100_JOINT_NAMES}"
            ) from e
    else:
        ordered = list(present_positions)
        if len(ordered) != 6:
            raise ValueError(
                f"present_positions sequence must be length 6; "
                f"got {len(ordered)}"
            )

    state = np.zeros(6, dtype=np.float64)
    for i, (raw, joint) in enumerate(zip(ordered, cfg.joints)):
        if i == GRIPPER_INDEX:
            state[i] = servo_units_to_normalized_gripper(int(raw), cfg)
        else:
            state[i] = servo_units_to_radians(int(raw), joint)
    return state


def chunk_to_servo_command_stream(
    action_chunk: np.ndarray,
    cfg: SOARM100Config,
) -> list[list[tuple[int, int]]]:
    """Translate a (T, 6) LeRobot action chunk into a list of per-step
    command lists. T = action chunk length.

    Used by the runtime to stream a chunk to the arm at cfg.control_frequency_hz.
    """
    if action_chunk.ndim != 2 or action_chunk.shape[1] != 6:
        raise ValueError(
            f"Expected (T, 6) action chunk; got shape {tuple(action_chunk.shape)}"
        )
    return [
        lerobot_action_to_servo_commands(action_chunk[t], cfg)
        for t in range(action_chunk.shape[0])
    ]


def gripper_should_close(action_value: float, cfg: SOARM100Config) -> bool:
    """Apply the gripper close-threshold rule to one scalar action value.

    Useful for discrete-gripper datasets (some LeRobot pipelines threshold
    the continuous gripper signal before publishing). The serve loop ALWAYS
    sends the continuous value; this is a helper for analytics + dataset
    conversion, not a runtime gate."""
    v = float(action_value)
    if cfg.gripper_inverted:
        v = 1.0 - v
    return v >= cfg.gripper_close_threshold


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


__all__ = [
    "RADIANS_PER_TICK",
    "TICKS_PER_REV",
    "chunk_to_servo_command_stream",
    "gripper_should_close",
    "lerobot_action_to_servo_commands",
    "normalized_gripper_to_servo_units",
    "radians_to_servo_units",
    "servo_state_to_lerobot_state",
    "servo_units_to_normalized_gripper",
    "servo_units_to_radians",
]
