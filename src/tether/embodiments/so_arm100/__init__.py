"""SO-ARM100 — first-class embodiment + LeRobot interop.

The SO-ARM100 (TheRobotStudio + HuggingFace LeRobot, ~$150-250 BOM, 6-DOF,
3D-printable) is the reference real-robot for the LeRobot ecosystem. This
module is Tether's first-class adapter so users with a SO-ARM100 can run:

    pip install tether[lerobot]
    tether export lerobot/smolvla_base --embodiment so_arm100 --output bundle/
    tether verify bundle/ --embodiment so_arm100 --num-episodes 30
    tether serve bundle/ --embodiment so_arm100 --port /dev/ttyUSB0

…against any LeRobot-format SmolVLA / pi0 / pi0.5 checkpoint.

Three submodules:

    config            SOARM100Config + JointConfig (frozen dataclasses; load once)
    lerobot_bridge    radians ↔ servo-units + chunk → wire stream
    adapter           SOARM100Adapter (the user-facing class)

The existing `tether.embodiments.so100` package (vendored from `auto_soarm`)
stays put — it covers the legacy tablet-tap calibration rig that some pilot
users depend on. `so_arm100` is the supported, LeRobot-aligned interface
going forward.

See docs/embodiments/so_arm100.md for the hardware + walkthrough, and
examples/so_arm100_smolvla.py for the end-to-end deploy script.
"""
from __future__ import annotations

from tether.embodiments.so_arm100.adapter import SOARM100Adapter
from tether.embodiments.so_arm100.config import (
    GRIPPER_INDEX,
    SO_ARM100_JOINT_NAMES,
    SO_ARM100_MOTOR_IDS,
    JointConfig,
    SOARM100Config,
)

__all__ = [
    "GRIPPER_INDEX",
    "JointConfig",
    "SO_ARM100_JOINT_NAMES",
    "SO_ARM100_MOTOR_IDS",
    "SOARM100Adapter",
    "SOARM100Config",
]
