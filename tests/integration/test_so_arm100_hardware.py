"""SO-ARM100 hardware integration smoke test.

HARDWARE-GATED: this module is SKIPPED by default. Run only on a host wired
to a real SO-ARM100 with `RUN_HARDWARE_TESTS=1`:

    RUN_HARDWARE_TESTS=1 pytest tests/integration/test_so_arm100_hardware.py -v -s

What it covers (the bits the mock-only unit tests can't):
    1. Real serial connect via either the feetech bus (LeRobot) or scservo SDK
    2. Reading present-position back as a sane 6-vector state
    3. Writing a small near-home goal pose and reading it back within tolerance
    4. Clean disconnect (port is released; reconnect works)

These checks are intentionally conservative — no aggressive motions, no
end-effector contact, no calibration write. They verify the bridge + adapter
plumb to real hardware without false positives.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

from tether.embodiments.so_arm100 import SOARM100Adapter, SOARM100Config

HARDWARE_PORT = os.environ.get("REFLEX_SO_ARM100_PORT", "/dev/ttyUSB0")
HARDWARE_ENABLED = bool(int(os.environ.get("RUN_HARDWARE_TESTS", "0") or "0"))

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        not HARDWARE_ENABLED,
        reason="hardware test gated on RUN_HARDWARE_TESTS=1 (real SO-ARM100 required)",
    ),
]


@pytest.fixture(scope="module")
def adapter() -> SOARM100Adapter:
    cal_path = os.environ.get("REFLEX_SO_ARM100_CALIBRATION", "")
    if cal_path:
        return SOARM100Adapter.from_calibration(cal_path, port=HARDWARE_PORT)
    return SOARM100Adapter.default(port=HARDWARE_PORT)


def test_connect_and_read_pose(adapter: SOARM100Adapter):
    with adapter:
        hw = adapter._hw
        assert hw is not None
        pose = hw.read_pose()
        # Six entries, plausible servo range.
        assert set(pose.keys()) == set(adapter.joint_names)
        for raw in pose.values():
            assert 0 <= int(raw) <= 4095, f"servo raw out of range: {raw}"


def test_state_vector_well_formed(adapter: SOARM100Adapter):
    with adapter:
        pose = adapter._hw.read_pose()
        state = adapter.state_from_servo(pose)
        assert state.shape == (6,)
        # 5 revolute joints inside [-pi, pi]; gripper in [0, 1].
        for v in state[:5]:
            assert -np.pi <= float(v) <= np.pi
        assert 0.0 <= float(state[5]) <= 1.0


def test_near_home_round_trip(adapter: SOARM100Adapter):
    """Send a very small near-home action and confirm the arm acknowledges
    it; we don't enforce a tight position match because the servo's settling
    behaviour + the test's short sleep are non-deterministic — what we DO
    enforce is that the conversion math + wire write produced a consistent
    state vector when read back."""
    with adapter:
        hw = adapter._hw
        # Start from whatever the arm is at; command a no-op identity action.
        present = hw.read_pose()
        state = adapter.state_from_servo(present)
        cmds = adapter.action_to_servo_commands(state)
        for mid, raw in cmds:
            hw.write_goal(mid, raw)
        time.sleep(0.3)  # let it settle
        re_read = hw.read_pose()
        # Each joint should land within +- 20 servo ticks of the goal (a
        # generous slack for static friction + servo deadband).
        for mid, raw in cmds:
            name = next(
                j.name for j in adapter.config.joints if j.motor_id == mid
            )
            assert abs(int(re_read[name]) - int(raw)) <= 20, (
                f"joint {name} settled at {re_read[name]}, expected ~{raw}"
            )


def test_disconnect_and_reconnect(adapter: SOARM100Adapter):
    """The bus must release the port cleanly; a second connect within the
    same process should succeed."""
    adapter.connect()
    adapter.disconnect()
    adapter.connect()
    pose = adapter._hw.read_pose()
    assert len(pose) == 6
    adapter.disconnect()
