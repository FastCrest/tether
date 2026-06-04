# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Arm setup helpers — motor IDs and PID configuration.

Calibration loaders (load_hover_cal, load_contact_offset,
append_calibration_correction) were moved to calibration/model.py for the unified API.
"""

import json
import struct

WRIST_FLEX_ID = 4


def setup_arm(ph, port):
    """Apply homing offsets, configure all motor settings, enable torque.

    Replicates lerobot's so_follower.configure() but uses P=32 (factory default)
    instead of lerobot's P=16 — this eliminates the ~22-tick dead-band that
    causes approach-dependent position error.
    """
    cal = json.load(open(
        "/home/user/.cache/huggingface/lerobot/calibration/robots/so_follower/None.json"))

    # Snapshot current positions so motors don't jump on torque toggle
    current_positions = {}
    for mid in [1, 2, 3, 4, 5, 6]:
        pos, _, _ = ph.read2ByteTxRx(port, mid, 56)
        current_positions[mid] = pos

    # Torque off
    for mid in [1, 2, 3, 4, 5, 6]:
        ph.write1ByteTxRx(port, mid, 40, 0)

    # Set goal = current so torque-on doesn't move
    for mid in [1, 2, 3, 4, 5, 6]:
        ph.write2ByteTxRx(port, mid, 42, current_positions[mid])

    # Apply homing offsets for pan + wrist_flex
    for mid, name in [(1, "shoulder_pan"), (4, "wrist_flex")]:
        offset = cal[name]["homing_offset"]
        unsigned = struct.unpack("H", struct.pack("h", offset))[0]
        ph.write1ByteTxRx(port, mid, 55, 0)
        ph.write2ByteTxRx(port, mid, 31, unsigned)

    # Configure motors (P=32 not lerobot's P=16)
    for mid in [1, 2, 3, 4, 5, 6]:
        ph.write1ByteTxRx(port, mid, 7, 0)
        ph.write1ByteTxRx(port, mid, 85, 254)
        ph.write1ByteTxRx(port, mid, 41, 254)
        ph.write1ByteTxRx(port, mid, 33, 0)
        ph.write1ByteTxRx(port, mid, 21, 32)
        ph.write1ByteTxRx(port, mid, 22, 32)
        ph.write1ByteTxRx(port, mid, 23, 0)

    # Gripper protection
    ph.write2ByteTxRx(port, 6, 48, 500)
    ph.write2ByteTxRx(port, 6, 71, 250)
    ph.write1ByteTxRx(port, 6, 37, 25)

    # Enable torque on all non-gripper motors
    for mid in [1, 2, 3, 4, 5]:
        ph.write1ByteTxRx(port, mid, 40, 1)
    ph.write1ByteTxRx(port, 6, 40, 0)
