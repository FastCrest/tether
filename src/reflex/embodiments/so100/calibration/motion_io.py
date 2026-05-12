# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Shared normalized state/action helpers for collection and eval.

The main repo learned this the hard way: collection and eval must share the
same joint order and raw<->normalized mapping or trained policies silently fail.
"""
import numpy as np

from reflex.embodiments.so100.calibration.tapper import read_pose

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
POSE_KEYS = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]

CAL_RANGES = {
    "shoulder_pan": (1171, 3064),
    "shoulder_lift": (0, 4095),
    "elbow_flex": (0, 4094),
    "wrist_flex": (0, 4095),
    "wrist_roll": (0, 4095),
    "gripper": (701, 2204),
}

TAP_FRAMES = 31
APPROACH_STEPS = 20
APPROACH_STEP_DELAY_S = 0.05
PRESS_HOLD_FRAMES = 6
RETRACT_HOLD_FRAMES = 4


def raw_to_norm(name, raw):
    rmin, rmax = CAL_RANGES[name]
    return (float(raw) - rmin) / (rmax - rmin) * 200.0 - 100.0


def norm_to_raw(name, norm):
    rmin, rmax = CAL_RANGES[name]
    return int(((float(norm) + 100.0) / 200.0) * (rmax - rmin) + rmin)


def raw_pose_to_state(raw_pose):
    return np.array([
        raw_to_norm(joint_name, raw_pose[pose_key])
        for joint_name, pose_key in zip(JOINT_NAMES, POSE_KEYS)
    ], dtype=np.float32)


def read_state(arm):
    return raw_pose_to_state(read_pose(arm))


def action_from_state(state, *, pan=None, lift=None, wrist_flex=None):
    action = np.array(state, dtype=np.float32).copy()
    if pan is not None:
        action[0] = raw_to_norm("shoulder_pan", pan)
    if lift is not None:
        action[1] = raw_to_norm("shoulder_lift", lift)
    if wrist_flex is not None:
        action[3] = raw_to_norm("wrist_flex", wrist_flex)
    return action


def make_frame_record(image_path, state, action, task="tap_circle_2d", **meta):
    if image_path is None:
        return None
    rec = {
        "observation.images.front": image_path,
        "observation.state": np.asarray(state, dtype=np.float32).tolist(),
        "action": np.asarray(action, dtype=np.float32).tolist(),
        "task": task,
    }
    rec.update(meta)
    return rec
