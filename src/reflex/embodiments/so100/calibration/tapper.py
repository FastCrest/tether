# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Shared calibrated tablet tapping primitives.

Game collectors should call this module instead of importing from calibration
wizards. It owns the physical loop: calibrated pixel -> motor pose -> hover ->
tap -> tablet touch report.
"""
import json
import math
import time
import urllib.request

from reflex.embodiments.so100.edge_runtime import LIFT_ID, PAN_ID, WRIST_FLEX_ID
from reflex.embodiments.so100.calibration.model import WristGridCal

HARD_UP_WF = 850
MOVE_STEPS = 14
MOVE_STEP_DELAY_S = 0.055
SETTLE_S = 0.20
# Keep the runtime tap close to a single-contact impulse. Calibration already
# finds first-contact wrist depth, so holding after arrival mainly creates
# tablet touchmove trails on asymmetric poses.
PRESS_HOLD_S = 0.0
RETRACT_HOLD_S = 0.25
FIRST_CONTACT_POLL_S = 0.03
WRIST_ARRIVAL_TOL = 25
WRIST_ARRIVAL_TIMEOUT_S = 1.4
MAX_TRAIL_POINTS = 12
MAX_TRAIL_PX = 5.0


def http_json(path, port, timeout=2.0):
    with urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def touch_count(port):
    return int(http_json("/touches?since=0", port=port).get("count", 0))


def touches_since(idx, port):
    return http_json(f"/touches?since={idx}", port=port).get("touches", [])


def read_pose(arm):
    out = {}
    for jid, name in [(1, "pan"), (2, "lift"), (3, "elbow"),
                      (4, "wrist_flex"), (5, "wrist_roll"), (6, "gripper")]:
        val, _, _ = arm.ph.read2ByteTxRx(arm.port_h, jid, 56)
        out[name] = val
    return out


def format_pose_delta(actual, target):
    parts = []
    for name in ("pan", "lift", "wrist_flex"):
        if name in target:
            parts.append(f"{name}={actual[name]} ({actual[name]-target[name]:+d})")
        else:
            parts.append(f"{name}={actual[name]}")
    return " ".join(parts)


def score_touch(touch, target):
    dx = float(touch.get("touch_x", 0)) - float(target["x"])
    dy = float(touch.get("touch_y", 0)) - float(target["y"])
    return math.sqrt(dx * dx + dy * dy)


def classify_touch(touches, target, *, allow_trail=False):
    if len(touches) != 1:
        return "reject", f"{len(touches)} touch entries", None
    touch = touches[0]
    trail_n = int(touch.get("trail_n", 0))
    trail_px = float(touch.get("trail_px", 0.0))
    if (not allow_trail
            and (trail_n > MAX_TRAIL_POINTS or trail_px > MAX_TRAIL_PX)):
        return "reject", f"trail n={trail_n} px={trail_px:.1f}", touch
    dist = score_touch(touch, target)
    if dist > float(target.get("r", 120)):
        return "miss", f"dist={dist:.1f}px", touch
    return "hit", f"dist={dist:.1f}px", touch


class CalibratedTapper:
    """Execute clean tablet taps using the wrist-grid calibration model."""

    def __init__(self, arm, port, model=None):
        self.arm = arm
        self.port = port
        self.model = model or WristGridCal()

    def contains(self, x, y, margin=20):
        return self.model.contains(x, y, margin=margin)

    def pose_for(self, x, y):
        return self.model.pose_for(x, y)

    def move_to_hover(self, pose, on_step=None):
        cur = read_pose(self.arm)
        t0 = touch_count(self.port)
        for step in range(1, MOVE_STEPS + 1):
            t = step / MOVE_STEPS
            p = int(round(cur["pan"] + (pose["pan"] - cur["pan"]) * t))
            l = int(round(cur["lift"] + (pose["lift"] - cur["lift"]) * t))
            w = int(round(cur["wrist_flex"] + (pose["hover_wf"] - cur["wrist_flex"]) * t))
            self.arm.write_goal(WRIST_FLEX_ID, w)
            time.sleep(0.02)
            self.arm.write_goal(PAN_ID, p)
            self.arm.write_goal(LIFT_ID, l)
            time.sleep(MOVE_STEP_DELAY_S)
            if on_step is not None:
                on_step("approach", pan=p, lift=l, wrist_flex=w)
            if touches_since(t0, self.port):
                self.arm.write_goal(WRIST_FLEX_ID, HARD_UP_WF)
                time.sleep(RETRACT_HOLD_S)
                return False, "touch during approach"
        time.sleep(SETTLE_S)
        return True, None

    def tap(self, pose, on_step=None):
        before = touch_count(self.port)
        self.arm.write_goal(WRIST_FLEX_ID, int(pose["wrist_flex"]))
        deadline = time.time() + WRIST_ARRIVAL_TIMEOUT_S
        actual = read_pose(self.arm)
        saw_contact = False
        while time.time() < deadline:
            actual = read_pose(self.arm)
            if [t for t in touches_since(before, self.port) if not t.get("_fs_event")]:
                saw_contact = True
                break
            if abs(actual["wrist_flex"] - int(pose["wrist_flex"])) <= WRIST_ARRIVAL_TOL:
                break
            time.sleep(0.04)
        if on_step is not None:
            on_step("press", pan=pose["pan"], lift=pose["lift"],
                    wrist_flex=pose["wrist_flex"])

        hold_deadline = time.time() + PRESS_HOLD_S
        while not saw_contact and time.time() < hold_deadline:
            if [t for t in touches_since(before, self.port) if not t.get("_fs_event")]:
                saw_contact = True
                break
            time.sleep(FIRST_CONTACT_POLL_S)

        actual = read_pose(self.arm)
        self.arm.write_goal(WRIST_FLEX_ID, int(pose["hover_wf"]))
        time.sleep(RETRACT_HOLD_S)
        if on_step is not None:
            on_step("retract", pan=pose["pan"], lift=pose["lift"],
                    wrist_flex=pose["hover_wf"])
        touches = [t for t in touches_since(before, self.port) if not t.get("_fs_event")]
        return touches, actual

    def run_target(self, target, on_step=None):
        if not self.contains(target["x"], target["y"], margin=20):
            return {
                "result": "reject",
                "reason": f"target outside calibrated touch box {target}",
                "target": target,
            }
        pose = self.pose_for(target["x"], target["y"])
        ok, note = self.move_to_hover(pose, on_step=on_step)
        if not ok:
            return {
                "result": "reject",
                "reason": note,
                "target": target,
                "command_pose": pose,
            }
        hover_actual = read_pose(self.arm)
        touches, actual = self.tap(pose, on_step=on_step)
        result, reason, touch = classify_touch(touches, target)
        return {
            "result": result,
            "reason": reason,
            "target": target,
            "command_pose": pose,
            "hover_actual": hover_actual,
            "tap_actual": actual,
            "tap_delta": format_pose_delta(actual, {
                "pan": pose["pan"],
                "lift": pose["lift"],
                "wrist_flex": pose["wrist_flex"],
            }),
            "touch": touch,
            "all_touches": touches,
        }
