# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Circle game task environment.

This owns the runtime mechanics shared by collection and eval:
server/tablet setup, arm lifecycle, target spawn handshake, frame capture,
and the calibrated-oracle action path.
"""
import time

from reflex.embodiments.so100.edge_runtime import SOArmHardware
from reflex.embodiments.so100.calibration.tapper import (
    CalibratedTapper,
    WRIST_FLEX_ID,
    classify_touch,
    format_pose_delta,
    touch_count,
    touches_since,
    read_pose,
)
from reflex.embodiments.so100.calibration.motion_io import (
    APPROACH_STEP_DELAY_S,
    APPROACH_STEPS,
    PRESS_HOLD_FRAMES,
    RETRACT_HOLD_FRAMES,
    action_from_state,
    make_frame_record,
    read_state,
)
from reflex.bench.games._base.circle_runtime import capture_frame, fullscreen_setup, open_camera
from reflex.bench.games._base.tablet_setup import request_respawn
from touch_server import start_background_server


class CircleTask:
    def __init__(self, port, npos, frame_dir, no_camera=False,
                 cam_width=640, cam_height=360):
        self.port = port
        self.npos = npos
        self.frame_dir = frame_dir
        self.no_camera = no_camera
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.server = None
        self.cam = None
        self.arm = None
        self.tapper = None

    def start(self):
        try:
            self.server = start_background_server(self.port)
            self.cam = None if self.no_camera else open_camera(
                self.cam_width, self.cam_height)
            if not fullscreen_setup(self.port):
                raise RuntimeError("could not launch circle game")
            self.arm = SOArmHardware()
            self.tapper = CalibratedTapper(self.arm, self.port)
            self.arm.home()
            time.sleep(1.0)
            return self
        except Exception:
            self.close()
            raise

    def close(self):
        try:
            if self.arm is not None:
                self.arm.home(settle_s=0.8)
                self.arm.close()
        except Exception:
            pass
        if self.cam is not None:
            self.cam.release()
        if self.server is not None:
            self.server.shutdown()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()

    def home_for_episode(self):
        self.arm.home(settle_s=0.6)

    def spawn_target(self):
        return request_respawn(
            port=self.port, npos=self.npos, require_drawn=True)

    def capture(self, ep, label):
        return capture_frame(
            self.cam,
            self.frame_dir / f"ep{ep:04d}_{label}.jpg",
            self.cam_width,
            self.cam_height,
        )

    def capture_demo_frame(self, ep, frame_idx, phase, *, pan=None, lift=None,
                           wrist_flex=None):
        state = read_state(self.arm)
        action = action_from_state(
            state, pan=pan, lift=lift, wrist_flex=wrist_flex)
        img = self.capture(ep, f"f{frame_idx:03d}_{phase}")
        return make_frame_record(
            img, state, action, phase=phase, frame_index=frame_idx)

    def prepare_episode(self, ep):
        self.home_for_episode()
        target = self.spawn_target()
        if target is None:
            return None, {
                "episode": ep,
                "result": "reject",
                "reason": "target was not draw-confirmed",
            }
        return target, {"before": self.capture(ep, "before")}

    def oracle_demo(self, target, ep, capture_hover=True,
                    capture_trajectory=False, allow_touch_trail=False):
        """Execute the calibrated baseline action and return one record."""
        if not self.tapper.contains(target["x"], target["y"], margin=20):
            return {
                "result": "reject",
                "reason": f"target outside calibrated touch box {target}",
                "target": target,
            }

        pose = self.tapper.pose_for(target["x"], target["y"])
        print(f"  target=({target['x']:.0f},{target['y']:.0f}) "
              f"token={target.get('token')} pose={pose}", flush=True)
        if capture_trajectory:
            rec = self._oracle_demo_trajectory(target, ep, pose)
            return rec
        ok, note = self.tapper.move_to_hover(pose)
        if not ok:
            return {
                "result": "reject",
                "reason": note,
                "target": target,
                "command_pose": pose,
            }

        hover_actual = read_pose(self.arm)
        hover_img = self.capture(ep, "hover") if capture_hover else None
        touches, actual = self.tapper.tap(pose)
        result, reason, touch = classify_touch(
            touches, target, allow_trail=allow_touch_trail)
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
            "hover_frame": hover_img,
        }

    def _oracle_demo_trajectory(self, target, ep, pose):
        frames = []

        def add_frame(phase, frame_idx, **action):
            rec = self.capture_demo_frame(ep, frame_idx, phase, **action)
            if rec is not None:
                frames.append(rec)

        add_frame("scout", 0)
        cur = read_pose(self.arm)
        approach_start = touch_count(self.port)
        for step in range(1, APPROACH_STEPS + 1):
            t = step / APPROACH_STEPS
            pan = int(round(cur["pan"] + (pose["pan"] - cur["pan"]) * t))
            lift = int(round(cur["lift"] + (pose["lift"] - cur["lift"]) * t))
            wf = int(round(cur["wrist_flex"] + (pose["hover_wf"] - cur["wrist_flex"]) * t))
            self.arm.write_goal(1, pan)
            self.arm.write_goal(2, lift)
            self.arm.write_goal(WRIST_FLEX_ID, wf)
            time.sleep(APPROACH_STEP_DELAY_S)
            add_frame("approach", step, pan=pan, lift=lift, wrist_flex=wf)
            if touches_since(approach_start, self.port):
                self.arm.write_goal(WRIST_FLEX_ID, pose["hover_wf"])
                return {
                    "result": "reject",
                    "reason": "touch during approach",
                    "target": target,
                    "command_pose": pose,
                    "demo_frames": [],
                }

        before_press = touch_count(self.port)
        frame_idx = APPROACH_STEPS
        ramp_steps = 3
        for s in range(1, ramp_steps + 1):
            frame_idx += 1
            wf = int(round(pose["hover_wf"] + (pose["wrist_flex"] - pose["hover_wf"]) * s / ramp_steps))
            self.arm.write_goal(WRIST_FLEX_ID, wf)
            time.sleep(0.1)
            add_frame("press", frame_idx, pan=pose["pan"], lift=pose["lift"], wrist_flex=wf)
        for _ in range(PRESS_HOLD_FRAMES - ramp_steps):
            frame_idx += 1
            time.sleep(0.1)
            add_frame("press_hold", frame_idx, pan=pose["pan"], lift=pose["lift"], wrist_flex=pose["wrist_flex"])

        for s in range(1, ramp_steps + 1):
            frame_idx += 1
            wf = int(round(pose["wrist_flex"] + (pose["hover_wf"] - pose["wrist_flex"]) * s / ramp_steps))
            self.arm.write_goal(WRIST_FLEX_ID, wf)
            time.sleep(0.1)
            add_frame("retract", frame_idx, pan=pose["pan"], lift=pose["lift"], wrist_flex=wf)
        for _ in range(max(0, RETRACT_HOLD_FRAMES - ramp_steps)):
            frame_idx += 1
            time.sleep(0.1)
            add_frame("retract_hold", frame_idx, pan=pose["pan"], lift=pose["lift"], wrist_flex=pose["hover_wf"])

        actual = read_pose(self.arm)
        touches = [t for t in touches_since(before_press, self.port) if not t.get("_fs_event")]
        result, reason, touch = classify_touch(touches, target)
        return {
            "result": result,
            "reason": reason,
            "target": target,
            "command_pose": pose,
            "tap_actual": actual,
            "tap_delta": format_pose_delta(actual, {
                "pan": pose["pan"],
                "lift": pose["lift"],
                "wrist_flex": pose["wrist_flex"],
            }),
            "touch": touch,
            "all_touches": touches,
            "demo_frames": frames,
        }
