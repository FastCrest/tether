# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Shared runtime helpers for circle collection/eval."""
import time
from pathlib import Path

import cv2

from reflex.bench.games._base.tablet_setup import adb_tap, ensure_circle_fullscreen

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 8186
COLLECT_MODES = {
    "random": 0,
    "2d": 2,
    "left-right": 2,
    "lr": 2,
    "left-right-alternating": 23,
    "lr-alt": 23,
    "left": 21,
    "right": 22,
    "center": 1,
}


def npos_for_mode(mode, npos=None):
    return npos if npos is not None else COLLECT_MODES[mode]


def open_camera(width, height):
    import subprocess
    # Auto-exposure works fine when there's enough light; manual boost was a
    # nighttime workaround for a black-frame regime. Default to auto and only
    # fall back to manual if the first frame is unusably dark.
    for cmd in (
        "v4l2-ctl -d /dev/video0 -c auto_exposure=3",
        "v4l2-ctl -d /dev/video0 -c gain=0",
        "v4l2-ctl -d /dev/video0 -c brightness=0",
    ):
        subprocess.run(cmd.split(), capture_output=True)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return None
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(8):
        cam.read()
    # Low-light fallback: if the first frame is essentially black, force manual
    # exposure so collection at night still works without a code edit.
    ok, probe = cam.read()
    if ok and probe is not None and probe.mean() < 20:
        for cmd in (
            "v4l2-ctl -d /dev/video0 -c auto_exposure=1",
            "v4l2-ctl -d /dev/video0 -c exposure_time_absolute=2500",
            "v4l2-ctl -d /dev/video0 -c gain=80",
            "v4l2-ctl -d /dev/video0 -c brightness=40",
        ):
            subprocess.run(cmd.split(), capture_output=True)
        for _ in range(8):
            cam.read()
    return cam


def read_camera_frame(cam, width, height, drain=3):
    if cam is None:
        return None
    for _ in range(drain):
        cam.read()
    ok, frame = cam.read()
    if not ok:
        return None
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))
    return frame


def capture_frame(cam, out_path, width, height):
    frame = read_camera_frame(cam, width, height)
    if frame is None:
        return None
    cv2.imwrite(str(out_path), frame)
    return str(out_path.relative_to(ROOT))


def fullscreen_setup(port):
    if not ensure_circle_fullscreen(port=port, relaunch=True, verbose=True):
        return False
    print("[fullscreen] 3 center taps while target is hidden", flush=True)
    for _ in range(3):
        adb_tap(400, 640)
        time.sleep(0.6)
    time.sleep(0.8)
    return True
