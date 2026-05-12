# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Preflight checks for a calibration-to-collection run.

This is intentionally non-invasive: it does not move the arm. It checks the
host, tablet, Python environments, expected calibration files, and optional
camera access.

Run:
    python calibration/preflight.py
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEROBOT_CAL_PATH = (
    Path.home()
    / ".cache/huggingface/lerobot/calibration/robots/so_follower/None.json"
)
LEROBOT_PYTHON = Path("/home/user/.venv/lerobot/bin/python")
DEFAULT_SERIAL_PORT = Path("/dev/ttyACM0")


def ok(msg):
    print(f"[ok]   {msg}")


def warn(msg):
    print(f"[warn] {msg}")


def fail(msg):
    print(f"[fail] {msg}")


def run_cmd(args, timeout=4):
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired as e:
        return e


def check_python_import(module_name):
    if importlib.util.find_spec(module_name) is None:
        fail(f"Python import missing: {module_name}")
        return False
    ok(f"Python import available: {module_name}")
    return True


def check_lerobot_env():
    if not LEROBOT_PYTHON.exists():
        fail(f"LeRobot Python not found: {LEROBOT_PYTHON}")
        return False
    res = run_cmd([
        str(LEROBOT_PYTHON),
        "-c",
        "import lerobot; print('lerobot ok')",
    ])
    if res is None or getattr(res, "returncode", 1) != 0:
        stderr = getattr(res, "stderr", "").strip()
        fail(f"LeRobot import failed in {LEROBOT_PYTHON}: {stderr}")
        return False
    ok(f"LeRobot import works in {LEROBOT_PYTHON}")
    return True


def check_adb():
    res = run_cmd(["adb", "devices"])
    if res is None:
        fail("adb not found")
        return False
    if res.returncode != 0:
        fail(f"adb devices failed: {res.stderr.strip()}")
        return False
    devices = [
        line for line in res.stdout.splitlines()[1:]
        if line.strip().endswith("\tdevice")
    ]
    if not devices:
        fail("adb sees no authorized device")
        return False
    ok(f"adb device authorized: {devices[0].split()[0]}")

    size = run_cmd(["adb", "shell", "wm", "size"])
    if size is not None and size.returncode == 0 and "size:" in size.stdout.lower():
        ok(size.stdout.strip().splitlines()[-1])
    else:
        warn("could not read tablet screen size via adb shell wm size")
    return True


def check_files():
    required = [
        ROOT / "calibration/calibrate_corners.py",
        ROOT / "calibration/calibrate_surface.py",
        ROOT / "experiments/001_circle_lr/collect.py",
        ROOT / "games/calibrate.html",
        ROOT / "games/circle.html",
    ]
    success = True
    for path in required:
        if path.exists():
            ok(f"found {path.relative_to(ROOT)}")
        else:
            fail(f"missing {path.relative_to(ROOT)}")
            success = False

    if LEROBOT_CAL_PATH.exists():
        ok(f"lerobot arm calibration exists: {LEROBOT_CAL_PATH}")
    else:
        fail(f"missing lerobot arm calibration: {LEROBOT_CAL_PATH}")
        success = False

    for path in (ROOT / "data/tablet_config.json",
                 ROOT / "data/last_wrist_grid_samples.json"):
        if path.exists():
            ok(f"existing calibration artifact: {path.relative_to(ROOT)}")
        else:
            warn(f"calibration artifact not present yet: {path.relative_to(ROOT)}")

    samples_path = ROOT / "data/last_wrist_grid_samples.json"
    if samples_path.exists():
        try:
            payload = json.loads(samples_path.read_text())
            clean = sum(1 for s in payload.get("samples", [])
                        if s.get("clean", True))
            ok(f"clean wrist-grid samples: {clean}")
        except json.JSONDecodeError:
            warn("data/last_wrist_grid_samples.json is not valid JSON")
    return success


def check_serial(port):
    if port.exists():
        ok(f"arm serial port exists: {port}")
        return True
    fail(f"arm serial port missing: {port}")
    return False


def check_camera(index):
    try:
        import cv2
    except ModuleNotFoundError:
        warn("cv2 not installed; skipping camera open check")
        return None

    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            fail(f"camera {index} did not open")
            return False
        ok(f"camera {index} opens")
        return True
    finally:
        cap.release()


def check_server_free(port):
    try:
        urllib.request.urlopen(f"http://localhost:{port}/circle/target", timeout=0.5)
    except Exception:
        ok(f"touch server port {port} appears free")
        return True
    warn(f"something is already responding on localhost:{port}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-port", default=str(DEFAULT_SERIAL_PORT))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--skip-camera", action="store_true")
    parser.add_argument("--touch-port", type=int, default=8186)
    args = parser.parse_args()

    checks = []
    checks.append(check_files())
    checks.append(check_python_import("scservo_sdk"))
    checks.append(check_lerobot_env())
    checks.append(check_adb())
    checks.append(check_serial(Path(args.serial_port)))
    checks.append(check_server_free(args.touch_port))
    if not args.skip_camera:
        cam_ok = check_camera(args.camera_index)
        if cam_ok is not None:
            checks.append(cam_ok)

    if all(checks):
        print("\nPreflight passed.")
        return 0
    print("\nPreflight found issues. Fix failures before calibration/collection.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
