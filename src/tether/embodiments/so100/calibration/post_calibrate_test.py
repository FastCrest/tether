# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Simple post-calibration tap test.

This wraps the experiment collector in log-only mode. It is intended to be run
immediately after calibration/calibrate_surface.py to validate that the shared calibrated
tapper can hit both left/right targets cleanly before training-data collection.

Run:
    python -u -B calibration/post_calibrate_test.py --episodes 10
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--mode", default="2d")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_name = args.run_name or time.strftime("post_calibrate_tap_%Y%m%d_%H%M%S")
    cmd = [
        sys.executable,
        "-u",
        "-B",
        "experiments/001_circle_lr/collect.py",
        "--episodes",
        str(args.episodes),
        "--mode",
        args.mode,
        "--run-name",
        run_name,
    ]
    print(f">>> {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
