# Adapted from auto_soarm by 0o8o0 (MIT-licensed)
# Source: https://github.com/0o8o0-blip/auto_soarm
# See THIRD_PARTY_LICENSES.md for the full license text.
"""Run the full interactive calibration sequence.

`calibration/calibrate_surface.py` depends on
`calibration/calibrate_corners.py` producing
data/tablet_config.json, so these steps cannot run in parallel. This wrapper
keeps the operator flow explicit and then optionally runs a tap smoke test.

Run:
    python -u -B calibration/calibrate_all.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(cmd):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-corners", action="store_true",
                        help="Reuse existing data/tablet_config.json.")
    parser.add_argument("--skip-surface", action="store_true",
                        help="Reuse existing data/last_wrist_grid_samples.json.")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--test-episodes", type=int, default=10)
    parser.add_argument("--test-run-name", default="post_calibrate_tap_check")
    args = parser.parse_args()

    py = sys.executable
    steps = []
    if not args.skip_preflight:
        steps.append([py, "calibration/preflight.py"])
    if not args.skip_corners:
        steps.append([py, "-u", "-B", "calibration/calibrate_corners.py"])
    if not args.skip_surface:
        steps.append([py, "-u", "-B", "calibration/calibrate_surface.py"])
    if not args.skip_test:
        steps.append([
            py,
            "-u",
            "-B",
            "calibration/post_calibrate_test.py",
            "--episodes",
            str(args.test_episodes),
            "--run-name",
            args.test_run_name,
        ])

    for cmd in steps:
        rc = run_step(cmd)
        if rc != 0:
            print(f"\nABORT: step failed with exit code {rc}")
            return rc

    print("\nCalibration sequence complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
