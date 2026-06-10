"""End-to-end SO-ARM100 + LeRobot SmolVLA pipeline.

What this does, top to bottom:

    1. Build a SOARM100Adapter from a LeRobot calibration file
    2. Export the LeRobot SmolVLA base checkpoint to an ONNX bundle, embedding
       the SO-ARM100 calibration so the runtime can stream commands to the arm
    3. Verify the export matches the original PyTorch policy on a small LIBERO
       sample (signed parity cert lands in ./verify_output/parity.cert.json)
    4. Serve the bundle to a physical SO-ARM100 over a USB serial port

The full chain mirrors the README's `Quickstart` exactly — only the
embodiment + hardware port flags are new. Everything else (export, verify,
serve) is the same Reflex surface you'd use for a Franka or UR5.

Hardware requirements:
    - SO-ARM100 assembled per https://github.com/TheRobotStudio/SO-ARM100
    - 6x Feetech STS3215 servos wired in the canonical 1..6 ID order
    - USB-to-serial bridge on /dev/ttyUSB0 (Linux) or /dev/tty.usbserial-* (Mac)

Python requirements:
    pip install 'tether[serve,gpu,monolithic,lerobot]'   # GPU host
    pip install 'tether[serve,onnx,lerobot,so100]'       # Mac / Pi at the arm

Calibration:
    If you already have a LeRobot calibration file (recorded via
    `lerobot-calibrate --robot so_follower`), point CAL_PATH at it.
    Otherwise, run:
        tether calibrate so_arm100 default --output calib.json
    and replace with a real calibration before running on hardware.

Usage:
    python examples/so_arm100_smolvla.py

To run only one phase, set the SO_ARM100_PHASE env var to
"export", "verify", or "serve".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from tether.embodiments.so_arm100 import SOARM100Adapter

# ─── Config ─────────────────────────────────────────────────────────────────

MODEL_ID = os.environ.get("REFLEX_MODEL_ID", "lerobot/smolvla_base")
CAL_PATH = os.environ.get("CAL_PATH", "calib.json")
BUNDLE_DIR = os.environ.get("BUNDLE_DIR", "bundle/")
VERIFY_OUT = os.environ.get("VERIFY_OUT", "verify_output/")
PORT = os.environ.get("SO_ARM100_PORT", "/dev/ttyUSB0")
PHASE = os.environ.get("SO_ARM100_PHASE", "all").lower()
N_EPISODES = int(os.environ.get("VERIFY_EPISODES", "10"))


def build_adapter() -> SOARM100Adapter:
    """Load the LeRobot calibration into an SOARM100Adapter."""
    if not Path(CAL_PATH).exists():
        print(
            f"[warn] {CAL_PATH} not found; using factory defaults. "
            f"Generate one with `tether calibrate so_arm100 default --output {CAL_PATH}` "
            f"or import an existing LeRobot calibration with "
            f"`tether calibrate so_arm100 import <path>`."
        )
        return SOARM100Adapter.default(port=PORT)
    return SOARM100Adapter.from_calibration(CAL_PATH, port=PORT)


def phase_export(adapter: SOARM100Adapter) -> None:
    """Run `tether export` with the SO-ARM100 calibration embedded.

    This drives the same code path as the CLI command:
        tether export lerobot/smolvla_base \
            --output bundle/ \
            --embodiment so_arm100 \
            --calibration calib.json
    """
    import subprocess
    cmd = [
        sys.executable, "-m", "tether.cli", "export", MODEL_ID,
        "--output", BUNDLE_DIR,
        "--embodiment", "so_arm100",
    ]
    if Path(CAL_PATH).exists():
        cmd += ["--calibration", CAL_PATH]
    print("[export]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Sanity: the bundle should now carry our embodiment dir.
    bundle_cal = Path(BUNDLE_DIR) / "embodiment" / "so_arm100" / "calibration.json"
    assert bundle_cal.exists(), f"Export did not write {bundle_cal}"
    print(f"[export] embodiment bundle written: {bundle_cal}")

    # Confirm round-trip — load the bundle as if we were the serve runtime.
    loaded = SOARM100Adapter.from_bundle(BUNDLE_DIR)
    print(
        f"[export] loaded back: "
        f"{[j.name for j in loaded.config.joints]} ({loaded.action_dim}-DOF)"
    )


def phase_verify() -> None:
    """Run `tether verify` on the exported bundle.

    The `--embodiment so_arm100` flag tells verify to record provenance for
    the parity cert; the numerical gates are unchanged.
    """
    import subprocess
    cmd = [
        sys.executable, "-m", "tether.cli", "verify", BUNDLE_DIR,
        "--num-episodes", str(N_EPISODES),
        "--output", VERIFY_OUT,
        "--embodiment", "so_arm100",
    ]
    print("[verify]", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc == 0:
        print("[verify] PASS")
        cert = Path(VERIFY_OUT) / "parity.cert.json"
        if cert.exists():
            print(f"[verify] signed parity cert: {cert}")
    else:
        print(f"[verify] verify exited with code {rc} — see {VERIFY_OUT}/PARITY.md")


def phase_serve(adapter: SOARM100Adapter) -> None:
    """Start `tether serve` against the bundle on the SO-ARM100.

    NOTE: this opens a live serial port + runs until killed. Don't fire-and-
    forget in headless scripts; we exec the CLI directly so Ctrl+C reaches
    the server.
    """
    import os
    print(
        f"[serve] launching: tether serve {BUNDLE_DIR} "
        f"--embodiment so_arm100 --port {PORT}"
    )
    print("[serve] Ctrl+C to stop. Endpoints: /act /health /config")
    os.execvp(
        sys.executable,
        [
            sys.executable, "-m", "tether.cli", "serve", BUNDLE_DIR,
            "--embodiment", "so_arm100",
        ],
    )


def main() -> None:
    print(f"[so_arm100_smolvla] phase={PHASE}, model={MODEL_ID}, "
          f"bundle={BUNDLE_DIR}, calibration={CAL_PATH}, port={PORT}")
    adapter = build_adapter()
    print(
        f"[so_arm100_smolvla] adapter: {adapter.embodiment_name}, "
        f"{adapter.action_dim}-DOF, source="
        f"{adapter.config._source_path or '(default)'}"
    )

    if PHASE in ("all", "export"):
        phase_export(adapter)
    if PHASE in ("all", "verify"):
        phase_verify()
    if PHASE in ("all", "serve"):
        phase_serve(adapter)


if __name__ == "__main__":
    main()
