"""SO-ARM 100 physical-arm calibration.

Adapted from auto_soarm (MIT, github.com/0o8o0-blip/auto_soarm). The original
project structures these as standalone scripts; we vendor them as a Python
package with adapted imports.

Calibration flow (per upstream README):
    1. corners  — hand-guide arm to 4 numbered tablet markers; record joint
                  positions; fit homography from joint-space to tablet pixels
    2. surface  — probe the tablet surface for tap depth; fit calibrated tap
                  model (hover height, press depth, contact backoff)
    3. all      — orchestrates corners + surface in sequence

Calibration outputs land at:
    ~/.reflex/calibration/so100/<calibration_id>/{corners,surface,model}.json

These are LOCAL files; never synced to Reflex servers. See `_compliance/`
for the data-handling story for any data downstream of calibration.

CLI surface (wired from src/reflex/cli.py):
    reflex calibrate so100 corners
    reflex calibrate so100 surface
    reflex calibrate so100 all
"""
from __future__ import annotations

# Re-exports for convenience. Submodule attributes are not re-exported wholesale
# to keep the public surface narrow; callers import what they need directly.

DEFAULT_CALIBRATION_DIR = "~/.reflex/calibration/so100"
