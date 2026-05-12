"""SO-ARM 100 embodiment-specific code.

Vendored substrate from auto_soarm (MIT, github.com/0o8o0-blip/auto_soarm).
Provides physical-arm calibration + Pi-edge runtime that LeRobot's joint-only
calibration doesn't cover.

Submodules:
    calibration   physical calibration (corners + surface + model + tapper)
    edge_runtime  no-torch arm driver for Pi deploy

CLI surface:
    reflex calibrate so100 corners      (hand-guide arm to 4 tablet corners)
    reflex calibrate so100 surface      (probe surface depth + fit tap model)
    reflex calibrate so100 all          (run the full sequence)

See ADR `01_decisions/2026-05-06-vendor-auto-soarm.md` for vendoring scope
+ attribution.
"""
from __future__ import annotations
