"""Reflex bench games — real-arm benchmark scaffolding.

Vendored substrate from auto_soarm (MIT, github.com/0o8o0-blip/auto_soarm).
Provides the "robot does a real task and we score pass/fail" rig that
existing bench/methodology covers only at the hardware level.

Submodules:
    _base/         generic game scaffold (base_game, tablet_setup,
                   circle_runtime, run_writer, touch_server)
    circle_lr/     canonical circle-tap benchmark (collect / eval / task /
                   demo_sanity)

CLI surface (wired from src/reflex/cli.py):
    reflex bench game circle_lr collect [--episodes N]
    reflex bench game circle_lr eval --ckpt <path> [--episodes N]

See ADR `01_decisions/2026-05-06-vendor-auto-soarm.md` for vendoring scope.
"""
from __future__ import annotations
