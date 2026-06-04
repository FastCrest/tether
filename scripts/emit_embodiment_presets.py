"""Generate the 3 shipped embodiment presets (Franka, SO-100, UR5).

One-shot: edit the dicts below, run the script, get freshly-validated
JSON files at configs/embodiments/{franka,so100,ur5}.json. Validates
each preset against the schema BEFORE writing — bad presets never land.

Run:
    python scripts/emit_embodiment_presets.py

Source-of-truth values come from TECHNICAL_PLAN.md §4.5 / Appendix D.2
(franka lines 1943-1974, so100 lines 1976-1999) plus per-robot URDF
references in tether-vla/reference/mujoco_menagerie/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `tether` importable when running from a fresh checkout
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tether.embodiments import EmbodimentConfig, _PRESETS_DIR  # noqa: E402
from tether.embodiments.validate import (  # noqa: E402
    format_errors,
    validate_embodiment_config,
)

# ---------------------------------------------------------------------------
# Preset definitions. Mirror TECHNICAL_PLAN D.2 examples verbatim where
# possible; supplement with URDF refs for ranges. Numbers chosen for
# safety-first defaults — customers can override per their site.
# ---------------------------------------------------------------------------

# Franka Emika Panda — 7-DOF arm, parallel-jaw gripper, wrist camera
# URDF: reference/mujoco_menagerie/franka_emika_panda/
FRANKA: dict = {
    "schema_version": 1,
    "embodiment": "franka",
    "action_space": {
        "type": "continuous",
        "dim": 7,
        # Joint limits from Franka official datasheet (rad). Last dim is gripper width [0,1].
        "ranges": [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [0.0, 1.0],
        ],
    },
    "normalization": {
        "mean_action": [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5],
        "std_action": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25],
        # State: end-effector x, y (planar workspace stats) — used by pi0/pi05 state head
        "mean_state": [0.5, 0.0],
        "std_state": [0.25, 0.25],
    },
    "gripper": {
        "component_idx": 6,
        "close_threshold": 0.5,
        "inverted": False,
    },
    "cameras": [
        {"name": "wrist", "resolution": [640, 480], "fps": 30.0, "color_space": "rgb8"},
    ],
    "control": {
        "frequency_hz": 20.0,
        "chunk_size": 50,
        "rtc_execution_horizon": 0.5,
    },
    "constraints": {
        "max_ee_velocity": 1.0,
        "max_gripper_velocity": 1.0,
        "collision_check": True,
    },
}

# SO-ARM-100 — 6-DOF research arm (TheRobotStudio), parallel-jaw, wrist camera
# Lower control rate, smaller chunk for compute-constrained Orin Nano targets.
# URDF: reference/mujoco_menagerie/trossen_arm/ (closest analogue) +
#       reference/SO-ARM100/
SO100: dict = {
    "schema_version": 1,
    "embodiment": "so100",
    "action_space": {
        "type": "continuous",
        "dim": 6,
        # SO-ARM-100 servo limits (rad), last dim is gripper [0,1]
        "ranges": [
            [-3.14, 3.14],
            [-1.57, 1.57],
            [-3.14, 3.14],
            [-1.57, 1.57],
            [-3.14, 3.14],
            [0.0, 1.0],
        ],
    },
    "normalization": {
        "mean_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        "std_action": [0.4, 0.4, 0.4, 0.4, 0.4, 0.25],
        "mean_state": [0.0, 0.0],
        "std_state": [0.2, 0.2],
    },
    "gripper": {
        "component_idx": 5,
        "close_threshold": 0.5,
        "inverted": False,
    },
    "cameras": [
        {"name": "wrist", "resolution": [640, 480], "fps": 30.0, "color_space": "rgb8"},
    ],
    "control": {
        # Lower frequency for compute-constrained Orin Nano deployments per
        # TECHNICAL_PLAN line 1987. Customers with desktop GPU can override.
        "frequency_hz": 15.0,
        "chunk_size": 30,
        "rtc_execution_horizon": 0.4,
    },
    "constraints": {
        "max_ee_velocity": 0.5,
        "max_gripper_velocity": 1.0,
        "collision_check": True,
    },
}

# Universal Robots UR5 — 6-DOF collaborative arm, optional onRobot RG6 gripper
# URDF: reference/mujoco_menagerie/universal_robots_ur5e/
UR5: dict = {
    "schema_version": 1,
    "embodiment": "ur5",
    "action_space": {
        "type": "continuous",
        "dim": 7,
        # UR5 joint limits (rad) + gripper width [0,1]
        "ranges": [
            [-6.2832, 6.2832],
            [-6.2832, 6.2832],
            [-3.1416, 3.1416],
            [-6.2832, 6.2832],
            [-6.2832, 6.2832],
            [-6.2832, 6.2832],
            [0.0, 1.0],
        ],
    },
    "normalization": {
        "mean_action": [0.0, -1.5, 1.5, -1.5, -1.5, 0.0, 0.5],
        "std_action": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.25],
        "mean_state": [0.5, 0.0],
        "std_state": [0.25, 0.25],
    },
    "gripper": {
        "component_idx": 6,
        # onRobot RG6 typical close-grip threshold — slightly tighter than Franka
        "close_threshold": 0.6,
        "inverted": False,
    },
    "cameras": [
        {"name": "wrist", "resolution": [640, 480], "fps": 30.0, "color_space": "rgb8"},
    ],
    "control": {
        # UR native is ~125 Hz; we run at 20 Hz to match our action chunk cadence
        "frequency_hz": 20.0,
        "chunk_size": 50,
        "rtc_execution_horizon": 0.5,
    },
    "constraints": {
        # UR5 rated for 1.0 m/s; conservative cap at 1.2 with collision check on
        "max_ee_velocity": 1.0,
        "max_gripper_velocity": 1.0,
        "collision_check": True,
    },
}


PRESETS: dict[str, dict] = {
    "franka": FRANKA,
    "so100": SO100,
    "ur5": UR5,
}


def main() -> int:
    _PRESETS_DIR.mkdir(parents=True, exist_ok=True)

    n_failed = 0
    for name, preset_dict in PRESETS.items():
        cfg = EmbodimentConfig.from_dict(preset_dict)
        ok, errors = validate_embodiment_config(cfg)
        if not ok:
            print(f"FAIL  {name}.json — validation failed:")
            print(format_errors(errors))
            n_failed += 1
            continue

        # Surface warnings even on pass
        warnings = [e for e in errors if e["severity"] == "warn"]
        if warnings:
            print(f"WARN  {name}.json — non-blocking warnings:")
            print(format_errors(warnings))

        out_path = _PRESETS_DIR / f"{name}.json"
        with out_path.open("w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
            f.write("\n")
        print(f"OK    {name}.json  →  {out_path}")

    if n_failed:
        print(f"\n{n_failed}/{len(PRESETS)} presets FAILED validation. Fix and re-run.")
        return 1
    print(f"\n{len(PRESETS)}/{len(PRESETS)} presets validated and written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
