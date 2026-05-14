"""Per-embodiment configs (Franka, SO-100, UR5, Trossen, Stretch, custom).

Read by `reflex serve --embodiment <name>` so the runtime knows the robot's
action space, normalization stats, gripper layout, control rate, and safety
constraints. Designed to be loaded once at server startup, passed to the
RTC adapter (B.3), action denormalization (B.6), and reflex doctor (D.1).

Pattern mirrors `src/reflex/config.py:HARDWARE_PROFILES` — module-level
registry + frozen dataclass + getter that raises a descriptive error.

Schema is at `schema.json` next to this file. Preset JSON files are at
`<repo>/configs/embodiments/{franka,so100,ur5}.json` (NOT in the package —
they're user-editable + ship in the repo, not in the wheel).

Plan: features/01_serve/subfeatures/_rtc_a2c2/per-embodiment-configs_plan.md
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Tracks which (source_path, embodiment) pairs have already emitted the
# rtc_execution_horizon migration warning, so we don't spam the operator
# on repeated loads.
_RTC_HORIZON_MIGRATION_WARNED: set[tuple[str, str]] = set()


def _warn_rtc_horizon_migration(
    *,
    source_path: str,
    embodiment: str,
    old_value: float,
    chunk_size: int,
    new_value: int,
) -> None:
    """One-time-per-(source, embodiment) deprecation warning for the
    fractional → integer rtc_execution_horizon migration."""
    key = (source_path, embodiment)
    if key in _RTC_HORIZON_MIGRATION_WARNED:
        return
    _RTC_HORIZON_MIGRATION_WARNED.add(key)
    logger.warning(
        "[deprecated] embodiment %s at %s stores rtc_execution_horizon as "
        "a fraction (%s) — converted to integer count %d via chunk_size=%d. "
        "Update the JSON to integer count to silence this warning. Schema "
        "v2 will reject fractional values.",
        embodiment, source_path or "(in-memory)", old_value, new_value, chunk_size,
    )

# Embodiment preset JSONs are bundled INSIDE the package so they ship with
# `pip install reflex-vla` (since v0.5.2). For dev workflows the repo also
# keeps editable copies in <repo>/configs/embodiments/ — those are checked
# as a fallback when running from source if the in-package presets are
# missing for some reason. Canonical runtime location is the package.
_PRESETS_DIR = Path(__file__).parent / "presets"

# Dev fallback: the editable copies in the repo (only used when the in-package
# presets dir doesn't exist, which shouldn't happen in a proper install).
_DEV_PRESETS_DIR = Path(__file__).resolve().parents[3] / "configs" / "embodiments"
if not _PRESETS_DIR.exists() and _DEV_PRESETS_DIR.exists():
    _PRESETS_DIR = _DEV_PRESETS_DIR

# Path to the JSON schema file (lives inside the package).
_SCHEMA_PATH = Path(__file__).parent / "schema.json"


@dataclass(frozen=True)
class EmbodimentConfig:
    """A robot's per-embodiment config. Frozen — load once, pass around safely."""

    schema_version: int
    embodiment: str
    action_space: dict[str, Any]
    normalization: dict[str, list[float]]
    cameras: list[dict[str, Any]]
    control: dict[str, float | int]
    constraints: dict[str, Any]
    gripper: dict[str, Any] = field(default_factory=dict)
    payload_release: dict[str, Any] = field(default_factory=dict)

    # Where this config came from (for debugging + audit trail). Not part
    # of the schema; populated by the loader.
    _source_path: str = field(default="")

    @classmethod
    def from_dict(cls, d: dict[str, Any], source_path: str = "") -> EmbodimentConfig:
        """Construct from a parsed JSON dict. Doesn't validate — call validate()
        separately if you want to know if it's well-formed.

        Migration (auto-calibration Day 6, per ADR 2026-04-25): the legacy
        `control.rtc_execution_horizon` field was historically stored as a
        fraction of `chunk_size` (e.g. 0.5 = half the chunk). The runtime
        code (rtc_adapter.RtcAdapterConfig) treats it as an integer COUNT
        of actions. Migrate fractional values silently AND emit a one-time
        deprecation warning so customers update their JSON to integer counts.
        """
        control = dict(d["control"])  # don't mutate the caller's dict
        horizon = control.get("rtc_execution_horizon")
        chunk_size = control.get("chunk_size", 0)
        if (
            horizon is not None
            and isinstance(horizon, (int, float))
            and 0 < horizon < 1.0
            and chunk_size >= 1
        ):
            converted = max(1, int(round(horizon * chunk_size)))
            _warn_rtc_horizon_migration(
                source_path=source_path, embodiment=d.get("embodiment", "?"),
                old_value=horizon, chunk_size=chunk_size, new_value=converted,
            )
            control["rtc_execution_horizon"] = converted
        return cls(
            schema_version=d["schema_version"],
            embodiment=d["embodiment"],
            action_space=d["action_space"],
            normalization=d["normalization"],
            gripper=d.get("gripper", {}),
            payload_release=d.get("payload_release", {}),
            cameras=d["cameras"],
            control=control,
            constraints=d["constraints"],
            _source_path=source_path,
        )

    @classmethod
    def load_preset(cls, name: str) -> EmbodimentConfig:
        """Load a shipped preset by name. Raises ValueError if unknown."""
        path = _PRESETS_DIR / f"{name}.json"
        if not path.exists():
            available = sorted(p.stem for p in _PRESETS_DIR.glob("*.json"))
            user_dir = Path.home() / ".cache" / "reflex" / "embodiments"
            raise ValueError(
                f"Unknown embodiment preset '{name}'.\n"
                f"  Available bundled presets: {available or '(none — package may be stale, try: pip install --upgrade reflex-vla)'}\n"
                f"  Workarounds:\n"
                f"    1. Drop --embodiment to run without normalization (raw actions).\n"
                f"    2. Use one of the bundled presets above.\n"
                f"    3. Drop your own JSON at {user_dir}/{name}.json\n"
                f"       (see docs/embodiment_schema.md for the format) and pass it via\n"
                f"       --custom-embodiment-config {user_dir}/{name}.json"
            )
        return cls.load_custom(str(path))

    @classmethod
    def load_custom(cls, path: str) -> EmbodimentConfig:
        """Load from an arbitrary JSON file path."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Embodiment config not found: {path}")
        with p.open() as f:
            data = json.load(f)
        return cls.from_dict(data, source_path=str(p))

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a dict matching the schema (drops _source_path)."""
        out = {
            "schema_version": self.schema_version,
            "embodiment": self.embodiment,
            "action_space": self.action_space,
            "normalization": self.normalization,
            "cameras": self.cameras,
            "control": self.control,
            "constraints": self.constraints,
        }
        if self.gripper:
            out["gripper"] = self.gripper
        if self.payload_release:
            out["payload_release"] = self.payload_release
        return out

    @property
    def action_dim(self) -> int:
        """Convenience accessor — number of action dimensions."""
        return int(self.action_space["dim"])

    @property
    def state_dim(self) -> int:
        """Convenience accessor — number of state dimensions
        (inferred from mean_state)."""
        return len(self.normalization["mean_state"])

    @property
    def gripper_idx(self) -> int:
        return int(self.gripper["component_idx"])


def list_presets() -> list[str]:
    """Return slugs of all shipped presets (alphabetical)."""
    if not _PRESETS_DIR.exists():
        return []
    return sorted(p.stem for p in _PRESETS_DIR.glob("*.json"))


def get_schema_path() -> Path:
    """Path to the embodiment JSON schema (for jsonschema validation +
    VSCode hookup)."""
    return _SCHEMA_PATH


__all__ = [
    "EmbodimentConfig",
    "list_presets",
    "get_schema_path",
]
