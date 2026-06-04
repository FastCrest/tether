"""SOARM100Adapter — first-class SO-ARM100 embodiment adapter.

Reflex's "embodiment" surface is split across two co-existing layers:

  1. `reflex.embodiments.EmbodimentConfig` — frozen-dataclass preset (JSON-
     backed) consumed by the serve runtime, action guard, and validation.
  2. `reflex.models.adapt.EmbodimentAdapter` — URDF-derived cross-embodiment
     remapping helper.

`SOARM100Adapter` is the SO-ARM100 specialization that:
  - composes (1) the bundled `so100.json` preset (action ranges + normalization)
  - PLUS (2) physical SOARM100Config (per-joint calibration, gripper toggle,
    serial protocol)
  - PLUS (3) the LeRobot bridge (action conversion math)

It implements a minimal informal protocol so the export / verify / serve
plumbing can call it uniformly:

    adapter.to_embodiment_config() -> EmbodimentConfig    (for the serve runtime)
    adapter.calibration_dict()      -> dict                (for the export bundle)
    adapter.action_to_servo_commands(action) -> list      (for the serve loop)
    adapter.state_from_servo(present_pose) -> np.ndarray  (for /act request building)
    adapter.connect(port?)          -> hw handle           (for serve --port)
    adapter.save_calibration(path)  -> None                (for export bundle write)

Importing SOARM100Adapter is cheap — the heavy hardware-side bits (scservo_sdk,
lerobot.motors.feetech) are lazy-imported only when the user actually opens a
serial port. CI / mac-dev hosts can construct an adapter, validate, run unit
tests, etc., without any hardware deps installed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tether.embodiments.so_arm100.config import (
    GRIPPER_INDEX,
    SO_ARM100_JOINT_NAMES,
    SOARM100Config,
)
from tether.embodiments.so_arm100.lerobot_bridge import (
    chunk_to_servo_command_stream,
    lerobot_action_to_servo_commands,
    servo_state_to_lerobot_state,
)

logger = logging.getLogger(__name__)

# Canonical preset name (matches the file at
# `reflex.embodiments.presets/so100.json`). The "so100" preset is the same
# physical arm as the "so_arm100" adapter — the preset uses the shorter slug
# for backward compatibility with v0.x users who already have it in scripts.
PRESET_NAME: str = "so100"


@dataclass
class SOARM100Adapter:
    """The user-facing adapter. See module docstring for the role."""

    config: SOARM100Config
    # Lazy-constructed hardware port. None until `connect()` is called.
    _hw: Any = None

    # ─── construction ────────────────────────────────────────────────────────

    @classmethod
    def default(cls, *, port: str = "/dev/ttyUSB0") -> SOARM100Adapter:
        """Build an adapter with factory-default calibration. Usable for unit
        tests + dry-run flows; SHOULD be replaced with `from_calibration` before
        running on real hardware."""
        return cls(config=SOARM100Config.default(port=port))

    @classmethod
    def from_calibration(
        cls,
        calibration: str | Path | SOARM100Config,
        *,
        port: str = "/dev/ttyUSB0",
        protocol: str = "feetech",
    ) -> SOARM100Adapter:
        """Build an adapter from a LeRobot calibration JSON path OR an
        already-constructed SOARM100Config.

        The path form is what users will hit most often:
            adapter = SOARM100Adapter.from_calibration("calib.json")

        We accept both forms so callers programmatically composing configs
        (notebooks, tests, sim harness) don't have to write the JSON to disk
        just to feed it in.
        """
        if isinstance(calibration, SOARM100Config):
            return cls(config=calibration)
        cfg = SOARM100Config.from_lerobot_calibration(
            str(calibration), port=port, protocol=protocol,
        )
        return cls(config=cfg)

    # ─── EmbodimentAdapter informal protocol ────────────────────────────────

    @property
    def embodiment_name(self) -> str:
        """Slug used in bundle metadata + telemetry. Matches what the user
        passes via `--embodiment so_arm100`."""
        return "so_arm100"

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    @property
    def gripper_idx(self) -> int:
        return GRIPPER_INDEX

    @property
    def joint_names(self) -> tuple[str, ...]:
        return SO_ARM100_JOINT_NAMES

    def to_embodiment_config(self):
        """Return the matching `reflex.embodiments.EmbodimentConfig` preset.

        SO-ARM100 reuses the bundled `so100.json` preset for action ranges +
        normalization stats — these are policy-agnostic and stable. Per-arm
        calibration (homing offsets, range_min/max) lives in this adapter's
        `config` field and is fed separately to the serve loop, NOT smuggled
        into the EmbodimentConfig.

        Returns a `reflex.embodiments.EmbodimentConfig` instance.
        """
        from tether.embodiments import EmbodimentConfig
        return EmbodimentConfig.load_preset(PRESET_NAME)

    # ─── runtime conversion API (what serve calls) ───────────────────────────

    def action_to_servo_commands(
        self,
        action: Sequence[float] | np.ndarray,
        *,
        apply_position_limits: bool = True,
    ) -> list[tuple[int, int]]:
        """Translate a 6-vector LeRobot action into raw servo commands.

        Returns: list of (motor_id, raw_servo_units) in canonical joint order.
        """
        return lerobot_action_to_servo_commands(
            action, self.config,
            apply_position_limits=apply_position_limits,
        )

    def chunk_to_servo_command_stream(
        self,
        action_chunk: np.ndarray,
    ) -> list[list[tuple[int, int]]]:
        """Translate a (T, 6) action chunk to a per-step stream of servo
        commands. Used by `reflex serve --embodiment so_arm100`'s wire loop."""
        return chunk_to_servo_command_stream(action_chunk, self.config)

    def state_from_servo(
        self,
        present_positions: dict[str, int] | Sequence[int],
    ) -> np.ndarray:
        """Build a LeRobot-compatible state vector from `read_pose()` output."""
        return servo_state_to_lerobot_state(present_positions, self.config)

    # ─── hardware lifecycle ─────────────────────────────────────────────────

    def connect(self, port: str | None = None):
        """Open the serial port + initialise the arm.

        Hardware deps (`scservo_sdk` / `lerobot.motors.feetech`) are only
        imported here, so this method raises ImportError when the user runs
        on a host without the SDK installed. Use `pip install
        'reflex-vla[so100]'` (existing extra) to pull `scservo_sdk` on the
        machine wired to the arm.
        """
        if self._hw is not None:
            return self._hw
        target_port = port or self.config.port
        if self.config.protocol == "scservo":
            from tether.embodiments.so100.edge_runtime import SOArmHardware
            self._hw = SOArmHardware(port=target_port, baud=self.config.baud)
        elif self.config.protocol == "feetech":
            self._hw = _FeetechAdapterRuntime(self.config, port=target_port)
            self._hw.connect()
        else:
            raise ValueError(
                f"Unknown protocol {self.config.protocol!r}; "
                f"supported: feetech, scservo"
            )
        return self._hw

    def disconnect(self) -> None:
        if self._hw is None:
            return
        try:
            close = getattr(self._hw, "close", None) or getattr(
                self._hw, "disconnect", None
            )
            if close is not None:
                close()
        finally:
            self._hw = None

    def __enter__(self) -> SOARM100Adapter:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    # ─── export-bundle helpers ──────────────────────────────────────────────

    def calibration_dict(self) -> dict[str, Any]:
        """Return the calibration-bundle dict that gets written into the
        export under `embodiment/so_arm100/calibration.json`.

        Structure:
            {
              "schema_version": 1,
              "embodiment": "so_arm100",
              "source": "<original calib path or empty>",
              "lerobot_calibration": {<same shape as LeRobot JSON>},
              "runtime": {control + gripper + safety fields},
              "joints": {<reflex-side extension fields per joint>}
            }
        """
        return {
            "schema_version": 1,
            "embodiment": self.embodiment_name,
            "source": self.config._source_path,
            "lerobot_calibration": self.config.to_lerobot_calibration(),
            "runtime": {
                "port": self.config.port,
                "baud": self.config.baud,
                "protocol": self.config.protocol,
                "control_frequency_hz": self.config.control_frequency_hz,
                "chunk_size": self.config.chunk_size,
                "rtc_execution_horizon": self.config.rtc_execution_horizon,
                "max_ee_velocity": self.config.max_ee_velocity,
                "max_gripper_velocity": self.config.max_gripper_velocity,
                "gripper_open_servo_units": self.config.gripper_open_servo_units,
                "gripper_closed_servo_units": self.config.gripper_closed_servo_units,
                "gripper_close_threshold": self.config.gripper_close_threshold,
                "gripper_inverted": self.config.gripper_inverted,
            },
            "joints": {
                j.name: {
                    "position_limits": list(j.position_limits),
                    "velocity_limit": j.velocity_limit,
                }
                for j in self.config.joints
            },
        }

    def save_calibration(self, path: str | Path) -> Path:
        """Write the calibration-bundle dict to disk. Parent dirs are created."""
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(self.calibration_dict(), f, indent=2)
        return p

    @classmethod
    def from_bundle(cls, bundle_dir: str | Path) -> SOARM100Adapter:
        """Load an adapter from a previously-exported reflex bundle.

        Looks for `embodiment/so_arm100/calibration.json` under the bundle.
        Raises FileNotFoundError if the bundle wasn't exported with the
        so_arm100 embodiment.
        """
        bundle = Path(bundle_dir).expanduser()
        candidates = [
            bundle / "embodiment" / "so_arm100" / "calibration.json",
            bundle / "so_arm100_calibration.json",
        ]
        for c in candidates:
            if c.exists():
                with c.open() as f:
                    data = json.load(f)
                # Round-trip via SOARM100Config: drop the JSON into a tmp file
                # and use the LeRobot loader path — guarantees the same code
                # path as fresh calibration imports.
                from tether.embodiments.so_arm100.config import (
                    JointConfig,
                    SO_ARM100_JOINT_NAMES,
                )
                lr_cal = data["lerobot_calibration"]
                joints_meta = data.get("joints", {})
                joints = tuple(
                    JointConfig.from_lerobot_calibration(
                        name=name,
                        cal=lr_cal[name],
                        position_limits=tuple(
                            joints_meta.get(name, {}).get(
                                "position_limits",
                                (0.0, 1.0) if name == "gripper" else (-3.14, 3.14),
                            )
                        ),
                        velocity_limit=joints_meta.get(name, {}).get(
                            "velocity_limit", 1.0 if name == "gripper" else 3.14,
                        ),
                    )
                    for name in SO_ARM100_JOINT_NAMES
                )
                runtime = data.get("runtime", {})
                cfg = SOARM100Config(
                    joints=joints,
                    port=runtime.get("port", "/dev/ttyUSB0"),
                    baud=runtime.get("baud", 1_000_000),
                    protocol=runtime.get("protocol", "feetech"),
                    gripper_open_servo_units=runtime.get(
                        "gripper_open_servo_units", 1024,
                    ),
                    gripper_closed_servo_units=runtime.get(
                        "gripper_closed_servo_units", 2400,
                    ),
                    gripper_close_threshold=runtime.get(
                        "gripper_close_threshold", 0.5,
                    ),
                    gripper_inverted=runtime.get("gripper_inverted", False),
                    control_frequency_hz=runtime.get("control_frequency_hz", 15.0),
                    chunk_size=runtime.get("chunk_size", 30),
                    rtc_execution_horizon=runtime.get("rtc_execution_horizon", 12),
                    max_ee_velocity=runtime.get("max_ee_velocity", 0.5),
                    max_gripper_velocity=runtime.get("max_gripper_velocity", 1.0),
                    _source_path=data.get("source", str(c)),
                )
                return cls(config=cfg)
        raise FileNotFoundError(
            f"No SO-ARM100 calibration found in bundle at {bundle}. "
            f"Expected one of: {[str(c) for c in candidates]}. "
            f"Re-export with `reflex export <model> --embodiment so_arm100`."
        )


# ─── feetech runtime (hardware-only) ────────────────────────────────────────


class _FeetechAdapterRuntime:
    """Thin wrapper around LeRobot's FeetechMotorsBus, applying our calibration.

    Public surface matches `SOArmHardware`:
        - read_pose() -> dict[str, int]
        - write_goal(motor_id, raw) -> None
        - close() -> None
    """

    def __init__(self, cfg: SOARM100Config, *, port: str):
        self._cfg = cfg
        self._port = port
        self._bus = None  # lazy

    def connect(self) -> None:
        try:
            from lerobot.motors import Motor, MotorCalibration, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus
        except ImportError as exc:
            raise ImportError(
                "Feetech runtime requires `lerobot>=0.5` installed. "
                "Install via `pip install 'reflex-vla[lerobot]'` on the host "
                "wired to the arm (Python >= 3.12)."
            ) from exc

        motors = {}
        calibration: dict[str, MotorCalibration] = {}
        for j in self._cfg.joints:
            norm_mode = (
                MotorNormMode.RANGE_0_100
                if j.name == "gripper"
                else MotorNormMode.RANGE_M100_100
            )
            motors[j.name] = Motor(j.motor_id, "sts3215", norm_mode)
            calibration[j.name] = MotorCalibration(
                id=j.motor_id,
                drive_mode=j.drive_mode,
                homing_offset=j.homing_offset,
                range_min=j.range_min,
                range_max=j.range_max,
            )
        self._bus = FeetechMotorsBus(
            port=self._port, motors=motors, calibration=calibration,
        )
        self._bus.connect()

    def read_pose(self) -> dict[str, int]:
        if self._bus is None:
            raise RuntimeError("Bus not connected; call connect() first.")
        # FeetechMotorsBus.sync_read returns name → present-position; cast to int.
        raw = self._bus.sync_read("Present_Position")
        return {name: int(v) for name, v in raw.items()}

    def write_goal(self, motor_id: int, raw: int) -> None:
        if self._bus is None:
            raise RuntimeError("Bus not connected; call connect() first.")
        for j in self._cfg.joints:
            if j.motor_id == motor_id:
                self._bus.write("Goal_Position", j.name, int(raw))
                return
        raise KeyError(f"No joint maps to motor_id={motor_id}")

    def write_goals(self, commands: Sequence[tuple[int, int]]) -> None:
        """Batch goal-position writes. Lower wire overhead than per-motor."""
        if self._bus is None:
            raise RuntimeError("Bus not connected; call connect() first.")
        by_name: dict[str, int] = {}
        for mid, raw in commands:
            for j in self._cfg.joints:
                if j.motor_id == mid:
                    by_name[j.name] = int(raw)
                    break
        self._bus.sync_write("Goal_Position", by_name)

    def close(self) -> None:
        if self._bus is None:
            return
        try:
            self._bus.disconnect()
        except Exception:  # noqa: BLE001 — never block teardown on driver error
            logger.warning("FeetechMotorsBus disconnect raised; ignoring.")
        self._bus = None


__all__ = ["SOARM100Adapter"]
