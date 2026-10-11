"""Tether Guard — runtime safety constraints for VLA actions.

Validates robot actions against configurable safety bounds before execution.
Clamps or rejects unsafe actions. Logs every inference for EU AI Act compliance.

Usage:
    from tether.safety import ActionGuard
    guard = ActionGuard.from_urdf("robot.urdf")
    safe_actions = guard.check(raw_actions)

Or via CLI:
    tether guard ./tether_export/ --urdf robot.urdf --port 8001
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# What the action values physically are. Decides what "stop" means: in a
# position space a stop is "hold the current target" (an all-zero target is a
# move to the zero pose); in velocity/effort spaces zero IS a stop.
ACTION_SPACES = ("position", "velocity", "effort")

_GRIPPER_UNITS = {"normalized", "gripper_0_1", "gripper_0_100"}
UNITS_BY_SPACE: dict[str, frozenset[str]] = {
    "position": frozenset({"radians", "degrees", "meters"} | _GRIPPER_UNITS),
    "velocity": frozenset(
        {"radians_per_second", "degrees_per_second", "meters_per_second"} | _GRIPPER_UNITS
    ),
    "effort": frozenset({"newton_meters", "newtons"} | _GRIPPER_UNITS),
}


class GuardError(RuntimeError):
    """Base class for ActionGuard refusals. Callers must not execute actions."""


class GuardUnitsError(GuardError, ValueError):
    """Limits or inputs have unknown/mismatched units; the guard refuses to clamp."""


class GuardStop(GuardError):
    """No safe target exists (e.g. NaN on the first chunk of a position space).

    The caller must stop motion: send no new target to the robot.
    """


def check_safety_units_against_embodiment(limits: SafetyLimits, embodiment_config: Any) -> None:
    """Raise GuardUnitsError when a safety config's units disagree with the
    action units an embodiment config declares (no-op when either is absent)."""
    if embodiment_config is None:
        return
    emb_units = list((getattr(embodiment_config, "action_space", None) or {}).get("units") or [])
    n = min(len(emb_units), len(limits.units))
    if n and list(limits.units)[:n] != emb_units[:n]:
        raise GuardUnitsError(
            f"safety-config units {list(limits.units)} do not match embodiment "
            f"action units {emb_units}: refusing to clamp"
        )


def validate_action_units(spec: Any, *, max_dim: int | None = None) -> dict[str, Any]:
    """Validate an ``action_units`` spec: ``{"kind": <space>, "units": [...]}``.

    Returns a normalized copy; raises GuardUnitsError when it is malformed.
    """
    if not isinstance(spec, dict):
        raise GuardUnitsError(f"action_units must be an object, got {spec!r}")
    kind = spec.get("kind")
    if kind not in ACTION_SPACES:
        raise GuardUnitsError(f"action_units.kind must be one of {ACTION_SPACES}, got {kind!r}")
    units = spec.get("units")
    if not isinstance(units, list) or not units or not all(isinstance(u, str) for u in units):
        raise GuardUnitsError("action_units.units must be a non-empty list of strings")
    bad = sorted({u for u in units if u not in UNITS_BY_SPACE[kind]})
    if bad:
        raise GuardUnitsError(
            f"action_units.units {bad} are not valid for a {kind} action space "
            f"(allowed: {sorted(UNITS_BY_SPACE[kind])})"
        )
    if max_dim is not None and len(units) > max_dim:
        raise GuardUnitsError(
            f"action_units has {len(units)} entries but the export's action_dim is {max_dim}"
        )
    return {"kind": kind, "units": list(units)}


def parse_action_units(text: str, kind: str = "position") -> dict[str, Any]:
    """Parse a CLI units list: comma-separated, ``unit*N`` repeats.

    Example: ``"radians*5,gripper_0_1"``.
    """
    units: list[str] = []
    for token in text.split(","):
        name, _, count = token.strip().partition("*")
        if not name.strip():
            raise GuardUnitsError(f"empty unit in {text!r}")
        try:
            repeat = int(count) if count else 1
        except ValueError as exc:
            raise GuardUnitsError(f"bad repeat count in {token!r}") from exc
        units.extend([name.strip()] * repeat)
    return validate_action_units({"kind": kind, "units": units})


def _require_same_units(label: str, spec: dict[str, Any], kind: str, units: list[str]) -> None:
    if spec["kind"] != kind:
        raise GuardUnitsError(
            f"policy action space {spec['kind']!r} does not match {label} action space {kind!r}"
        )
    n = min(len(spec["units"]), len(units))
    if spec["units"][:n] != list(units)[:n]:
        raise GuardUnitsError(
            f"policy action units {spec['units']} do not match {label} units "
            f"{list(units)}: refusing to serve"
        )


def resolve_serve_action_units(
    *,
    export_config: Any,
    explicit: dict[str, Any] | None,
    safety_limits: SafetyLimits | None,
    embodiment_config: Any,
) -> dict[str, Any] | None:
    """Decide the policy's action units at serve time and check every guard.

    - The export's ``action_units`` (written at export time) and an
      operator-supplied ``explicit`` spec must agree when both exist.
    - When a guard is configured (safety config or embodiment) and neither
      source declares units (an export from before action_units existed),
      raise: the operator must pass units explicitly. No default is assumed.
    - The resolved units must match the safety config and embodiment units.
    """
    exported_raw = export_config.get("action_units") if export_config else None
    exported = (
        validate_action_units(exported_raw, max_dim=export_config.get("action_dim"))
        if exported_raw is not None
        else None
    )
    if explicit is not None:
        explicit = validate_action_units(explicit)
    if exported is not None and explicit is not None and exported != explicit:
        raise GuardUnitsError(
            f"--action-units {explicit} contradict the export's action_units {exported}"
        )
    resolved = exported or explicit
    if safety_limits is None and embodiment_config is None:
        return resolved
    if resolved is None:
        raise GuardUnitsError(
            "this export has no action_units (exported before units were recorded); "
            "pass --action-units (and --action-space if not position) so the guard "
            "knows what the policy's actions are in"
        )
    if safety_limits is not None:
        _require_same_units(
            "safety-config", resolved, safety_limits.action_space, safety_limits.units
        )
    if embodiment_config is not None:
        space = getattr(embodiment_config, "action_space", None) or {}
        emb_units = list(space.get("units") or [])
        if not emb_units:
            raise GuardUnitsError("embodiment config has no action_space.units")
        _require_same_units(
            "embodiment", resolved, str(space.get("kind", "position")), emb_units
        )
    return resolved


def guard_stop_response(exc: BaseException) -> dict[str, Any]:
    """Standard serving envelope for a guard refusal: an error with no actions."""
    return {
        "error": "guard_stop",
        "reason": f"{type(exc).__name__}: {exc}",
        "hint": "No safe action was produced; do not move the robot. "
                "Fix the upstream cause (model output, units, config).",
    }


@dataclass
class SafetyLimits:
    """Per-joint safety limits."""

    joint_names: list[str] = field(default_factory=list)
    position_min: list[float] = field(default_factory=list)
    position_max: list[float] = field(default_factory=list)
    velocity_max: list[float] = field(default_factory=list)
    effort_max: list[float] = field(default_factory=list)
    workspace_min: list[float] = field(default_factory=lambda: [-1.0, -1.0, 0.0])
    workspace_max: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.5])
    workspace_indices: list[int] = field(default_factory=list)
    # What the action values are ("position" | "velocity" | "effort") and
    # their per-axis units (see UNITS_BY_SPACE). ActionGuard refuses limits
    # whose units are unknown, so a degree action is never "clamped" against
    # radian limits.
    action_space: str = "position"
    units: list[str] = field(default_factory=list)

    @classmethod
    def from_urdf(cls, urdf_path: str | Path) -> SafetyLimits:
        """Extract safety limits from a URDF file."""
        try:
            import yourdfpy

            urdf = yourdfpy.URDF.load(str(urdf_path))
            names, pos_min, pos_max, vel_max, eff_max, units = [], [], [], [], [], []

            for joint_name, joint in urdf.joint_map.items():
                if joint.type in ("revolute", "prismatic"):
                    names.append(joint_name)
                    units.append("radians" if joint.type == "revolute" else "meters")
                    if joint.limit is not None:
                        pos_min.append(joint.limit.lower)
                        pos_max.append(joint.limit.upper)
                        vel_max.append(joint.limit.velocity if joint.limit.velocity else 3.14)
                        eff_max.append(joint.limit.effort if joint.limit.effort else 100.0)
                    else:
                        pos_min.append(-3.14)
                        pos_max.append(3.14)
                        vel_max.append(3.14)
                        eff_max.append(100.0)

            return cls(
                joint_names=names,
                position_min=pos_min,
                position_max=pos_max,
                velocity_max=vel_max,
                effort_max=eff_max,
                units=units,
            )
        except ImportError:
            logger.warning("yourdfpy not installed. Install with: pip install 'fastcrest-tether[safety]'")
            return cls()

    @classmethod
    def from_json(cls, path: str | Path) -> SafetyLimits:
        """Load limits from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(**data)

    @classmethod
    def default(cls, num_joints: int = 6) -> SafetyLimits:
        """Reasonable defaults for a 6-DOF robot arm."""
        return cls(
            joint_names=[f"joint_{i}" for i in range(num_joints)],
            position_min=[-3.14] * num_joints,
            position_max=[3.14] * num_joints,
            velocity_max=[2.0] * num_joints,
            effort_max=[50.0] * num_joints,
            units=["radians"] * num_joints,
        )

    @classmethod
    def from_embodiment_config(cls, cfg: Any) -> SafetyLimits:
        """Build SafetyLimits from a per-embodiment config (B.1 + B.6).

        Maps `action_space.ranges` → position_min/max (per-axis joint limits)
        and `constraints.max_ee_velocity` / `max_gripper_velocity` → velocity_max
        (broadcast across joints; gripper dim gets the gripper velocity cap).

        Effort/torque limits aren't in the embodiment config schema yet —
        defaulted to 50 N·m per joint pending B.6 v2 (never compared against
        position actions; see ActionGuard).

        `action_space.kind` (default "position") and `action_space.units`
        (per-dim) carry over; missing units make ActionGuard refuse the config.
        """
        action_space = cfg.action_space
        kind = str(action_space.get("kind", "position"))
        if kind != "position":
            raise ValueError(
                f"embodiment action_space.kind={kind!r}: only 'position' embodiment "
                "configs are supported; use an explicit SafetyLimits JSON for "
                "velocity/effort action spaces"
            )
        ranges = action_space["ranges"]
        action_dim = int(action_space["dim"])
        constraints = cfg.constraints

        max_ee_vel = float(constraints["max_ee_velocity"])
        # max_gripper_velocity is only present for embodiments with a gripper
        # (arms). For drones and other gripper-less embodiments, broadcast
        # max_ee_velocity across all action dims. gripper_idx = -1 ensures
        # no axis matches the gripper-specific path below.
        if cfg.has_gripper:
            gripper_idx = cfg.gripper_idx
            max_gripper_vel = float(constraints["max_gripper_velocity"])
        else:
            gripper_idx = -1
            max_gripper_vel = max_ee_vel  # unused when gripper_idx == -1

        position_min = [float(r[0]) for r in ranges]
        position_max = [float(r[1]) for r in ranges]
        velocity_max = [
            max_gripper_vel if i == gripper_idx else max_ee_vel
            for i in range(action_dim)
        ]
        effort_max = [50.0] * action_dim  # default until B.6 v2 adds torque to schema

        return cls(
            joint_names=[f"joint_{i}" for i in range(action_dim)],
            position_min=position_min,
            position_max=position_max,
            velocity_max=velocity_max,
            effort_max=effort_max,
            action_space=kind,
            units=[str(u) for u in action_space.get("units", [])],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


@dataclass
class SafetyCheckResult:
    """Result of a safety check on a single action."""

    safe: bool
    violations: list[str]
    clamped: bool
    original_action: list[float]
    safe_action: list[float]
    check_time_ms: float
    # True when safe_action is a hold/stop target rather than a clamped
    # version of the input (reject mode or non-finite input).
    hold: bool = False


@dataclass
class InferenceLog:
    """EU AI Act Article 12 compliant inference record."""

    timestamp: str
    input_hash: str
    actions_raw: list[list[float]]
    actions_safe: list[list[float]]
    violations: list[str]
    clamped: bool
    model_version: str
    latency_ms: float


class ActionGuard:
    """Runtime safety layer for VLA action outputs."""

    def __init__(
        self,
        limits: SafetyLimits,
        mode: str = "clamp",
        log_dir: str | Path | None = None,
        model_version: str = "unknown",
        max_consecutive_clamps: int = 10,
    ):
        """
        Args:
            limits: Safety limits for joints and workspace. ``limits.units``
                must name a known unit for every axis of
                ``limits.action_space``; otherwise GuardUnitsError is raised
                here (the guard refuses to clamp values whose units it does
                not know).
            mode: "clamp" (adjust to nearest safe value) or "reject" (do not
                execute the chunk; hold/stop instead). Reject and non-finite
                inputs never synthesize a new target: position spaces hold the
                observed position or the last emitted safe action (or raise
                GuardStop when neither exists); velocity/effort spaces return
                zeros, which is a stop there.
            log_dir: Directory for EU AI Act compliance logs (None = no logging)
            model_version: Model identifier for audit trail
            max_consecutive_clamps: staleness kill-switch. After N consecutive
                chunks that required clamping or contained NaN/Inf, the guard
                "trips" — `tripped` becomes True and callers (e.g. `tether
                serve`) should stop serving actions until `reset()` is called.
                Set to 0 to disable. This protects against stale or runaway
                policies that keep emitting invalid actions — e.g. a model
                that's producing NaN due to numerical divergence, or a
                degenerate output mode where every chunk hits clamp limits.
        """
        if mode not in ("clamp", "reject"):
            raise ValueError(f"mode must be 'clamp' or 'reject', got {mode!r}")
        self.limits = limits
        self.mode = mode
        self._validate_space_and_units()
        self.model_version = model_version
        self._log_dir = Path(log_dir) if log_dir else None
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        self._inference_count = 0
        self.max_consecutive_clamps = max_consecutive_clamps
        self._consecutive_clamps = 0
        self._tripped = False
        self._trip_reason: str | None = None
        # Last row this guard emitted: the hold target when no observation
        # is supplied. Cleared by reset().
        self._last_safe_action: np.ndarray | None = None

    @property
    def action_space(self) -> str:
        return self.limits.action_space

    def _validate_space_and_units(self) -> None:
        space = self.limits.action_space
        if space not in ACTION_SPACES:
            raise ValueError(
                f"SafetyLimits.action_space must be one of {ACTION_SPACES}, got {space!r}"
            )
        axes = {
            "position": self.limits.position_max,
            "velocity": self.limits.velocity_max,
            "effort": self.limits.effort_max,
        }[space]
        units = list(self.limits.units or [])
        if not units:
            raise GuardUnitsError(
                "SafetyLimits.units is missing: refusing to clamp actions whose "
                f"units are unknown. Add a per-axis \"units\" list (allowed for "
                f"{space}: {sorted(UNITS_BY_SPACE[space])})."
            )
        if len(units) != len(axes):
            raise GuardUnitsError(
                f"SafetyLimits.units has {len(units)} entries but the {space} "
                f"limits cover {len(axes)} axes"
            )
        bad = sorted({u for u in units if u not in UNITS_BY_SPACE[space]})
        if bad:
            raise GuardUnitsError(
                f"units {bad} are not valid for a {space} action space "
                f"(allowed: {sorted(UNITS_BY_SPACE[space])})"
            )

    def _check_input_units(self, units: list[str] | tuple[str, ...] | None) -> None:
        """Refuse inputs whose declared units differ from the limits' units."""
        if units is None:
            return
        if list(units) != list(self.limits.units):
            raise GuardUnitsError(
                f"action units {list(units)} do not match safety-limit units "
                f"{list(self.limits.units)}: refusing to clamp"
            )

    @classmethod
    def from_urdf(cls, urdf_path: str | Path, **kwargs) -> ActionGuard:
        limits = SafetyLimits.from_urdf(urdf_path)
        return cls(limits=limits, **kwargs)

    @classmethod
    def default(cls, num_joints: int = 6, **kwargs) -> ActionGuard:
        limits = SafetyLimits.default(num_joints)
        return cls(limits=limits, **kwargs)

    @classmethod
    def from_embodiment_config(cls, cfg: Any, **kwargs) -> ActionGuard:
        """Build an ActionGuard from a per-embodiment config (B.6).

        The everyday safety path — uses the lightweight per-axis ranges +
        velocity caps from configs/embodiments/*.json. No URDF required.
        For the full URDF physics path, use from_urdf() instead.
        """
        limits = SafetyLimits.from_embodiment_config(cfg)
        return cls(limits=limits, **kwargs)

    @staticmethod
    def _clamp_value(value: float, lower: float, upper: float) -> float:
        return float(min(max(value, lower), upper))

    @staticmethod
    def _interval_margin(value: float, lower: float, upper: float) -> float | None:
        """Normalized distance to the nearest interval boundary.

        Returns 1.0 at interval center, 0.0 at or outside a boundary, and None
        for invalid intervals.
        """
        if upper <= lower:
            return None
        half_span = (upper - lower) / 2.0
        if half_span <= 0:
            return None
        distance = min(value - lower, upper - value)
        return float(min(max(distance / half_span, 0.0), 1.0))

    @staticmethod
    def _magnitude_margin(value: float, limit: float) -> float | None:
        if limit <= 0:
            return None
        return float(min(max((limit - abs(value)) / limit, 0.0), 1.0))

    def safety_margin(self, actions: np.ndarray) -> float | None:
        """Return the closest normalized margin to any configured limit.

        The value is in [0, 1], where 0 means at/outside a configured safety
        boundary and 1 means centered under every applicable bound. Only the
        limits that apply to ``limits.action_space`` count: position bounds,
        per-step deltas against velocity_max and workspace bounds for position
        spaces; |value| against velocity_max / effort_max for velocity / effort
        spaces. Returns None when no comparable limits apply.
        """
        arr = np.asarray(actions, dtype=np.float32)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or not np.isfinite(arr).all():
            return 0.0

        space = self.limits.action_space
        margins: list[float] = []
        for row in arr:
            if space == "position":
                num_joints = min(arr.shape[1], len(self.limits.position_max))
                for i in range(num_joints):
                    m = self._interval_margin(
                        float(row[i]),
                        float(self.limits.position_min[i]),
                        float(self.limits.position_max[i]),
                    )
                    if m is not None:
                        margins.append(m)
                for workspace_axis, action_idx in enumerate(self.limits.workspace_indices):
                    if action_idx < 0 or action_idx >= arr.shape[1]:
                        continue
                    if (
                        workspace_axis >= len(self.limits.workspace_min)
                        or workspace_axis >= len(self.limits.workspace_max)
                    ):
                        continue
                    m = self._interval_margin(
                        float(row[action_idx]),
                        float(self.limits.workspace_min[workspace_axis]),
                        float(self.limits.workspace_max[workspace_axis]),
                    )
                    if m is not None:
                        margins.append(m)
            else:
                bound = self.limits.velocity_max if space == "velocity" else self.limits.effort_max
                for i in range(min(arr.shape[1], len(bound))):
                    m = self._magnitude_margin(float(row[i]), float(bound[i]))
                    if m is not None:
                        margins.append(m)

        if space == "position" and arr.shape[0] >= 2:
            num_velocity = min(arr.shape[1], len(self.limits.velocity_max))
            for prev, cur in zip(arr[:-1], arr[1:]):
                for i in range(num_velocity):
                    m = self._magnitude_margin(
                        float(cur[i] - prev[i]), float(self.limits.velocity_max[i])
                    )
                    if m is not None:
                        margins.append(m)

        if not margins:
            return None
        return min(margins)

    def _evaluate(
        self,
        action: np.ndarray,
        previous_action: np.ndarray | None,
    ) -> tuple[list[str], bool, np.ndarray]:
        """Find violations; in clamp mode also return the clamped action."""
        violations: list[str] = []
        clamped = False
        safe_action = np.array(action, copy=True)
        clamp = self.mode == "clamp"
        space = self.limits.action_space

        if space != "position":
            # Velocity/effort spaces: the values themselves are bounded.
            name = "velocity" if space == "velocity" else "effort"
            bound = self.limits.velocity_max if space == "velocity" else self.limits.effort_max
            for i in range(min(len(safe_action), len(bound))):
                limit = float(bound[i])
                if limit > 0 and abs(float(safe_action[i])) > limit:
                    violations.append(
                        f"joint_{i} {name} limit: |{safe_action[i]:.3f}| > {limit:.3f}"
                    )
                    if clamp:
                        safe_action[i] = self._clamp_value(float(safe_action[i]), -limit, limit)
                        clamped = True
            return violations, clamped, safe_action

        # Position space. effort_max is NOT compared here: a joint position is
        # not a torque.
        num_joints = min(len(action), len(self.limits.position_max))
        for i in range(num_joints):
            if safe_action[i] < self.limits.position_min[i]:
                violations.append(
                    f"joint_{i} below min: "
                    f"{safe_action[i]:.3f} < {self.limits.position_min[i]:.3f}"
                )
                if clamp:
                    safe_action[i] = self.limits.position_min[i]
                    clamped = True
            elif safe_action[i] > self.limits.position_max[i]:
                violations.append(
                    f"joint_{i} above max: "
                    f"{safe_action[i]:.3f} > {self.limits.position_max[i]:.3f}"
                )
                if clamp:
                    safe_action[i] = self.limits.position_max[i]
                    clamped = True

        # Compose the velocity constraint with the position bounds. The caller
        # executes ``safe_action``, so its delta from the previously emitted
        # action must remain within the configured velocity limit even when
        # this joint also needed a static clamp.
        if previous_action is not None:
            num_velocity = min(
                len(safe_action),
                len(previous_action),
                len(self.limits.velocity_max),
            )
            for i in range(num_velocity):
                velocity_limit = self.limits.velocity_max[i]
                delta = float(safe_action[i] - previous_action[i])
                if velocity_limit > 0 and abs(delta) > velocity_limit:
                    violations.append(
                        f"joint_{i} velocity limit: "
                        f"|delta {delta:.3f}| > {velocity_limit:.3f}"
                    )
                    if clamp:
                        safe_action[i] = float(previous_action[i]) + self._clamp_value(
                            delta, -velocity_limit, velocity_limit
                        )
                        clamped = True

        for workspace_axis, action_idx in enumerate(self.limits.workspace_indices):
            if action_idx < 0 or action_idx >= len(safe_action):
                continue
            if (
                workspace_axis >= len(self.limits.workspace_min)
                or workspace_axis >= len(self.limits.workspace_max)
            ):
                continue
            lower = self.limits.workspace_min[workspace_axis]
            upper = self.limits.workspace_max[workspace_axis]
            if safe_action[action_idx] < lower:
                violations.append(
                    f"workspace_axis_{workspace_axis} below min: "
                    f"action[{action_idx}]={safe_action[action_idx]:.3f} < {lower:.3f}"
                )
                if clamp:
                    safe_action[action_idx] = lower
                    clamped = True
            elif safe_action[action_idx] > upper:
                violations.append(
                    f"workspace_axis_{workspace_axis} above max: "
                    f"action[{action_idx}]={safe_action[action_idx]:.3f} > {upper:.3f}"
                )
                if clamp:
                    safe_action[action_idx] = upper
                    clamped = True

        return violations, clamped, safe_action

    def _hold_target(self, like: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
        """The action that stops motion without synthesizing a new target.

        Velocity/effort: zeros. Position: ``reference`` (an observed position
        or the last emitted safe action). Raises GuardStop when a position
        space has no reference: the caller must send no new target.
        """
        if self.limits.action_space != "position":
            return np.zeros_like(like)
        if reference is None:
            raise GuardStop(
                "position action space has no safe hold target (no observed "
                "position and no previously emitted safe action)"
            )
        ref = np.asarray(reference, dtype=like.dtype).reshape(-1)
        if ref.shape != like.shape or not np.isfinite(ref).all():
            raise GuardStop(
                f"hold reference shape {ref.shape} / finiteness does not match "
                f"action shape {like.shape}"
            )
        return ref.copy()

    def check_single(
        self,
        action: np.ndarray,
        *,
        previous_action: np.ndarray | None = None,
        units: list[str] | None = None,
    ) -> SafetyCheckResult:
        """Check a single action vector against safety limits.

        Position and workspace bounds are single-action checks. Velocity (in a
        position space) is a delta check and only runs when the caller
        provides ``previous_action``. In reject mode a violating action is
        replaced by the hold target (``previous_action`` for position spaces,
        zeros for velocity/effort); GuardStop is raised when a position space
        has no previous action.
        """
        start = time.perf_counter()
        self._check_input_units(units)
        action = np.asarray(action)
        violations, clamped, safe_action = self._evaluate(action, previous_action)
        hold = False
        if self.mode == "reject" and violations:
            safe_action = self._hold_target(action, previous_action)
            hold = True

        elapsed = (time.perf_counter() - start) * 1000

        return SafetyCheckResult(
            safe=len(violations) == 0,
            violations=violations,
            clamped=clamped,
            original_action=action.tolist(),
            safe_action=safe_action.tolist(),
            check_time_ms=elapsed,
            hold=hold,
        )

    def check(
        self,
        actions: np.ndarray,
        *,
        observed_position: np.ndarray | None = None,
        units: list[str] | None = None,
    ) -> tuple[np.ndarray, list[SafetyCheckResult]]:
        """Check a batch of actions (action chunk).

        Args:
            actions: [chunk_size, action_dim] array
            observed_position: optional current measured position (position
                spaces). Preferred hold target over the last emitted action.
            units: optional per-axis units of ``actions``. When given and
                different from ``limits.units`` the guard raises
                GuardUnitsError instead of clamping.

        Returns:
            (safe_actions, results). When the chunk is a hold/stop, every row
            of safe_actions is the hold target and every result has
            ``hold=True``.

        Raises:
            GuardStop: position space, the chunk must not run (NaN/Inf, or a
                violation in reject mode) and no hold reference exists. The
                caller must stop motion.
            GuardUnitsError: declared input units differ from the limits'.

        Non-finite handling: any NaN or Inf in the input is a hard reject of
        the whole chunk (a single violation record, not per-joint) in both
        modes. It counts as a "clamp event" for the staleness kill-switch.
        """
        self._check_input_units(units)
        actions = np.asarray(actions)
        reference = (
            np.asarray(observed_position).reshape(-1)
            if observed_position is not None
            else self._last_safe_action
        )
        results: list[SafetyCheckResult] = []
        non_finite_mask = ~np.isfinite(actions)
        had_non_finite = bool(non_finite_mask.any())
        hold_reason: str | None = None

        if had_non_finite:
            num_bad = int(non_finite_mask.sum())
            hold_reason = (
                f"non_finite_action: {num_bad} NaN/Inf value(s) detected — "
                f"chunk not executed, holding"
            )
            all_violations = [hold_reason]
            chunk_clamped = True
        else:
            safe_actions = np.array(actions, copy=True)
            previous_safe: np.ndarray | None = None
            for i in range(len(actions)):
                row_start = time.perf_counter()
                violations, clamped, safe_row = self._evaluate(actions[i], previous_safe)
                results.append(
                    SafetyCheckResult(
                        safe=not violations,
                        violations=violations,
                        clamped=clamped,
                        original_action=actions[i].tolist(),
                        safe_action=safe_row.tolist(),
                        check_time_ms=(time.perf_counter() - row_start) * 1000,
                    )
                )
                safe_actions[i] = safe_row
                # The returned action is what the caller will execute.
                previous_safe = safe_actions[i]
            all_violations = [v for r in results for v in r.violations]
            chunk_clamped = any(r.clamped for r in results)
            if self.mode == "reject" and all_violations:
                hold_reason = "reject_mode: chunk had violations — not executed, holding"

        if hold_reason is not None:
            row_like = actions[0] if len(actions) else np.zeros(0, dtype=actions.dtype)
            try:
                target = self._hold_target(row_like, reference)
            except GuardStop:
                if self._log_dir:
                    self._log_inference(actions, np.zeros((0,)), all_violations, True)
                self._count_clamp_event(True)
                self._inference_count += 1
                raise
            safe_actions = np.broadcast_to(target, actions.shape).astype(actions.dtype, copy=True)
            if had_non_finite:
                results = [
                    SafetyCheckResult(
                        safe=False,
                        violations=[hold_reason],
                        clamped=True,
                        original_action=actions[0].tolist() if len(actions) else [],
                        safe_action=target.tolist(),
                        check_time_ms=0.0,
                        hold=True,
                    )
                ]
            else:
                for r in results:
                    r.safe_action = target.tolist()
                    r.hold = True

        if len(safe_actions):
            self._last_safe_action = np.array(safe_actions[-1], copy=True)

        if self._log_dir:
            self._log_inference(actions, safe_actions, all_violations, chunk_clamped)

        self._count_clamp_event(chunk_clamped or hold_reason is not None)
        self._inference_count += 1
        return safe_actions, results

    def _count_clamp_event(self, clamped: bool) -> None:
        """Staleness kill-switch — trip after N consecutive clamp/NaN chunks."""
        if self.max_consecutive_clamps <= 0:
            return
        if not clamped:
            self._consecutive_clamps = 0
            return
        self._consecutive_clamps += 1
        if self._consecutive_clamps >= self.max_consecutive_clamps and not self._tripped:
            self._tripped = True
            self._trip_reason = (
                f"consecutive_clamp_limit_exceeded: "
                f"{self._consecutive_clamps} chunks in a row required "
                f"clamping or contained NaN/Inf (limit "
                f"{self.max_consecutive_clamps})"
            )
            logger.error(self._trip_reason)

    @property
    def tripped(self) -> bool:
        """True when the consecutive-clamp kill-switch has fired.

        Callers (e.g. `tether serve`) should stop serving actions and raise a
        loud error until `reset()` is called.
        """
        return self._tripped

    @property
    def trip_reason(self) -> str | None:
        """Human-readable reason the guard tripped, or None if not tripped."""
        return self._trip_reason

    @property
    def consecutive_clamps(self) -> int:
        """Current count of consecutive clamped or NaN/Inf-rejected chunks."""
        return self._consecutive_clamps

    def reset(self) -> None:
        """Clear the tripped state and consecutive-clamp counter.

        Call after investigating the upstream cause (bad inputs, model drift,
        sensor failure, etc.) and confirming it's safe to resume.
        """
        self._tripped = False
        self._trip_reason = None
        self._consecutive_clamps = 0
        # Forget the hold target: after an investigation the next NaN/reject
        # must not hold a pose from before the reset.
        self._last_safe_action = None

    def _log_inference(
        self,
        raw_actions: np.ndarray,
        safe_actions: np.ndarray,
        violations: list[str],
        clamped: bool,
    ) -> None:
        """Log inference for EU AI Act Article 12 compliance."""
        import hashlib

        input_hash = hashlib.sha256(raw_actions.tobytes()).hexdigest()[:16]

        log_entry = InferenceLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash=input_hash,
            actions_raw=raw_actions.tolist(),
            actions_safe=safe_actions.tolist(),
            violations=violations,
            clamped=clamped,
            model_version=self.model_version,
            latency_ms=0.0,
        )

        log_file = self._log_dir / f"inference_log_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(log_entry)) + "\n")

    @property
    def inference_count(self) -> int:
        return self._inference_count
