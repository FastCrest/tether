"""Reflex Guard — runtime safety constraints for VLA actions.

Validates robot actions against configurable safety bounds before execution.
Clamps or rejects unsafe actions. Logs every inference for EU AI Act compliance.

Usage:
    from reflex.safety import ActionGuard
    guard = ActionGuard.from_urdf("robot.urdf")
    safe_actions = guard.check(raw_actions)

Or via CLI:
    reflex guard ./reflex_export/ --urdf robot.urdf --port 8001
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

    @classmethod
    def from_urdf(cls, urdf_path: str | Path) -> SafetyLimits:
        """Extract safety limits from a URDF file."""
        try:
            import yourdfpy

            urdf = yourdfpy.URDF.load(str(urdf_path))
            names, pos_min, pos_max, vel_max, eff_max = [], [], [], [], []

            for joint_name, joint in urdf.joint_map.items():
                if joint.type in ("revolute", "prismatic"):
                    names.append(joint_name)
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
            )
        except ImportError:
            logger.warning("yourdfpy not installed. Install with: pip install 'reflex-vla[safety]'")
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
        )

    @classmethod
    def from_embodiment_config(cls, cfg: Any) -> SafetyLimits:
        """Build SafetyLimits from a per-embodiment config (B.1 + B.6).

        Maps `action_space.ranges` → position_min/max (per-axis joint limits)
        and `constraints.max_ee_velocity` / `max_gripper_velocity` → velocity_max
        (broadcast across joints; gripper dim gets the gripper velocity cap).

        Effort/torque limits aren't in the embodiment config schema yet —
        defaulted to 50 N·m per joint pending B.6 v2.
        """
        action_space = cfg.action_space
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
            limits: Safety limits for joints and workspace
            mode: "clamp" (adjust to nearest safe value) or "reject" (return zeros)
            log_dir: Directory for EU AI Act compliance logs (None = no logging)
            model_version: Model identifier for audit trail
            max_consecutive_clamps: staleness kill-switch. After N consecutive
                chunks that required clamping or contained NaN/Inf, the guard
                "trips" — `tripped` becomes True and callers (e.g. `reflex
                serve`) should stop serving actions until `reset()` is called.
                Set to 0 to disable. This protects against stale or runaway
                policies that keep emitting invalid actions — e.g. a model
                that's producing NaN due to numerical divergence, or a
                degenerate output mode where every chunk hits clamp limits.
        """
        self.limits = limits
        self.mode = mode
        self.model_version = model_version
        self._log_dir = Path(log_dir) if log_dir else None
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        self._inference_count = 0
        self.max_consecutive_clamps = max_consecutive_clamps
        self._consecutive_clamps = 0
        self._tripped = False
        self._trip_reason: str | None = None

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

    def check_single(self, action: np.ndarray) -> SafetyCheckResult:
        """Check a single action vector against safety limits."""
        start = time.perf_counter()
        violations = []
        clamped = False
        safe_action = action.copy()
        num_joints = min(len(action), len(self.limits.position_max))

        for i in range(num_joints):
            # Position bounds
            if action[i] < self.limits.position_min[i]:
                violations.append(f"joint_{i} below min: {action[i]:.3f} < {self.limits.position_min[i]:.3f}")
                if self.mode == "clamp":
                    safe_action[i] = self.limits.position_min[i]
                    clamped = True
            elif action[i] > self.limits.position_max[i]:
                violations.append(f"joint_{i} above max: {action[i]:.3f} > {self.limits.position_max[i]:.3f}")
                if self.mode == "clamp":
                    safe_action[i] = self.limits.position_max[i]
                    clamped = True

        if self.mode == "reject" and violations:
            safe_action = np.zeros_like(action)

        elapsed = (time.perf_counter() - start) * 1000

        return SafetyCheckResult(
            safe=len(violations) == 0,
            violations=violations,
            clamped=clamped,
            original_action=action.tolist(),
            safe_action=safe_action.tolist(),
            check_time_ms=elapsed,
        )

    def check(self, actions: np.ndarray) -> tuple[np.ndarray, list[SafetyCheckResult]]:
        """Check a batch of actions (action chunk).

        Args:
            actions: [chunk_size, action_dim] array

        Returns:
            (safe_actions, results) where safe_actions is the clamped/rejected array

        Non-finite handling: any NaN or Inf (i.e. any nan/inf value) in the
        input array is a hard reject — the whole chunk is replaced with zeros
        and a single violation record is appended (not per-joint). This counts
        as a "clamp event" for the staleness kill-switch.
        """
        results = []
        non_finite_mask = ~np.isfinite(actions)
        had_non_finite = bool(non_finite_mask.any())

        if had_non_finite:
            num_bad = int(non_finite_mask.sum())
            safe_actions = np.zeros_like(actions)
            violation_msg = (
                f"non_finite_action: {num_bad} NaN/Inf value(s) detected — "
                f"entire chunk zeroed"
            )
            check_result = SafetyCheckResult(
                safe=False,
                violations=[violation_msg],
                clamped=True,
                original_action=actions[0].tolist() if len(actions) else [],
                safe_action=safe_actions[0].tolist() if len(safe_actions) else [],
                check_time_ms=0.0,
            )
            results.append(check_result)
            all_violations = [violation_msg]
            chunk_clamped = True
        else:
            safe_actions = actions.copy()
            for i in range(len(actions)):
                result = self.check_single(actions[i])
                results.append(result)
                safe_actions[i] = np.array(result.safe_action)
            all_violations = [v for r in results for v in r.violations]
            chunk_clamped = any(r.clamped for r in results)

        if self._log_dir:
            self._log_inference(actions, safe_actions, all_violations, chunk_clamped)

        # Staleness kill-switch — trip after N consecutive clamp/NaN chunks.
        if self.max_consecutive_clamps > 0:
            if chunk_clamped:
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
            else:
                self._consecutive_clamps = 0

        self._inference_count += 1
        return safe_actions, results

    @property
    def tripped(self) -> bool:
        """True when the consecutive-clamp kill-switch has fired.

        Callers (e.g. `reflex serve`) should stop serving actions and raise a
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
