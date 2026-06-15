"""Tether Adapt — cross-embodiment transfer tools.

Generates configuration files for deploying a VLA trained on one robot
to a different robot, given its URDF description.

Usage:
    from tether.models.adapt import EmbodimentAdapter
    adapter = EmbodimentAdapter.from_urdf("new_robot.urdf")
    config = adapter.generate_config(source_action_dim=7, source_format="openpi")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbodimentConfig:
    """Configuration for a specific robot embodiment."""

    name: str
    num_joints: int
    joint_names: list[str]
    joint_types: list[str]
    action_dim: int
    position_limits: list[tuple[float, float]]
    velocity_limits: list[float]
    action_space: str = "delta_joint"  # "delta_joint", "absolute_joint", "ee_pose"
    gripper_indices: list[int] | None = None
    arm_indices: list[int] | None = None
    normalization_range: tuple[float, float] = (-1.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> EmbodimentConfig:
        data = json.loads(Path(path).read_text())
        # Convert position_limits from list of lists to list of tuples
        if "position_limits" in data:
            data["position_limits"] = [tuple(lim) for lim in data["position_limits"]]
        return cls(**data)


@dataclass
class ActionMapping:
    """Mapping between source and target action spaces."""

    source_dim: int
    target_dim: int
    mapping_type: str  # "pad", "truncate", "project"
    index_map: list[int]  # target_index -> source_index (-1 = zero-fill)
    scale_factors: list[float]

    def apply(self, source_actions: np.ndarray) -> np.ndarray:
        """Map source actions to target action space."""
        if source_actions.ndim == 1:
            source_actions = source_actions.reshape(1, -1)

        batch_size, src_dim = source_actions.shape
        target = np.zeros((batch_size, self.target_dim), dtype=source_actions.dtype)

        for t_idx, s_idx in enumerate(self.index_map):
            if t_idx >= self.target_dim:
                break
            if s_idx >= 0 and s_idx < src_dim:
                target[:, t_idx] = source_actions[:, s_idx] * self.scale_factors[t_idx]

        return target


class EmbodimentAdapter:
    """Generate embodiment configs and action mappings for cross-robot transfer."""

    def __init__(self, config: EmbodimentConfig):
        self.config = config

    @classmethod
    def from_urdf(cls, urdf_path: str | Path) -> EmbodimentAdapter:
        """Extract embodiment config from URDF."""
        try:
            import yourdfpy

            urdf = yourdfpy.URDF.load(str(urdf_path))
            names, types, pos_limits, vel_limits = [], [], [], []
            arm_indices, gripper_indices = [], []

            idx = 0
            for joint_name, joint in urdf.joint_map.items():
                if joint.type in ("revolute", "prismatic"):
                    names.append(joint_name)
                    types.append(joint.type)

                    if joint.limit is not None:
                        pos_limits.append((joint.limit.lower, joint.limit.upper))
                        vel_limits.append(joint.limit.velocity or 3.14)
                    else:
                        pos_limits.append((-3.14, 3.14))
                        vel_limits.append(3.14)

                    # Heuristic: gripper joints have small range or "grip/finger" in name
                    name_lower = joint_name.lower()
                    if any(g in name_lower for g in ["grip", "finger", "hand", "jaw"]):
                        gripper_indices.append(idx)
                    else:
                        arm_indices.append(idx)
                    idx += 1

            config = EmbodimentConfig(
                name=Path(urdf_path).stem,
                num_joints=len(names),
                joint_names=names,
                joint_types=types,
                action_dim=len(names),
                position_limits=pos_limits,
                velocity_limits=vel_limits,
                arm_indices=arm_indices if arm_indices else None,
                gripper_indices=gripper_indices if gripper_indices else None,
            )
            return cls(config)

        except ImportError:
            logger.warning("yourdfpy not installed. Install with: pip install 'fastcrest-tether[safety]'")
            raise

    @classmethod
    def default(cls, name: str = "generic_6dof", num_joints: int = 6) -> EmbodimentAdapter:
        config = EmbodimentConfig(
            name=name,
            num_joints=num_joints,
            joint_names=[f"joint_{i}" for i in range(num_joints)],
            joint_types=["revolute"] * num_joints,
            action_dim=num_joints,
            position_limits=[(-3.14, 3.14)] * num_joints,
            velocity_limits=[2.0] * num_joints,
        )
        return cls(config)

    def create_mapping(
        self,
        source_dim: int,
        source_joint_names: list[str] | None = None,
    ) -> ActionMapping:
        """Create an action mapping from source to this embodiment."""
        target_dim = self.config.action_dim

        if source_dim == target_dim:
            # Same dimension — direct mapping
            return ActionMapping(
                source_dim=source_dim,
                target_dim=target_dim,
                mapping_type="direct",
                index_map=list(range(target_dim)),
                scale_factors=[1.0] * target_dim,
            )

        if source_dim < target_dim:
            # Source is smaller — pad with zeros
            index_map = list(range(source_dim)) + [-1] * (target_dim - source_dim)
            return ActionMapping(
                source_dim=source_dim,
                target_dim=target_dim,
                mapping_type="pad",
                index_map=index_map,
                scale_factors=[1.0] * target_dim,
            )

        # Source is larger — truncate
        index_map = list(range(target_dim))
        return ActionMapping(
            source_dim=source_dim,
            target_dim=target_dim,
            mapping_type="truncate",
            index_map=index_map,
            scale_factors=[1.0] * target_dim,
        )

    def generate_framework_config(self, framework: str = "lerobot") -> dict[str, Any]:
        """Generate framework-specific config for this embodiment."""
        base = {
            "embodiment": self.config.name,
            "action_dim": self.config.action_dim,
            "state_dim": self.config.action_dim,
            "joint_names": self.config.joint_names,
            "action_space": self.config.action_space,
        }

        if framework == "lerobot":
            base["max_action_dim"] = 32
            base["max_state_dim"] = 32
            base["chunk_size"] = 50

        elif framework == "openpi":
            base["max_action_dim"] = 32
            base["normalization"] = {"type": "percentile", "low": 1, "high": 99}

        elif framework == "gr00t":
            base["embodiment_tag"] = self.config.name
            base["data_config_name"] = f"{self.config.name}_arms"

        return base
