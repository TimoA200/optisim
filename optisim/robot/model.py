"""Robot model, joint constraints, and forward kinematics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import pi
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from optisim.math3d import Pose, Quaternion, dh_transform, pose_from_matrix, vec3

Matrix = NDArray[np.float64]
Vector3 = NDArray[np.float64]


@dataclass(slots=True)
class LinkSpec:
    """Visual and topological metadata for a robot link."""

    name: str
    parent_joint: str | None = None
    visual_extent: tuple[float, float, float] = (0.08, 0.08, 0.08)


@dataclass(slots=True)
class JointSpec:
    """Joint definition with limits and kinematic parameters."""

    name: str
    parent: str
    child: str
    joint_type: str = "revolute"
    origin: Pose = field(default_factory=Pose.identity)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    limit_lower: float = -pi
    limit_upper: float = pi
    velocity_limit: float = 2.0
    dh_a: float = 0.0
    dh_alpha: float = 0.0
    dh_d: float = 0.0
    dh_theta_offset: float = 0.0

    def clamp(self, value: float) -> float:
        """Clamp a joint position to this joint's configured limits."""

        return float(np.clip(value, self.limit_lower, self.limit_upper))


@dataclass
class RobotModel:
    """Tree-structured kinematic robot model with FK and constraints."""

    name: str
    links: dict[str, LinkSpec]
    joints: dict[str, JointSpec]
    root_link: str
    end_effectors: dict[str, str]
    base_pose: Pose = field(default_factory=Pose.identity)
    joint_positions: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize initial joint state and cache traversal helpers."""

        for name, joint in self.joints.items():
            self.joint_positions.setdefault(name, 0.0)
            self.joint_positions[name] = joint.clamp(self.joint_positions[name])
        self._child_map: dict[str, list[JointSpec]] = defaultdict(list)
        self._joint_by_child: dict[str, JointSpec] = {}
        for joint in self.joints.values():
            self._child_map[joint.parent].append(joint)
            self._joint_by_child[joint.child] = joint

    def max_reach(self) -> float:
        """Estimate the maximum reach of arm-like end effectors."""

        reach = 0.0
        for effector, link_name in self.end_effectors.items():
            if "palm" not in effector and "gripper" not in effector:
                continue
            chain = self.joint_chain(link_name)
            chain_reach = sum(abs(joint.dh_a) + abs(joint.dh_d) for joint in chain)
            reach = max(reach, chain_reach)
        torso_bonus = max(link.visual_extent[2] for link in self.links.values()) * 0.25
        return max(reach + torso_bonus, 0.75)

    def set_joint_positions(self, values: dict[str, float]) -> None:
        """Update one or more joint positions while enforcing limits."""

        for name, value in values.items():
            self.joint_positions[name] = self.joints[name].clamp(value)

    def forward_kinematics(self, joint_positions: dict[str, float] | None = None) -> dict[str, Pose]:
        """Compute world poses for every reachable link in the robot tree."""

        positions = self._merged_positions(joint_positions)
        poses: dict[str, Pose] = {self.root_link: self.base_pose}

        def visit(link_name: str) -> None:
            parent_pose = poses[link_name]
            for joint in self._child_map.get(link_name, []):
                transform = self._joint_transform(joint, positions[joint.name], parent_pose.matrix())
                poses[joint.child] = pose_from_matrix(transform)
                visit(joint.child)

        visit(self.root_link)
        return poses

    def end_effector_pose(self, effector: str, joint_positions: dict[str, float] | None = None) -> Pose:
        """Return the pose of a named end effector."""

        link_name = self.end_effectors[effector]
        return self.forward_kinematics(joint_positions)[link_name]

    def link_aabbs(self, joint_positions: dict[str, float] | None = None) -> dict[str, tuple[Vector3, Vector3]]:
        """Compute axis-aligned bounding boxes for every link."""

        poses = self.forward_kinematics(joint_positions)
        aabbs: dict[str, tuple[Vector3, Vector3]] = {}
        for name, link in self.links.items():
            pose = poses.get(name, self.base_pose)
            extents = np.asarray(link.visual_extent, dtype=np.float64) / 2.0
            aabbs[name] = (pose.position - extents, pose.position + extents)
        return aabbs

    def joint_chain(self, target_link: str) -> list[JointSpec]:
        """Return the ordered joint chain from the root to a target link."""

        chain: list[JointSpec] = []
        current_link = target_link
        while current_link != self.root_link:
            joint = self._joint_by_child.get(current_link)
            if joint is None:
                raise KeyError(f"no kinematic chain from '{self.root_link}' to '{target_link}'")
            chain.append(joint)
            current_link = joint.parent
        chain.reverse()
        return chain

    def joint_chain_for_effector(self, effector: str) -> list[JointSpec]:
        """Return the ordered joint chain that drives a named end effector."""

        return self.joint_chain(self.end_effectors[effector])

    def joint_frames(
        self,
        joint_positions: dict[str, float] | None = None,
    ) -> dict[str, tuple[Matrix, Matrix]]:
        """Return origin and child frames for every joint in the current pose."""

        positions = self._merged_positions(joint_positions)
        frames: dict[str, tuple[Matrix, Matrix]] = {}

        def visit(link_name: str, parent_matrix: Matrix) -> None:
            for joint in self._child_map.get(link_name, []):
                origin_matrix = parent_matrix @ joint.origin.matrix()
                child_matrix = self._joint_transform(joint, positions[joint.name], parent_matrix)
                frames[joint.name] = (origin_matrix, child_matrix)
                visit(joint.child, child_matrix)

        visit(self.root_link, self.base_pose.matrix())
        return frames

    def _merged_positions(self, joint_positions: dict[str, float] | None) -> dict[str, float]:
        positions = dict(self.joint_positions)
        if joint_positions:
            for name, value in joint_positions.items():
                positions[name] = self.joints[name].clamp(value)
        return positions

    def _joint_transform(self, joint: JointSpec, value: float, parent_matrix: Matrix) -> Matrix:
        transform = parent_matrix @ joint.origin.matrix()
        if joint.joint_type == "revolute":
            return transform @ dh_transform(
                joint.dh_a,
                joint.dh_alpha,
                joint.dh_d,
                value + joint.dh_theta_offset,
            )
        if joint.joint_type == "prismatic":
            return transform @ dh_transform(
                joint.dh_a,
                joint.dh_alpha,
                joint.dh_d + value,
                joint.dh_theta_offset,
            )
        return transform @ dh_transform(joint.dh_a, joint.dh_alpha, joint.dh_d, joint.dh_theta_offset)
