"""Robot model, joint constraints, and forward kinematics."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from optisim.math3d import Pose, Quaternion, dh_transform, pose_from_matrix, vec3

Matrix = NDArray[np.float64]


@dataclass(slots=True)
class LinkSpec:
    name: str
    parent_joint: str | None = None
    visual_extent: tuple[float, float, float] = (0.08, 0.08, 0.08)


@dataclass(slots=True)
class JointSpec:
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
        return float(np.clip(value, self.limit_lower, self.limit_upper))


@dataclass(slots=True)
class DemoHumanoidSpec:
    shoulder_height: float = 1.38
    upper_arm_length: float = 0.34
    forearm_length: float = 0.30
    hand_length: float = 0.12
    shoulder_offset_y: float = 0.19
    torso_height: float = 0.95


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
        for name, joint in self.joints.items():
            self.joint_positions.setdefault(name, 0.0)
            self.joint_positions[name] = joint.clamp(self.joint_positions[name])

    def max_reach(self) -> float:
        arm_reach: dict[str, float] = {"left": 0.0, "right": 0.0}
        for joint in self.joints.values():
            if any(token in joint.name for token in ("shoulder", "elbow", "wrist")):
                side = "left" if joint.name.startswith("left_") else "right"
                arm_reach[side] += abs(joint.dh_a) + abs(joint.dh_d)
        torso_bonus = max(link.visual_extent[2] for link in self.links.values()) * 0.25
        return max(max(arm_reach.values()) + torso_bonus, 0.75)

    def set_joint_positions(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.joint_positions[name] = self.joints[name].clamp(value)

    def forward_kinematics(self, joint_positions: dict[str, float] | None = None) -> dict[str, Pose]:
        positions = dict(self.joint_positions)
        if joint_positions:
            for name, value in joint_positions.items():
                positions[name] = self.joints[name].clamp(value)

        child_map: dict[str, list[JointSpec]] = {}
        for joint in self.joints.values():
            child_map.setdefault(joint.parent, []).append(joint)

        poses: dict[str, Pose] = {self.root_link: self.base_pose}

        def visit(link_name: str) -> None:
            parent_pose = poses[link_name]
            for joint in child_map.get(link_name, []):
                transform = parent_pose.matrix() @ joint.origin.matrix()
                if joint.joint_type == "revolute":
                    transform = transform @ dh_transform(
                        joint.dh_a,
                        joint.dh_alpha,
                        joint.dh_d,
                        positions[joint.name] + joint.dh_theta_offset,
                    )
                elif joint.joint_type == "prismatic":
                    transform = transform @ dh_transform(
                        joint.dh_a,
                        joint.dh_alpha,
                        joint.dh_d + positions[joint.name],
                        joint.dh_theta_offset,
                    )
                else:
                    transform = transform @ dh_transform(
                        joint.dh_a, joint.dh_alpha, joint.dh_d, joint.dh_theta_offset
                    )
                poses[joint.child] = pose_from_matrix(transform)
                visit(joint.child)

        visit(self.root_link)
        return poses

    def end_effector_pose(self, effector: str, joint_positions: dict[str, float] | None = None) -> Pose:
        link_name = self.end_effectors[effector]
        return self.forward_kinematics(joint_positions)[link_name]

    def link_aabbs(self, joint_positions: dict[str, float] | None = None) -> dict[str, tuple[Vector3, Vector3]]:
        poses = self.forward_kinematics(joint_positions)
        aabbs: dict[str, tuple[Vector3, Vector3]] = {}
        for name, link in self.links.items():
            pose = poses.get(name, self.base_pose)
            extents = np.asarray(link.visual_extent, dtype=np.float64) / 2.0
            aabbs[name] = (pose.position - extents, pose.position + extents)
        return aabbs


Vector3 = NDArray[np.float64]


def build_demo_humanoid(spec: DemoHumanoidSpec | None = None) -> RobotModel:
    """Construct a compact humanoid with DH-based arms for demos."""

    spec = spec or DemoHumanoidSpec()
    links = {
        "pelvis": LinkSpec("pelvis", visual_extent=(0.24, 0.18, 0.16)),
        "torso": LinkSpec("torso", visual_extent=(0.28, 0.22, 0.48)),
        "right_upper_arm": LinkSpec("right_upper_arm", visual_extent=(0.12, 0.12, spec.upper_arm_length)),
        "right_forearm": LinkSpec("right_forearm", visual_extent=(0.10, 0.10, spec.forearm_length)),
        "right_palm": LinkSpec("right_palm", visual_extent=(0.10, 0.14, spec.hand_length)),
        "left_upper_arm": LinkSpec("left_upper_arm", visual_extent=(0.12, 0.12, spec.upper_arm_length)),
        "left_forearm": LinkSpec("left_forearm", visual_extent=(0.10, 0.10, spec.forearm_length)),
        "left_palm": LinkSpec("left_palm", visual_extent=(0.10, 0.14, spec.hand_length)),
    }
    joints = {
        "torso_yaw": JointSpec(
            name="torso_yaw",
            parent="pelvis",
            child="torso",
            origin=Pose(position=vec3([0.0, 0.0, spec.torso_height]), orientation=Quaternion.identity()),
            limit_lower=-0.6,
            limit_upper=0.6,
            dh_d=0.0,
        ),
        "right_shoulder_pitch": JointSpec(
            name="right_shoulder_pitch",
            parent="torso",
            child="right_upper_arm",
            origin=Pose.from_xyz_rpy([0.0, -spec.shoulder_offset_y, spec.shoulder_height - spec.torso_height], [0.0, 0.0, -pi / 2]),
            limit_lower=-1.7,
            limit_upper=1.5,
            velocity_limit=2.5,
            dh_a=spec.upper_arm_length,
            dh_alpha=0.0,
        ),
        "right_elbow_pitch": JointSpec(
            name="right_elbow_pitch",
            parent="right_upper_arm",
            child="right_forearm",
            limit_lower=0.0,
            limit_upper=2.4,
            velocity_limit=2.5,
            dh_a=spec.forearm_length,
            dh_alpha=0.0,
        ),
        "right_wrist_pitch": JointSpec(
            name="right_wrist_pitch",
            parent="right_forearm",
            child="right_palm",
            limit_lower=-1.1,
            limit_upper=1.1,
            velocity_limit=3.0,
            dh_a=spec.hand_length,
            dh_alpha=0.0,
        ),
        "left_shoulder_pitch": JointSpec(
            name="left_shoulder_pitch",
            parent="torso",
            child="left_upper_arm",
            origin=Pose.from_xyz_rpy([0.0, spec.shoulder_offset_y, spec.shoulder_height - spec.torso_height], [0.0, 0.0, pi / 2]),
            limit_lower=-1.7,
            limit_upper=1.5,
            velocity_limit=2.5,
            dh_a=spec.upper_arm_length,
            dh_alpha=0.0,
        ),
        "left_elbow_pitch": JointSpec(
            name="left_elbow_pitch",
            parent="left_upper_arm",
            child="left_forearm",
            limit_lower=0.0,
            limit_upper=2.4,
            velocity_limit=2.5,
            dh_a=spec.forearm_length,
            dh_alpha=0.0,
        ),
        "left_wrist_pitch": JointSpec(
            name="left_wrist_pitch",
            parent="left_forearm",
            child="left_palm",
            limit_lower=-1.1,
            limit_upper=1.1,
            velocity_limit=3.0,
            dh_a=spec.hand_length,
            dh_alpha=0.0,
        ),
    }
    return RobotModel(
        name="optisim_demo_humanoid",
        links=links,
        joints=joints,
        root_link="pelvis",
        end_effectors={"right_palm": "right_palm", "left_palm": "left_palm", "right_gripper": "right_palm"},
        base_pose=Pose(position=vec3([0.0, 0.0, 0.0]), orientation=Quaternion.identity()),
    )
