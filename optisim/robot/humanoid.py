"""Built-in humanoid robot definition inspired by modern general-purpose robots."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

from optisim.math3d import Pose, Quaternion, vec3
from optisim.robot.model import JointSpec, LinkSpec, RobotModel


@dataclass(slots=True)
class HumanoidSpec:
    """Anthropomorphic dimensions for the built-in humanoid."""

    height_m: float = 1.75
    shoulder_height_m: float = 1.43
    hip_height_m: float = 0.98
    shoulder_offset_y_m: float = 0.22
    hip_offset_y_m: float = 0.10
    upper_arm_m: float = 0.31
    forearm_m: float = 0.28
    hand_m: float = 0.14
    thigh_m: float = 0.42
    shin_m: float = 0.41
    foot_m: float = 0.24
    neck_m: float = 0.12


@dataclass(slots=True)
class BuiltInHumanoid:
    """Factory object that produces the built-in humanoid as a ``RobotModel``."""

    spec: HumanoidSpec = field(default_factory=HumanoidSpec)

    def as_robot_model(self) -> RobotModel:
        """Build the configured built-in humanoid as a ``RobotModel``."""

        spec = self.spec
        links: dict[str, LinkSpec] = {
            "pelvis": LinkSpec("pelvis", visual_extent=(0.28, 0.22, 0.16)),
            "torso_yaw_link": LinkSpec("torso_yaw_link", visual_extent=(0.22, 0.18, 0.10)),
            "torso_roll_link": LinkSpec("torso_roll_link", visual_extent=(0.24, 0.20, 0.14)),
            "chest": LinkSpec("chest", visual_extent=(0.34, 0.26, 0.44)),
            "neck": LinkSpec("neck", visual_extent=(0.12, 0.12, spec.neck_m)),
            "head": LinkSpec("head", visual_extent=(0.18, 0.18, 0.24)),
            "right_clavicle": LinkSpec("right_clavicle", visual_extent=(0.14, 0.08, 0.08)),
            "right_upper_arm": LinkSpec("right_upper_arm", visual_extent=(0.10, 0.10, spec.upper_arm_m)),
            "right_upper_arm_roll": LinkSpec("right_upper_arm_roll", visual_extent=(0.10, 0.10, 0.10)),
            "right_forearm": LinkSpec("right_forearm", visual_extent=(0.09, 0.09, spec.forearm_m)),
            "right_forearm_roll": LinkSpec("right_forearm_roll", visual_extent=(0.09, 0.09, 0.10)),
            "right_hand": LinkSpec("right_hand", visual_extent=(0.10, 0.12, spec.hand_m)),
            "right_palm": LinkSpec("right_palm", visual_extent=(0.10, 0.12, 0.06)),
            "left_clavicle": LinkSpec("left_clavicle", visual_extent=(0.14, 0.08, 0.08)),
            "left_upper_arm": LinkSpec("left_upper_arm", visual_extent=(0.10, 0.10, spec.upper_arm_m)),
            "left_upper_arm_roll": LinkSpec("left_upper_arm_roll", visual_extent=(0.10, 0.10, 0.10)),
            "left_forearm": LinkSpec("left_forearm", visual_extent=(0.09, 0.09, spec.forearm_m)),
            "left_forearm_roll": LinkSpec("left_forearm_roll", visual_extent=(0.09, 0.09, 0.10)),
            "left_hand": LinkSpec("left_hand", visual_extent=(0.10, 0.12, spec.hand_m)),
            "left_palm": LinkSpec("left_palm", visual_extent=(0.10, 0.12, 0.06)),
            "right_hip_yaw_link": LinkSpec("right_hip_yaw_link", visual_extent=(0.14, 0.10, 0.10)),
            "right_hip_roll_link": LinkSpec("right_hip_roll_link", visual_extent=(0.14, 0.10, 0.10)),
            "right_thigh": LinkSpec("right_thigh", visual_extent=(0.12, 0.12, spec.thigh_m)),
            "right_shin": LinkSpec("right_shin", visual_extent=(0.11, 0.11, spec.shin_m)),
            "right_ankle_pitch_link": LinkSpec("right_ankle_pitch_link", visual_extent=(0.10, 0.10, 0.08)),
            "right_foot": LinkSpec("right_foot", visual_extent=(spec.foot_m, 0.11, 0.06)),
            "left_hip_yaw_link": LinkSpec("left_hip_yaw_link", visual_extent=(0.14, 0.10, 0.10)),
            "left_hip_roll_link": LinkSpec("left_hip_roll_link", visual_extent=(0.14, 0.10, 0.10)),
            "left_thigh": LinkSpec("left_thigh", visual_extent=(0.12, 0.12, spec.thigh_m)),
            "left_shin": LinkSpec("left_shin", visual_extent=(0.11, 0.11, spec.shin_m)),
            "left_ankle_pitch_link": LinkSpec("left_ankle_pitch_link", visual_extent=(0.10, 0.10, 0.08)),
            "left_foot": LinkSpec("left_foot", visual_extent=(spec.foot_m, 0.11, 0.06)),
        }
        joints: dict[str, JointSpec] = {}

        def add_joint(joint: JointSpec) -> None:
            joints[joint.name] = joint
            links[joint.child].parent_joint = joint.name

        add_joint(
            JointSpec(
                name="torso_yaw",
                parent="pelvis",
                child="torso_yaw_link",
                origin=Pose(position=vec3([0.0, 0.0, spec.hip_height_m]), orientation=Quaternion.identity()),
                limit_lower=-0.8,
                limit_upper=0.8,
                velocity_limit=1.5,
            )
        )
        add_joint(
            JointSpec(
                name="torso_roll",
                parent="torso_yaw_link",
                child="torso_roll_link",
                origin=Pose.from_xyz_rpy([0.0, 0.0, 0.0], [pi / 2, 0.0, 0.0]),
                limit_lower=-0.45,
                limit_upper=0.45,
                velocity_limit=1.5,
            )
        )
        add_joint(
            JointSpec(
                name="torso_pitch",
                parent="torso_roll_link",
                child="chest",
                origin=Pose.from_xyz_rpy([0.0, 0.0, 0.0], [0.0, pi / 2, 0.0]),
                limit_lower=-0.7,
                limit_upper=0.6,
                velocity_limit=1.5,
                dh_d=spec.shoulder_height_m - spec.hip_height_m,
            )
        )
        add_joint(
            JointSpec(
                name="neck_yaw",
                parent="chest",
                child="neck",
                origin=Pose(position=vec3([0.0, 0.0, 0.26]), orientation=Quaternion.identity()),
                limit_lower=-1.2,
                limit_upper=1.2,
                velocity_limit=2.0,
            )
        )
        add_joint(
            JointSpec(
                name="neck_pitch",
                parent="neck",
                child="head",
                origin=Pose.from_xyz_rpy([0.0, 0.0, spec.neck_m], [0.0, pi / 2, 0.0]),
                limit_lower=-0.75,
                limit_upper=0.95,
                velocity_limit=2.0,
            )
        )

        for side, sign in (("right", -1.0), ("left", 1.0)):
            shoulder_origin = [0.02, sign * spec.shoulder_offset_y_m, 0.16]
            elbow_twist = -pi / 2 if side == "right" else pi / 2
            add_joint(
                JointSpec(
                    name=f"{side}_clavicle_pitch",
                    parent="chest",
                    child=f"{side}_clavicle",
                    origin=Pose.from_xyz_rpy(shoulder_origin, [0.0, pi / 2, elbow_twist]),
                    limit_lower=-0.45,
                    limit_upper=0.55,
                    velocity_limit=2.0,
                    dh_a=0.08,
                )
            )
            add_joint(
                JointSpec(
                    name=f"{side}_shoulder_pitch",
                    parent=f"{side}_clavicle",
                    child=f"{side}_upper_arm",
                    origin=Pose.from_xyz_rpy([0.0, 0.0, 0.0], [0.0, pi / 2, 0.0]),
                    limit_lower=-2.6,
                    limit_upper=1.5,
                    velocity_limit=2.5,
                    dh_a=spec.upper_arm_m,
                )
            )
            add_joint(
                JointSpec(
                    name=f"{side}_shoulder_roll",
                    parent=f"{side}_upper_arm",
                    child=f"{side}_upper_arm_roll",
                    origin=Pose.from_xyz_rpy([0.0, 0.0, 0.0], [pi / 2, 0.0, 0.0]),
                    limit_lower=-1.35 if side == "right" else -0.35,
                    limit_upper=0.35 if side == "right" else 1.35,
                    velocity_limit=2.5,
                )
            )
            add_joint(
                JointSpec(
                    name=f"{side}_shoulder_yaw",
                    parent=f"{side}_upper_arm_roll",
                    child=f"{side}_forearm",
                    limit_lower=-1.5,
                    limit_upper=1.5,
                    velocity_limit=2.5,
                    dh_a=spec.forearm_m,
                )
            )
            add_joint(
                JointSpec(
                    name=f"{side}_elbow_pitch",
                    parent=f"{side}_forearm",
                    child=f"{side}_forearm_roll",
                    origin=Pose.from_xyz_rpy([0.0, 0.0, 0.0], [0.0, pi / 2, 0.0]),
                    limit_lower=0.0,
                    limit_upper=2.5,
                    velocity_limit=2.7,
                )
            )
            add_joint(
                JointSpec(
                    name=f"{side}_forearm_yaw",
                    parent=f"{side}_forearm_roll",
                    child=f"{side}_hand",
                    limit_lower=-1.8,
                    limit_upper=1.8,
                    velocity_limit=3.0,
                    dh_a=spec.hand_m,
                )
            )
            add_joint(
                JointSpec(
                    name=f"{side}_wrist_pitch",
                    parent=f"{side}_hand",
                    child=f"{side}_palm",
                    origin=Pose.from_xyz_rpy([0.0, 0.0, 0.0], [0.0, pi / 2, 0.0]),
                    limit_lower=-1.0,
                    limit_upper=1.0,
                    velocity_limit=3.2,
                    dh_a=0.05,
                )
            )
        for side, sign in (("right", -1.0), ("left", 1.0)):
            hip_origin = [0.0, sign * spec.hip_offset_y_m, spec.hip_height_m]
            add_joint(
                JointSpec(
                    name=f"{side}_hip_yaw",
                    parent="pelvis",
                    child=f"{side}_hip_yaw_link",
                    origin=Pose(position=vec3(hip_origin), orientation=Quaternion.identity()),
                    limit_lower=-0.75,
                    limit_upper=0.75,
                    velocity_limit=2.5,
                )
            )
            add_joint(
                JointSpec(
                    name=f"{side}_hip_roll",
                    parent=f"{side}_hip_yaw_link",
                    child=f"{side}_hip_roll_link",
                    origin=Pose.from_xyz_rpy([0.0, 0.0, 0.0], [pi / 2, 0.0, 0.0]),
                    limit_lower=-0.45 if side == "right" else -0.25,
                    limit_upper=0.25 if side == "right" else 0.45,
                    velocity_limit=2.5,
                )
            )
            add_joint(
                JointSpec(
                    name=f"{side}_hip_pitch",
                    parent=f"{side}_hip_roll_link",
                    child=f"{side}_thigh",
                    origin=Pose.from_xyz_rpy([0.0, 0.0, 0.0], [0.0, -pi / 2, 0.0]),
                    limit_lower=-1.6,
                    limit_upper=0.8,
                    velocity_limit=2.5,
                    dh_d=-spec.thigh_m,
                )
            )
            add_joint(
                JointSpec(
                    name=f"{side}_knee_pitch",
                    parent=f"{side}_thigh",
                    child=f"{side}_shin",
                    origin=Pose.from_xyz_rpy([0.0, 0.0, 0.0], [0.0, -pi / 2, 0.0]),
                    limit_lower=0.0,
                    limit_upper=2.6,
                    velocity_limit=3.0,
                    dh_d=-spec.shin_m,
                )
            )
            add_joint(
                JointSpec(
                    name=f"{side}_ankle_pitch",
                    parent=f"{side}_shin",
                    child=f"{side}_ankle_pitch_link",
                    origin=Pose.from_xyz_rpy([0.0, 0.0, 0.0], [0.0, -pi / 2, 0.0]),
                    limit_lower=-0.8,
                    limit_upper=0.6,
                    velocity_limit=2.5,
                    dh_d=-0.05,
                )
            )
            add_joint(
                JointSpec(
                    name=f"{side}_ankle_roll",
                    parent=f"{side}_ankle_pitch_link",
                    child=f"{side}_foot",
                    origin=Pose.from_xyz_rpy([0.0, 0.0, -0.04], [pi / 2, 0.0, 0.0]),
                    limit_lower=-0.35,
                    limit_upper=0.35,
                    velocity_limit=2.5,
                    dh_a=0.08,
                )
            )

        return RobotModel(
            name="optisim_optimus_humanoid",
            links=links,
            joints=joints,
            root_link="pelvis",
            end_effectors={
                "right_palm": "right_palm",
                "left_palm": "left_palm",
                "right_gripper": "right_palm",
                "left_gripper": "left_palm",
                "right_foot": "right_foot",
                "left_foot": "left_foot",
            },
            base_pose=Pose(position=vec3([0.0, 0.0, 0.0]), orientation=Quaternion.identity()),
        )


DemoHumanoidSpec = HumanoidSpec


def build_humanoid_model(spec: HumanoidSpec | None = None) -> RobotModel:
    """Build the built-in humanoid robot model."""

    return BuiltInHumanoid(spec or HumanoidSpec()).as_robot_model()


def build_demo_humanoid(spec: HumanoidSpec | None = None) -> RobotModel:
    """Backward-compatible alias for the built-in humanoid model."""

    return build_humanoid_model(spec)
