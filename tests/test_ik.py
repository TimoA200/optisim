from __future__ import annotations

import numpy as np

from optisim.robot import IKOptions, build_humanoid_model, solve_inverse_kinematics


def test_position_only_ik_converges_to_reachable_target() -> None:
    robot = build_humanoid_model()
    target = robot.end_effector_pose(
        "right_palm",
        {
            "torso_yaw": -0.08,
            "right_shoulder_pitch": -1.1,
            "right_shoulder_yaw": 0.45,
            "right_elbow_pitch": 1.7,
        },
    )

    result = solve_inverse_kinematics(
        robot,
        "right_palm",
        target,
        options=IKOptions(max_iterations=120, convergence_threshold=1e-3, position_only=True),
    )

    solved_pose = robot.end_effector_pose("right_palm", result.joint_positions)
    assert result.success
    assert result.iterations <= 120
    assert np.linalg.norm(solved_pose.position - target.position) < 2e-3


def test_pose_ik_converges_for_position_and_orientation() -> None:
    robot = build_humanoid_model()
    desired_joints = {
        "torso_yaw": -0.1,
        "torso_pitch": 0.12,
        "right_clavicle_pitch": 0.1,
        "right_shoulder_pitch": -0.9,
        "right_shoulder_roll": -0.25,
        "right_shoulder_yaw": 0.6,
        "right_elbow_pitch": 1.2,
        "right_forearm_yaw": -0.35,
        "right_wrist_pitch": 0.2,
    }
    target = robot.end_effector_pose("right_palm", desired_joints)

    result = solve_inverse_kinematics(
        robot,
        "right_palm",
        target,
        options=IKOptions(max_iterations=150, convergence_threshold=1e-3, damping=0.12),
    )

    solved_pose = robot.end_effector_pose("right_palm", result.joint_positions)
    assert result.success
    assert np.linalg.norm(solved_pose.position - target.position) < 2e-3
    assert result.orientation_error < 2e-3


def test_ik_respects_joint_limits_when_target_is_far() -> None:
    robot = build_humanoid_model()
    target = robot.end_effector_pose(
        "right_palm",
        {
            "right_shoulder_pitch": -2.0,
            "right_shoulder_yaw": 1.3,
            "right_elbow_pitch": 2.2,
            "right_forearm_yaw": -1.4,
        },
    )

    result = solve_inverse_kinematics(
        robot,
        "right_palm",
        target,
        initial_positions={"right_shoulder_pitch": 10.0, "right_elbow_pitch": -10.0},
        options=IKOptions(max_iterations=100, convergence_threshold=1e-3),
    )

    assert result.joint_positions
    for joint_name, value in result.joint_positions.items():
        spec = robot.joints[joint_name]
        assert spec.limit_lower <= value <= spec.limit_upper
