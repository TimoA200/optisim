from __future__ import annotations

import numpy as np

from optisim.robot import IKOptions, build_humanoid_model, solve_inverse_kinematics


def test_humanoid_model_construction_counts_match_expected_topology() -> None:
    robot = build_humanoid_model()

    assert len(robot.joints) == 31
    assert len(robot.links) == 32
    assert robot.root_link == "pelvis"
    assert set(robot.end_effectors) == {
        "right_palm",
        "left_palm",
        "right_gripper",
        "left_gripper",
        "right_foot",
        "left_foot",
    }


def test_forward_kinematics_places_end_effectors_in_reasonable_workspace() -> None:
    robot = build_humanoid_model()
    poses = robot.forward_kinematics(
        {
            "torso_pitch": 0.1,
            "right_shoulder_pitch": -1.0,
            "right_shoulder_yaw": 0.35,
            "right_elbow_pitch": 1.45,
            "left_shoulder_pitch": -0.95,
            "left_shoulder_yaw": -0.35,
            "left_elbow_pitch": 1.35,
        }
    )

    right_palm = poses["right_palm"].position
    left_palm = poses["left_palm"].position
    right_foot = poses["right_foot"].position
    left_foot = poses["left_foot"].position

    assert 0.3 < right_palm[0] < 1.2
    assert -0.6 < right_palm[1] < -0.05
    assert 0.7 < right_palm[2] < 1.7
    assert 0.3 < left_palm[0] < 1.2
    assert 0.05 < left_palm[1] < 0.6
    assert 0.7 < left_palm[2] < 1.7
    assert 0.2 < right_foot[0] < 0.5
    assert -0.5 < right_foot[1] < -0.1
    assert 0.8 < right_foot[2] < 1.1
    assert 0.2 < left_foot[0] < 0.5
    assert -0.3 < left_foot[1] < 0.0
    assert 0.8 < left_foot[2] < 1.1


def test_all_end_effectors_are_reachable_from_their_own_recorded_poses() -> None:
    robot = build_humanoid_model()
    targets = {
        "right_palm": {
            "torso_yaw": -0.08,
            "torso_pitch": 0.05,
            "right_clavicle_pitch": 0.12,
            "right_shoulder_pitch": -1.0,
            "right_shoulder_roll": -0.22,
            "right_shoulder_yaw": 0.45,
            "right_elbow_pitch": 1.35,
            "right_forearm_yaw": -0.25,
        },
        "left_palm": {
            "torso_yaw": 0.08,
            "torso_pitch": 0.04,
            "left_clavicle_pitch": 0.10,
            "left_shoulder_pitch": -1.0,
            "left_shoulder_roll": 0.22,
            "left_shoulder_yaw": -0.45,
            "left_elbow_pitch": 1.35,
            "left_forearm_yaw": 0.25,
        },
        "right_foot": {
            "right_hip_pitch": -0.35,
            "right_knee_pitch": 0.7,
            "right_ankle_pitch": -0.3,
        },
        "left_foot": {
            "left_hip_pitch": -0.35,
            "left_knee_pitch": 0.7,
            "left_ankle_pitch": -0.3,
        },
    }

    for effector, seed in targets.items():
        robot = build_humanoid_model()
        target_pose = robot.end_effector_pose(effector, seed)
        result = solve_inverse_kinematics(
            robot,
            effector,
            target_pose,
            initial_positions=seed,
            options=IKOptions(max_iterations=160, convergence_threshold=2e-3, damping=0.12),
        )
        solved_pose = robot.end_effector_pose(effector, result.joint_positions)

        assert result.success, effector
        assert np.linalg.norm(solved_pose.position - target_pose.position) < 6e-3
