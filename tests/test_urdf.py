from __future__ import annotations

from pathlib import Path

import numpy as np

from optisim.math3d import Pose
from optisim.robot import IKOptions, load_urdf, solve_inverse_kinematics

__all__ = [
    "test_load_bundled_simple_arm_urdf",
    "test_loaded_simple_arm_forward_kinematics",
    "test_loaded_simple_arm_inverse_kinematics",
]


def test_load_bundled_simple_arm_urdf() -> None:
    robot = load_urdf(_bundled_simple_arm_path())

    assert robot.name == "simple_arm"
    assert robot.root_link == "base_link"
    assert list(robot.end_effectors.values()) == ["tool_link"]
    assert len(robot.joints) == 4


def test_loaded_simple_arm_forward_kinematics() -> None:
    robot = load_urdf(_bundled_simple_arm_path())

    poses = robot.forward_kinematics(
        {
            "joint1": 0.35,
            "joint2": -0.55,
            "joint3": 0.40,
            "joint4": 0.15,
        }
    )

    tool_pose = poses["tool_link"]
    assert np.isfinite(tool_pose.position).all()
    assert tool_pose.position[0] > 0.4
    assert tool_pose.position[2] != robot.base_pose.position[2]


def test_loaded_simple_arm_inverse_kinematics() -> None:
    robot = load_urdf(_bundled_simple_arm_path())
    target_pose = robot.end_effector_pose(
        "tool_link",
        {
            "joint1": 0.45,
            "joint2": -0.60,
            "joint3": 0.35,
            "joint4": -0.20,
        },
    )

    result = solve_inverse_kinematics(
        robot,
        "tool_link",
        Pose(position=target_pose.position.copy(), orientation=target_pose.orientation),
        options=IKOptions(max_iterations=160, convergence_threshold=1e-4, damping=0.05, position_only=True),
    )

    solved_pose = robot.end_effector_pose("tool_link", result.joint_positions)
    assert result.success
    assert np.linalg.norm(solved_pose.position - target_pose.position) < 2e-3


def _bundled_simple_arm_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "robots" / "simple_arm.urdf"
