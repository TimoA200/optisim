from __future__ import annotations

import numpy as np

from optisim.robot import build_demo_humanoid


def test_forward_kinematics_produces_end_effector_pose() -> None:
    robot = build_demo_humanoid()
    pose = robot.end_effector_pose("right_palm")
    assert pose.position.shape == (3,)
    assert np.isfinite(pose.position).all()
    assert pose.position[2] > 0.5


def test_joint_limits_are_respected() -> None:
    robot = build_demo_humanoid()
    robot.set_joint_positions({"right_elbow_pitch": 100.0})
    assert robot.joint_positions["right_elbow_pitch"] <= robot.joints["right_elbow_pitch"].limit_upper
