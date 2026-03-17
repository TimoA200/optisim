"""Robot modeling and loading."""

from optisim.robot.controller import JointController
from optisim.robot.humanoid import (
    BuiltInHumanoid,
    DemoHumanoidSpec,
    HumanoidSpec,
    build_demo_humanoid,
    build_humanoid_model,
)
from optisim.robot.ik import IKOptions, IKResult, solve_inverse_kinematics
from optisim.robot.model import JointSpec, LinkSpec, RobotModel
from optisim.robot.urdf import load_urdf

__all__ = [
    "JointController",
    "BuiltInHumanoid",
    "DemoHumanoidSpec",
    "HumanoidSpec",
    "JointSpec",
    "LinkSpec",
    "RobotModel",
    "IKOptions",
    "IKResult",
    "build_demo_humanoid",
    "build_humanoid_model",
    "load_urdf",
    "solve_inverse_kinematics",
]
