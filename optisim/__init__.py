"""optisim public package interface."""

from optisim.analytics import TrajectoryMetrics, analyze_trajectory, compare_trajectories
from optisim.core import ActionPrimitive, TaskComposer, TaskDefinition, ValidationReport
from optisim.robot import DemoHumanoidSpec, HumanoidSpec, RobotModel, build_demo_humanoid, build_humanoid_model
from optisim.sim import ExecutionEngine, WorldState

__all__ = [
    "TrajectoryMetrics",
    "analyze_trajectory",
    "compare_trajectories",
    "ActionPrimitive",
    "TaskComposer",
    "TaskDefinition",
    "ValidationReport",
    "DemoHumanoidSpec",
    "HumanoidSpec",
    "RobotModel",
    "build_demo_humanoid",
    "build_humanoid_model",
    "ExecutionEngine",
    "WorldState",
]

__version__ = "0.1.0"
