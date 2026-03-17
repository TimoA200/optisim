"""optisim public package interface."""

from optisim.analytics import TrajectoryMetrics, analyze_trajectory, compare_trajectories
from optisim.behavior import BTStatus, BehaviorTreeBuilder, BehaviorTreeExecutor
from optisim.core import ActionPrimitive, TaskComposer, TaskDefinition, ValidationReport
from optisim.library import TaskCatalog
from optisim.multi import AssignmentValidator, Dependency, MultiRobotRecord, RobotFleet, TaskAssignment, TaskCoordinator
from optisim.planning import MotionPlanner, PlanningResult, RRTConfig
from optisim.robot import DemoHumanoidSpec, HumanoidSpec, RobotModel, build_demo_humanoid, build_humanoid_model
from optisim.sim import ExecutionEngine, WorldState

__all__ = [
    "TrajectoryMetrics",
    "analyze_trajectory",
    "compare_trajectories",
    "BTStatus",
    "BehaviorTreeBuilder",
    "BehaviorTreeExecutor",
    "ActionPrimitive",
    "TaskComposer",
    "TaskDefinition",
    "TaskCatalog",
    "AssignmentValidator",
    "Dependency",
    "MultiRobotRecord",
    "RobotFleet",
    "TaskAssignment",
    "TaskCoordinator",
    "ValidationReport",
    "MotionPlanner",
    "PlanningResult",
    "RRTConfig",
    "DemoHumanoidSpec",
    "HumanoidSpec",
    "RobotModel",
    "build_demo_humanoid",
    "build_humanoid_model",
    "ExecutionEngine",
    "WorldState",
]

__version__ = "0.1.0"
