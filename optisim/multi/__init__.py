"""Public multi-robot coordination interface."""

from optisim.multi.assignment import AssignmentValidator, Dependency, TaskAssignment
from optisim.multi.collision import InterRobotCollision, inter_robot_collisions
from optisim.multi.coordinator import MultiRobotRecord, RobotTrace, TaskCoordinator
from optisim.multi.fleet import RobotFleet

__all__ = [
    "AssignmentValidator",
    "Dependency",
    "TaskAssignment",
    "InterRobotCollision",
    "inter_robot_collisions",
    "MultiRobotRecord",
    "RobotTrace",
    "TaskCoordinator",
    "RobotFleet",
]
