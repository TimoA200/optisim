"""Task assignment and dependency types for multi-robot execution."""

from __future__ import annotations

from dataclasses import dataclass, field

from optisim.core import TaskDefinition, ValidationIssue, ValidationReport
from optisim.sim import ExecutionEngine


@dataclass(frozen=True, slots=True)
class Dependency:
    """Wait until another robot completes a specific action index."""

    robot_name: str
    action_index: int

    def to_dict(self) -> dict[str, int | str]:
        """Serialize the dependency into a plain mapping."""

        return {"robot_name": self.robot_name, "action_index": self.action_index}


@dataclass(slots=True)
class TaskAssignment:
    """Assign a task definition to a named robot with optional dependencies."""

    robot_name: str
    task: TaskDefinition
    dependencies: list[Dependency] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize the assignment into a plain mapping."""

        return {
            "robot_name": self.robot_name,
            "task": self.task.to_dict(),
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
        }


class AssignmentValidator:
    """Validate multi-robot assignments against a fleet and dependency graph."""

    def validate(self, fleet: "RobotFleet", assignments: list[TaskAssignment]) -> ValidationReport:
        """Validate the supplied assignments against fleet membership and task feasibility."""

        report = ValidationReport()
        assignments_by_robot: dict[str, TaskAssignment] = {}

        for assignment in assignments:
            if assignment.robot_name not in fleet.robots:
                report.errors.append(
                    ValidationIssue(f"assignment references unknown robot '{assignment.robot_name}'")
                )
                continue
            if assignment.robot_name in assignments_by_robot:
                report.errors.append(
                    ValidationIssue(f"robot '{assignment.robot_name}' has multiple assignments")
                )
                continue
            assignments_by_robot[assignment.robot_name] = assignment

        for assignment in assignments_by_robot.values():
            engine = ExecutionEngine(
                robot=fleet.get_robot(assignment.robot_name),
                world=fleet.world,
                holder_prefix=assignment.robot_name,
            )
            task_report = engine.validate(assignment.task)
            report.errors.extend(task_report.errors)
            report.warnings.extend(task_report.warnings)

        for assignment in assignments_by_robot.values():
            for dependency in assignment.dependencies:
                upstream = assignments_by_robot.get(dependency.robot_name)
                if upstream is None:
                    report.errors.append(
                        ValidationIssue(
                            f"dependency for '{assignment.robot_name}' references unknown robot "
                            f"'{dependency.robot_name}'"
                        )
                    )
                    continue
                if dependency.action_index < 0 or dependency.action_index >= len(upstream.task.actions):
                    report.errors.append(
                        ValidationIssue(
                            f"dependency for '{assignment.robot_name}' references invalid action index "
                            f"{dependency.action_index} on robot '{dependency.robot_name}'"
                        )
                    )

        self._validate_dependency_cycles(assignments_by_robot, report)
        return report

    @staticmethod
    def _validate_dependency_cycles(
        assignments_by_robot: dict[str, TaskAssignment],
        report: ValidationReport,
    ) -> None:
        graph = {
            robot_name: {dependency.robot_name for dependency in assignment.dependencies}
            for robot_name, assignment in assignments_by_robot.items()
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(robot_name: str) -> bool:
            if robot_name in visited:
                return False
            if robot_name in visiting:
                return True
            visiting.add(robot_name)
            for dependency_name in graph.get(robot_name, set()):
                if visit(dependency_name):
                    return True
            visiting.remove(robot_name)
            visited.add(robot_name)
            return False

        for robot_name in graph:
            if visit(robot_name):
                report.errors.append(ValidationIssue("assignment dependencies contain a cycle"))
                return


from optisim.multi.fleet import RobotFleet

__all__ = ["Dependency", "TaskAssignment", "AssignmentValidator"]
