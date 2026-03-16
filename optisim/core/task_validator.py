"""Task validation with lightweight physical plausibility checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optisim.core.action_primitives import ActionPrimitive, ActionType
from optisim.core.task_definition import TaskDefinition


@dataclass(slots=True)
class ValidationIssue:
    message: str
    action_index: int | None = None
    severity: str = "error"


@dataclass(slots=True)
class ValidationReport:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        if self.is_valid:
            return f"valid ({len(self.warnings)} warnings)"
        return f"invalid ({len(self.errors)} errors, {len(self.warnings)} warnings)"


class TaskValidator:
    """Checks task semantics against world and robot capabilities."""

    def validate(self, task: TaskDefinition, world: "WorldState", robot: "RobotModel") -> ValidationReport:
        report = ValidationReport()
        carried_object: str | None = None
        max_reach = robot.max_reach()

        for index, action in enumerate(task.actions):
            self._validate_action_shape(action, index, report)

            if action.target not in world.objects:
                report.errors.append(
                    ValidationIssue(f"target '{action.target}' not present in world", action_index=index)
                )
                continue

            target_position = world.objects[action.target].pose.position
            distance = float(np.linalg.norm(target_position - robot.base_pose.position))

            if action.action_type is ActionType.REACH and distance > max_reach * 1.1:
                report.errors.append(
                    ValidationIssue(
                        f"target '{action.target}' is beyond robot reach ({distance:.2f}m > {max_reach:.2f}m)",
                        action_index=index,
                    )
                )
            if action.action_type is ActionType.GRASP and carried_object is not None:
                report.errors.append(
                    ValidationIssue(
                        f"cannot grasp '{action.target}' while already holding '{carried_object}'",
                        action_index=index,
                    )
                )
            if action.action_type is ActionType.GRASP:
                carried_object = action.target
            if action.action_type is ActionType.MOVE and carried_object != action.target:
                report.errors.append(
                    ValidationIssue(
                        f"move action for '{action.target}' requires prior grasp",
                        action_index=index,
                    )
                )
            if action.action_type is ActionType.PLACE:
                if carried_object != action.target:
                    report.errors.append(
                        ValidationIssue(
                            f"place action for '{action.target}' requires prior grasp",
                            action_index=index,
                        )
                    )
                elif action.support not in world.surfaces:
                    report.errors.append(
                        ValidationIssue(
                            f"support '{action.support}' not present in world surfaces",
                            action_index=index,
                        )
                    )
                carried_object = None
            if action.action_type in {ActionType.PUSH, ActionType.PULL, ActionType.ROTATE} and not action.axis:
                report.errors.append(
                    ValidationIssue("interaction action requires a non-zero axis", action_index=index)
                )
            if action.axis is not None and np.linalg.norm(np.asarray(action.axis, dtype=np.float64)) < 1e-6:
                report.errors.append(ValidationIssue("axis magnitude must be non-zero", action_index=index))

        if carried_object is not None:
            report.warnings.append(
                ValidationIssue(f"task ends while still holding '{carried_object}'", severity="warning")
            )
        return report

    @staticmethod
    def _validate_action_shape(action: ActionPrimitive, index: int, report: ValidationReport) -> None:
        if action.speed <= 0.0:
            report.errors.append(ValidationIssue("action speed must be positive", action_index=index))
        if action.action_type is ActionType.MOVE and action.destination is None:
            report.errors.append(ValidationIssue("move action requires destination", action_index=index))
        if action.action_type is ActionType.PLACE and action.support is None:
            report.errors.append(ValidationIssue("place action requires support", action_index=index))
        if action.action_type is ActionType.ROTATE and action.angle_rad is None:
            report.errors.append(ValidationIssue("rotate action requires angle_rad", action_index=index))
