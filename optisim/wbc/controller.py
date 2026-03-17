"""Whole-body control tasks and hierarchical controller."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from optisim.math3d import Pose
from optisim.robot.model import RobotModel

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

__all__ = [
    "WBCTask",
    "PostureTask",
    "EndEffectorTask",
    "BalanceTask",
    "JointLimitTask",
    "WBCController",
    "WBCSolution",
    "build_wbc_controller",
]


class WBCTask(ABC):
    """Priority-ordered objective used by the WBC controller."""

    def __init__(self, priority: int = 0, weight: float = 1.0) -> None:
        self.priority = int(priority)
        self.weight = float(weight)

    @abstractmethod
    def compute_jacobian_and_error(self, robot: RobotModel) -> tuple[Matrix, Vector]:
        """Return the task Jacobian and task-space error."""


class PostureTask(WBCTask):
    """Track a desired joint posture in joint space."""

    def __init__(self, desired_joint_positions: dict[str, float], priority: int = 0, weight: float = 1.0) -> None:
        super().__init__(priority=priority, weight=weight)
        self.desired_joint_positions = dict(desired_joint_positions)

    def compute_jacobian_and_error(self, robot: RobotModel) -> tuple[Matrix, Vector]:
        joint_names = list(robot.joints)
        jacobian = np.eye(len(joint_names), dtype=np.float64)
        error = np.asarray(
            [
                robot.joints[name].clamp(self.desired_joint_positions.get(name, robot.joint_positions[name]))
                - robot.joint_positions[name]
                for name in joint_names
            ],
            dtype=np.float64,
        )
        return jacobian, error


class EndEffectorTask(WBCTask):
    """Track a target pose for a named end effector."""

    def __init__(
        self,
        end_effector: str,
        target_pose: Pose,
        *,
        position_only: bool = False,
        priority: int = 0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(priority=priority, weight=weight)
        self.end_effector = end_effector
        self.target_pose = target_pose
        self.position_only = position_only

    def compute_jacobian_and_error(self, robot: RobotModel) -> tuple[Matrix, Vector]:
        joint_names = list(robot.joints)
        joint_index = {name: index for index, name in enumerate(joint_names)}
        active_joint_names = [joint.name for joint in robot.joint_chain_for_effector(self.end_effector)]
        jacobian = np.zeros((3 if self.position_only else 6, len(joint_names)), dtype=np.float64)
        active_jacobian = _numerical_end_effector_jacobian(
            robot,
            self.end_effector,
            active_joint_names,
            robot.joint_positions,
            self.position_only,
        )
        for column_index, joint_name in enumerate(active_joint_names):
            jacobian[:, joint_index[joint_name]] = active_jacobian[:, column_index]

        current_pose = robot.end_effector_pose(self.end_effector)
        position_error = self.target_pose.position - current_pose.position
        if self.position_only:
            return jacobian, position_error.astype(np.float64)
        orientation_error = _orientation_error(
            current_pose.orientation.to_rotation_matrix(),
            self.target_pose.orientation.to_rotation_matrix(),
        )
        return jacobian, np.concatenate([position_error, orientation_error], dtype=np.float64)


class BalanceTask(WBCTask):
    """Track a center-of-mass proxy target."""

    def __init__(self, target_position: Vector | None = None, priority: int = 0, weight: float = 1.0) -> None:
        super().__init__(priority=priority, weight=weight)
        self.target_position = None if target_position is None else np.asarray(target_position, dtype=np.float64)

    def compute_jacobian_and_error(self, robot: RobotModel) -> tuple[Matrix, Vector]:
        if self.target_position is None:
            self.target_position = _center_of_mass_proxy(robot)
        joint_names = list(robot.joints)
        base_com = _center_of_mass_proxy(robot)
        jacobian = np.zeros((3, len(joint_names)), dtype=np.float64)
        epsilon = 1e-5

        for index, joint_name in enumerate(joint_names):
            shifted_positions = dict(robot.joint_positions)
            shifted_positions[joint_name] = robot.joints[joint_name].clamp(shifted_positions[joint_name] + epsilon)
            shifted_com = _center_of_mass_proxy(robot, shifted_positions)
            jacobian[:, index] = (shifted_com - base_com) / epsilon

        return jacobian, (self.target_position - base_com).astype(np.float64)


class JointLimitTask(WBCTask):
    """Push joints away from their configured limits."""

    def __init__(self, activation_ratio: float = 0.2, priority: int = 0, weight: float = 1.0) -> None:
        super().__init__(priority=priority, weight=weight)
        self.activation_ratio = float(activation_ratio)

    def compute_jacobian_and_error(self, robot: RobotModel) -> tuple[Matrix, Vector]:
        joint_names = list(robot.joints)
        jacobian = np.eye(len(joint_names), dtype=np.float64)
        error = np.zeros(len(joint_names), dtype=np.float64)

        for index, joint_name in enumerate(joint_names):
            joint = robot.joints[joint_name]
            span = joint.limit_upper - joint.limit_lower
            if span <= 1e-9:
                continue
            activation_band = max(span * self.activation_ratio, 1e-6)
            distance_to_lower = robot.joint_positions[joint_name] - joint.limit_lower
            distance_to_upper = joint.limit_upper - robot.joint_positions[joint_name]
            if distance_to_lower < activation_band:
                error[index] = (activation_band - distance_to_lower) / activation_band
            elif distance_to_upper < activation_band:
                error[index] = -(activation_band - distance_to_upper) / activation_band

        return jacobian, error


@dataclass(slots=True)
class WBCSolution:
    """Result returned by the iterative WBC solve loop."""

    iterations: int
    converged: bool
    task_errors: dict[str, float]
    joint_positions: dict[str, float]


@dataclass
class WBCController:
    """Hierarchical whole-body controller with null-space task stacking."""

    tasks: list[WBCTask] = field(default_factory=list)
    damping: float = 0.01

    def __post_init__(self) -> None:
        self.tasks.sort(key=lambda task: task.priority)

    def add_task(self, task: WBCTask) -> None:
        """Insert a task and keep the list priority ordered."""

        self.tasks.append(task)
        self.tasks.sort(key=lambda entry: entry.priority)

    def compute_joint_velocities(self, robot: RobotModel, dt: float) -> dict[str, float]:
        """Compute a velocity command by stacking tasks hierarchically."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")

        joint_names = list(robot.joints)
        degrees_of_freedom = len(joint_names)
        if degrees_of_freedom == 0:
            return {}

        joint_velocity = np.zeros(degrees_of_freedom, dtype=np.float64)
        nullspace = np.eye(degrees_of_freedom, dtype=np.float64)

        for task in self.tasks:
            jacobian, error = task.compute_jacobian_and_error(robot)
            if jacobian.size == 0 or error.size == 0:
                continue
            scale = max(task.weight, 0.0) ** 0.5
            weighted_jacobian = np.asarray(jacobian, dtype=np.float64) * scale
            weighted_error = (np.asarray(error, dtype=np.float64) * scale) / dt
            projected_jacobian = weighted_jacobian @ nullspace
            if not np.any(projected_jacobian):
                continue
            projected_pinv = _damped_pseudoinverse(projected_jacobian, self.damping)
            joint_velocity = joint_velocity + projected_pinv @ (weighted_error - weighted_jacobian @ joint_velocity)
            nullspace = nullspace @ (np.eye(degrees_of_freedom, dtype=np.float64) - projected_pinv @ projected_jacobian)

        return {
            joint_name: float(
                np.clip(
                    joint_velocity[index],
                    -robot.joints[joint_name].velocity_limit,
                    robot.joints[joint_name].velocity_limit,
                )
            )
            for index, joint_name in enumerate(joint_names)
        }

    def step(self, robot: RobotModel, dt: float) -> dict[str, float]:
        """Apply one WBC velocity step and return the updated joint positions."""

        velocities = self.compute_joint_velocities(robot, dt)
        updated_positions = {
            joint_name: robot.joints[joint_name].clamp(robot.joint_positions[joint_name] + velocity * dt)
            for joint_name, velocity in velocities.items()
        }
        robot.set_joint_positions(updated_positions)
        return dict(robot.joint_positions)

    def solve(
        self,
        robot: RobotModel,
        dt: float,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
    ) -> WBCSolution:
        """Iterate WBC steps until all tasks are within tolerance or the budget is exhausted."""

        task_errors = self._task_errors(robot)
        if all(error <= tolerance for error in task_errors.values()):
            return WBCSolution(
                iterations=0,
                converged=True,
                task_errors=task_errors,
                joint_positions=dict(robot.joint_positions),
            )

        for iteration in range(1, max_iterations + 1):
            self.step(robot, dt)
            task_errors = self._task_errors(robot)
            if all(error <= tolerance for error in task_errors.values()):
                return WBCSolution(
                    iterations=iteration,
                    converged=True,
                    task_errors=task_errors,
                    joint_positions=dict(robot.joint_positions),
                )

        return WBCSolution(
            iterations=max_iterations,
            converged=False,
            task_errors=task_errors,
            joint_positions=dict(robot.joint_positions),
        )

    def _task_errors(self, robot: RobotModel) -> dict[str, float]:
        return {
            _task_name(task): float(np.linalg.norm(task.compute_jacobian_and_error(robot)[1]))
            for task in self.tasks
        }


def build_wbc_controller(tasks: list[WBCTask]) -> WBCController:
    """Build a controller from an initial task list."""

    return WBCController(tasks=list(tasks))


def _task_name(task: WBCTask) -> str:
    if isinstance(task, EndEffectorTask):
        return f"{task.__class__.__name__}[{task.end_effector}]"
    return task.__class__.__name__


def _damped_pseudoinverse(matrix: Matrix, damping: float) -> Matrix:
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.zeros((matrix.shape[1], matrix.shape[0]), dtype=np.float64)
    regularized = matrix @ matrix.T + (damping**2) * np.eye(matrix.shape[0], dtype=np.float64)
    return matrix.T @ np.linalg.solve(regularized, np.eye(matrix.shape[0], dtype=np.float64))


def _orientation_error(current_rotation: Matrix, target_rotation: Matrix) -> Vector:
    return 0.5 * (
        np.cross(current_rotation[:, 0], target_rotation[:, 0])
        + np.cross(current_rotation[:, 1], target_rotation[:, 1])
        + np.cross(current_rotation[:, 2], target_rotation[:, 2])
    )


def _numerical_end_effector_jacobian(
    robot: RobotModel,
    end_effector: str,
    joint_names: list[str],
    positions: dict[str, float],
    position_only: bool,
    epsilon: float = 1e-5,
) -> Matrix:
    rows = 3 if position_only else 6
    jacobian = np.zeros((rows, len(joint_names)), dtype=np.float64)
    base_pose = robot.end_effector_pose(end_effector, positions)
    base_rotation = base_pose.orientation.to_rotation_matrix()

    for index, joint_name in enumerate(joint_names):
        shifted_positions = dict(positions)
        shifted_positions[joint_name] = robot.joints[joint_name].clamp(shifted_positions[joint_name] + epsilon)
        shifted_pose = robot.end_effector_pose(end_effector, shifted_positions)
        jacobian[:3, index] = (shifted_pose.position - base_pose.position) / epsilon
        if not position_only:
            shifted_rotation = shifted_pose.orientation.to_rotation_matrix()
            jacobian[3:, index] = _orientation_error(base_rotation, shifted_rotation) / epsilon

    return jacobian


def _center_of_mass_proxy(robot: RobotModel, positions: dict[str, float] | None = None) -> Vector:
    poses = robot.forward_kinematics(positions)
    support_positions = [
        pose.position
        for name, pose in poses.items()
        if any(token in name for token in ("foot", "ankle", "leg"))
    ]
    if support_positions:
        return np.mean(np.asarray(support_positions, dtype=np.float64), axis=0)
    link_positions = [pose.position for pose in poses.values()]
    return np.mean(np.asarray(link_positions, dtype=np.float64), axis=0)
