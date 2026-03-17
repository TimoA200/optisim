"""Iterative inverse kinematics for built-in robot models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optisim.math3d import Pose
from optisim.robot.model import RobotModel

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(slots=True)
class IKOptions:
    """Numerical settings for damped least-squares IK."""

    max_iterations: int = 100
    convergence_threshold: float = 1e-3
    damping: float = 0.08
    step_scale: float = 1.0
    orientation_weight: float = 0.35
    position_only: bool = False


@dataclass(slots=True)
class IKResult:
    """Result payload returned by the IK solver."""

    success: bool
    iterations: int
    position_error: float
    orientation_error: float
    joint_positions: dict[str, float]


def solve_inverse_kinematics(
    robot: RobotModel,
    effector: str,
    target: Pose,
    *,
    joint_names: list[str] | None = None,
    initial_positions: dict[str, float] | None = None,
    options: IKOptions | None = None,
) -> IKResult:
    """Solve IK for the requested end-effector pose."""

    options = options or IKOptions()
    active_chain = joint_names or [joint.name for joint in robot.joint_chain_for_effector(effector)]
    positions = dict(robot.joint_positions)
    if initial_positions:
        for name, value in initial_positions.items():
            positions[name] = robot.joints[name].clamp(value)

    best_positions = dict(positions)
    best_position_error = float("inf")
    best_orientation_error = float("inf")

    for iteration in range(1, options.max_iterations + 1):
        current_pose = robot.end_effector_pose(effector, positions)
        error_vector = _pose_error(current_pose, target, options)
        position_error = float(np.linalg.norm(target.position - current_pose.position))
        orientation_error = 0.0 if options.position_only else float(np.linalg.norm(error_vector[3:]))
        total_error = float(np.linalg.norm(error_vector))

        if total_error < best_position_error + best_orientation_error:
            best_positions = dict(positions)
            best_position_error = position_error
            best_orientation_error = orientation_error

        if total_error <= options.convergence_threshold:
            return IKResult(
                success=True,
                iterations=iteration,
                position_error=position_error,
                orientation_error=orientation_error,
                joint_positions=_extract_positions(positions, active_chain),
            )

        jacobian = _numerical_jacobian(robot, effector, active_chain, positions, options.position_only)
        lhs = jacobian @ jacobian.T + (options.damping**2) * np.eye(jacobian.shape[0], dtype=np.float64)
        delta_task = np.linalg.solve(lhs, error_vector)
        delta_joints = jacobian.T @ delta_task

        if np.linalg.norm(delta_joints) < options.convergence_threshold * 0.1:
            break

        for joint_name, delta in zip(active_chain, delta_joints, strict=True):
            spec = robot.joints[joint_name]
            positions[joint_name] = spec.clamp(positions[joint_name] + options.step_scale * float(delta))

    return IKResult(
        success=False,
        iterations=options.max_iterations,
        position_error=best_position_error,
        orientation_error=best_orientation_error,
        joint_positions=_extract_positions(best_positions, active_chain),
    )


def _extract_positions(positions: dict[str, float], joint_names: list[str]) -> dict[str, float]:
    return {name: positions[name] for name in joint_names}


def _pose_error(current: Pose, target: Pose, options: IKOptions) -> Vector:
    position_error = target.position - current.position
    if options.position_only:
        return position_error.astype(np.float64)
    orientation_error = _orientation_error(
        current.orientation.to_rotation_matrix(),
        target.orientation.to_rotation_matrix(),
    )
    return np.concatenate(
        [position_error, orientation_error * options.orientation_weight],
        dtype=np.float64,
    )


def _orientation_error(current_rotation: Matrix, target_rotation: Matrix) -> Vector:
    # SO(3) error expressed as a world-frame rotation vector approximation.
    return 0.5 * (
        np.cross(current_rotation[:, 0], target_rotation[:, 0])
        + np.cross(current_rotation[:, 1], target_rotation[:, 1])
        + np.cross(current_rotation[:, 2], target_rotation[:, 2])
    )


def _numerical_jacobian(
    robot: RobotModel,
    effector: str,
    joint_names: list[str],
    positions: dict[str, float],
    position_only: bool,
    epsilon: float = 1e-4,
) -> Matrix:
    rows = 3 if position_only else 6
    jacobian = np.zeros((rows, len(joint_names)), dtype=np.float64)
    base_pose = robot.end_effector_pose(effector, positions)
    base_rotation = base_pose.orientation.to_rotation_matrix()

    for index, joint_name in enumerate(joint_names):
        shifted_positions = dict(positions)
        shifted_positions[joint_name] = robot.joints[joint_name].clamp(shifted_positions[joint_name] + epsilon)
        shifted_pose = robot.end_effector_pose(effector, shifted_positions)
        jacobian[:3, index] = (shifted_pose.position - base_pose.position) / epsilon
        if not position_only:
            shifted_rotation = shifted_pose.orientation.to_rotation_matrix()
            jacobian[3:, index] = _orientation_error(base_rotation, shifted_rotation) / epsilon

    return jacobian
