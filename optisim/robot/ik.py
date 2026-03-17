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
    failure_reason: str | None = None


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
    best_total_error = float("inf")
    terminated_due_to_small_step = False

    for iteration in range(1, options.max_iterations + 1):
        current_pose = robot.end_effector_pose(effector, positions)
        error_vector = _pose_error(current_pose, target, options)
        position_error = float(np.linalg.norm(target.position - current_pose.position))
        orientation_error = 0.0 if options.position_only else float(np.linalg.norm(error_vector[3:]))
        total_error = float(np.linalg.norm(error_vector))

        if total_error < best_total_error:
            best_positions = dict(positions)
            best_position_error = position_error
            best_orientation_error = orientation_error
            best_total_error = total_error

        if total_error <= options.convergence_threshold:
            return IKResult(
                success=True,
                iterations=iteration,
                position_error=position_error,
                orientation_error=orientation_error,
                joint_positions=_extract_positions(positions, active_chain),
                failure_reason=None,
            )

        jacobian = _numerical_jacobian(robot, effector, active_chain, positions, options.position_only)
        lhs = jacobian @ jacobian.T + (options.damping**2) * np.eye(jacobian.shape[0], dtype=np.float64)
        delta_task = np.linalg.solve(lhs, error_vector)
        delta_joints = jacobian.T @ delta_task

        if np.linalg.norm(delta_joints) < options.convergence_threshold * 0.1:
            terminated_due_to_small_step = True
            break

        for joint_name, delta in zip(active_chain, delta_joints, strict=True):
            spec = robot.joints[joint_name]
            positions[joint_name] = spec.clamp(positions[joint_name] + options.step_scale * float(delta))

    return IKResult(
        success=False,
        iterations=min(iteration, options.max_iterations),
        position_error=best_position_error,
        orientation_error=best_orientation_error,
        joint_positions=_extract_positions(best_positions, active_chain),
        failure_reason=_classify_failure(
            robot,
            effector,
            active_chain,
            target,
            best_positions,
            best_position_error,
            terminated_due_to_small_step=terminated_due_to_small_step,
            convergence_threshold=options.convergence_threshold,
            position_only=options.position_only,
        ),
    )


def _classify_failure(
    robot: RobotModel,
    effector: str,
    active_chain: list[str],
    target: Pose,
    positions: dict[str, float],
    position_error: float,
    *,
    terminated_due_to_small_step: bool,
    convergence_threshold: float,
    position_only: bool,
) -> str:
    """Provide a human-readable explanation for an IK failure."""

    current_pose = robot.end_effector_pose(effector, positions)
    target_distance = float(np.linalg.norm(target.position - robot.base_pose.position))
    max_reach = robot.max_reach()
    if target_distance > max_reach * 1.05:
        return (
            f"target is out of reach ({target_distance:.3f}m from base, "
            f"robot reach is about {max_reach:.3f}m)"
        )

    saturated_joints = [
        name
        for name in active_chain
        if np.isclose(positions[name], robot.joints[name].limit_lower, atol=1e-3)
        or np.isclose(positions[name], robot.joints[name].limit_upper, atol=1e-3)
    ]
    if saturated_joints and position_error > convergence_threshold * 2.0:
        joint_list = ", ".join(saturated_joints[:4])
        suffix = "..." if len(saturated_joints) > 4 else ""
        return f"joint limits prevented convergence ({joint_list}{suffix})"

    if float(np.linalg.norm(target.position - current_pose.position)) > convergence_threshold * 2.0 and terminated_due_to_small_step:
        return "solver stalled before reaching the target pose"

    if terminated_due_to_small_step:
        mode = "position" if position_only else "pose"
        return f"{mode} target did not converge before the solver stalled"
    return "solver exceeded the iteration limit before convergence"


def _extract_positions(positions: dict[str, float], joint_names: list[str]) -> dict[str, float]:
    """Return a filtered joint-position mapping for the active IK joints."""

    return {name: positions[name] for name in joint_names}


def _pose_error(current: Pose, target: Pose, options: IKOptions) -> Vector:
    """Compute the task-space error vector for the current IK iteration."""

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
    """Approximate rotational error as a world-frame rotation vector."""

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
    """Estimate the task Jacobian with forward finite differences."""

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
