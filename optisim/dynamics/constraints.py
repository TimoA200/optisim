"""Constraint models and lightweight physical plausibility checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from optisim.math3d import vec3
from optisim.robot import RobotModel
from optisim.sim.world import ObjectState


@dataclass(slots=True)
class JointTorqueLimit:
    """Maximum allowable torque for a single robot joint."""

    joint_name: str
    max_torque: float


@dataclass(slots=True)
class PayloadConstraint:
    """Maximum payload that an end effector is allowed to carry."""

    max_payload_kg: float
    end_effector: str


@dataclass(slots=True)
class WorkspaceConstraint:
    """Axis-aligned workspace bounds expressed in world coordinates."""

    bounds_min: np.ndarray
    bounds_max: np.ndarray

    def __post_init__(self) -> None:
        self.bounds_min = vec3(self.bounds_min)
        self.bounds_max = vec3(self.bounds_max)


@dataclass(slots=True)
class ConstraintViolation:
    """Single constraint violation emitted by the dynamics validator."""

    constraint_type: str
    message: str
    severity: str = "error"
    joint_name: str | None = None
    value: float | None = None
    limit: float | None = None


@dataclass(slots=True)
class ConstraintSet:
    """Grouped dynamics constraints evaluated during task validation."""

    joint_torque_limits: list[JointTorqueLimit] = field(default_factory=list)
    payload_constraints: list[PayloadConstraint] = field(default_factory=list)
    workspace_constraints: list[WorkspaceConstraint] = field(default_factory=list)


def check_joint_torques(
    robot: RobotModel,
    joint_positions: dict[str, float],
    payload: float | ObjectState | None,
    *,
    torque_limits: Sequence[JointTorqueLimit] | None = None,
    end_effector: str | None = None,
    g: float = 9.81,
) -> list[ConstraintViolation]:
    """Estimate joint loading from a payload and report exceeded limits."""

    limits = list(torque_limits or getattr(robot, "joint_torque_limits", []))
    if not limits:
        return []

    payload_mass = _payload_mass(payload)
    if payload_mass <= 0.0:
        return []

    frames = robot.joint_frames(joint_positions)
    poses = robot.forward_kinematics(joint_positions)
    effector_links = _resolve_effector_links(robot, end_effector)
    if not effector_links:
        effector_links = list(robot.end_effectors.values())

    violations: list[ConstraintViolation] = []
    for limit in limits:
        if limit.joint_name not in frames:
            continue
        joint_origin = frames[limit.joint_name][0][:3, 3]
        lever_arm = max(
            float(np.linalg.norm(poses[link].position - joint_origin))
            for link in effector_links
            if link in poses
        )
        estimated_torque = payload_mass * float(g) * lever_arm
        if estimated_torque > float(limit.max_torque):
            violations.append(
                ConstraintViolation(
                    constraint_type="joint_torque",
                    message=(
                        f"joint '{limit.joint_name}' torque estimate {estimated_torque:.2f}Nm "
                        f"exceeds limit {limit.max_torque:.2f}Nm"
                    ),
                    joint_name=limit.joint_name,
                    value=estimated_torque,
                    limit=float(limit.max_torque),
                )
            )
    return violations


def check_payload(
    robot: RobotModel,
    payload_mass: float,
    end_effector: str,
    constraint: PayloadConstraint | None = None,
) -> ConstraintViolation | None:
    """Validate a payload mass against a payload constraint."""

    if end_effector not in robot.end_effectors and end_effector not in robot.end_effectors.values():
        return ConstraintViolation(
            constraint_type="payload",
            message=f"end effector '{end_effector}' not present in robot model",
            severity="warning",
        )

    if constraint is None:
        return None
    if _canonical_effector(robot, end_effector) != _canonical_effector(
        robot, constraint.end_effector
    ):
        return None
    if float(payload_mass) <= float(constraint.max_payload_kg):
        return None
    return ConstraintViolation(
        constraint_type="payload",
        message=(
            f"payload {float(payload_mass):.2f}kg exceeds "
            f"limit {constraint.max_payload_kg:.2f}kg for '{constraint.end_effector}'"
        ),
        value=float(payload_mass),
        limit=float(constraint.max_payload_kg),
    )


def check_workspace_bounds(
    position: np.ndarray | list[float] | tuple[float, float, float],
    bounds: WorkspaceConstraint,
) -> ConstraintViolation | None:
    """Validate that a position lies inside workspace bounds."""

    point = vec3(position)
    if np.all(point >= bounds.bounds_min) and np.all(point <= bounds.bounds_max):
        return None
    distance = float(np.max(np.maximum(bounds.bounds_min - point, point - bounds.bounds_max)))
    return ConstraintViolation(
        constraint_type="workspace",
        message=(
            f"position {point.tolist()} lies outside workspace bounds "
            f"{bounds.bounds_min.tolist()} .. {bounds.bounds_max.tolist()}"
        ),
        value=distance,
        limit=0.0,
    )


def _payload_mass(payload: float | ObjectState | None) -> float:
    if payload is None:
        return 0.0
    if isinstance(payload, ObjectState):
        return float(payload.mass_kg)
    return float(payload)


def _canonical_effector(robot: RobotModel, end_effector: str) -> str:
    return robot.end_effectors.get(end_effector, end_effector)


def _resolve_effector_links(robot: RobotModel, end_effector: str | None) -> list[str]:
    if end_effector is None:
        return []
    canonical = _canonical_effector(robot, end_effector)
    return [canonical]
