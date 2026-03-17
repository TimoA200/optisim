"""Rigid-body state and deterministic Newton-Euler integration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from optisim.math3d import vec3

Matrix3 = NDArray[np.float64]
Vector3 = NDArray[np.float64]


def _matrix3(values: NDArray[np.float64] | list[list[float]] | tuple[tuple[float, ...], ...]) -> Matrix3:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"expected 3x3 matrix, received shape {matrix.shape}")
    return matrix


@dataclass(slots=True)
class RigidBodyState:
    """Compact rigid-body state used for lightweight dynamics checks."""

    mass: float
    inertia_tensor: Matrix3
    position: Vector3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity: Vector3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    angular_velocity: Vector3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    forces: Vector3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    torques: Vector3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self) -> None:
        self.mass = float(self.mass)
        if self.mass < 0.0:
            raise ValueError("mass must be non-negative")
        self.inertia_tensor = _matrix3(self.inertia_tensor)
        self.position = vec3(self.position)
        self.velocity = vec3(self.velocity)
        self.angular_velocity = vec3(self.angular_velocity)
        self.forces = vec3(self.forces)
        self.torques = vec3(self.torques)


def step_dynamics(state: RigidBodyState, dt: float) -> RigidBodyState:
    """Advance a rigid body one explicit Euler step."""

    step = float(dt)
    if step < 0.0:
        raise ValueError("dt must be non-negative")

    linear_acceleration = np.zeros(3, dtype=np.float64)
    if state.mass > 0.0:
        linear_acceleration = state.forces / state.mass

    angular_acceleration = np.linalg.pinv(state.inertia_tensor) @ state.torques

    return RigidBodyState(
        mass=state.mass,
        inertia_tensor=state.inertia_tensor.copy(),
        position=state.position + state.velocity * step,
        velocity=state.velocity + linear_acceleration * step,
        angular_velocity=state.angular_velocity + angular_acceleration * step,
        forces=np.zeros(3, dtype=np.float64),
        torques=np.zeros(3, dtype=np.float64),
    )


def compute_inertia_box(mass: float, size: tuple[float, float, float] | list[float] | Vector3) -> Matrix3:
    """Return the diagonal inertia tensor for a solid box."""

    sx, sy, sz = vec3(size)
    value = float(mass) / 12.0
    return np.diag(
        [
            value * (sy**2 + sz**2),
            value * (sx**2 + sz**2),
            value * (sx**2 + sy**2),
        ]
    ).astype(np.float64)


def compute_inertia_cylinder(mass: float, radius: float, height: float) -> Matrix3:
    """Return the diagonal inertia tensor for a solid cylinder aligned with z."""

    m = float(mass)
    r = float(radius)
    h = float(height)
    radial = m * (3.0 * r**2 + h**2) / 12.0
    axial = 0.5 * m * r**2
    return np.diag([radial, radial, axial]).astype(np.float64)


def gravitational_force(mass: float, g: float = 9.81) -> Vector3:
    """Return the gravitational force vector for a mass."""

    return vec3([0.0, 0.0, -float(mass) * float(g)])

__all__ = ["Matrix3", "Vector3", "RigidBodyState", "step_dynamics", "compute_inertia_box", "compute_inertia_cylinder", "gravitational_force"]
