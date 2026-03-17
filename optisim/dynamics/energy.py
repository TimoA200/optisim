"""Energy analysis helpers for lightweight dynamics validation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optisim.dynamics.rigid_body import RigidBodyState


def kinetic_energy(state: RigidBodyState) -> float:
    """Return linear plus rotational kinetic energy."""

    linear = 0.5 * state.mass * float(np.dot(state.velocity, state.velocity))
    rotational = 0.5 * float(state.angular_velocity @ state.inertia_tensor @ state.angular_velocity)
    return float(linear + rotational)


def potential_energy(mass: float, height: float, g: float = 9.81) -> float:
    """Return gravitational potential energy relative to zero height."""

    return float(mass) * float(g) * float(height)


def total_mechanical_energy(state: RigidBodyState, height: float, g: float = 9.81) -> float:
    """Return the total mechanical energy of a rigid body."""

    return kinetic_energy(state) + potential_energy(state.mass, height, g=g)


def joint_power(torque: float, angular_velocity: float) -> float:
    """Return instantaneous joint power."""

    return float(torque) * float(angular_velocity)


@dataclass(slots=True)
class TaskEnergyProfile:
    """Aggregate task-level energy metrics."""

    total_energy: float
    peak_power: float
    energy_per_action: dict[str, float] = field(default_factory=dict)

__all__ = ["kinetic_energy", "potential_energy", "total_mechanical_energy", "joint_power", "TaskEnergyProfile"]
