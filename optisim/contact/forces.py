"""Contact force models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optisim.contact.geometry import ContactPair


def _vec3(value: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("expected a 3D vector")
    return array


@dataclass(slots=True)
class ContactParams:
    """Parameters for spring-damper contact and Coulomb friction."""

    stiffness: float = 1000.0
    damping: float = 10.0
    friction_coeff: float = 0.5
    restitution: float = 0.2


def compute_normal_force(depth: float, depth_dot: float, params: ContactParams) -> float:
    """Return the scalar normal force from a linear spring-damper model."""

    if depth <= 0.0:
        return 0.0
    return float(max(0.0, params.stiffness * depth + params.damping * depth_dot))


def compute_friction_force(
    normal_force: float,
    tangential_velocity: np.ndarray,
    params: ContactParams,
) -> np.ndarray:
    """Return Coulomb friction opposing tangential slip."""

    tangential_velocity = _vec3(tangential_velocity)
    speed = float(np.linalg.norm(tangential_velocity))
    if normal_force <= 0.0 or speed <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    limit = params.friction_coeff * normal_force
    return -(tangential_velocity / speed) * limit


@dataclass(slots=True)
class ContactForceModel:
    """Compute net contact forces over a contact manifold."""

    params: ContactParams = field(default_factory=ContactParams)

    def apply(
        self,
        pair: ContactPair,
        velocity_a: np.ndarray,
        velocity_b: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        """Return equal-and-opposite forces for a contact pair."""

        velocity_a = _vec3(velocity_a)
        velocity_b = _vec3(velocity_b)
        total_force_on_a = np.zeros(3, dtype=np.float64)
        total_normal_force = 0.0
        total_friction_force = np.zeros(3, dtype=np.float64)
        relative_velocity = velocity_a - velocity_b

        for contact in pair.contacts:
            depth_dot = float(np.dot(relative_velocity, contact.normal))
            normal_force = compute_normal_force(contact.depth, depth_dot, self.params)
            tangential_velocity = relative_velocity - np.dot(relative_velocity, contact.normal) * contact.normal
            friction_force = compute_friction_force(normal_force, tangential_velocity, self.params)
            force_on_a = -normal_force * contact.normal + friction_force
            total_force_on_a += force_on_a
            total_normal_force += normal_force
            total_friction_force += friction_force

        return {
            "force_on_a": total_force_on_a,
            "force_on_b": -total_force_on_a,
            "normal_force": float(total_normal_force),
            "friction_force": total_friction_force,
        }


__all__ = [
    "ContactParams",
    "compute_normal_force",
    "compute_friction_force",
    "ContactForceModel",
]
