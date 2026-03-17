"""Grasp stability metrics and force-closure approximations."""

from __future__ import annotations

import numpy as np

from optisim.grasp.contact import ContactPoint
from optisim.math3d import normalize


def force_closure(contacts: list[ContactPoint]) -> bool:
    """Return a conservative approximation of whether contacts achieve force closure."""

    if len(contacts) < 2:
        return False
    normals = np.asarray([contact.normal for contact in contacts], dtype=np.float64)
    opposition = min(float(np.dot(a, b)) for index, a in enumerate(normals) for b in normals[index + 1 :])
    mean_friction = float(np.mean([contact.friction_coeff for contact in contacts]))
    wrench_rank = int(np.linalg.matrix_rank(grasp_wrench_space(contacts)))
    if len(contacts) == 2:
        return opposition < -0.9 and mean_friction >= 0.2
    return opposition < -0.2 and mean_friction >= 0.2 and wrench_rank >= 4


def grasp_wrench_space(contacts: list[ContactPoint]) -> np.ndarray:
    """Construct a linearized grasp wrench basis from contact normals and friction edges."""

    if not contacts:
        return np.zeros((6, 0), dtype=np.float64)
    center = np.mean([contact.position for contact in contacts], axis=0)
    columns: list[np.ndarray] = []
    for contact in contacts:
        inward = -contact.normal
        tangent_a, tangent_b = _tangent_basis(contact.normal)
        directions = [
            inward,
            normalize(inward + contact.friction_coeff * tangent_a),
            normalize(inward - contact.friction_coeff * tangent_a),
            normalize(inward + contact.friction_coeff * tangent_b),
            normalize(inward - contact.friction_coeff * tangent_b),
        ]
        lever_arm = contact.position - center
        for direction in directions:
            torque = np.cross(lever_arm, direction)
            columns.append(np.concatenate([direction, torque]))
    return np.column_stack(columns) if columns else np.zeros((6, 0), dtype=np.float64)


def min_resisted_wrench(contacts: list[ContactPoint]) -> float:
    """Return a Ferrari-Canny-like scalar from the wrench basis singular values."""

    wrench_space = grasp_wrench_space(contacts)
    if wrench_space.size == 0:
        return 0.0
    singular_values = np.linalg.svd(wrench_space, compute_uv=False)
    if singular_values.size == 0:
        return 0.0
    non_zero = singular_values[singular_values > 1e-9]
    if non_zero.size == 0:
        return 0.0
    return float(np.min(non_zero))


def slip_margin(contact: ContactPoint, load: np.ndarray) -> float:
    """Return the positive slack before the supplied load would slip."""

    force = np.asarray(load, dtype=np.float64)
    inward_normal_force = max(-float(np.dot(force, contact.normal)), 0.0)
    tangential = force + inward_normal_force * contact.normal
    tangential_norm = float(np.linalg.norm(tangential))
    return contact.friction_coeff * inward_normal_force - tangential_norm


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(seed, normal))) > 0.9:
        seed = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    tangent_a = normalize(np.cross(normal, seed))
    tangent_b = normalize(np.cross(normal, tangent_a))
    return tangent_a, tangent_b

__all__ = ["force_closure", "grasp_wrench_space", "min_resisted_wrench", "slip_margin"]
