"""Basic contact geometry routines."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _vec3(value: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("expected a 3D vector")
    return array


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return value / norm


@dataclass(slots=True)
class ContactPoint:
    """Single contact point between two bodies."""

    position: np.ndarray
    normal: np.ndarray
    depth: float
    body_a: str
    body_b: str

    def __post_init__(self) -> None:
        self.position = _vec3(self.position)
        self.normal = _normalize(_vec3(self.normal))
        self.depth = float(max(self.depth, 0.0))
        self.body_a = str(self.body_a)
        self.body_b = str(self.body_b)


@dataclass(slots=True)
class ContactPair:
    """Contact manifold between two bodies."""

    body_a: str
    body_b: str
    contacts: list[ContactPoint] = field(default_factory=list)


def sphere_sphere_contact(
    center_a: np.ndarray,
    radius_a: float,
    center_b: np.ndarray,
    radius_b: float,
    name_a: str = "A",
    name_b: str = "B",
) -> ContactPair | None:
    """Return the contact manifold for two spheres if they overlap or touch."""

    point_a = _vec3(center_a)
    point_b = _vec3(center_b)
    radius_a = float(radius_a)
    radius_b = float(radius_b)
    delta = point_b - point_a
    distance = float(np.linalg.norm(delta))
    total_radius = radius_a + radius_b
    depth = total_radius - distance
    if depth < 0.0:
        return None
    normal = _normalize(delta if distance > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=np.float64))
    position = point_a + normal * (radius_a - max(depth, 0.0) * 0.5)
    return ContactPair(
        body_a=name_a,
        body_b=name_b,
        contacts=[ContactPoint(position=position, normal=normal, depth=depth, body_a=name_a, body_b=name_b)],
    )


def box_sphere_contact(
    box_center: np.ndarray,
    box_half_extents: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
    name_box: str = "box",
    name_sphere: str = "sphere",
) -> ContactPair | None:
    """Return the contact manifold for an axis-aligned box and sphere."""

    center_box = _vec3(box_center)
    half_extents = _vec3(box_half_extents)
    center_sphere = _vec3(sphere_center)
    sphere_radius = float(sphere_radius)

    local = center_sphere - center_box
    closest_local = np.clip(local, -half_extents, half_extents)
    closest_world = center_box + closest_local
    delta = center_sphere - closest_world
    distance = float(np.linalg.norm(delta))

    if distance > sphere_radius:
        return None

    if distance > 1e-12:
        normal = _normalize(delta)
        depth = sphere_radius - distance
        position = closest_world
    else:
        margins = half_extents - np.abs(local)
        axis = int(np.argmin(margins))
        sign = 1.0 if local[axis] >= 0.0 else -1.0
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = sign
        face_local = local.copy()
        face_local[axis] = sign * half_extents[axis]
        position = center_box + face_local
        depth = sphere_radius + float(margins[axis])

    return ContactPair(
        body_a=name_box,
        body_b=name_sphere,
        contacts=[ContactPoint(position=position, normal=normal, depth=depth, body_a=name_box, body_b=name_sphere)],
    )


def aabb_aabb_contact(
    center_a: np.ndarray,
    half_a: np.ndarray,
    center_b: np.ndarray,
    half_b: np.ndarray,
    name_a: str = "A",
    name_b: str = "B",
) -> ContactPair | None:
    """Return the contact manifold for two axis-aligned boxes."""

    point_a = _vec3(center_a)
    point_b = _vec3(center_b)
    half_a = _vec3(half_a)
    half_b = _vec3(half_b)
    delta = point_b - point_a
    overlap = half_a + half_b - np.abs(delta)
    if np.any(overlap < 0.0):
        return None

    axis = int(np.argmin(overlap))
    sign = 1.0 if delta[axis] >= 0.0 else -1.0
    normal = np.zeros(3, dtype=np.float64)
    normal[axis] = sign
    position = (point_a + point_b) * 0.5
    depth = float(overlap[axis])
    return ContactPair(
        body_a=name_a,
        body_b=name_b,
        contacts=[ContactPoint(position=position, normal=normal, depth=depth, body_a=name_a, body_b=name_b)],
    )


__all__ = [
    "ContactPoint",
    "ContactPair",
    "sphere_sphere_contact",
    "box_sphere_contact",
    "aabb_aabb_contact",
]
