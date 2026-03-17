"""Broad-phase and lightweight narrow-phase collision checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optisim.sim.world import ObjectState, Surface


@dataclass(slots=True)
class Collision:
    """Collision report between two named entities."""

    entity_a: str
    entity_b: str
    penetration_depth: float


def intersect_aabb(
    a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray
) -> Collision | None:
    """Return an overlap report for two axis-aligned bounding boxes."""

    overlaps = np.minimum(a_max, b_max) - np.maximum(a_min, b_min)
    if np.any(overlaps <= 0.0):
        return None
    return Collision(entity_a="", entity_b="", penetration_depth=float(np.min(overlaps)))


def object_surface_collision(obj: ObjectState, surface: Surface) -> Collision | None:
    """Check whether an object intersects a support surface AABB."""

    obj_min, obj_max = obj.aabb
    half = np.asarray(surface.size, dtype=np.float64) / 2.0
    surf_min = surface.pose.position - half
    surf_max = surface.pose.position + half
    hit = intersect_aabb(obj_min, obj_max, surf_min, surf_max)
    if hit is None:
        return None
    return Collision(entity_a=obj.name, entity_b=surface.name, penetration_depth=hit.penetration_depth)


def mesh_hint_collision(vertices_a: np.ndarray, vertices_b: np.ndarray) -> bool:
    """Very lightweight mesh-style overlap using vertex cloud bounds."""

    if vertices_a.size == 0 or vertices_b.size == 0:
        return False
    a_min = vertices_a.min(axis=0)
    a_max = vertices_a.max(axis=0)
    b_min = vertices_b.min(axis=0)
    b_max = vertices_b.max(axis=0)
    return intersect_aabb(a_min, a_max, b_min, b_max) is not None
