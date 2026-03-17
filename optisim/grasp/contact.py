"""Contact point and contact patch modeling for grasp analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from optisim.math3d import Pose, Quaternion, normalize, vec3


@dataclass(slots=True)
class ContactPoint:
    """Single point contact between a gripper and an object surface."""

    position: np.ndarray
    normal: np.ndarray
    friction_coeff: float = 0.5
    force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self) -> None:
        self.position = vec3(self.position)
        self.normal = normalize(vec3(self.normal))
        self.force = vec3(self.force)
        self.friction_coeff = float(max(self.friction_coeff, 0.0))


@dataclass(slots=True)
class ContactPatch:
    """Group of related contacts treated as a single support patch."""

    points: list[ContactPoint] = field(default_factory=list)

    def add(self, contact: ContactPoint) -> None:
        """Append a contact to the patch."""

        self.points.append(contact)

    @property
    def centroid(self) -> np.ndarray:
        """Return the centroid of all contact positions."""

        if not self.points:
            return np.zeros(3, dtype=np.float64)
        return np.mean([point.position for point in self.points], axis=0)

    @property
    def total_force(self) -> np.ndarray:
        """Return the aggregate force over the contact patch."""

        if not self.points:
            return np.zeros(3, dtype=np.float64)
        return np.sum([point.force for point in self.points], axis=0)


def surface_contacts(obj_geometry: Any, gripper_geometry: Any) -> list[ContactPoint]:
    """Return deterministic contact hypotheses for a gripper pose on an object."""

    obj_pose, obj_size, _ = _coerce_object_geometry(obj_geometry)
    grip_pose, grip_type, aperture, friction = _coerce_gripper_geometry(gripper_geometry)
    face_centers = _box_faces(obj_pose, obj_size)

    if grip_type == "parallel_jaw":
        closing_axis = grip_pose.orientation.to_rotation_matrix()[:, 0]
        required_span = _box_projection_width(obj_pose.orientation.to_rotation_matrix(), obj_size, closing_axis)
        if aperture is not None and required_span > aperture + 1e-9:
            return []
        positive_face = max(face_centers, key=lambda item: float(np.dot(item[1], closing_axis)))
        negative_face = min(face_centers, key=lambda item: float(np.dot(item[1], closing_axis)))
        return [
            ContactPoint(position=positive_face[0], normal=positive_face[1], friction_coeff=friction),
            ContactPoint(position=negative_face[0], normal=negative_face[1], friction_coeff=friction),
        ]

    if grip_type == "suction":
        face = min(face_centers, key=lambda item: float(np.linalg.norm(item[0] - grip_pose.position)))
        return [ContactPoint(position=face[0], normal=face[1], friction_coeff=friction)]

    closing_axis = grip_pose.orientation.to_rotation_matrix()[:, 0]
    lateral_axis = grip_pose.orientation.to_rotation_matrix()[:, 1]
    required_span = _box_projection_width(obj_pose.orientation.to_rotation_matrix(), obj_size, closing_axis)
    if aperture is not None and required_span > aperture + 1e-9:
        return []
    positive_closing = max(face_centers, key=lambda item: float(np.dot(item[1], closing_axis)))
    negative_closing = min(face_centers, key=lambda item: float(np.dot(item[1], closing_axis)))
    positive_lateral = max(face_centers, key=lambda item: float(np.dot(item[1], lateral_axis)))
    return [
        ContactPoint(position=positive_closing[0], normal=positive_closing[1], friction_coeff=friction),
        ContactPoint(position=negative_closing[0], normal=negative_closing[1], friction_coeff=friction),
        ContactPoint(position=positive_lateral[0], normal=positive_lateral[1], friction_coeff=friction),
    ]


def friction_cone_check(contact: ContactPoint, applied_force: np.ndarray) -> bool:
    """Return whether an applied force stays within a Coulomb friction cone."""

    force = vec3(applied_force)
    inward_normal_force = max(-float(np.dot(force, contact.normal)), 0.0)
    tangential = force + inward_normal_force * contact.normal
    tangential_norm = float(np.linalg.norm(tangential))
    if inward_normal_force == 0.0:
        return tangential_norm <= 1e-9
    return tangential_norm <= contact.friction_coeff * inward_normal_force + 1e-9


def _coerce_object_geometry(obj_geometry: Any) -> tuple[Pose, tuple[float, float, float], float]:
    if hasattr(obj_geometry, "pose") and hasattr(obj_geometry, "size"):
        mass_kg = float(getattr(obj_geometry, "mass_kg", 1.0))
        return obj_geometry.pose, tuple(float(v) for v in obj_geometry.size), mass_kg
    if isinstance(obj_geometry, dict):
        pose_payload = obj_geometry.get("pose", {})
        pose = Pose.from_xyz_rpy(
            pose_payload.get("position", [0.0, 0.0, 0.0]),
            pose_payload.get("rpy", [0.0, 0.0, 0.0]),
        )
        return pose, tuple(float(v) for v in obj_geometry["size"]), float(obj_geometry.get("mass_kg", 1.0))
    raise TypeError("object geometry must expose pose and size")


def _coerce_gripper_geometry(gripper_geometry: Any) -> tuple[Pose, str, float | None, float]:
    pose = getattr(gripper_geometry, "pose", None)
    if pose is None and hasattr(gripper_geometry, "position"):
        pose = Pose(
            position=vec3(getattr(gripper_geometry, "position")),
            orientation=getattr(gripper_geometry, "orientation", Quaternion.identity()),
        )
    if pose is None and isinstance(gripper_geometry, dict):
        pose = Pose(
            position=vec3(gripper_geometry.get("position", [0.0, 0.0, 0.0])),
            orientation=gripper_geometry.get("orientation", Quaternion.identity()),
        )
    if pose is None:
        raise TypeError("gripper geometry must expose a pose or position/orientation")
    raw_type = getattr(gripper_geometry, "gripper_type", None) or getattr(gripper_geometry, "type", None)
    if raw_type is None and isinstance(gripper_geometry, dict):
        raw_type = gripper_geometry.get("gripper_type") or gripper_geometry.get("type")
    grip_type = str(raw_type or "parallel_jaw")
    aperture = getattr(gripper_geometry, "aperture", None)
    if aperture is None and isinstance(gripper_geometry, dict):
        aperture = gripper_geometry.get("aperture")
    friction = getattr(gripper_geometry, "friction_coeff", None)
    if friction is None and isinstance(gripper_geometry, dict):
        friction = gripper_geometry.get("friction_coeff", 0.5)
    return pose, grip_type, None if aperture is None else float(aperture), float(friction or 0.5)


def _box_faces(pose: Pose, size: tuple[float, float, float]) -> list[tuple[np.ndarray, np.ndarray]]:
    half = np.asarray(size, dtype=np.float64) / 2.0
    rotation = pose.orientation.to_rotation_matrix()
    faces: list[tuple[np.ndarray, np.ndarray]] = []
    for axis_index in range(3):
        local_normal = np.zeros(3, dtype=np.float64)
        local_normal[axis_index] = 1.0
        world_normal = rotation @ local_normal
        offset = rotation @ (local_normal * half[axis_index])
        faces.append((pose.position + offset, normalize(world_normal)))
        faces.append((pose.position - offset, normalize(-world_normal)))
    return faces


def _box_projection_width(rotation: np.ndarray, size: tuple[float, float, float], axis: np.ndarray) -> float:
    unit_axis = normalize(vec3(axis))
    extents = np.asarray(size, dtype=np.float64)
    world_axes = [rotation[:, index] for index in range(3)]
    return float(sum(abs(np.dot(unit_axis, basis)) * extent for basis, extent in zip(world_axes, extents, strict=True)))
