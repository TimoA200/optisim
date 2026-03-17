"""Deterministic grasp candidate generation and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optisim.grasp.contact import ContactPoint, surface_contacts
from optisim.grasp.gripper import Gripper, GripperType
from optisim.grasp.stability import force_closure, min_resisted_wrench, slip_margin
from optisim.math3d import Pose, Quaternion, vec3
from optisim.robot import RobotModel


@dataclass(slots=True)
class GraspPose:
    """Candidate grasp pose paired with contacts and a quality score."""

    position: np.ndarray
    orientation: Quaternion
    aperture: float
    contact_points: list[ContactPoint] = field(default_factory=list)
    quality_score: float = 0.0
    gripper_type: str = GripperType.PARALLEL_JAW.value

    @property
    def pose(self) -> Pose:
        """Return the grasp pose as a ``Pose`` instance."""

        return Pose(position=vec3(self.position), orientation=self.orientation)


@dataclass
class GraspPlanner:
    """Simple deterministic grasp planner over box-like object geometry."""

    robot: RobotModel | None = None

    def plan_grasps(self, obj, gripper: Gripper, n_candidates: int = 10) -> list[GraspPose]:
        """Generate and rank grasp poses for an object and gripper model."""

        if gripper.type is GripperType.PARALLEL_JAW:
            candidates = self.antipodal_grasps(obj, gripper)
        elif gripper.type is GripperType.SUCTION:
            candidates = self._suction_grasps(obj, gripper)
        else:
            candidates = self._three_finger_grasps(obj, gripper)

        for candidate in candidates:
            candidate.quality_score = self.evaluate_grasp(candidate, obj)
        candidates = [candidate for candidate in candidates if np.isfinite(candidate.quality_score)]
        candidates.sort(
            key=lambda grasp: (
                -grasp.quality_score,
                round(float(grasp.position[2]), 6),
                round(float(grasp.aperture), 6),
            )
        )
        return candidates[: max(int(n_candidates), 0)]

    def antipodal_grasps(self, obj, gripper: Gripper) -> list[GraspPose]:
        """Return antipodal parallel-jaw grasp candidates aligned to object principal axes."""

        if gripper.type is GripperType.SUCTION:
            return []
        obj_pose, obj_size, _ = self._object_geometry(obj)
        local_rotations = (
            Quaternion.identity(),
            Quaternion.from_euler(0.0, 0.0, np.pi / 2.0),
            Quaternion.from_euler(0.0, -np.pi / 2.0, 0.0),
        )
        candidates: list[GraspPose] = []
        for axis_index, local_rotation in enumerate(local_rotations):
            width = float(obj_size[axis_index])
            aperture = width + 2.0 * gripper.finger_width
            if aperture > gripper.max_aperture + 1e-9:
                continue
            candidate = GraspPose(
                position=obj_pose.position.copy(),
                orientation=obj_pose.orientation * local_rotation,
                aperture=aperture,
                gripper_type=gripper.type.value,
            )
            if not self._reachable(candidate):
                continue
            candidate.contact_points = surface_contacts(
                obj,
                {
                    "position": candidate.position,
                    "orientation": candidate.orientation,
                    "aperture": gripper.max_aperture,
                    "gripper_type": gripper.type.value,
                },
            )
            if len(candidate.contact_points) >= 2:
                candidates.append(candidate)
        return candidates

    def evaluate_grasp(self, grasp_pose: GraspPose, obj) -> float:
        """Return a deterministic quality score based on closure and stability."""

        contacts = grasp_pose.contact_points
        if not contacts:
            return float("-inf")
        obj_pose, obj_size, mass_kg = self._object_geometry(obj)
        closure_bonus = 1.0 if force_closure(contacts) else 0.0
        resisted = min_resisted_wrench(contacts)
        load = np.asarray([0.0, 0.0, -9.81 * mass_kg / max(len(contacts), 1)], dtype=np.float64)
        mean_slip_margin = float(np.mean([slip_margin(contact, load) for contact in contacts]))
        contact_center = np.mean([contact.position for contact in contacts], axis=0)
        center_offset = float(np.linalg.norm(contact_center - obj_pose.position))
        scale = max(float(np.linalg.norm(np.asarray(obj_size, dtype=np.float64))), 1e-6)
        aperture_penalty = max(grasp_pose.aperture, 0.0) * 0.1
        return closure_bonus + resisted + 0.05 * mean_slip_margin - (center_offset / scale) - aperture_penalty

    def _suction_grasps(self, obj, gripper: Gripper) -> list[GraspPose]:
        obj_pose, obj_size, _ = self._object_geometry(obj)
        rotation = obj_pose.orientation.to_rotation_matrix()
        half = np.asarray(obj_size, dtype=np.float64) / 2.0
        candidates: list[GraspPose] = []
        face_rotations = (
            Quaternion.identity(),
            Quaternion.from_euler(np.pi, 0.0, 0.0),
            Quaternion.from_euler(0.0, np.pi / 2.0, 0.0),
            Quaternion.from_euler(0.0, -np.pi / 2.0, 0.0),
            Quaternion.from_euler(-np.pi / 2.0, 0.0, 0.0),
            Quaternion.from_euler(np.pi / 2.0, 0.0, 0.0),
        )
        local_normals = (
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
        )
        for local_normal, local_rotation in zip(local_normals, face_rotations, strict=True):
            face_span = _face_area_for_normal(local_normal, obj_size)
            if face_span + 1e-9 < gripper.contact_area:
                continue
            offset = rotation @ (local_normal * half[np.argmax(np.abs(local_normal))])
            position = obj_pose.position + offset
            candidate = GraspPose(
                position=position,
                orientation=obj_pose.orientation * local_rotation,
                aperture=min(gripper.max_aperture, float(np.sqrt(face_span))),
                gripper_type=gripper.type.value,
            )
            if not self._reachable(candidate):
                continue
            candidate.contact_points = surface_contacts(
                obj,
                {
                    "position": candidate.position,
                    "orientation": candidate.orientation,
                    "aperture": gripper.max_aperture,
                    "gripper_type": gripper.type.value,
                },
            )
            if candidate.contact_points:
                candidates.append(candidate)
        return candidates

    def _three_finger_grasps(self, obj, gripper: Gripper) -> list[GraspPose]:
        obj_pose, obj_size, _ = self._object_geometry(obj)
        local_rotations = (
            Quaternion.identity(),
            Quaternion.from_euler(0.0, 0.0, np.pi / 2.0),
            Quaternion.from_euler(0.0, -np.pi / 2.0, 0.0),
        )
        candidates: list[GraspPose] = []
        for axis_index, local_rotation in enumerate(local_rotations):
            width = float(obj_size[axis_index])
            aperture = width + 2.0 * gripper.finger_width
            if aperture > gripper.max_aperture + 1e-9:
                continue
            candidate = GraspPose(
                position=obj_pose.position.copy(),
                orientation=obj_pose.orientation * local_rotation,
                aperture=aperture,
                gripper_type=gripper.type.value,
            )
            if not self._reachable(candidate):
                continue
            candidate.contact_points = surface_contacts(
                obj,
                {
                    "position": candidate.position,
                    "orientation": candidate.orientation,
                    "aperture": gripper.max_aperture,
                    "gripper_type": gripper.type.value,
                },
            )
            if len(candidate.contact_points) == 3:
                candidates.append(candidate)
        return candidates

    def _reachable(self, grasp_pose: GraspPose) -> bool:
        if self.robot is None:
            return True
        distance = float(np.linalg.norm(grasp_pose.position - self.robot.base_pose.position))
        return distance <= self.robot.max_reach() * 1.05

    @staticmethod
    def _object_geometry(obj) -> tuple[Pose, tuple[float, float, float], float]:
        if hasattr(obj, "pose") and hasattr(obj, "size"):
            return obj.pose, tuple(float(v) for v in obj.size), float(getattr(obj, "mass_kg", 1.0))
        raise TypeError("grasp planning currently expects an object with pose and size attributes")


def _face_area_for_normal(local_normal: np.ndarray, size: tuple[float, float, float]) -> float:
    axis_index = int(np.argmax(np.abs(local_normal)))
    extents = [float(value) for value in size]
    del extents[axis_index]
    return extents[0] * extents[1]

__all__ = ["GraspPose", "GraspPlanner"]
