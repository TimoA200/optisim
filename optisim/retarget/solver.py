"""Human-to-humanoid motion retargeting solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optisim.retarget.mapping import JointMapping, RetargetMapping
from optisim.retarget.skeleton import ReferenceSkeleton
from optisim.robot import build_humanoid_model

__all__ = ["RetargetResult", "RetargetSolver"]


@dataclass(slots=True)
class RetargetResult:
    """Dense humanoid retargeting output."""

    joint_angles: np.ndarray
    residuals: np.ndarray
    n_mapped: int
    coverage: float


class RetargetSolver:
    """Heuristic motion retargeting from a reference skeleton pose."""

    def __init__(self, mapping: RetargetMapping | None = None) -> None:
        self.mapping = mapping or RetargetMapping.HUMANOID_DEFAULT()
        self.robot = build_humanoid_model()
        self._joint_index = {name: index for index, name in enumerate(self.robot.joint_names)}
        self._forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self._left = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self._up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def retarget(self, skeleton: ReferenceSkeleton) -> RetargetResult:
        joint_angles = np.array(self.robot.home_config, dtype=np.float64)
        residuals: list[float] = []

        for mapping in self.mapping.mappings:
            if mapping.target_joint not in self._joint_index:
                continue
            angle, residual = self._compute_joint_value(skeleton, mapping)
            value = mapping.scale * angle + mapping.offset
            spec = self.robot.joints[mapping.target_joint]
            joint_angles[self._joint_index[mapping.target_joint]] = spec.clamp(value)
            residuals.append(residual)

        n_mapped = len(residuals)
        coverage = float(n_mapped / self.robot.dof) if self.robot.dof else 0.0
        return RetargetResult(
            joint_angles=joint_angles,
            residuals=np.asarray(residuals, dtype=np.float64),
            n_mapped=n_mapped,
            coverage=coverage,
        )

    def retarget_sequence(self, skeletons: list[ReferenceSkeleton]) -> list[RetargetResult]:
        return [self.retarget(skeleton) for skeleton in skeletons]

    def smooth_sequence(self, results: list[RetargetResult], window: int = 3) -> list[RetargetResult]:
        if not results or window <= 1:
            return list(results)

        smoothed: list[RetargetResult] = []
        radius = max(window // 2, 0)
        for index, result in enumerate(results):
            start = max(0, index - radius)
            stop = min(len(results), index + radius + 1)
            averaged = np.mean([candidate.joint_angles for candidate in results[start:stop]], axis=0)
            smoothed.append(
                RetargetResult(
                    joint_angles=np.asarray(averaged, dtype=np.float64),
                    residuals=result.residuals.copy(),
                    n_mapped=result.n_mapped,
                    coverage=result.coverage,
                )
            )
        return smoothed

    def _compute_joint_value(self, skeleton: ReferenceSkeleton, mapping: JointMapping) -> tuple[float, float]:
        joint = mapping.target_joint
        if joint.startswith("torso_"):
            return self._torso_angle(skeleton, joint), 0.0
        if joint.startswith("neck_"):
            return self._neck_angle(skeleton, joint), 0.0
        if joint.startswith("left_") or joint.startswith("right_"):
            side = "left" if joint.startswith("left_") else "right"
            if "_hip_" in joint or "_knee_" in joint or "_ankle_" in joint:
                return self._leg_angle(skeleton, joint, side), 0.0
            return self._arm_angle(skeleton, joint, side), 0.0
        return 0.0, 0.0

    def _torso_angle(self, skeleton: ReferenceSkeleton, joint: str) -> float:
        torso = self._unit(skeleton.bone_vector("pelvis", "chest"))
        shoulders = self._unit(skeleton.bone_vector("right_shoulder", "left_shoulder"))
        if joint == "torso_yaw":
            return float(np.arctan2(np.dot(shoulders, self._forward), np.dot(shoulders, self._left)))
        if joint == "torso_roll":
            return float(np.arctan2(np.dot(torso, self._left), np.dot(torso, self._up)))
        return float(-np.arctan2(np.dot(torso, self._forward), np.dot(torso, self._up)))

    def _neck_angle(self, skeleton: ReferenceSkeleton, joint: str) -> float:
        head = self._unit(skeleton.bone_vector("neck", "head"))
        if joint == "neck_yaw":
            return float(np.arctan2(np.dot(head, self._left), np.dot(head, self._forward)))
        return float(-np.arctan2(np.dot(head, self._forward), np.dot(head, self._up)))

    def _arm_angle(self, skeleton: ReferenceSkeleton, joint: str, side: str) -> float:
        shoulder = skeleton.get_position(f"{side}_shoulder")
        elbow = skeleton.get_position(f"{side}_elbow")
        wrist = skeleton.get_position(f"{side}_wrist")
        upper = self._unit(elbow - shoulder)
        fore = self._unit(wrist - elbow)
        side_sign = 1.0 if side == "left" else -1.0
        arm_plane = self._unit(np.cross(upper, fore))

        if joint.endswith("clavicle_pitch"):
            return float(-np.arctan2(np.dot(upper, self._forward), -np.dot(upper, self._up)))
        if joint.endswith("shoulder_pitch"):
            return float(-np.arctan2(np.dot(upper, self._forward), -np.dot(upper, self._up)))
        if joint.endswith("shoulder_roll"):
            lateral = side_sign * np.dot(upper, self._left)
            return float(side_sign * np.arctan2(lateral, -np.dot(upper, self._up)))
        if joint.endswith("shoulder_yaw"):
            return float(np.arctan2(np.dot(arm_plane, self._forward), side_sign * np.dot(arm_plane, self._left)))
        if joint.endswith("elbow_pitch"):
            return self._hinge_angle(shoulder, elbow, wrist)
        if joint.endswith("forearm_yaw"):
            torso_normal = self._unit(np.cross(self._up, upper))
            return float(np.arctan2(np.dot(arm_plane, upper), np.dot(arm_plane, torso_normal)))
        if joint.endswith("wrist_pitch"):
            return float(-np.arctan2(np.dot(fore, self._forward), -np.dot(fore, self._up)))
        return 0.0

    def _leg_angle(self, skeleton: ReferenceSkeleton, joint: str, side: str) -> float:
        hip = skeleton.get_position(f"{side}_hip")
        knee = skeleton.get_position(f"{side}_knee")
        ankle = skeleton.get_position(f"{side}_ankle")
        thigh = self._unit(knee - hip)
        shin = self._unit(ankle - knee)
        side_sign = 1.0 if side == "left" else -1.0

        if joint.endswith("hip_yaw"):
            horizontal = thigh.copy()
            horizontal[2] = 0.0
            horizontal = self._unit(horizontal)
            return float(np.arctan2(np.dot(horizontal, self._left), np.dot(horizontal, self._forward)))
        if joint.endswith("hip_roll"):
            lateral = side_sign * np.dot(thigh, self._left)
            return float(side_sign * np.arctan2(lateral, -np.dot(thigh, self._up)))
        if joint.endswith("hip_pitch"):
            return float(-np.arctan2(np.dot(thigh, self._forward), -np.dot(thigh, self._up)))
        if joint.endswith("knee_pitch"):
            return self._hinge_angle(hip, knee, ankle)
        if joint.endswith("ankle_pitch"):
            return float(-np.arctan2(np.dot(shin, self._forward), -np.dot(shin, self._up)))
        if joint.endswith("ankle_roll"):
            lateral = side_sign * np.dot(shin, self._left)
            return float(-side_sign * np.arctan2(lateral, -np.dot(shin, self._up)))
        return 0.0

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(vector, dtype=np.float64) / norm

    @staticmethod
    def _hinge_angle(parent: np.ndarray, joint: np.ndarray, child: np.ndarray) -> float:
        incoming = parent - joint
        outgoing = child - joint
        incoming_norm = float(np.linalg.norm(incoming))
        outgoing_norm = float(np.linalg.norm(outgoing))
        if incoming_norm < 1e-8 or outgoing_norm < 1e-8:
            return 0.0
        cosine = np.clip(float(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm)), -1.0, 1.0)
        return float(np.pi - np.arccos(cosine))
