"""Reference-to-humanoid retarget mapping definitions."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["JointMapping", "RetargetMapping"]


@dataclass(slots=True)
class JointMapping:
    """Mapping from source skeleton joints to a robot joint."""

    source_joints: list[str]
    target_joint: str
    scale: float = 1.0
    offset: float = 0.0


class RetargetMapping:
    """Collection of source-to-target joint mappings."""

    def __init__(self, mappings: list[JointMapping]) -> None:
        self.mappings = list(mappings)

    @classmethod
    def HUMANOID_DEFAULT(cls) -> "RetargetMapping":
        mappings = [
            JointMapping(["left_hip", "right_hip", "left_shoulder", "right_shoulder"], "torso_yaw"),
            JointMapping(["pelvis", "chest"], "torso_roll"),
            JointMapping(["pelvis", "chest"], "torso_pitch"),
            JointMapping(["neck", "head"], "neck_yaw"),
            JointMapping(["neck", "head"], "neck_pitch"),
        ]
        for side in ("left", "right"):
            mappings.extend(
                [
                    JointMapping([f"{side}_shoulder", f"{side}_elbow"], f"{side}_clavicle_pitch", scale=0.35),
                    JointMapping([f"{side}_shoulder", f"{side}_elbow"], f"{side}_shoulder_pitch"),
                    JointMapping([f"{side}_shoulder", f"{side}_elbow"], f"{side}_shoulder_roll"),
                    JointMapping([f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"], f"{side}_shoulder_yaw"),
                    JointMapping([f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"], f"{side}_elbow_pitch"),
                    JointMapping([f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"], f"{side}_forearm_yaw"),
                    JointMapping([f"{side}_elbow", f"{side}_wrist"], f"{side}_wrist_pitch", scale=0.5),
                    JointMapping([f"{side}_hip", f"{side}_knee"], f"{side}_hip_yaw"),
                    JointMapping([f"{side}_hip", f"{side}_knee"], f"{side}_hip_roll"),
                    JointMapping([f"{side}_hip", f"{side}_knee"], f"{side}_hip_pitch"),
                    JointMapping([f"{side}_hip", f"{side}_knee", f"{side}_ankle"], f"{side}_knee_pitch"),
                    JointMapping([f"{side}_knee", f"{side}_ankle"], f"{side}_ankle_pitch"),
                    JointMapping([f"{side}_knee", f"{side}_ankle"], f"{side}_ankle_roll"),
                ]
            )
        return cls(mappings)

    def get_mapping(self, target_joint: str) -> JointMapping | None:
        for mapping in self.mappings:
            if mapping.target_joint == target_joint:
                return mapping
        return None

    @property
    def joint_names(self) -> list[str]:
        return [mapping.target_joint for mapping in self.mappings]
