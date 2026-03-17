"""Reference skeleton definitions for human-to-robot retargeting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Joint3D", "ReferenceSkeleton"]


@dataclass(slots=True)
class Joint3D:
    """Single reference skeleton joint in world coordinates."""

    name: str
    position: np.ndarray
    parent: str | None = None

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)


class ReferenceSkeleton:
    """Named reference skeleton with utility accessors."""

    def __init__(self, joints: list[Joint3D]) -> None:
        self.joints: dict[str, Joint3D] = {}
        for joint in joints:
            if joint.name in self.joints:
                raise ValueError(f"duplicate joint '{joint.name}'")
            self.joints[joint.name] = joint

    def get_position(self, name: str) -> np.ndarray:
        return self.joints[name].position.copy()

    def bone_vector(self, parent: str, child: str) -> np.ndarray:
        return self.get_position(child) - self.get_position(parent)

    def bone_length(self, parent: str, child: str) -> float:
        return float(np.linalg.norm(self.bone_vector(parent, child)))

    @classmethod
    def HUMAN_SKELETON(cls) -> "ReferenceSkeleton":
        joints = [
            Joint3D("pelvis", [0.0, 0.0, 1.00]),
            Joint3D("spine", [0.0, 0.0, 1.16], parent="pelvis"),
            Joint3D("chest", [0.0, 0.0, 1.38], parent="spine"),
            Joint3D("neck", [0.0, 0.0, 1.54], parent="chest"),
            Joint3D("head", [0.0, 0.0, 1.72], parent="neck"),
            Joint3D("left_shoulder", [0.0, 0.19, 1.47], parent="chest"),
            Joint3D("left_elbow", [0.0, 0.48, 1.25], parent="left_shoulder"),
            Joint3D("left_wrist", [0.0, 0.72, 1.05], parent="left_elbow"),
            Joint3D("right_shoulder", [0.0, -0.19, 1.47], parent="chest"),
            Joint3D("right_elbow", [0.0, -0.48, 1.25], parent="right_shoulder"),
            Joint3D("right_wrist", [0.0, -0.72, 1.05], parent="right_elbow"),
            Joint3D("left_hip", [0.0, 0.10, 1.00], parent="pelvis"),
            Joint3D("left_knee", [0.0, 0.10, 0.56], parent="left_hip"),
            Joint3D("left_ankle", [0.0, 0.10, 0.11], parent="left_knee"),
            Joint3D("right_hip", [0.0, -0.10, 1.00], parent="pelvis"),
            Joint3D("right_knee", [0.0, -0.10, 0.56], parent="right_hip"),
            Joint3D("right_ankle", [0.0, -0.10, 0.11], parent="right_knee"),
        ]
        return cls(joints)

    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceSkeleton":
        parent_map = {
            "spine": "pelvis",
            "chest": "spine",
            "neck": "chest",
            "head": "neck",
            "left_shoulder": "chest",
            "left_elbow": "left_shoulder",
            "left_wrist": "left_elbow",
            "right_shoulder": "chest",
            "right_elbow": "right_shoulder",
            "right_wrist": "right_elbow",
            "left_hip": "pelvis",
            "left_knee": "left_hip",
            "left_ankle": "left_knee",
            "right_hip": "pelvis",
            "right_knee": "right_hip",
            "right_ankle": "right_knee",
        }
        return cls([Joint3D(name=str(name), position=value, parent=parent_map.get(str(name))) for name, value in data.items()])
