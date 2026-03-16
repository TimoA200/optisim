"""Action primitives used by the planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from optisim.math3d import Pose, Quaternion, vec3


class ActionType(StrEnum):
    REACH = "reach"
    GRASP = "grasp"
    MOVE = "move"
    PLACE = "place"
    PUSH = "push"
    PULL = "pull"
    ROTATE = "rotate"


@dataclass(slots=True)
class ActionPrimitive:
    """An atomic manipulation intent with explicit execution metadata."""

    action_type: ActionType
    target: str
    end_effector: str = "right_palm"
    pose: Pose | None = None
    destination: tuple[float, float, float] | None = None
    support: str | None = None
    axis: tuple[float, float, float] | None = None
    angle_rad: float | None = None
    force_newtons: float | None = None
    speed: float = 0.25
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.action_type.value,
            "target": self.target,
            "end_effector": self.end_effector,
            "speed": self.speed,
        }
        if self.pose is not None:
            payload["pose"] = {
                "position": self.pose.position.tolist(),
                "orientation": self.pose.orientation.as_np().tolist(),
            }
        if self.destination is not None:
            payload["destination"] = list(self.destination)
        if self.support is not None:
            payload["support"] = self.support
        if self.axis is not None:
            payload["axis"] = list(self.axis)
        if self.angle_rad is not None:
            payload["angle_rad"] = self.angle_rad
        if self.force_newtons is not None:
            payload["force_newtons"] = self.force_newtons
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionPrimitive":
        pose = None
        if "pose" in payload:
            pose_data = payload["pose"]
            orientation = pose_data.get("orientation", [1.0, 0.0, 0.0, 0.0])
            pose = Pose(
                position=vec3(pose_data["position"]),
                orientation=Quaternion(*orientation).normalized(),
            )
        destination = tuple(payload["destination"]) if "destination" in payload else None
        axis = tuple(payload["axis"]) if "axis" in payload else None
        return cls(
            action_type=ActionType(payload["type"]),
            target=payload["target"],
            end_effector=payload.get("end_effector", "right_palm"),
            pose=pose,
            destination=destination,
            support=payload.get("support"),
            axis=axis,
            angle_rad=payload.get("angle_rad"),
            force_newtons=payload.get("force_newtons"),
            speed=float(payload.get("speed", 0.25)),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def reach(cls, target: str, end_effector: str, pose: Pose | None = None) -> "ActionPrimitive":
        return cls(action_type=ActionType.REACH, target=target, end_effector=end_effector, pose=pose)

    @classmethod
    def grasp(cls, target: str, gripper: str) -> "ActionPrimitive":
        return cls(action_type=ActionType.GRASP, target=target, end_effector=gripper)

    @classmethod
    def move(
        cls,
        target: str,
        destination: list[float] | tuple[float, float, float],
        end_effector: str = "right_palm",
    ) -> "ActionPrimitive":
        return cls(
            action_type=ActionType.MOVE,
            target=target,
            end_effector=end_effector,
            destination=tuple(float(v) for v in destination),
        )

    @classmethod
    def place(cls, target: str, support: str, end_effector: str = "right_palm") -> "ActionPrimitive":
        return cls(action_type=ActionType.PLACE, target=target, end_effector=end_effector, support=support)

    @classmethod
    def push(
        cls,
        target: str,
        direction: list[float] | tuple[float, float, float],
        force_newtons: float,
        end_effector: str = "right_palm",
    ) -> "ActionPrimitive":
        return cls(
            action_type=ActionType.PUSH,
            target=target,
            end_effector=end_effector,
            axis=tuple(float(v) for v in direction),
            force_newtons=force_newtons,
        )

    @classmethod
    def pull(
        cls,
        target: str,
        direction: list[float] | tuple[float, float, float],
        force_newtons: float,
        end_effector: str = "right_palm",
    ) -> "ActionPrimitive":
        return cls(
            action_type=ActionType.PULL,
            target=target,
            end_effector=end_effector,
            axis=tuple(float(v) for v in direction),
            force_newtons=force_newtons,
        )

    @classmethod
    def rotate(
        cls,
        target: str,
        axis: list[float] | tuple[float, float, float],
        angle_rad: float,
        end_effector: str = "right_palm",
    ) -> "ActionPrimitive":
        return cls(
            action_type=ActionType.ROTATE,
            target=target,
            end_effector=end_effector,
            axis=tuple(float(v) for v in axis),
            angle_rad=angle_rad,
        )
