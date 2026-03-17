"""Action primitives used by the planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from optisim.math3d import Pose, Quaternion, vec3


class ActionType(StrEnum):
    """Enumeration of built-in atomic manipulation action types."""

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
        """Serialize the action into a task-file-friendly dictionary."""

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
        """Build an action primitive from a serialized mapping payload."""

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
        """Create a reach action for a target object or explicit pose."""

        return cls(action_type=ActionType.REACH, target=target, end_effector=end_effector, pose=pose)

    @classmethod
    def grasp(cls, target: str, gripper: str) -> "ActionPrimitive":
        """Create a grasp action for the requested target and gripper."""

        return cls(action_type=ActionType.GRASP, target=target, end_effector=gripper)

    @classmethod
    def move(
        cls,
        target: str,
        destination: list[float] | tuple[float, float, float],
        end_effector: str = "right_palm",
    ) -> "ActionPrimitive":
        """Create a move action that transports a grasped target to a destination."""

        return cls(
            action_type=ActionType.MOVE,
            target=target,
            end_effector=end_effector,
            destination=tuple(float(v) for v in destination),
        )

    @classmethod
    def place(cls, target: str, support: str, end_effector: str = "right_palm") -> "ActionPrimitive":
        """Create a place action that releases a held object onto a support surface."""

        return cls(action_type=ActionType.PLACE, target=target, end_effector=end_effector, support=support)

    @classmethod
    def push(
        cls,
        target: str,
        direction: list[float] | tuple[float, float, float],
        force_newtons: float,
        end_effector: str = "right_palm",
    ) -> "ActionPrimitive":
        """Create a push action with a direction vector and nominal force."""

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
        """Create a pull action with a direction vector and nominal force."""

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
        """Create a rotate action about the supplied axis and angle."""

        return cls(
            action_type=ActionType.ROTATE,
            target=target,
            end_effector=end_effector,
            axis=tuple(float(v) for v in axis),
            angle_rad=angle_rad,
        )

__all__ = ["ActionType", "ActionPrimitive"]
