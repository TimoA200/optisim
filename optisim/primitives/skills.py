"""Concrete semantic motion primitives."""

from __future__ import annotations

from typing import Any

import numpy as np

from optisim.primitives.base import MotionPrimitive, PrimitiveResult, PrimitiveStatus
from optisim.scene import SceneGraph, SceneQuery


def _require_node(scene: SceneGraph, node_id: str, kind: str) -> tuple[bool, str]:
    if node_id not in scene.nodes:
        return False, f"{kind} {node_id!r} does not exist"
    return True, ""


def _has_relation(scene: SceneGraph, subject_id: str, predicate: str, object_id: str) -> bool:
    return bool(scene.get_relations(subject_id=subject_id, predicate=predicate, object_id=object_id))


def _trajectory(
    robot_joints: np.ndarray,
    target_joints: np.ndarray,
    steps: int,
) -> list[np.ndarray]:
    start = np.asarray(robot_joints, dtype=float)
    goal = np.asarray(target_joints, dtype=float)
    return [((1.0 - alpha) * start + alpha * goal).copy() for alpha in np.linspace(0.0, 1.0, steps)]


def _bounded_target(
    scene: SceneGraph,
    robot_id: str,
    target_id: str,
    robot_joints: np.ndarray,
    *,
    scale: float,
) -> np.ndarray | None:
    reachable_ids = {node.id for node in SceneQuery.find_reachable(scene, robot_id)}
    if target_id not in reachable_ids:
        return None
    start = np.asarray(robot_joints, dtype=float)
    target = start.copy()
    offset_seed = float(np.sum(scene.get_node(target_id).pose[:3, 3]))
    base = np.linspace(0.2, 1.0, start.shape[0], dtype=float)
    delta = scale * np.sin(base + offset_seed)
    target += delta
    return target


class ReachPrimitive(MotionPrimitive):
    name = "reach"

    def check_preconditions(self, scene: SceneGraph, robot_id: str) -> tuple[bool, str]:
        ok, reason = _require_node(scene, robot_id, "robot")
        if not ok:
            return ok, reason
        return _require_node(scene, str(self.params["target_id"]), "target")

    def execute(self, scene: SceneGraph, robot_id: str, robot_joints: np.ndarray) -> PrimitiveResult:
        ok, reason = self.check_preconditions(scene, robot_id)
        if not ok:
            return PrimitiveResult(status=PrimitiveStatus.FAILURE, message=reason, duration_s=0.0)

        target_id = str(self.params["target_id"])
        target_joints = _bounded_target(scene, robot_id, target_id, robot_joints, scale=0.08)
        if target_joints is None:
            return PrimitiveResult(
                status=PrimitiveStatus.FAILURE,
                message=f"target {target_id!r} is not reachable",
                duration_s=0.0,
            )

        return PrimitiveResult(
            status=PrimitiveStatus.SUCCESS,
            message=f"Reached toward {target_id}",
            joint_trajectory=_trajectory(robot_joints, target_joints, 10),
            duration_s=1.0,
            metadata={"target_id": target_id, "end_effector": self.params.get("end_effector", "right")},
        )

    def get_effects(self, scene: SceneGraph, robot_id: str) -> list[dict[str, Any]]:
        del scene
        return [{"name": "near", "args": [robot_id, str(self.params["target_id"])], "value": True}]


class GraspPrimitive(MotionPrimitive):
    name = "grasp"

    def check_preconditions(self, scene: SceneGraph, robot_id: str) -> tuple[bool, str]:
        ok, reason = _require_node(scene, robot_id, "robot")
        if not ok:
            return ok, reason

        target_id = str(self.params["target_id"])
        ok, reason = _require_node(scene, target_id, "target")
        if not ok:
            return ok, reason
        if not bool(scene.get_node(target_id).properties.get("graspable", False)):
            return False, f"target {target_id!r} is not graspable"
        if not _has_relation(scene, robot_id, "near", target_id):
            return False, f"robot {robot_id!r} is not near {target_id!r}"
        return True, ""

    def execute(self, scene: SceneGraph, robot_id: str, robot_joints: np.ndarray) -> PrimitiveResult:
        ok, reason = self.check_preconditions(scene, robot_id)
        if not ok:
            return PrimitiveResult(status=PrimitiveStatus.FAILURE, message=reason, duration_s=0.0)

        start = np.asarray(robot_joints, dtype=float)
        target = start.copy()
        target[-2:] += np.array([0.015, -0.015], dtype=float)
        return PrimitiveResult(
            status=PrimitiveStatus.SUCCESS,
            message=f"Grasped {self.params['target_id']}",
            joint_trajectory=_trajectory(start, target, 5),
            duration_s=0.5,
            metadata={
                "target_id": str(self.params["target_id"]),
                "grasp_force": float(self.params.get("grasp_force", 10.0)),
            },
        )

    def get_effects(self, scene: SceneGraph, robot_id: str) -> list[dict[str, Any]]:
        del scene
        target_id = str(self.params["target_id"])
        return [
            {"name": "held_by", "args": [target_id, robot_id], "value": True},
            {"name": "near", "args": [robot_id, target_id], "value": False},
        ]


class PlacePrimitive(MotionPrimitive):
    name = "place"

    def check_preconditions(self, scene: SceneGraph, robot_id: str) -> tuple[bool, str]:
        object_id = str(self.params["object_id"])
        surface_id = str(self.params["surface_id"])
        ok, reason = _require_node(scene, object_id, "object")
        if not ok:
            return ok, reason
        ok, reason = _require_node(scene, surface_id, "surface")
        if not ok:
            return ok, reason
        if not _has_relation(scene, object_id, "held_by", robot_id):
            return False, f"object {object_id!r} is not held by {robot_id!r}"
        return True, ""

    def execute(self, scene: SceneGraph, robot_id: str, robot_joints: np.ndarray) -> PrimitiveResult:
        ok, reason = self.check_preconditions(scene, robot_id)
        if not ok:
            return PrimitiveResult(status=PrimitiveStatus.FAILURE, message=reason, duration_s=0.0)

        surface_id = str(self.params["surface_id"])
        target_joints = _bounded_target(scene, robot_id, surface_id, robot_joints, scale=0.05)
        if target_joints is None:
            start = np.asarray(robot_joints, dtype=float)
            target_joints = start + np.linspace(0.01, 0.03, start.shape[0], dtype=float) * 0.1
        return PrimitiveResult(
            status=PrimitiveStatus.SUCCESS,
            message=f"Placed {self.params['object_id']} on {surface_id}",
            joint_trajectory=_trajectory(robot_joints, target_joints, 8),
            duration_s=0.8,
            metadata={"surface_id": surface_id},
        )

    def get_effects(self, scene: SceneGraph, robot_id: str) -> list[dict[str, Any]]:
        del scene
        return [
            {"name": "held_by", "args": [str(self.params["object_id"]), robot_id], "value": False},
            {"name": "on", "args": [str(self.params["object_id"]), str(self.params["surface_id"])], "value": True},
        ]


class PushPrimitive(MotionPrimitive):
    name = "push"

    def check_preconditions(self, scene: SceneGraph, robot_id: str) -> tuple[bool, str]:
        target_id = str(self.params["target_id"])
        ok, reason = _require_node(scene, target_id, "target")
        if not ok:
            return ok, reason
        if not _has_relation(scene, robot_id, "near", target_id):
            return False, f"robot {robot_id!r} is not near {target_id!r}"
        return True, ""

    def execute(self, scene: SceneGraph, robot_id: str, robot_joints: np.ndarray) -> PrimitiveResult:
        ok, reason = self.check_preconditions(scene, robot_id)
        if not ok:
            return PrimitiveResult(status=PrimitiveStatus.FAILURE, message=reason, duration_s=0.0)

        direction = np.asarray(self.params["direction"], dtype=float)
        direction_norm = float(np.linalg.norm(direction))
        scaled = direction if direction_norm == 0.0 else direction / direction_norm
        start = np.asarray(robot_joints, dtype=float)
        target = start.copy()
        target[: min(3, target.shape[0])] += scaled[: min(3, scaled.shape[0])] * float(self.params.get("distance", 0.1))
        return PrimitiveResult(
            status=PrimitiveStatus.SUCCESS,
            message=f"Pushed {self.params['target_id']}",
            joint_trajectory=_trajectory(start, target, 6),
            duration_s=0.6,
            metadata={"direction": scaled, "distance": float(self.params.get("distance", 0.1))},
        )

    def get_effects(self, scene: SceneGraph, robot_id: str) -> list[dict[str, Any]]:
        del scene, robot_id
        return [{"name": "displaced", "args": [str(self.params["target_id"])], "value": True}]


class HandoverPrimitive(MotionPrimitive):
    name = "handover"

    def check_preconditions(self, scene: SceneGraph, robot_id: str) -> tuple[bool, str]:
        object_id = str(self.params["object_id"])
        ok, reason = _require_node(scene, object_id, "object")
        if not ok:
            return ok, reason
        if not _has_relation(scene, object_id, "held_by", robot_id):
            return False, f"object {object_id!r} is not held by {robot_id!r}"
        from_arm = str(self.params["from_arm"])
        to_arm = str(self.params["to_arm"])
        if from_arm == to_arm:
            return False, "from_arm and to_arm must be different"
        return True, ""

    def execute(self, scene: SceneGraph, robot_id: str, robot_joints: np.ndarray) -> PrimitiveResult:
        ok, reason = self.check_preconditions(scene, robot_id)
        if not ok:
            return PrimitiveResult(status=PrimitiveStatus.FAILURE, message=reason, duration_s=0.0)

        start = np.asarray(robot_joints, dtype=float)
        phases = np.linspace(0.0, np.pi, start.shape[0], dtype=float)
        target = start + 0.04 * np.sin(phases)
        return PrimitiveResult(
            status=PrimitiveStatus.SUCCESS,
            message=f"Handed over {self.params['object_id']}",
            joint_trajectory=_trajectory(start, target, 12),
            duration_s=1.2,
            metadata={"from_arm": self.params["from_arm"], "to_arm": self.params["to_arm"]},
        )

    def get_effects(self, scene: SceneGraph, robot_id: str) -> list[dict[str, Any]]:
        del scene
        return [{"name": "held_by", "args": [str(self.params["object_id"]), robot_id], "value": True}]


class NavigatePrimitive(MotionPrimitive):
    name = "navigate"

    def check_preconditions(self, scene: SceneGraph, robot_id: str) -> tuple[bool, str]:
        del robot_id
        return _require_node(scene, str(self.params["target_id"]), "target")

    def execute(self, scene: SceneGraph, robot_id: str, robot_joints: np.ndarray) -> PrimitiveResult:
        ok, reason = self.check_preconditions(scene, robot_id)
        if not ok:
            return PrimitiveResult(status=PrimitiveStatus.FAILURE, message=reason, duration_s=0.0)

        target_id = str(self.params["target_id"])
        start = np.asarray(robot_joints, dtype=float)
        target = start.copy()
        target[0] += 0.15
        if target.shape[0] > 1:
            target[1] -= 0.05
        return PrimitiveResult(
            status=PrimitiveStatus.SUCCESS,
            message=f"Navigated near {target_id}",
            joint_trajectory=_trajectory(start, target, 15),
            duration_s=1.5,
            metadata={"target_id": target_id, "stop_distance": float(self.params.get("stop_distance", 0.5))},
        )

    def get_effects(self, scene: SceneGraph, robot_id: str) -> list[dict[str, Any]]:
        del scene
        return [{"name": "near", "args": [robot_id, str(self.params["target_id"])], "value": True}]


__all__ = [
    "ReachPrimitive",
    "GraspPrimitive",
    "PlacePrimitive",
    "PushPrimitive",
    "HandoverPrimitive",
    "NavigatePrimitive",
]
