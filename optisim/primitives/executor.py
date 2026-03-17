"""Primitive registry and sequence execution."""

from __future__ import annotations

from typing import Type

import numpy as np

from optisim.primitives.base import MotionPrimitive, PrimitiveResult, PrimitiveStatus
from optisim.primitives.effects import apply_effects
from optisim.primitives import skills as primitive_skills
from optisim.scene import SceneGraph


def _build_registry() -> dict[str, Type[MotionPrimitive]]:
    registry: dict[str, Type[MotionPrimitive]] = {}
    for value in vars(primitive_skills).values():
        if not isinstance(value, type):
            continue
        if not issubclass(value, MotionPrimitive) or value is MotionPrimitive:
            continue
        if value.name:
            registry[value.name] = value
    return registry


class PrimitiveExecutor:
    """Factory and executor for motion primitives."""

    registry: dict[str, type[MotionPrimitive]] = _build_registry()

    def get(self, name: str, params: dict) -> MotionPrimitive:
        primitive_cls = self.registry.get(name)
        if primitive_cls is None:
            raise KeyError(f"unknown primitive {name!r}")
        return primitive_cls(params)

    def execute_sequence(
        self,
        scene: SceneGraph,
        robot_id: str,
        robot_joints: np.ndarray,
        sequence: list[dict],
    ) -> list[PrimitiveResult]:
        results: list[PrimitiveResult] = []
        current_joints = np.asarray(robot_joints, dtype=float).copy()
        for step in sequence:
            primitive = self.get(str(step["primitive"]), dict(step.get("params", {})))
            result = primitive.execute(scene, robot_id, current_joints)
            results.append(result)
            if result.status is not PrimitiveStatus.SUCCESS:
                break
            apply_effects(scene, primitive.get_effects(scene, robot_id))
            if result.joint_trajectory:
                current_joints = np.asarray(result.joint_trajectory[-1], dtype=float).copy()
        return results

    def available_primitives(self) -> list[str]:
        return sorted(self.registry)


__all__ = ["PrimitiveExecutor"]
