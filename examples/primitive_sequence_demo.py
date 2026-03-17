"""Showcase semantic primitive execution on top of a kitchen scene."""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optisim.primitives import PrimitiveExecutor
from optisim.scene import SceneBuilder


def print_relation_block(scene, title: str) -> None:
    """Print scene relations in a predictable order."""

    print(title)
    for relation in sorted(scene.relations, key=lambda item: (item.subject_id, item.predicate, item.object_id)):
        print(f"  - {relation.subject_id} --{relation.predicate}--> {relation.object_id}")


def main() -> None:
    """Run a reach, grasp, and place sequence and report each step."""

    scene = SceneBuilder.build_kitchen()
    robot_pose = np.eye(4, dtype=float)
    robot_pose[:3, 3] = [0.2, 0.0, 0.75]
    SceneBuilder.add_robot(scene, pose=robot_pose)

    sequence = [
        {"primitive": "reach", "params": {"target_id": "cup", "end_effector": "right"}},
        {"primitive": "grasp", "params": {"target_id": "cup", "end_effector": "right", "grasp_force": 12.0}},
        {"primitive": "place", "params": {"object_id": "cup", "surface_id": "countertop"}},
    ]

    executor = PrimitiveExecutor()
    results = executor.execute_sequence(
        scene=scene,
        robot_id="humanoid",
        robot_joints=np.zeros(31, dtype=float),
        sequence=sequence,
    )

    print("optisim.scene + optisim.primitives demo")
    print("=======================================")
    print()
    print(f"Executed {len(results)} primitive(s).")
    for index, step in enumerate(sequence, start=1):
        primitive_name = step["primitive"]
        if index > len(results):
            print(f"Step {index}: {primitive_name}")
            print("  status: skipped")
            print("  message: sequence stopped after an earlier failure")
            print("  trajectory steps: 0")
            print("  duration_s: 0.00")
            print("  metadata: {}")
            print("  effects applied: none")
            print()
            continue

        result = results[index - 1]
        primitive = executor.get(primitive_name, step.get("params", {}))
        effects = primitive.get_effects(scene, "humanoid") if result.status.value == "success" else []
        print(f"Step {index}: {primitive_name}")
        print(f"  status: {result.status.value}")
        print(f"  message: {result.message}")
        print(f"  trajectory steps: {len(result.joint_trajectory or [])}")
        print(f"  duration_s: {result.duration_s:.2f}")
        print(f"  metadata: {result.metadata}")
        print(f"  effects applied: {effects or 'none'}")
        print()

    print_relation_block(scene, "Final scene relations:")


if __name__ == "__main__":
    main()
