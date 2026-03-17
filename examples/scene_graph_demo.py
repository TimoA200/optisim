"""Showcase the semantic scene graph helpers for kitchen manipulation."""

from __future__ import annotations

from collections import Counter
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optisim.scene import SceneBuilder, SceneQuery


def format_node_list(title: str, node_ids: list[str]) -> str:
    """Render a compact bullet list for scene query results."""

    lines = [f"{title} ({len(node_ids)}):"]
    for node_id in node_ids:
        lines.append(f"  - {node_id}")
    return "\n".join(lines)


def main() -> None:
    """Build a kitchen scene, query it, and print a readable summary."""

    scene = SceneBuilder.build_kitchen()
    robot_pose = np.eye(4, dtype=float)
    robot_pose[:3, 3] = [0.2, 0.0, 0.75]
    SceneBuilder.add_robot(scene, pose=robot_pose)

    graspable = sorted(node.id for node in SceneQuery.find_graspable(scene))
    on_table = sorted(node.id for node in SceneQuery.find_on_surface(scene, "kitchen_table"))
    reachable = sorted(node.id for node in SceneQuery.find_reachable(scene, "humanoid"))
    tamp_predicates = SceneQuery.to_tamp_predicates(scene)

    print("optisim.scene demo")
    print("==================")
    print()
    print(f"Nodes: {len(scene.nodes)}")
    print(f"Relations: {len(scene.relations)}")
    print(
        "Categories:",
        ", ".join(
            f"{category}={count}" for category, count in sorted(Counter(node.category for node in scene.nodes.values()).items())
        ),
    )
    print()
    print(format_node_list("Graspable objects", graspable))
    print()
    print(format_node_list("Objects on kitchen_table", on_table))
    print()
    print(format_node_list("Reachable from humanoid", reachable))
    print()
    print("Scene relations:")
    for relation in sorted(scene.relations, key=lambda item: (item.subject_id, item.predicate, item.object_id)):
        print(
            f"  - {relation.subject_id} --{relation.predicate}"
            f" ({relation.confidence:.2f})--> {relation.object_id}"
        )
    print()
    print("TAMP predicates:")
    for predicate in tamp_predicates:
        print(f"  - {predicate['name']}({', '.join(predicate['args'])}) = {predicate['value']}")
    print()
    print("Scene summary:")
    for node in sorted(scene.nodes.values(), key=lambda item: item.id):
        parent_text = node.parent_id or "-"
        print(
            f"  - {node.id}: label={node.label!r}, category={node.category}, "
            f"parent={parent_text}, properties={node.properties}"
        )


if __name__ == "__main__":
    main()
