"""Scene export helpers."""

from __future__ import annotations

import json
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

from optisim.export.formats import SceneExport
from optisim.export.trajectory import TrajectoryExporter
from optisim.scene import SceneGraph


class SceneExporter:
    """Serialize scene graphs and task annotations."""

    @staticmethod
    def to_json(export: SceneExport) -> str:
        """Serialize a scene export payload to JSON."""

        trajectory_payload = None
        if export.robot_trajectory is not None:
            trajectory_payload = TrajectoryExporter._to_dict(export.robot_trajectory)
        payload = {
            "task_name": export.task_name,
            "scene": export.scene_graph.to_dict(),
            "trajectory": trajectory_payload,
            "metadata": _jsonify(export.metadata),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @staticmethod
    def from_json(json_str: str) -> SceneExport:
        """Deserialize a scene export payload from JSON."""

        payload = json.loads(json_str)
        trajectory = payload.get("trajectory")
        return SceneExport(
            scene_graph=SceneGraph.from_dict(payload.get("scene", {})),
            robot_trajectory=None if trajectory is None else TrajectoryExporter._from_dict(trajectory),
            task_name=str(payload.get("task_name", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    @staticmethod
    def to_urdf_annotation(export: SceneExport) -> str:
        """Serialize a scene export payload to a minimal XML task annotation."""

        root = Element("task_annotation", {"name": export.task_name})
        objects = SubElement(root, "objects")
        for node in export.scene_graph.nodes.values():
            position = node.pose[:3, 3]
            SubElement(
                objects,
                "object",
                {
                    "id": node.id,
                    "category": node.category,
                    "label": node.label,
                    "x": f"{float(position[0]):.12g}",
                    "y": f"{float(position[1]):.12g}",
                    "z": f"{float(position[2]):.12g}",
                },
            )
        relations = SubElement(root, "relations")
        for relation in export.scene_graph.relations:
            SubElement(
                relations,
                "relation",
                {
                    "subject": relation.subject_id,
                    "predicate": relation.predicate,
                    "object": relation.object_id,
                    "confidence": f"{float(relation.confidence):.12g}",
                },
            )
        return tostring(root, encoding="unicode")


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value
