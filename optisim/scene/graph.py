"""Semantic scene graph primitives for robot task planning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _identity_pose() -> np.ndarray:
    """Return a default homogeneous identity transform."""

    return np.eye(4, dtype=float)


def _validate_pose(pose: np.ndarray) -> np.ndarray:
    """Return a validated homogeneous transform."""

    pose_array = np.asarray(pose, dtype=float)
    if pose_array.shape != (4, 4):
        raise ValueError(f"pose must have shape (4, 4), got {pose_array.shape}")
    return pose_array.copy()


def _validate_bbox(bbox: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return bbox half-extents as floats."""

    if len(bbox) != 3:
        raise ValueError("bbox must contain exactly three half-extents")
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]))


@dataclass(slots=True)
class SceneNode:
    """Object or entity in a semantic scene."""

    id: str
    label: str
    category: str
    pose: np.ndarray = field(default_factory=_identity_pose)
    bbox: tuple[float, float, float] = (0.0, 0.0, 0.0)
    properties: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None

    def __post_init__(self) -> None:
        self.pose = _validate_pose(self.pose)
        self.bbox = _validate_bbox(self.bbox)
        self.properties = dict(self.properties)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the node to a plain dictionary."""

        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "pose": self.pose.tolist(),
            "bbox": list(self.bbox),
            "properties": dict(self.properties),
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneNode:
        """Construct a node from a dictionary."""

        return cls(
            id=str(data["id"]),
            label=str(data["label"]),
            category=str(data["category"]),
            pose=np.asarray(data.get("pose", _identity_pose()), dtype=float),
            bbox=tuple(data.get("bbox", (0.0, 0.0, 0.0))),
            properties=dict(data.get("properties", {})),
            parent_id=data.get("parent_id"),
        )


@dataclass(slots=True)
class SceneRelation:
    """Directed semantic relation between two nodes."""

    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.confidence = float(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the relation to a plain dictionary."""

        return {
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneRelation:
        """Construct a relation from a dictionary."""

        return cls(
            subject_id=str(data["subject_id"]),
            predicate=str(data["predicate"]),
            object_id=str(data["object_id"]),
            confidence=float(data.get("confidence", 1.0)),
        )


class SceneGraph:
    """Semantic scene graph with typed nodes and directed relations."""

    def __init__(
        self,
        nodes: dict[str, SceneNode] | None = None,
        relations: list[SceneRelation] | None = None,
    ) -> None:
        self._nodes: dict[str, SceneNode] = {}
        self._relations: list[SceneRelation] = []
        if nodes is not None:
            self._add_nodes_bulk(list(nodes.values()))
        if relations is not None:
            for relation in relations:
                self.add_relation(relation)

    @property
    def nodes(self) -> dict[str, SceneNode]:
        """Return the graph nodes keyed by node id."""

        return self._nodes

    @property
    def relations(self) -> list[SceneRelation]:
        """Return the graph relations."""

        return self._relations

    def add_node(self, node: SceneNode) -> None:
        """Insert a node into the graph."""

        if node.id in self._nodes:
            raise ValueError(f"node {node.id!r} already exists")
        if node.parent_id is not None and node.parent_id not in self._nodes:
            raise KeyError(f"parent node {node.parent_id!r} does not exist")
        self._nodes[node.id] = SceneNode.from_dict(node.to_dict())

    def _add_nodes_bulk(self, nodes: list[SceneNode]) -> None:
        """Insert a batch of nodes while respecting parent-child ordering."""

        pending = {node.id: SceneNode.from_dict(node.to_dict()) for node in nodes}
        if len(pending) != len(nodes):
            raise ValueError("duplicate node ids are not allowed")

        while pending:
            progressed = False
            for node_id, node in list(pending.items()):
                if node_id in self._nodes:
                    raise ValueError(f"node {node_id!r} already exists")
                if node.parent_id is None or node.parent_id in self._nodes:
                    self._nodes[node_id] = node
                    del pending[node_id]
                    progressed = True
                    continue
                if node.parent_id not in pending:
                    raise KeyError(f"parent node {node.parent_id!r} does not exist")
            if progressed:
                continue
            raise ValueError("scene hierarchy contains a cycle")

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its attached relations."""

        if node_id not in self._nodes:
            raise KeyError(f"node {node_id!r} does not exist")
        del self._nodes[node_id]
        self._relations = [
            relation
            for relation in self._relations
            if relation.subject_id != node_id and relation.object_id != node_id
        ]
        for node in self._nodes.values():
            if node.parent_id == node_id:
                node.parent_id = None

    def get_node(self, node_id: str) -> SceneNode:
        """Return a node by id."""

        if node_id not in self._nodes:
            raise KeyError(f"node {node_id!r} does not exist")
        return self._nodes[node_id]

    def update_pose(self, node_id: str, pose: np.ndarray) -> None:
        """Replace a node pose."""

        self.get_node(node_id).pose = _validate_pose(pose)

    def add_relation(self, relation: SceneRelation) -> None:
        """Insert or replace a semantic relation."""

        if relation.subject_id not in self._nodes:
            raise KeyError(f"subject node {relation.subject_id!r} does not exist")
        if relation.object_id not in self._nodes:
            raise KeyError(f"object node {relation.object_id!r} does not exist")

        for index, existing in enumerate(self._relations):
            if (
                existing.subject_id == relation.subject_id
                and existing.predicate == relation.predicate
                and existing.object_id == relation.object_id
            ):
                self._relations[index] = SceneRelation.from_dict(relation.to_dict())
                return

        self._relations.append(SceneRelation.from_dict(relation.to_dict()))

    def remove_relation(self, subject_id: str, predicate: str, object_id: str) -> None:
        """Remove a relation by triple."""

        retained = [
            relation
            for relation in self._relations
            if (relation.subject_id, relation.predicate, relation.object_id)
            != (subject_id, predicate, object_id)
        ]
        if len(retained) == len(self._relations):
            raise KeyError(f"relation ({subject_id!r}, {predicate!r}, {object_id!r}) does not exist")
        self._relations = retained

    def get_relations(
        self,
        subject_id: str | None = None,
        predicate: str | None = None,
        object_id: str | None = None,
    ) -> list[SceneRelation]:
        """Return relations matching the requested filters."""

        return [
            relation
            for relation in self._relations
            if (subject_id is None or relation.subject_id == subject_id)
            and (predicate is None or relation.predicate == predicate)
            and (object_id is None or relation.object_id == object_id)
        ]

    def neighbors(self, node_id: str) -> list[str]:
        """Return all related node ids for the requested node."""

        self.get_node(node_id)
        neighbor_ids = {
            relation.object_id
            for relation in self._relations
            if relation.subject_id == node_id and relation.object_id != node_id
        }
        neighbor_ids.update(
            relation.subject_id
            for relation in self._relations
            if relation.object_id == node_id and relation.subject_id != node_id
        )
        return sorted(neighbor_ids)

    def subgraph(self, root_id: str, depth: int = 2) -> SceneGraph:
        """Return a relation-local subgraph around the requested root node."""

        if depth < 0:
            raise ValueError("depth must be non-negative")

        self.get_node(root_id)
        included = {root_id}
        queue: deque[tuple[str, int]] = deque([(root_id, 0)])

        while queue:
            node_id, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            for neighbor_id in self.neighbors(node_id):
                if neighbor_id in included:
                    continue
                included.add(neighbor_id)
                queue.append((neighbor_id, current_depth + 1))

        subgraph = SceneGraph()
        subgraph_nodes = []
        for node_id in sorted(included):
            node_data = self._nodes[node_id].to_dict()
            if node_data["parent_id"] not in included:
                node_data["parent_id"] = None
            subgraph_nodes.append(SceneNode.from_dict(node_data))
        subgraph._add_nodes_bulk(subgraph_nodes)
        for relation in self._relations:
            if relation.subject_id in included and relation.object_id in included:
                subgraph.add_relation(SceneRelation.from_dict(relation.to_dict()))
        return subgraph

    def to_dict(self) -> dict[str, Any]:
        """Serialize the scene graph to a plain dictionary."""

        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self._nodes.items()},
            "relations": [relation.to_dict() for relation in self._relations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneGraph:
        """Construct a scene graph from a dictionary."""

        graph = cls()
        nodes: list[SceneNode] = []
        for node_id, node_data in data.get("nodes", {}).items():
            if "id" not in node_data:
                node_data = {**node_data, "id": node_id}
            nodes.append(SceneNode.from_dict(node_data))
        graph._add_nodes_bulk(nodes)
        for relation_data in data.get("relations", []):
            graph.add_relation(SceneRelation.from_dict(relation_data))
        return graph


class SceneBuilder:
    """Helper for constructing reusable example scenes."""

    @staticmethod
    def build_kitchen() -> SceneGraph:
        """Return a household kitchen scene."""

        graph = SceneGraph()
        graph.add_node(
            SceneNode(
                id="kitchen_table",
                label="table",
                category="surface",
                pose=_make_pose(0.8, 0.0, 0.75),
                bbox=(0.6, 0.4, 0.05),
                properties={"supports_objects": True},
            )
        )
        graph.add_node(
            SceneNode(
                id="countertop",
                label="countertop",
                category="surface",
                pose=_make_pose(0.0, 1.0, 0.9),
                bbox=(0.8, 0.3, 0.05),
                properties={"supports_objects": True},
            )
        )
        graph.add_node(
            SceneNode(
                id="cup",
                label="cup",
                category="container",
                pose=_make_pose(0.75, -0.05, 0.82),
                bbox=(0.04, 0.04, 0.06),
                properties={"graspable": True, "mass": 0.3},
            )
        )
        graph.add_node(
            SceneNode(
                id="bowl",
                label="bowl",
                category="container",
                pose=_make_pose(0.95, 0.1, 0.81),
                bbox=(0.08, 0.08, 0.04),
                properties={"graspable": True, "mass": 0.4},
            )
        )
        graph.add_node(
            SceneNode(
                id="knife",
                label="knife",
                category="tool",
                pose=_make_pose(0.82, 0.18, 0.8),
                bbox=(0.12, 0.01, 0.01),
                properties={"graspable": True, "sharp": True, "mass": 0.2},
            )
        )
        graph.add_node(
            SceneNode(
                id="sink",
                label="sink",
                category="container",
                pose=_make_pose(-0.3, 1.05, 0.85),
                bbox=(0.25, 0.2, 0.12),
                properties={"wash_station": True},
            )
        )

        graph.add_relation(SceneRelation("cup", "on", "kitchen_table"))
        graph.add_relation(SceneRelation("bowl", "on", "kitchen_table"))
        graph.add_relation(SceneRelation("knife", "on", "kitchen_table"))
        graph.add_relation(SceneRelation("kitchen_table", "near", "countertop", confidence=0.9))
        graph.add_relation(SceneRelation("sink", "attached_to", "countertop"))
        graph.add_relation(SceneRelation("countertop", "supports", "sink"))
        return graph

    @staticmethod
    def build_warehouse() -> SceneGraph:
        """Return a warehouse handling scene."""

        graph = SceneGraph()
        graph.add_node(
            SceneNode(
                id="pallet",
                label="pallet",
                category="surface",
                pose=_make_pose(1.2, -0.2, 0.15),
                bbox=(0.6, 0.5, 0.07),
                properties={"supports_objects": True},
            )
        )
        graph.add_node(
            SceneNode(
                id="shelf",
                label="shelf",
                category="surface",
                pose=_make_pose(2.0, 0.8, 1.2),
                bbox=(0.8, 0.3, 1.2),
                properties={"supports_objects": True},
            )
        )
        graph.add_node(
            SceneNode(
                id="box",
                label="box",
                category="container",
                pose=_make_pose(1.15, -0.2, 0.32),
                bbox=(0.2, 0.2, 0.2),
                properties={"graspable": True, "mass": 4.0},
            )
        )
        graph.add_node(
            SceneNode(
                id="conveyor",
                label="conveyor",
                category="surface",
                pose=_make_pose(0.2, -1.0, 0.5),
                bbox=(1.2, 0.3, 0.1),
                properties={"moving_surface": True},
            )
        )
        graph.add_node(
            SceneNode(
                id="safety_barrier",
                label="safety barrier",
                category="obstacle",
                pose=_make_pose(0.9, 0.45, 0.6),
                bbox=(0.05, 1.0, 0.6),
                properties={"passable": False},
            )
        )

        graph.add_relation(SceneRelation("box", "on", "pallet"))
        graph.add_relation(SceneRelation("pallet", "near", "conveyor", confidence=0.85))
        graph.add_relation(SceneRelation("shelf", "near", "pallet", confidence=0.8))
        graph.add_relation(SceneRelation("safety_barrier", "near", "conveyor", confidence=0.95))
        return SceneBuilder.add_robot(graph)

    @staticmethod
    def add_robot(
        graph: SceneGraph,
        robot_id: str = "humanoid",
        pose: np.ndarray | None = None,
    ) -> SceneGraph:
        """Insert a robot node into an existing scene graph."""

        graph.add_node(
            SceneNode(
                id=robot_id,
                label="humanoid robot",
                category="robot",
                pose=_identity_pose() if pose is None else pose,
                bbox=(0.3, 0.3, 0.9),
                properties={"mobile": True, "manipulator": True},
            )
        )
        return graph


class SceneQuery:
    """Search and conversion helpers for scene graphs."""

    @staticmethod
    def find_by_category(graph: SceneGraph, category: str) -> list[SceneNode]:
        """Return all nodes in the requested category."""

        return [node for node in graph.nodes.values() if node.category == category]

    @staticmethod
    def find_graspable(graph: SceneGraph) -> list[SceneNode]:
        """Return nodes marked as graspable."""

        return [node for node in graph.nodes.values() if bool(node.properties.get("graspable", False))]

    @staticmethod
    def find_on_surface(graph: SceneGraph, surface_id: str) -> list[SceneNode]:
        """Return nodes that are on the requested surface."""

        if surface_id not in graph.nodes:
            raise KeyError(f"node {surface_id!r} does not exist")
        node_ids = [relation.subject_id for relation in graph.get_relations(predicate="on", object_id=surface_id)]
        return [graph.get_node(node_id) for node_id in node_ids]

    @staticmethod
    def find_path(graph: SceneGraph, from_id: str, to_id: str) -> list[str]:
        """Return the shortest undirected relation path between two nodes."""

        graph.get_node(from_id)
        graph.get_node(to_id)
        if from_id == to_id:
            return [from_id]

        queue: deque[list[str]] = deque([[from_id]])
        visited = {from_id}
        while queue:
            path = queue.popleft()
            for neighbor_id in graph.neighbors(path[-1]):
                if neighbor_id in visited:
                    continue
                next_path = [*path, neighbor_id]
                if neighbor_id == to_id:
                    return next_path
                visited.add(neighbor_id)
                queue.append(next_path)
        return []

    @staticmethod
    def find_reachable(
        graph: SceneGraph,
        robot_id: str,
        reach_radius: float = 0.8,
    ) -> list[SceneNode]:
        """Return nodes inside the robot reach sphere based on pose translations."""

        if reach_radius < 0.0:
            raise ValueError("reach_radius must be non-negative")

        robot = graph.get_node(robot_id)
        robot_position = robot.pose[:3, 3]
        reachable: list[SceneNode] = []
        for node in graph.nodes.values():
            if node.id == robot_id:
                continue
            distance = float(np.linalg.norm(node.pose[:3, 3] - robot_position))
            if distance <= reach_radius:
                reachable.append(node)
        return reachable

    @staticmethod
    def to_tamp_predicates(graph: SceneGraph) -> list[dict[str, Any]]:
        """Convert scene relations into TAMP-style predicate dictionaries."""

        return [
            {
                "name": relation.predicate,
                "args": [relation.subject_id, relation.object_id],
                "value": relation.confidence > 0.0,
            }
            for relation in graph.relations
        ]


def _make_pose(x: float, y: float, z: float) -> np.ndarray:
    """Return a simple translation-only homogeneous transform."""

    pose = _identity_pose()
    pose[:3, 3] = [x, y, z]
    return pose
