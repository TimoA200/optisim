"""World-state representations and encoders for learned transition models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optisim.scene import SceneGraph

_CATEGORIES = ("robot", "surface", "container", "tool", "object", "obstacle")
_PREDICATES = (
    "on",
    "near",
    "held_by",
    "displaced",
    "attached_to",
    "supports",
    "inside",
    "under",
    "aligned_with",
    "reachable",
)
_PRIMITIVES = ("reach", "grasp", "place", "push", "handover", "navigate")


def _stable_hash(value: str) -> int:
    text = str(value)
    total = 0
    for index, char in enumerate(text):
        total += (index + 1) * ord(char)
    return total


@dataclass(slots=True)
class WorldState:
    """Encoded scene state paired with the robot joint configuration."""

    joint_positions: np.ndarray
    scene_features: np.ndarray
    relation_vector: np.ndarray
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.joint_positions = np.asarray(self.joint_positions, dtype=float).reshape(-1)
        self.scene_features = np.asarray(self.scene_features, dtype=float).reshape(-1)
        self.relation_vector = np.asarray(self.relation_vector, dtype=float).reshape(-1)
        self.timestamp = float(self.timestamp)
        if self.joint_positions.shape != (31,):
            raise ValueError(f"joint_positions must have shape (31,), got {self.joint_positions.shape}")

    def as_vector(self) -> np.ndarray:
        """Return the learnable scene state vector."""

        return np.concatenate([self.scene_features, self.relation_vector]).astype(float, copy=False)


class StateEncoder:
    """Convert semantic scenes and primitive actions into fixed-size vectors."""

    def __init__(self, max_nodes: int = 20, max_relations: int = 50) -> None:
        self.max_nodes = int(max_nodes)
        self.max_relations = int(max_relations)
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        if self.max_relations <= 0:
            raise ValueError("max_relations must be positive")

    @property
    def state_dim(self) -> int:
        return self.max_nodes * 13

    @property
    def relation_dim(self) -> int:
        return self.max_relations

    @property
    def action_dim(self) -> int:
        return 8

    def encode_scene(self, scene: SceneGraph) -> np.ndarray:
        """Encode a scene graph into a fixed-size flattened node-feature vector."""

        features = np.zeros(self.state_dim, dtype=float)
        nodes = sorted(scene.nodes.values(), key=lambda node: node.id)[: self.max_nodes]
        for index, node in enumerate(nodes):
            base = index * 13
            position = np.asarray(node.pose[:3, 3], dtype=float)
            bbox = np.asarray(node.bbox, dtype=float)
            category_vec = np.zeros(len(_CATEGORIES), dtype=float)
            if node.category in _CATEGORIES:
                category_vec[_CATEGORIES.index(node.category)] = 1.0
            graspable = float(bool(node.properties.get("graspable", False)))
            features[base : base + 3] = position
            features[base + 3 : base + 6] = bbox
            features[base + 6 : base + 12] = category_vec
            features[base + 12] = graspable
        return features

    def encode_relations(self, scene: SceneGraph) -> np.ndarray:
        """Encode semantic relations using a fixed hash-based sparse vector."""

        vector = np.zeros(self.relation_dim, dtype=float)
        node_ids = set(sorted(scene.nodes)[: self.max_nodes])
        for relation in scene.relations:
            if relation.subject_id not in node_ids or relation.object_id not in node_ids:
                continue
            predicate_key = relation.predicate if relation.predicate in _PREDICATES else "unknown"
            triple_key = f"{relation.subject_id}:{predicate_key}:{relation.object_id}"
            hashed_index = _stable_hash(triple_key) % self.relation_dim
            vector[hashed_index] = 1.0
        return vector

    def encode_action(self, primitive_name: str, params: dict) -> np.ndarray:
        """Encode a primitive invocation into a compact action vector."""

        vec = np.zeros(self.action_dim, dtype=float)
        if primitive_name in _PRIMITIVES:
            vec[_PRIMITIVES.index(primitive_name)] = 1.0
        target_id = str(params.get("target_id", params.get("object_id", "")))
        surface_id = str(params.get("surface_id", ""))
        vec[6] = (_stable_hash(target_id) % 1000) / 999.0 if target_id else 0.0
        vec[7] = (_stable_hash(surface_id) % 1000) / 999.0 if surface_id else 0.0
        return vec

    def decode_relation_vector(self, vec: np.ndarray, scene: SceneGraph) -> list[tuple[str, str, str]]:
        """Decode active relation slots back to scene triples when possible."""

        decoded: list[tuple[str, str, str]] = []
        active = np.asarray(vec, dtype=float).reshape(-1)
        node_ids = set(sorted(scene.nodes)[: self.max_nodes])
        for relation in scene.relations:
            if relation.subject_id not in node_ids or relation.object_id not in node_ids:
                continue
            predicate_key = relation.predicate if relation.predicate in _PREDICATES else "unknown"
            triple_key = f"{relation.subject_id}:{predicate_key}:{relation.object_id}"
            hashed_index = _stable_hash(triple_key) % self.relation_dim
            if active[hashed_index] > 0.5:
                decoded.append((relation.subject_id, relation.predicate, relation.object_id))
        return decoded


__all__ = ["WorldState", "StateEncoder"]
