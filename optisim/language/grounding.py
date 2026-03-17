"""Ground natural language intents into scene entities and primitives."""

from __future__ import annotations

from dataclasses import dataclass, field

from optisim.language.parser import InstructionParser, ParsedIntent
from optisim.scene import SceneGraph, SceneNode


@dataclass(slots=True)
class GroundedInstruction:
    intent: ParsedIntent
    target_node_id: str | None
    destination_node_id: str | None
    primitive_sequence: list[dict]
    grounding_confidence: float
    notes: list[str] = field(default_factory=list)


class Grounder:
    """Resolve parsed intents against a scene graph."""

    def __init__(self, scene: SceneGraph) -> None:
        self.scene = scene
        self.parser = InstructionParser(scene=scene)

    def ground(self, intent: ParsedIntent) -> GroundedInstruction:
        notes: list[str] = []
        target_node = self._match_node(intent.target)
        destination_node = self._match_node(intent.destination)

        if intent.target and target_node is None:
            notes.append("target not found in scene")
        if intent.destination and destination_node is None:
            notes.append("destination not found in scene")
        if intent.target and target_node is not None and self._is_fuzzy_match(intent.target, target_node):
            notes.append("using closest match")
        if intent.destination and destination_node is not None and self._is_fuzzy_match(intent.destination, destination_node):
            notes.append("using closest match")

        primitive_sequence = self._build_primitive_sequence(intent, target_node, destination_node, notes)
        grounding_confidence = self._compute_confidence(intent, target_node, destination_node)
        if intent.target and target_node is None:
            grounding_confidence = min(grounding_confidence, 0.2)

        return GroundedInstruction(
            intent=intent,
            target_node_id=None if target_node is None else target_node.id,
            destination_node_id=None if destination_node is None else destination_node.id,
            primitive_sequence=primitive_sequence,
            grounding_confidence=grounding_confidence,
            notes=notes,
        )

    def ground_instruction(self, text: str) -> GroundedInstruction:
        return self.ground(self.parser.parse(text))

    def suggest_targets(self, intent: ParsedIntent) -> list[str]:
        if not intent.target:
            return []
        needle = self._normalize(intent.target)
        matches: list[str] = []
        for node in self.scene.nodes.values():
            label = self._normalize(node.label)
            node_id = self._normalize(node.id)
            if needle in label or needle in node_id or label in needle or node_id in needle:
                matches.append(node.id)
        return matches

    def _build_primitive_sequence(
        self,
        intent: ParsedIntent,
        target_node: SceneNode | None,
        destination_node: SceneNode | None,
        notes: list[str],
    ) -> list[dict]:
        target_id = None if target_node is None else target_node.id
        destination_id = None if destination_node is None else destination_node.id

        if intent.action == "pick" and target_id is not None:
            return [
                {"primitive": "navigate", "params": {"target_id": target_id}},
                {"primitive": "reach", "params": {"target_id": target_id, "end_effector": "right"}},
                {"primitive": "grasp", "params": {"target_id": target_id, "end_effector": "right"}},
            ]
        if intent.action == "place":
            if destination_id is None:
                notes.append("destination required for place action")
                return []
            if target_id is None:
                notes.append("target required for place action")
                return [{"primitive": "navigate", "params": {"target_id": destination_id}}]
            return [
                {"primitive": "navigate", "params": {"target_id": destination_id}},
                {"primitive": "place", "params": {"object_id": target_id, "surface_id": destination_id}},
            ]
        if intent.action == "push" and target_id is not None:
            direction = [1.0, 0.0, 0.0]
            if intent.modifier == "left":
                direction = [0.0, 1.0, 0.0]
            elif intent.modifier == "right":
                direction = [0.0, -1.0, 0.0]
            return [
                {"primitive": "navigate", "params": {"target_id": target_id}},
                {"primitive": "push", "params": {"target_id": target_id, "direction": direction}},
            ]
        if intent.action == "go":
            go_target = destination_id or target_id
            if go_target is not None:
                return [{"primitive": "navigate", "params": {"target_id": go_target}}]
            return []
        if intent.action == "handover":
            params: dict[str, str] = {"from_arm": "left", "to_arm": "right"}
            if target_id is not None:
                params["object_id"] = target_id
            return [{"primitive": "handover", "params": params}]
        return []

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().replace("_", " ").split())

    def _match_node(self, text: str | None) -> SceneNode | None:
        if not text:
            return None
        needle = self._normalize(text)
        exact_match: SceneNode | None = None
        partial_match: SceneNode | None = None
        for node in self.scene.nodes.values():
            label = self._normalize(node.label)
            node_id = self._normalize(node.id)
            if needle == label or needle == node_id:
                exact_match = node
                break
            if needle in label or needle in node_id or label in needle or node_id in needle:
                if partial_match is None:
                    partial_match = node
        return exact_match or partial_match

    def _is_fuzzy_match(self, text: str, node: SceneNode) -> bool:
        needle = self._normalize(text)
        return needle not in {self._normalize(node.id), self._normalize(node.label)}

    @staticmethod
    def _compute_confidence(
        intent: ParsedIntent,
        target_node: SceneNode | None,
        destination_node: SceneNode | None,
    ) -> float:
        confidence = intent.confidence
        if intent.target:
            confidence *= 1.0 if target_node is not None else 0.4
        if intent.destination:
            confidence *= 1.0 if destination_node is not None else 0.7
        return max(0.0, min(1.0, confidence))


__all__ = ["GroundedInstruction", "Grounder"]
