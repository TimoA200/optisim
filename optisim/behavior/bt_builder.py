"""Behavior tree builder utilities and YAML loading."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from optisim.core import ActionPrimitive

from .bt_nodes import (
    ActionNode,
    BTNode,
    BehaviorContext,
    Condition,
    Fallback,
    Inverter,
    Parallel,
    Predicate,
    Repeat,
    RetryUntilSuccess,
    Sequence,
    Timeout,
    ensure_childless,
)


@dataclass(slots=True)
class BehaviorTreeDefinition:
    """Serializable behavior tree document used by the CLI and examples."""

    name: str
    root: BTNode
    world: dict[str, Any] = field(default_factory=dict)
    robot: dict[str, Any] = field(default_factory=dict)
    blackboard: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BehaviorTreeDefinition":
        """Build a behavior tree definition from a mapping payload."""

        if "tree" not in payload:
            raise ValueError("behavior tree document requires a 'tree' field")
        root = load_node_from_dict(payload["tree"])
        definition = cls(
            name=str(payload.get("name", "behavior_tree")),
            root=root,
            world=dict(payload.get("world", {})),
            robot=dict(payload.get("robot", {})),
            blackboard=dict(payload.get("blackboard", {})),
            metadata=dict(payload.get("metadata", {})),
        )
        definition.validate()
        return definition

    @classmethod
    def from_file(cls, path: str | Path) -> "BehaviorTreeDefinition":
        """Load a behavior tree definition from YAML or JSON."""

        source = Path(path)
        raw = source.read_text(encoding="utf-8")
        payload = json.loads(raw) if source.suffix.lower() == ".json" else yaml.safe_load(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"behavior tree file {source} does not contain a mapping")
        return cls.from_dict(payload)

    def validate(self) -> None:
        """Validate the in-memory tree structure by traversing it recursively."""

        _validate_node(self.root)


class BehaviorTreeBuilder:
    """Fluent builder for constructing behavior trees in Python."""

    def __init__(self) -> None:
        self._root: BTNode | None = None
        self._stack: list[_BuilderFrame] = []

    def sequence(self, name: str | None = None) -> "BehaviorTreeBuilder":
        """Open a sequence composite."""

        return self._push(
            _BuilderFrame(kind="composite", node_factory=lambda children: Sequence(name=name, child_nodes=tuple(children)))
        )

    def selector(self, name: str | None = None) -> "BehaviorTreeBuilder":
        """Open a selector composite."""

        return self._push(
            _BuilderFrame(kind="composite", node_factory=lambda children: Fallback(name=name, child_nodes=tuple(children)))
        )

    def fallback(self, name: str | None = None) -> "BehaviorTreeBuilder":
        """Open a fallback composite."""

        return self.selector(name=name)

    def parallel(self, success_policy: str = "require_all", name: str | None = None) -> "BehaviorTreeBuilder":
        """Open a parallel composite."""

        return self._push(
            _BuilderFrame(
                kind="composite",
                node_factory=lambda children: Parallel(
                    name=name,
                    child_nodes=tuple(children),
                    success_policy=_validate_parallel_policy(success_policy),
                ),
            )
        )

    def inverter(self, name: str | None = None) -> "BehaviorTreeBuilder":
        """Open an inverter decorator."""

        return self._push(
            _BuilderFrame(
                kind="decorator",
                node_factory=lambda children: Inverter(name=name, child=_single_child(name or "Inverter", children)),
            )
        )

    def repeat(self, count: int, name: str | None = None) -> "BehaviorTreeBuilder":
        """Open a repeat decorator."""

        return self._push(
            _BuilderFrame(
                kind="decorator",
                node_factory=lambda children: Repeat(
                    name=name,
                    child=_single_child(name or "Repeat", children),
                    count=count,
                ),
            )
        )

    def retry_until_success(self, max_attempts: int, name: str | None = None) -> "BehaviorTreeBuilder":
        """Open a retry decorator."""

        return self._push(
            _BuilderFrame(
                kind="decorator",
                node_factory=lambda children: RetryUntilSuccess(
                    name=name,
                    child=_single_child(name or "RetryUntilSuccess", children),
                    max_attempts=max_attempts,
                ),
            )
        )

    def timeout(self, steps: int, name: str | None = None) -> "BehaviorTreeBuilder":
        """Open a timeout decorator."""

        return self._push(
            _BuilderFrame(
                kind="decorator",
                node_factory=lambda children: Timeout(
                    name=name,
                    child=_single_child(name or "Timeout", children),
                    steps=steps,
                ),
            )
        )

    def condition(self, predicate: Predicate, name: str | None = None) -> "BehaviorTreeBuilder":
        """Append a condition leaf node."""

        return self._append(Condition(name=name, predicate=predicate))

    def yaml_condition(self, spec: dict[str, Any], name: str | None = None) -> "BehaviorTreeBuilder":
        """Append a built-in YAML-backed condition leaf."""

        return self._append(Condition(name=name, predicate=build_condition_predicate(spec)))

    def action(
        self,
        action: ActionPrimitive | None = None,
        *,
        name: str | None = None,
        action_type: str | None = None,
        **kwargs: Any,
    ) -> "BehaviorTreeBuilder":
        """Append an action leaf node."""

        resolved_action = action if action is not None else ActionPrimitive.from_dict({"type": action_type, **kwargs})
        return self._append(ActionNode(name=name, action=resolved_action))

    def node(self, node: BTNode) -> "BehaviorTreeBuilder":
        """Append an existing node instance to the current builder scope."""

        return self._append(node)

    def end(self) -> "BehaviorTreeBuilder":
        """Close the current composite or decorator scope."""

        if not self._stack:
            raise ValueError("builder stack is empty")
        frame = self._stack.pop()
        return self._append(frame.node_factory(frame.children))

    def build(self) -> BTNode:
        """Finalize the builder and return the root node."""

        if self._stack:
            raise ValueError("builder contains unclosed composite or decorator scopes")
        if self._root is None:
            raise ValueError("builder does not contain a root node")
        return self._root

    def _push(self, frame: "_BuilderFrame") -> "BehaviorTreeBuilder":
        self._stack.append(frame)
        return self

    def _append(self, node: BTNode) -> "BehaviorTreeBuilder":
        if self._stack:
            self._stack[-1].children.append(node)
            return self
        if self._root is not None:
            raise ValueError("builder already contains a root node")
        self._root = node
        return self


@dataclass(slots=True)
class _BuilderFrame:
    kind: str
    node_factory: Callable[[list[BTNode]], BTNode]
    children: list[BTNode] = field(default_factory=list)


def load_node_from_dict(payload: dict[str, Any]) -> BTNode:
    """Build a behavior tree node from a serialized mapping."""

    if not isinstance(payload, dict):
        raise ValueError("tree node must be a mapping")
    node_type = str(payload.get("type", "")).lower()
    name = payload.get("name")
    children_payload = payload.get("children", [])
    if children_payload is None:
        children_payload = []
    if not isinstance(children_payload, list):
        raise ValueError("node 'children' must be a list")
    children = [load_node_from_dict(child) for child in children_payload]

    if node_type == "sequence":
        return Sequence(name=name, child_nodes=tuple(children))
    if node_type in {"selector", "fallback"}:
        return Fallback(name=name, child_nodes=tuple(children))
    if node_type == "parallel":
        return Parallel(
            name=name,
            child_nodes=tuple(children),
            success_policy=_validate_parallel_policy(str(payload.get("success_policy", "require_all"))),
        )
    if node_type == "condition":
        ensure_childless("condition", children)
        spec = payload.get("condition", payload)
        return Condition(name=name, predicate=build_condition_predicate(spec))
    if node_type == "action":
        ensure_childless("action", children)
        action_payload = payload.get("action")
        if not isinstance(action_payload, dict):
            raise ValueError("action node requires an 'action' mapping")
        return ActionNode(name=name, action=ActionPrimitive.from_dict(action_payload))
    if node_type == "inverter":
        return Inverter(name=name, child=_single_child("Inverter", children))
    if node_type == "repeat":
        return Repeat(name=name, child=_single_child("Repeat", children), count=int(payload["count"]))
    if node_type == "retry_until_success":
        return RetryUntilSuccess(
            name=name,
            child=_single_child("RetryUntilSuccess", children),
            max_attempts=int(payload["max_attempts"]),
        )
    if node_type == "timeout":
        return Timeout(name=name, child=_single_child("Timeout", children), steps=int(payload["steps"]))
    raise ValueError(f"unsupported behavior tree node type '{node_type}'")


def build_condition_predicate(spec: dict[str, Any]) -> Predicate:
    """Create a runtime predicate from a YAML condition specification."""

    predicate_name = str(spec.get("predicate", "")).lower()
    if predicate_name == "object_exists":
        target = str(spec["object"])
        return lambda context: target in context.world.objects
    if predicate_name == "surface_exists":
        target = str(spec["surface"])
        return lambda context: target in context.world.surfaces
    if predicate_name == "object_reachable":
        target = str(spec["object"])
        reach_scale = float(spec.get("reach_scale", 1.0))
        explicit_max_distance = spec.get("max_distance")
        return lambda context: _object_reachable(
            context,
            target,
            float(explicit_max_distance)
            if explicit_max_distance is not None
            else float(context.engine.robot.max_reach()) * reach_scale,
        )
    if predicate_name == "object_held":
        target = str(spec["object"])
        holder = spec.get("by")
        return lambda context: _object_held(context, target, holder)
    if predicate_name == "blackboard_equals":
        key = str(spec["key"])
        value = spec.get("value")
        return lambda context: context.blackboard.get(key) == value
    if predicate_name == "blackboard_truthy":
        key = str(spec["key"])
        return lambda context: bool(context.blackboard.get(key))
    if predicate_name == "blackboard_has":
        key = str(spec["key"])
        return lambda context: key in context.blackboard
    if predicate_name == "object_on_surface":
        target = str(spec["object"])
        surface_name = str(spec["surface"])
        tolerance = float(spec.get("tolerance", 0.03))
        return lambda context: _object_on_surface(context, target, surface_name, tolerance)
    if predicate_name == "world_time_less_than":
        limit_s = float(spec["seconds"])
        return lambda context: context.world.time_s < limit_s
    raise ValueError(f"unsupported condition predicate '{predicate_name}'")


def _validate_node(node: BTNode) -> None:
    children = node.children()
    if isinstance(node, (Inverter, Repeat, RetryUntilSuccess, Timeout)) and len(children) != 1:
        raise ValueError(f"{node.__class__.__name__} requires exactly one child")
    if isinstance(node, (Sequence, Fallback, Parallel)):
        for child in children:
            _validate_node(child)
    elif isinstance(node, (Inverter, Repeat, RetryUntilSuccess, Timeout)):
        _validate_node(children[0])


def _single_child(label: str, children: list[BTNode]) -> BTNode:
    if len(children) != 1:
        raise ValueError(f"{label} requires exactly one child")
    return children[0]


def _validate_parallel_policy(policy: str) -> str:
    if policy not in {"require_all", "require_one"}:
        raise ValueError("parallel success_policy must be 'require_all' or 'require_one'")
    return policy


def _object_reachable(context: BehaviorContext, target: str, max_distance: float) -> bool:
    if target not in context.world.objects:
        return False
    position = context.world.objects[target].pose.position
    distance = float(np.linalg.norm(position - context.engine.robot.base_pose.position))
    return distance <= max_distance


def _object_held(context: BehaviorContext, target: str, holder: Any | None) -> bool:
    if target not in context.world.objects:
        return False
    held_by = context.world.objects[target].held_by
    return held_by is not None if holder is None else held_by == holder


def _object_on_surface(context: BehaviorContext, target: str, surface_name: str, tolerance: float) -> bool:
    if target not in context.world.objects or surface_name not in context.world.surfaces:
        return False
    obj = context.world.objects[target]
    surface = context.world.surfaces[surface_name]
    expected_z = surface.pose.position[2] + surface.size[2] / 2.0 + obj.size[2] / 2.0
    return abs(float(obj.pose.position[2] - expected_z)) <= tolerance
