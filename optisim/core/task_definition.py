"""Serializable task definition."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import yaml
from yaml import YAMLError

from optisim.core.action_primitives import ActionPrimitive
from optisim.core.task_composer import TaskComposer


@dataclass(slots=True)
class TaskDefinition:
    """Top-level task document exchanged between CLI and runtime."""

    name: str
    actions: list[ActionPrimitive]
    world: dict[str, Any] = field(default_factory=dict)
    robot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the task definition into a plain Python mapping."""

        return {
            "name": self.name,
            "metadata": self.metadata,
            "robot": self.robot,
            "world": self.world,
            "actions": [action.to_dict() for action in self.actions],
        }

    def dump(self, path: str | Path) -> None:
        """Write the task definition to JSON or YAML based on file extension."""

        destination = Path(path)
        payload = self.to_dict()
        if destination.suffix.lower() == ".json":
            destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return
        destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskDefinition":
        """Create a task definition from an in-memory mapping payload."""

        if not isinstance(payload, dict):
            raise ValueError("task payload must be a mapping")
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("task file must define a non-empty string field 'name'")
        raw_actions = payload.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise ValueError("task file must define a non-empty 'actions' list")
        actions: list[ActionPrimitive] = []
        for index, item in enumerate(raw_actions):
            if not isinstance(item, dict):
                raise ValueError(f"action[{index}] must be a mapping")
            try:
                actions.append(ActionPrimitive.from_dict(item))
            except KeyError as exc:
                missing = exc.args[0]
                raise ValueError(f"action[{index}] is missing required field '{missing}'") from exc
            except ValueError as exc:
                raise ValueError(f"action[{index}] is invalid: {exc}") from exc

        world = payload.get("world", {})
        if not isinstance(world, dict):
            raise ValueError("task field 'world' must be a mapping")
        robot = payload.get("robot", {})
        if not isinstance(robot, dict):
            raise ValueError("task field 'robot' must be a mapping")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("task field 'metadata' must be a mapping")

        return cls(name=name, actions=actions, world=dict(world), robot=dict(robot), metadata=dict(metadata))

    @classmethod
    def from_file(cls, path: str | Path) -> "TaskDefinition":
        """Load a task definition from a YAML or JSON file on disk."""

        source = Path(path)
        try:
            raw = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"failed to read task file {source}: {exc}") from exc
        try:
            if source.suffix.lower() == ".json":
                payload = json.loads(raw)
            else:
                payload = yaml.safe_load(raw)
        except JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in task file {source}: {exc.msg}") from exc
        except YAMLError as exc:
            raise ValueError(f"invalid YAML in task file {source}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"task file {source} does not contain a mapping")
        try:
            return cls.from_dict(payload)
        except ValueError as exc:
            raise ValueError(f"invalid task file {source}: {exc}") from exc

    @classmethod
    def from_composer(
        cls,
        composer: TaskComposer,
        *,
        world: dict[str, Any] | None = None,
        robot: dict[str, Any] | None = None,
    ) -> "TaskDefinition":
        """Build a task definition from a fluent ``TaskComposer`` instance."""

        return cls(
            name=composer.name,
            actions=list(composer.actions),
            metadata=dict(composer.metadata),
            world=dict(world or {}),
            robot=dict(robot or {}),
        )

__all__ = ["TaskDefinition"]
