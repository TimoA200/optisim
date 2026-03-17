"""Serializable task definition."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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

        actions = [ActionPrimitive.from_dict(item) for item in payload.get("actions", [])]
        return cls(
            name=payload["name"],
            actions=actions,
            world=dict(payload.get("world", {})),
            robot=dict(payload.get("robot", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "TaskDefinition":
        """Load a task definition from a YAML or JSON file on disk."""

        source = Path(path)
        raw = source.read_text(encoding="utf-8")
        if source.suffix.lower() == ".json":
            payload = json.loads(raw)
        else:
            payload = yaml.safe_load(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"task file {source} does not contain a mapping")
        return cls.from_dict(payload)

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
