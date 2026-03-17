"""Task template catalog for pre-built optisim scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Any, Callable

from optisim.core import TaskDefinition


class DifficultyLevel(StrEnum):
    """Supported difficulty levels for bundled task templates."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass(frozen=True, slots=True)
class TemplateParameter:
    """Describes a configurable template parameter and its default value."""

    name: str
    description: str
    default: Any


@dataclass(frozen=True, slots=True)
class TemplateInfo:
    """User-facing metadata for a task template."""

    name: str
    description: str
    tags: tuple[str, ...]
    difficulty: DifficultyLevel
    parameters: tuple[TemplateParameter, ...] = ()


TaskBuilder = Callable[..., TaskDefinition]


@dataclass(frozen=True, slots=True)
class TaskTemplate:
    """Callable task template plus catalog metadata."""

    name: str
    description: str
    tags: tuple[str, ...]
    difficulty: DifficultyLevel
    build: TaskBuilder
    parameters: tuple[TemplateParameter, ...] = ()

    def info(self) -> TemplateInfo:
        """Return the user-facing metadata view of the template."""

        return TemplateInfo(
            name=self.name,
            description=self.description,
            tags=self.tags,
            difficulty=self.difficulty,
            parameters=self.parameters,
        )


@dataclass
class TaskCatalog:
    """Registry of reusable task templates with discovery and lookup helpers."""

    _templates: dict[str, TaskTemplate] = field(default_factory=dict)

    def __init__(self, templates: list[TaskTemplate] | None = None) -> None:
        """Create a catalog, preloading the bundled templates by default."""

        self._templates = {}
        if templates is None:
            from optisim.library.templates import builtin_templates

            templates = builtin_templates()
        for template in templates:
            self.register(template)

    def register(self, template: TaskTemplate) -> None:
        """Register or replace a task template in the catalog."""

        self._templates[template.name] = template

    def list(self) -> list[TemplateInfo]:
        """Return all available templates sorted by name."""

        return [self._templates[name].info() for name in sorted(self._templates)]

    def info(self, name: str) -> TemplateInfo:
        """Return metadata for a named template."""

        return self._get_template(name).info()

    def get(self, name: str, **parameters: Any) -> TaskDefinition:
        """Build and return a ready-to-run task definition for ``name``."""

        return self._get_template(name).build(**parameters)

    def search(self, keyword: str) -> list[TemplateInfo]:
        """Return fuzzy matches across name, description, and tags."""

        needle = keyword.strip().lower()
        if not needle:
            return self.list()

        matches: list[tuple[float, TaskTemplate]] = []
        for template in self._templates.values():
            normalized_name = template.name.replace("_", " ").lower()
            description = template.description.lower()
            tags = [tag.lower() for tag in template.tags]
            if needle in normalized_name or needle in description or any(needle in tag for tag in tags):
                score = 2.0 + self._similarity(needle, normalized_name)
                matches.append((score, template))
                continue

            name_score = self._similarity(needle, normalized_name)
            token_score = max(
                [name_score, *(self._similarity(needle, token) for token in description.split()), *(self._similarity(needle, tag) for tag in tags)],
                default=0.0,
            )
            if max(name_score, token_score) >= 0.72:
                matches.append((max(name_score * 1.2, token_score), template))
        matches.sort(key=lambda item: (-item[0], item[1].name))
        return [template.info() for _, template in matches]

    def _get_template(self, name: str) -> TaskTemplate:
        try:
            return self._templates[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._templates))
            raise KeyError(f"unknown task template '{name}'. Available templates: {available}") from exc

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        """Return a normalized similarity score between two strings."""

        return SequenceMatcher(a=left, b=right).ratio()
