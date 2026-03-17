"""Instruction templates for generating simple robot commands."""

from __future__ import annotations

import random


class InstructionTemplate:
    PICK_TEMPLATES = [
        "pick up the {object}",
        "grab {object}",
        "grasp the {object}",
        "take the {object}",
        "grab {object} from {surface}",
    ]
    PLACE_TEMPLATES = [
        "put {object} on {surface}",
        "place {object} onto {destination}",
        "set the {object} on {surface}",
        "drop the {object} onto {destination}",
    ]
    GO_TEMPLATES = [
        "walk to {location}",
        "navigate to {object}",
        "go to {location}",
    ]
    PUSH_TEMPLATES = [
        "push the {object}",
        "slide the {object}",
    ]
    HANDOVER_TEMPLATES = [
        "hand over the {object}",
        "give me the {object}",
    ]

    @classmethod
    def generate(
        cls,
        action: str,
        target: str | None = None,
        destination: str | None = None,
    ) -> str:
        templates = cls.all_templates().get(action, [])
        if not templates:
            raise ValueError(f"unknown action {action!r}")
        template = random.choice(templates)
        replacements = {
            "object": target or "object",
            "surface": destination or target or "surface",
            "destination": destination or target or "destination",
            "location": target or destination or "location",
        }
        return template.format(**replacements)

    @classmethod
    def all_templates(cls) -> dict[str, list[str]]:
        return {
            "pick": list(cls.PICK_TEMPLATES),
            "place": list(cls.PLACE_TEMPLATES),
            "go": list(cls.GO_TEMPLATES),
            "push": list(cls.PUSH_TEMPLATES),
            "handover": list(cls.HANDOVER_TEMPLATES),
        }


__all__ = ["InstructionTemplate"]
