"""Natural language instruction parsing for semantic robot commands."""

from __future__ import annotations

from dataclasses import dataclass
import re

from optisim.scene import SceneGraph

_TOKEN_RE = re.compile(r"[a-z0-9_']+")

_ACTION_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pick", ("pick up", "pick", "grab", "grasp", "take", "lift")),
    ("place", ("place", "put", "set", "drop")),
    ("go", ("move to", "go to", "walk to", "navigate to", "go", "walk", "navigate")),
    ("push", ("push", "slide", "shove", "move")),
    ("handover", ("hand", "give", "pass", "transfer")),
    ("open", ("open",)),
    ("close", ("close",)),
    ("turn", ("turn", "rotate")),
)
_ACTION_TOKENS = {
    "close",
    "drop",
    "give",
    "go",
    "grab",
    "grasp",
    "hand",
    "lift",
    "move",
    "navigate",
    "open",
    "pass",
    "pick",
    "place",
    "push",
    "put",
    "rotate",
    "set",
    "shove",
    "slide",
    "take",
    "transfer",
    "turn",
    "walk",
}

_DESTINATION_PREPOSITIONS = {"on", "onto", "to", "in", "into"}
_MODIFIER_ALIASES = {
    "carefully": "carefully",
    "slowly": "carefully",
    "gently": "carefully",
    "quickly": "quickly",
    "fast": "quickly",
    "left": "left",
    "right": "right",
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "down",
    "from",
    "my",
    "near",
    "of",
    "please",
    "the",
    "then",
    "up",
    "with",
}
_DEFAULT_NOUNS = {
    "bottle",
    "bowl",
    "box",
    "cabinet",
    "counter",
    "countertop",
    "cup",
    "door",
    "drawer",
    "fork",
    "hand",
    "jar",
    "knife",
    "location",
    "mug",
    "object",
    "plate",
    "room",
    "shelf",
    "sink",
    "spoon",
    "table",
    "target",
    "tray",
}


@dataclass(slots=True)
class ParsedIntent:
    action: str
    target: str | None
    destination: str | None
    modifier: str | None
    raw_text: str
    confidence: float


class InstructionParser:
    """Rule-based natural language parser for robot instructions."""

    def __init__(
        self,
        scene: SceneGraph | None = None,
        known_nouns: list[str] | None = None,
    ) -> None:
        self.scene = scene
        self.known_nouns = set(_DEFAULT_NOUNS)
        if known_nouns is not None:
            self.known_nouns.update(self._normalize_phrase(noun) for noun in known_nouns if noun.strip())
        if scene is not None:
            self.known_nouns.update(self._scene_nouns(scene))

    def parse(self, text: str) -> ParsedIntent:
        raw_text = text
        normalized = self._normalize_phrase(text)
        tokens = self._tokenize(normalized)
        if not tokens:
            return ParsedIntent(
                action="",
                target=None,
                destination=None,
                modifier=None,
                raw_text=raw_text,
                confidence=0.0,
            )

        modifier = self._extract_modifier(tokens)
        action = self._match_action(normalized, tokens)
        destination = self._extract_destination(tokens)
        target = self._extract_target(tokens, action, destination)

        if action:
            confidence = 1.0
        elif target is not None or destination is not None:
            confidence = 0.5
        else:
            confidence = 0.0

        return ParsedIntent(
            action=action,
            target=target,
            destination=destination,
            modifier=modifier,
            raw_text=raw_text,
            confidence=confidence,
        )

    def parse_batch(self, texts: list[str]) -> list[ParsedIntent]:
        return [self.parse(text) for text in texts]

    def _match_action(self, normalized: str, tokens: list[str]) -> str:
        for action, patterns in _ACTION_PATTERNS:
            for pattern in patterns:
                if " " in pattern:
                    if pattern in normalized:
                        return action
                    continue
                if pattern in tokens:
                    if pattern == "move" and "to" in tokens[tokens.index("move") + 1 :]:
                        return "go"
                    return action
        return ""

    def _extract_modifier(self, tokens: list[str]) -> str | None:
        for token in tokens:
            modifier = _MODIFIER_ALIASES.get(token)
            if modifier is not None:
                return modifier
        return None

    def _extract_destination(self, tokens: list[str]) -> str | None:
        for index, token in enumerate(tokens):
            if token not in _DESTINATION_PREPOSITIONS:
                continue
            phrase = self._match_longest_noun(tokens[index + 1 :])
            if phrase is not None:
                return phrase
        return None

    def _extract_target(
        self,
        tokens: list[str],
        action: str,
        destination: str | None,
    ) -> str | None:
        destination_tokens = set(destination.split()) if destination is not None else set()
        if action == "go":
            for index, token in enumerate(tokens):
                if token in {"to", "toward", "towards"}:
                    phrase = self._match_longest_noun(tokens[index + 1 :])
                    if phrase is not None:
                        return phrase

        search_limit = len(tokens)
        for index, token in enumerate(tokens):
            if token in _DESTINATION_PREPOSITIONS:
                search_limit = index
                break

        phrase = self._match_longest_noun(
            tokens[:search_limit],
            destination_tokens=destination_tokens,
            excluded_tokens=_ACTION_TOKENS,
        )
        if phrase is not None:
            return phrase

        if action == "handover":
            segment = tokens[:search_limit]
        else:
            segment = tokens[:search_limit]

        if action:
            fallback = self._fallback_phrase(segment, destination_tokens=destination_tokens)
            if fallback is not None:
                return fallback

        return None

    def _match_longest_noun(
        self,
        tokens: list[str],
        *,
        destination_tokens: set[str] | None = None,
        excluded_tokens: set[str] | None = None,
    ) -> str | None:
        excluded_tokens = set() if excluded_tokens is None else excluded_tokens
        cleaned = [
            token
            for token in tokens
            if token not in _STOPWORDS and token not in _MODIFIER_ALIASES and token not in excluded_tokens
        ]
        if not cleaned:
            return None

        destination_tokens = set() if destination_tokens is None else destination_tokens
        max_phrase_len = min(4, len(cleaned))
        for phrase_len in range(max_phrase_len, 0, -1):
            for start in range(0, len(cleaned) - phrase_len + 1):
                phrase_tokens = cleaned[start : start + phrase_len]
                if destination_tokens and set(phrase_tokens) == destination_tokens:
                    continue
                phrase = " ".join(phrase_tokens)
                if phrase in self.known_nouns:
                    return phrase

        for token in cleaned:
            if token in destination_tokens:
                continue
            if token in self.known_nouns:
                return token
        return None

    def _fallback_phrase(
        self,
        tokens: list[str],
        *,
        destination_tokens: set[str],
    ) -> str | None:
        cleaned = [
            token
            for token in tokens
            if token not in _STOPWORDS and token not in _MODIFIER_ALIASES and token not in _ACTION_TOKENS
        ]
        if not cleaned:
            return None
        if destination_tokens and set(cleaned) == destination_tokens:
            return None
        return " ".join(cleaned)

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        return " ".join(_TOKEN_RE.findall(text.lower()))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    @staticmethod
    def _scene_nouns(scene: SceneGraph) -> set[str]:
        nouns: set[str] = set()
        for node in scene.nodes.values():
            normalized_id = InstructionParser._normalize_phrase(node.id)
            normalized_label = InstructionParser._normalize_phrase(node.label)
            if normalized_id:
                nouns.add(normalized_id)
            if normalized_label:
                nouns.add(normalized_label)
        return nouns


__all__ = ["ParsedIntent", "InstructionParser"]
