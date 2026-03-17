"""Core primitives for simulation hooks."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

LOGGER = logging.getLogger(__name__)
EVENT_KINDS = ("task_start", "action_start", "action_end", "collision", "task_end", "step")


@dataclass(slots=True)
class SimEvent:
    """A single simulation event, passed to hook callbacks."""

    kind: str
    task_name: str
    step: int
    data: dict[str, Any] = field(default_factory=dict)


HookCallback = Callable[[SimEvent], None]


class TaskHook(Protocol):
    """Attachable hook interface for reusable event consumers."""

    def attach(self, registry: "HookRegistry") -> None:
        """Attach the hook to a registry."""

    def detach(self, registry: "HookRegistry") -> None:
        """Detach the hook from a registry."""


@dataclass
class HookRegistry:
    """Central registry for simulation event hooks."""

    _callbacks: dict[str, list[HookCallback]] = field(default_factory=lambda: defaultdict(list))

    def on(self, event_kind: str | list[str]) -> Callable[[HookCallback], HookCallback]:
        """Decorator to register a callback for one or more event kinds."""

        event_kinds = [event_kind] if isinstance(event_kind, str) else list(event_kind)

        def decorator(callback: HookCallback) -> HookCallback:
            for kind in event_kinds:
                self.register(kind, callback)
            return callback

        return decorator

    def register(self, event_kind: str, callback: HookCallback) -> None:
        """Register a callback for an event kind."""

        callbacks = self._callbacks[event_kind]
        if callback not in callbacks:
            callbacks.append(callback)

    def unregister(self, event_kind: str, callback: HookCallback) -> None:
        """Remove a previously registered callback."""

        callbacks = self._callbacks.get(event_kind)
        if callbacks is None:
            return
        try:
            callbacks.remove(callback)
        except ValueError:
            return
        if not callbacks:
            self._callbacks.pop(event_kind, None)

    def clear(self, event_kind: str | None = None) -> None:
        """Remove all callbacks, optionally filtered to one event kind."""

        if event_kind is None:
            self._callbacks.clear()
            return
        self._callbacks.pop(event_kind, None)

    def emit(self, event: SimEvent) -> None:
        """Fire all callbacks registered for event.kind. Errors are logged and suppressed."""

        for callback in tuple(self._callbacks.get(event.kind, ())):
            try:
                callback(event)
            except Exception:
                LOGGER.exception("hook callback failed for event '%s': %r", event.kind, callback)

    def __len__(self) -> int:
        """Total number of registered callbacks across all event kinds."""

        return sum(len(callbacks) for callbacks in self._callbacks.values())


__all__ = ["EVENT_KINDS", "HookCallback", "HookRegistry", "SimEvent", "TaskHook"]
