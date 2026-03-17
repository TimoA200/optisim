"""Built-in hook implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import time_ns
from typing import Callable

from optisim.hooks.core import EVENT_KINDS, HookRegistry, SimEvent


@dataclass
class EventLogger:
    """Hook that writes all events to a JSONL file."""

    path: str | Path
    events: list[str] | None = None
    _attached: dict[int, list[tuple[str, Callable[[SimEvent], None]]]] = field(default_factory=dict, init=False, repr=False)

    def attach(self, registry: HookRegistry) -> None:
        """Attach this logger to the supplied registry."""

        registry_key = id(registry)
        if registry_key in self._attached:
            return
        destination = Path(self.path)
        callbacks: list[tuple[str, Callable[[SimEvent], None]]] = []
        for event_kind in self.events or list(EVENT_KINDS):
            callback = self._make_callback(destination)
            registry.register(event_kind, callback)
            callbacks.append((event_kind, callback))
        self._attached[registry_key] = callbacks

    def detach(self, registry: HookRegistry) -> None:
        """Detach this logger from the supplied registry."""

        for event_kind, callback in self._attached.pop(id(registry), []):
            registry.unregister(event_kind, callback)

    @staticmethod
    def _make_callback(path: Path) -> Callable[[SimEvent], None]:
        def callback(event: SimEvent) -> None:
            payload = {
                "ts": time_ns() // 1_000_000,
                "kind": event.kind,
                "task": event.task_name,
                "step": event.step,
                "data": event.data,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

        return callback


@dataclass
class CollisionRecorder:
    """Hook that accumulates all collision events into a list for inspection."""

    _events: list[SimEvent] = field(default_factory=list, init=False, repr=False)

    def attach(self, registry: HookRegistry) -> None:
        """Attach this recorder to the supplied registry."""

        registry.register("collision", self._record)

    def detach(self, registry: HookRegistry) -> None:
        """Detach this recorder from the supplied registry."""

        registry.unregister("collision", self._record)

    @property
    def events(self) -> list[SimEvent]:
        """Return a snapshot of recorded collision events."""

        return list(self._events)

    def clear(self) -> None:
        """Discard all recorded collision events."""

        self._events.clear()

    def _record(self, event: SimEvent) -> None:
        self._events.append(event)


@dataclass
class StepSampler:
    """Hook that samples step events at a configurable interval."""

    interval: int = 10
    _samples: list[SimEvent] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("interval must be positive")

    def attach(self, registry: HookRegistry) -> None:
        """Attach this sampler to the supplied registry."""

        registry.register("step", self._record)

    def detach(self, registry: HookRegistry) -> None:
        """Detach this sampler from the supplied registry."""

        registry.unregister("step", self._record)

    @property
    def samples(self) -> list[SimEvent]:
        """Return a snapshot of sampled step events."""

        return list(self._samples)

    def _record(self, event: SimEvent) -> None:
        if event.step % self.interval == 0:
            self._samples.append(event)


__all__ = ["CollisionRecorder", "EventLogger", "StepSampler"]
