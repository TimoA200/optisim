"""Public hook API."""

from __future__ import annotations

from optisim.hooks.built_ins import CollisionRecorder, EventLogger, StepSampler
from optisim.hooks.core import HookRegistry, SimEvent, TaskHook

hook_registry: HookRegistry = HookRegistry()

__all__ = [
    "CollisionRecorder",
    "EventLogger",
    "HookRegistry",
    "SimEvent",
    "StepSampler",
    "TaskHook",
    "hook_registry",
]
