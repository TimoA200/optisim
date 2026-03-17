"""Base interfaces for semantic motion primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from optisim.scene import SceneGraph


class PrimitiveStatus(Enum):
    """Execution state for a motion primitive."""

    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(slots=True)
class PrimitiveResult:
    """Result bundle returned by motion primitives."""

    status: PrimitiveStatus
    message: str = ""
    joint_trajectory: list[np.ndarray] | None = None
    duration_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class MotionPrimitive(ABC):
    """Abstract base class for semantic-to-motion skills."""

    name: str = ""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = {} if params is None else dict(params)

    @abstractmethod
    def check_preconditions(self, scene: SceneGraph, robot_id: str) -> tuple[bool, str]:
        """Return whether the primitive can run in the current scene."""

    @abstractmethod
    def execute(self, scene: SceneGraph, robot_id: str, robot_joints: np.ndarray) -> PrimitiveResult:
        """Generate a motion for the primitive."""

    @abstractmethod
    def get_effects(self, scene: SceneGraph, robot_id: str) -> list[dict[str, Any]]:
        """Return symbolic effects to apply after a successful execution."""
