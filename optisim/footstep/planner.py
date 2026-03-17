"""Footstep planning primitives for simple bipedal walking patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


def _as_position_2d(value: np.ndarray | None, *, default: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=float)
    position = np.asarray(value, dtype=float)
    if position.shape != (2,):
        raise ValueError(f"position must have shape (2,), got {position.shape}")
    return position.copy()


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _forward_vector(heading: float) -> np.ndarray:
    return np.asarray([np.cos(heading), np.sin(heading)], dtype=float)


def _lateral_vector(heading: float) -> np.ndarray:
    return np.asarray([-np.sin(heading), np.cos(heading)], dtype=float)


class FootstepSide(Enum):
    """Side of the biped support foot."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(slots=True)
class Footstep:
    """One grounded footstep target."""

    position: np.ndarray
    heading: float
    side: FootstepSide
    contact_duration: float = 0.6
    swing_duration: float = 0.4

    def __post_init__(self) -> None:
        self.position = _as_position_2d(self.position)
        self.heading = float(self.heading)
        self.contact_duration = float(self.contact_duration)
        self.swing_duration = float(self.swing_duration)


@dataclass(slots=True)
class FootstepPlan:
    """Sequence of footsteps for a walking pattern."""

    steps: list[Footstep] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return float(sum(step.contact_duration + step.swing_duration for step in self.steps))

    @property
    def n_steps(self) -> int:
        return len(self.steps)


class FootstepPlanner:
    """Generate lightweight alternating footstep plans."""

    def __init__(self, step_length: float = 0.3, step_width: float = 0.18, max_step_angle: float = 0.4) -> None:
        self.step_length = float(step_length)
        self.step_width = float(step_width)
        self.max_step_angle = float(max_step_angle)

    def plan_straight(
        self,
        n_steps: int,
        direction: float = 0.0,
        start_pos: np.ndarray | None = None,
    ) -> FootstepPlan:
        """Plan alternating left-right steps in a fixed heading."""

        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        center = _as_position_2d(start_pos)
        heading = float(direction)
        steps: list[Footstep] = []
        for step_index in range(n_steps):
            center = center + self.step_length * _forward_vector(heading)
            side = FootstepSide.LEFT if step_index % 2 == 0 else FootstepSide.RIGHT
            steps.append(self._make_step(center, heading, side))
        return FootstepPlan(steps=steps)

    def plan_turn(
        self,
        n_steps: int,
        turn_rate: float = 0.2,
        start_pos: np.ndarray | None = None,
    ) -> FootstepPlan:
        """Plan a turning walk with gradually changing heading."""

        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        center = _as_position_2d(start_pos)
        heading = 0.0
        delta = float(np.clip(turn_rate, -self.max_step_angle, self.max_step_angle))
        steps: list[Footstep] = []
        for step_index in range(n_steps):
            heading = _wrap_angle(heading + delta)
            center = center + self.step_length * _forward_vector(heading)
            side = FootstepSide.LEFT if step_index % 2 == 0 else FootstepSide.RIGHT
            steps.append(self._make_step(center, heading, side))
        return FootstepPlan(steps=steps)

    def plan_sidestep(
        self,
        n_steps: int,
        side: str = "left",
        start_pos: np.ndarray | None = None,
    ) -> FootstepPlan:
        """Plan a lateral shuffle while keeping the heading fixed."""

        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        direction = str(side).strip().lower()
        if direction not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        center = _as_position_2d(start_pos)
        lateral_sign = 1.0 if direction == "left" else -1.0
        steps: list[Footstep] = []
        for step_index in range(n_steps):
            center = center + lateral_sign * (self.step_width * 0.5) * _lateral_vector(0.0)
            foot_side = FootstepSide.LEFT if step_index % 2 == 0 else FootstepSide.RIGHT
            steps.append(self._make_step(center, 0.0, foot_side))
        return FootstepPlan(steps=steps)

    def plan_to_target(
        self,
        target: np.ndarray,
        start_pos: np.ndarray | None = None,
        max_steps: int = 20,
    ) -> FootstepPlan:
        """Greedily plan footsteps from the start position toward a target."""

        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        target_position = _as_position_2d(np.asarray(target, dtype=float))
        center = _as_position_2d(start_pos)
        steps: list[Footstep] = []

        for step_index in range(max_steps):
            delta = target_position - center
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-9:
                break
            heading = float(np.arctan2(delta[1], delta[0]))
            center = center + min(self.step_length, distance) * _forward_vector(heading)
            side = FootstepSide.LEFT if step_index % 2 == 0 else FootstepSide.RIGHT
            steps.append(self._make_step(center, heading, side))
        return FootstepPlan(steps=steps)

    def _make_step(self, center: np.ndarray, heading: float, side: FootstepSide) -> Footstep:
        offset = _lateral_vector(heading) * (self.step_width * 0.5)
        position = center + offset if side is FootstepSide.LEFT else center - offset
        return Footstep(position=position, heading=heading, side=side)


__all__ = [
    "Footstep",
    "FootstepPlan",
    "FootstepPlanner",
    "FootstepSide",
]
