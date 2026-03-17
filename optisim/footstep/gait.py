"""Gait scheduling and swing trajectory helpers."""

from __future__ import annotations

from enum import Enum

import numpy as np

from optisim.footstep.planner import FootstepPlan, FootstepSide


def _smoothstep5(s: float) -> float:
    return float(10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)


def _smoothstep5_derivative(s: float) -> float:
    return float(30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4)


class GaitPhase(Enum):
    """Current support configuration of the biped."""

    DOUBLE_SUPPORT = "double_support"
    LEFT_SWING = "left_swing"
    RIGHT_SWING = "right_swing"


class SwingTrajectory:
    """Smooth swing-foot trajectory between two footsteps."""

    def __init__(self, start: np.ndarray, end: np.ndarray, height: float = 0.08, duration: float = 0.4) -> None:
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        if self.start.shape != (3,) or self.end.shape != (3,):
            raise ValueError("start and end must have shape (3,)")
        self.height = float(height)
        self.duration = float(duration)

    def position_at(self, t: float) -> np.ndarray:
        """Return the swing foot position at time t."""

        local_t = float(np.clip(t, 0.0, self.duration))
        if self.duration <= 0.0:
            return self.end.copy()
        s = local_t / self.duration
        blend = _smoothstep5(s)
        horizontal = self.start[:2] + blend * (self.end[:2] - self.start[:2])
        z = self.start[2] + blend * (self.end[2] - self.start[2]) + self.height * np.sin(np.pi * s)
        return np.asarray([horizontal[0], horizontal[1], z], dtype=float)

    def velocity_at(self, t: float) -> np.ndarray:
        """Return the swing foot velocity at time t."""

        local_t = float(np.clip(t, 0.0, self.duration))
        if self.duration <= 0.0:
            return np.zeros(3, dtype=float)
        s = local_t / self.duration
        ds_dt = 1.0 / self.duration
        blend_rate = _smoothstep5_derivative(s) * ds_dt
        horizontal = blend_rate * (self.end[:2] - self.start[:2])
        vertical = blend_rate * (self.end[2] - self.start[2]) + self.height * np.pi * np.cos(np.pi * s) * ds_dt
        return np.asarray([horizontal[0], horizontal[1], vertical], dtype=float)


class GaitSchedule:
    """Time-indexed gait schedule derived from a footstep plan."""

    def __init__(self, plan: FootstepPlan) -> None:
        self.plan = plan
        self._timings: list[tuple[float, float, float]] = []
        elapsed = 0.0
        for step in self.plan.steps:
            ds_end = elapsed + step.contact_duration
            swing_end = ds_end + step.swing_duration
            self._timings.append((elapsed, ds_end, swing_end))
            elapsed = swing_end
        self._initial_feet = self._infer_initial_feet()

    def current_phase(self, t: float) -> GaitPhase:
        """Return the support phase at time t."""

        if not self.plan.steps:
            return GaitPhase.DOUBLE_SUPPORT
        local_t = float(np.clip(t, 0.0, self.plan.total_duration))
        for step, (_, ds_end, swing_end) in zip(self.plan.steps, self._timings):
            if local_t < ds_end:
                return GaitPhase.DOUBLE_SUPPORT
            if local_t <= swing_end:
                return GaitPhase.LEFT_SWING if step.side is FootstepSide.LEFT else GaitPhase.RIGHT_SWING
        return GaitPhase.DOUBLE_SUPPORT

    def active_swing(self, t: float) -> tuple[int, float] | None:
        """Return the active swing step index and local time within the swing."""

        local_t = float(np.clip(t, 0.0, self.plan.total_duration))
        for index, (_, ds_end, swing_end) in enumerate(self._timings):
            if ds_end <= local_t <= swing_end:
                return index, local_t - ds_end
        return None

    def cop_position(self, t: float) -> np.ndarray:
        """Return a simple center-of-pressure estimate in the support polygon."""

        feet = self.feet_positions(t)
        phase = self.current_phase(t)
        if phase is GaitPhase.LEFT_SWING:
            return feet["right"][:2].copy()
        if phase is GaitPhase.RIGHT_SWING:
            return feet["left"][:2].copy()
        return 0.5 * (feet["left"][:2] + feet["right"][:2])

    def feet_positions(self, t: float) -> dict[str, np.ndarray]:
        """Return the left and right foot positions at time t."""

        left = self._initial_feet["left"].copy()
        right = self._initial_feet["right"].copy()
        local_t = float(np.clip(t, 0.0, self.plan.total_duration))

        for step, (_, ds_end, swing_end) in zip(self.plan.steps, self._timings):
            target = np.asarray([step.position[0], step.position[1], 0.0], dtype=float)
            if local_t < ds_end:
                break
            if local_t <= swing_end:
                start = left if step.side is FootstepSide.LEFT else right
                swing = SwingTrajectory(start=start, end=target, duration=step.swing_duration)
                moved = swing.position_at(local_t - ds_end)
                if step.side is FootstepSide.LEFT:
                    left = moved
                else:
                    right = moved
                break
            if step.side is FootstepSide.LEFT:
                left = target
            else:
                right = target

        return {"left": left, "right": right}

    def _infer_initial_feet(self) -> dict[str, np.ndarray]:
        if not self.plan.steps:
            width = 0.18
            return {
                "left": np.asarray([0.0, 0.5 * width, 0.0], dtype=float),
                "right": np.asarray([0.0, -0.5 * width, 0.0], dtype=float),
            }

        first = self.plan.steps[0]
        heading = first.heading
        lateral = np.asarray([-np.sin(heading), np.cos(heading), 0.0], dtype=float)
        width = self._estimate_step_width(default=0.18)
        anchor = np.asarray([first.position[0], first.position[1], 0.0], dtype=float)
        if first.side is FootstepSide.LEFT:
            left = anchor
            right = anchor - width * lateral
        else:
            right = anchor
            left = anchor + width * lateral
        return {"left": left, "right": right}

    def _estimate_step_width(self, default: float) -> float:
        for index in range(1, len(self.plan.steps)):
            current = self.plan.steps[index]
            previous = self.plan.steps[index - 1]
            if current.side is previous.side:
                continue
            return float(np.linalg.norm(current.position - previous.position))
        return float(default)


__all__ = [
    "GaitPhase",
    "GaitSchedule",
    "SwingTrajectory",
]
