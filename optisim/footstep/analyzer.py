"""Analysis helpers for footstep plans."""

from __future__ import annotations

import numpy as np

from optisim.footstep.planner import FootstepPlan


class FootstepAnalyzer:
    """Compute basic metrics over a footstep plan."""

    def step_lengths(self, plan: FootstepPlan) -> list[float]:
        lengths: list[float] = []
        by_side: dict[str, np.ndarray] = {}
        for step in plan.steps:
            previous = by_side.get(step.side.value)
            if previous is not None:
                lengths.append(float(np.linalg.norm(step.position - previous)))
            by_side[step.side.value] = step.position
        return lengths

    def step_widths(self, plan: FootstepPlan) -> list[float]:
        widths: list[float] = []
        for previous, current in zip(plan.steps, plan.steps[1:]):
            if previous.side is current.side:
                continue
            heading = 0.5 * (previous.heading + current.heading)
            lateral = np.asarray([-np.sin(heading), np.cos(heading)], dtype=float)
            widths.append(abs(float(np.dot(current.position - previous.position, lateral))))
        return widths

    def average_cadence(self, plan: FootstepPlan) -> float:
        if plan.total_duration <= 0.0:
            return 0.0
        return float(plan.n_steps / plan.total_duration)

    def path_length(self, plan: FootstepPlan) -> float:
        if plan.n_steps < 2:
            return 0.0
        return float(
            sum(np.linalg.norm(current.position - previous.position) for previous, current in zip(plan.steps, plan.steps[1:]))
        )

    def heading_changes(self, plan: FootstepPlan) -> list[float]:
        changes: list[float] = []
        previous_heading: float | None = None
        for step in plan.steps:
            if previous_heading is None:
                changes.append(0.0)
            else:
                delta = float(np.arctan2(np.sin(step.heading - previous_heading), np.cos(step.heading - previous_heading)))
                changes.append(delta)
            previous_heading = step.heading
        return changes


__all__ = ["FootstepAnalyzer"]
