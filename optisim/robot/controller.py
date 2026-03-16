"""Joint-level controller utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optisim.robot.model import RobotModel


@dataclass
class JointController:
    """Velocity-limited joint controller for deterministic stepping."""

    robot: RobotModel

    def step_towards(self, targets: dict[str, float], dt: float) -> dict[str, float]:
        updated = dict(self.robot.joint_positions)
        for joint_name, target in targets.items():
            spec = self.robot.joints[joint_name]
            current = updated[joint_name]
            delta = target - current
            max_delta = spec.velocity_limit * dt
            updated[joint_name] = spec.clamp(current + float(np.clip(delta, -max_delta, max_delta)))
        self.robot.set_joint_positions(updated)
        return updated
