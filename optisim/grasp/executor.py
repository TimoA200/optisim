"""Execution helpers bridging planned grasps and the simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optisim.core import ActionPrimitive, TaskDefinition
from optisim.grasp.contact import ContactPoint, surface_contacts
from optisim.grasp.planner import GraspPose
from optisim.grasp.stability import force_closure, min_resisted_wrench, slip_margin
from optisim.robot import RobotModel
from optisim.sim import ExecutionEngine


@dataclass(slots=True)
class GraspResult:
    """Outcome of executing a grasp pose in the simulation."""

    success: bool
    contact_points: list[ContactPoint] = field(default_factory=list)
    stability_score: float = 0.0
    slip_detected: bool = False


@dataclass
class GraspExecutor:
    """Run a planned grasp through the existing execution pipeline."""

    reach_effector: str = "right_palm"
    grasp_effector: str = "right_gripper"

    def execute_grasp(self, engine: ExecutionEngine, robot: RobotModel, grasp_pose: GraspPose, obj) -> GraspResult:
        """Execute a reach-then-grasp sequence and evaluate the resulting hold."""

        del robot
        task = TaskDefinition(
            name=f"grasp_{getattr(obj, 'name', 'object')}",
            actions=[
                ActionPrimitive.reach(target=obj.name, end_effector=self.reach_effector, pose=grasp_pose.pose),
                ActionPrimitive.grasp(target=obj.name, gripper=self.grasp_effector),
            ],
            world={},
            robot={},
            metadata={"mode": "grasp_execution"},
        )
        try:
            engine.run(task)
        except ValueError:
            return GraspResult(success=False)

        contact_points = grasp_pose.contact_points or surface_contacts(
            obj,
            {
                "position": grasp_pose.position,
                "orientation": grasp_pose.orientation,
                "aperture": grasp_pose.aperture,
                "gripper_type": grasp_pose.gripper_type,
            },
        )
        stability_score = min_resisted_wrench(contact_points)
        gravity_load = 9.81 * obj.mass_kg / max(len(contact_points), 1)
        mean_friction = float(np.mean([contact.friction_coeff for contact in contact_points])) if contact_points else 0.0
        preload = gravity_load / max(mean_friction, 1e-6) * 1.1 if mean_friction > 0.0 else 0.0
        slip_detected = any(
            slip_margin(
                contact,
                np.asarray([-preload * contact.normal[0], -preload * contact.normal[1], -gravity_load], dtype=np.float64),
            )
            < 0.0
            for contact in contact_points
        )
        success = (
            bool(contact_points)
            and engine.world.objects[obj.name].held_by is not None
            and force_closure(contact_points)
            and not slip_detected
        )
        return GraspResult(
            success=success,
            contact_points=contact_points,
            stability_score=stability_score,
            slip_detected=slip_detected,
        )
