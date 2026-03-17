"""Dynamics validation integrated with task execution and semantic validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from optisim.core import TaskDefinition, TaskValidator
from optisim.dynamics.constraints import (
    ConstraintSet,
    ConstraintViolation,
    JointTorqueLimit,
    check_joint_torques,
    check_payload,
    check_workspace_bounds,
)
from optisim.dynamics.energy import TaskEnergyProfile, joint_power, total_mechanical_energy
from optisim.dynamics.rigid_body import RigidBodyState, compute_inertia_box
from optisim.robot import RobotModel
from optisim.sim.world import WorldState


@dataclass(slots=True)
class DynamicsReport:
    """Result of a dynamics validation pass."""

    violations: list[ConstraintViolation] = field(default_factory=list)
    energy_profile: TaskEnergyProfile = field(
        default_factory=lambda: TaskEnergyProfile(
            total_energy=0.0,
            peak_power=0.0,
            energy_per_action={},
        )
    )
    feasible: bool = True
    warnings: list[str] = field(default_factory=list)


class DynamicsValidator:
    """Run lightweight dynamics and physical plausibility checks for a task."""

    def __init__(self, base_validator: TaskValidator | None = None) -> None:
        self.base_validator = base_validator or TaskValidator()

    def validate_task(
        self,
        task: TaskDefinition,
        robot: RobotModel,
        world: WorldState,
        constraint_set: ConstraintSet | None = None,
    ) -> DynamicsReport:
        """Validate task dynamics against robot, world, and constraint policies."""

        constraints = constraint_set or ConstraintSet()
        semantic = self.base_validator.validate(task=task, world=world, robot=robot)
        warnings = [issue.message for issue in semantic.warnings]
        violations: list[ConstraintViolation] = [
            ConstraintViolation(
                constraint_type="task_validation",
                message=issue.message,
                severity=issue.severity,
            )
            for issue in semantic.errors
        ]
        if semantic.errors:
            return DynamicsReport(
                violations=violations,
                energy_profile=TaskEnergyProfile(
                    total_energy=0.0,
                    peak_power=0.0,
                    energy_per_action={},
                ),
                feasible=False,
                warnings=warnings,
            )

        robot_copy = deepcopy(robot)
        world_copy = deepcopy(world)
        from optisim.sim.engine import ExecutionEngine

        engine = ExecutionEngine(robot=robot_copy, world=world_copy)
        record = engine.run(task)
        recording = record.recording
        if recording is None or not recording.frames:
            return DynamicsReport(
                violations=[],
                energy_profile=TaskEnergyProfile(
                    total_energy=0.0,
                    peak_power=0.0,
                    energy_per_action={},
                ),
                feasible=True,
                warnings=warnings,
            )

        energy_profile = self._build_energy_profile(
            robot_copy,
            recording.frames,
            recording.dt,
            constraints,
        )
        violations.extend(self._check_recording(robot_copy, recording.frames, constraints))
        return DynamicsReport(
            violations=violations,
            energy_profile=energy_profile,
            feasible=not any(item.severity == "error" for item in violations),
            warnings=warnings,
        )

    @staticmethod
    def default_torque_limits(robot: RobotModel) -> list[JointTorqueLimit]:
        """Create deterministic heuristic torque limits from the robot model."""

        return [
            JointTorqueLimit(joint_name=name, max_torque=max(10.0, joint.velocity_limit * 25.0))
            for name, joint in robot.joints.items()
        ]

    def _check_recording(
        self,
        robot: RobotModel,
        frames: list,
        constraints: ConstraintSet,
    ) -> list[ConstraintViolation]:
        violations: list[ConstraintViolation] = []
        seen: set[tuple[str, str | None, float | None, float | None]] = set()
        torque_limits = constraints.joint_torque_limits

        for frame in frames:
            joint_positions = frame.joint_positions
            held_payloads = self._held_payloads(frame.objects)

            for held_by, payload_mass in held_payloads.items():
                for payload_constraint in constraints.payload_constraints:
                    violation = check_payload(robot, payload_mass, held_by, payload_constraint)
                    self._append_unique(violations, seen, violation)
                for violation in check_joint_torques(
                    robot,
                    joint_positions,
                    payload_mass,
                    torque_limits=torque_limits,
                    end_effector=held_by,
                ):
                    self._append_unique(violations, seen, violation)

            for bounds in constraints.workspace_constraints:
                for item_name, obj in frame.objects.items():
                    position = obj["pose"]["position"]
                    violation = check_workspace_bounds(position, bounds)
                    if violation is not None:
                        violation.message = (
                            f"object '{item_name}' {violation.message}"
                        )
                    self._append_unique(violations, seen, violation)
                for effector_name, link_name in robot.end_effectors.items():
                    if link_name not in frame.link_positions:
                        continue
                    violation = check_workspace_bounds(frame.link_positions[link_name], bounds)
                    if violation is not None:
                        violation.message = (
                            f"end effector '{effector_name}' {violation.message}"
                        )
                    self._append_unique(violations, seen, violation)

        return violations

    def _build_energy_profile(
        self,
        robot: RobotModel,
        frames: list,
        dt: float,
        constraints: ConstraintSet,
    ) -> TaskEnergyProfile:
        total_energy = 0.0
        peak_power = 0.0
        energy_per_action: dict[str, float] = {}
        torque_limits = {
            item.joint_name: item.max_torque for item in constraints.joint_torque_limits
        }

        for previous, current in zip(frames, frames[1:]):
            active_action = current.active_action or previous.active_action or "idle"
            held_payloads = self._held_payloads(current.objects)
            action_energy = 0.0

            for object_name, current_object in current.objects.items():
                previous_object = previous.objects.get(object_name, current_object)
                state = self._object_state(previous_object, current_object, dt)
                action_energy += total_mechanical_energy(state, height=state.position[2])

            total_energy += action_energy * dt
            energy_per_action[active_action] = (
                energy_per_action.get(active_action, 0.0) + action_energy * dt
            )

            if held_payloads:
                dominant_payload = max(held_payloads.values())
                for joint_name, current_position in current.joint_positions.items():
                    previous_position = previous.joint_positions.get(joint_name, current_position)
                    angular_velocity = (current_position - previous_position) / max(dt, 1e-9)
                    torque = torque_limits.get(
                        joint_name,
                        max(10.0, robot.joints[joint_name].velocity_limit * 10.0)
                        * (dominant_payload / 5.0),
                    )
                    peak_power = max(peak_power, abs(joint_power(torque, angular_velocity)))

        return TaskEnergyProfile(
            total_energy=total_energy,
            peak_power=peak_power,
            energy_per_action=energy_per_action,
        )

    @staticmethod
    def _held_payloads(objects: dict[str, dict[str, object]]) -> dict[str, float]:
        held_payloads: dict[str, float] = {}
        for obj in objects.values():
            held_by = obj.get("held_by")
            if held_by is None:
                continue
            held_payloads[str(held_by)] = held_payloads.get(str(held_by), 0.0) + float(
                obj.get("mass_kg", 0.0)
            )
        return held_payloads

    @staticmethod
    def _object_state(
        previous_object: dict[str, object],
        current_object: dict[str, object],
        dt: float,
    ) -> RigidBodyState:
        previous_position = np.asarray(previous_object["pose"]["position"], dtype=np.float64)
        current_position = np.asarray(current_object["pose"]["position"], dtype=np.float64)
        velocity = (current_position - previous_position) / max(dt, 1e-9)
        mass = float(current_object.get("mass_kg", 0.0))
        size = tuple(
            float(value) for value in current_object.get("size", [0.0, 0.0, 0.0])
        )
        return RigidBodyState(
            mass=mass,
            inertia_tensor=compute_inertia_box(mass, size),
            position=current_position,
            velocity=velocity,
            angular_velocity=np.zeros(3, dtype=np.float64),
            forces=np.zeros(3, dtype=np.float64),
            torques=np.zeros(3, dtype=np.float64),
        )

    @staticmethod
    def _append_unique(
        violations: list[ConstraintViolation],
        seen: set[tuple[str, str | None, float | None, float | None]],
        violation: ConstraintViolation | None,
    ) -> None:
        if violation is None:
            return
        key = (
            violation.constraint_type,
            violation.joint_name,
            None if violation.value is None else round(violation.value, 6),
            None if violation.limit is None else round(violation.limit, 6),
        )
        if key in seen:
            return
        seen.add(key)
        violations.append(violation)
