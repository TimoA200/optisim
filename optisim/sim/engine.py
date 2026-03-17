"""Step-based simulator and task execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from optisim.core.action_primitives import ActionPrimitive, ActionType
from optisim.core.task_definition import TaskDefinition
from optisim.core.task_validator import TaskValidator, ValidationReport
from optisim.dynamics import ConstraintSet, DynamicsReport, DynamicsValidator
from optisim.math3d import Pose, Quaternion, normalize, vec3
from optisim.robot import IKOptions, JointController, RobotModel, build_humanoid_model, solve_inverse_kinematics
from optisim.sim.collision import Collision, object_surface_collision
from optisim.sim.recording import SimulationRecording
from optisim.sim.world import WorldState


@dataclass(slots=True)
class SimulationRecord:
    """Summary of a completed simulation run."""

    steps: int
    duration_s: float
    executed_actions: list[str] = field(default_factory=list)
    collisions: list[Collision] = field(default_factory=list)
    recording: SimulationRecording | None = None


@dataclass
class ExecutionEngine:
    """Deterministic task execution engine coupling robot and world state."""

    robot: RobotModel = field(default_factory=build_humanoid_model)
    world: WorldState = field(default_factory=WorldState.with_defaults)
    dt: float = 0.05
    holder_prefix: str | None = None

    def __post_init__(self) -> None:
        """Create helper subsystems after engine construction."""

        self.controller = JointController(self.robot)
        self.validator = TaskValidator()
        self.dynamics_validator = DynamicsValidator(self.validator)

    def validate(self, task: TaskDefinition) -> ValidationReport:
        """Validate a task against the engine's current robot and world."""

        return self.validator.validate(task=task, world=self.world, robot=self.robot)

    def validate_dynamics(self, task: TaskDefinition, constraint_set: ConstraintSet | None = None) -> DynamicsReport:
        """Run lightweight dynamics validation against the engine's current state."""

        return self.dynamics_validator.validate_task(
            task=task,
            world=self.world,
            robot=self.robot,
            constraint_set=constraint_set,
        )

    def run(self, task: TaskDefinition, visualize: "Visualizer | None" = None) -> SimulationRecord:
        """Execute a validated task and return a simulation record."""

        report = self.validate(task)
        if not report.is_valid:
            details = "; ".join(issue.message for issue in report.errors)
            raise ValueError(f"task validation failed: {details}")

        executed_actions: list[str] = []
        collisions: list[Collision] = []
        steps = 0
        recording = SimulationRecording.from_robot(
            self.robot,
            task_name=task.name,
            dt=self.dt,
            metadata={"world_time_start_s": float(self.world.time_s)},
        )

        if visualize is not None:
            visualize.start_task(task, self.world, self.robot)
        self._emit_frame(visualize=visualize, recording=recording, active_action=None, collisions=[])

        for index, action in enumerate(task.actions, start=1):
            action_label = f"{action.action_type.value} {action.target}"
            executed_actions.append(action.action_type.value)
            if visualize is not None:
                visualize.start_action(action, index=index, total_actions=len(task.actions))
            frame_count_before = recording.frame_count()
            steps += self._execute_action(action, visualize, recording, action_label)
            current_collisions = self._check_collisions()
            collisions.extend(current_collisions)
            if visualize is not None:
                visualize.update_collisions(current_collisions)
            if recording.frame_count() == frame_count_before:
                self._emit_frame(
                    visualize=visualize,
                    recording=recording,
                    active_action=action_label,
                    collisions=current_collisions,
                )

        if visualize is not None:
            visualize.finish(task, self.world, self.robot, collisions)

        return SimulationRecord(
            steps=steps,
            duration_s=self.world.time_s,
            executed_actions=executed_actions,
            collisions=collisions,
            recording=recording,
        )

    def step(
        self,
        visualize: "Visualizer | None" = None,
        *,
        recording: SimulationRecording | None = None,
        active_action: str | None = None,
    ) -> list[Collision]:
        """Advance simulated time by one fixed step and refresh visualization."""

        self.world.time_s += self.dt
        collisions = self._check_collisions()
        self._emit_frame(
            visualize=visualize,
            recording=recording,
            active_action=active_action,
            collisions=collisions,
        )
        return collisions

    def _execute_action(
        self,
        action: ActionPrimitive,
        visualize: "Visualizer | None",
        recording: SimulationRecording | None,
        active_action: str | None,
    ) -> int:
        if action.action_type is ActionType.REACH:
            return self._reach(action, visualize, recording, active_action)
        if action.action_type is ActionType.GRASP:
            self.world.objects[action.target].held_by = self._held_by_name(action.end_effector)
            return 1
        if action.action_type is ActionType.MOVE:
            return self._move_object(action, visualize, recording, active_action)
        if action.action_type is ActionType.PLACE:
            return self._place(action, visualize, recording, active_action)
        if action.action_type in {ActionType.PUSH, ActionType.PULL}:
            return self._translate_object(action, action.action_type is ActionType.PUSH, visualize, recording, active_action)
        if action.action_type is ActionType.ROTATE:
            return self._rotate_object(action, visualize, recording, active_action)
        raise NotImplementedError(action.action_type)

    def _reach(
        self,
        action: ActionPrimitive,
        visualize: "Visualizer | None",
        recording: SimulationRecording | None,
        active_action: str | None,
    ) -> int:
        target_pose = action.pose or self.world.objects[action.target].pose
        ik_result = solve_inverse_kinematics(
            self.robot,
            action.end_effector,
            target_pose,
            options=IKOptions(
                max_iterations=120,
                convergence_threshold=2e-3,
                damping=0.12,
                position_only=action.pose is None,
            ),
        )
        targets = ik_result.joint_positions
        steps = 0
        while any(abs(self.robot.joint_positions[name] - value) > 1e-3 for name, value in targets.items()):
            self.controller.step_towards(targets, self.dt)
            self.step(visualize, recording=recording, active_action=active_action)
            steps += 1
            if steps > 200:
                break
        if not self._ik_result_is_actionable(ik_result, position_only=action.pose is None):
            reason = ik_result.failure_reason or "unknown reason"
            raise ValueError(
                f"ik failed for action targeting '{action.target}' with end effector "
                f"'{action.end_effector}': {reason}"
            )
        return max(steps, 1)

    @staticmethod
    def _ik_result_is_actionable(ik_result: "IKResult", *, position_only: bool) -> bool:
        """Accept numerically imperfect IK results when they are still usable for execution."""

        if ik_result.success:
            return True
        if position_only:
            return ik_result.position_error <= 0.05
        return ik_result.position_error <= 0.01 and ik_result.orientation_error <= 0.05

    def _move_object(
        self,
        action: ActionPrimitive,
        visualize: "Visualizer | None",
        recording: SimulationRecording | None,
        active_action: str | None,
    ) -> int:
        obj = self.world.objects[action.target]
        if obj.held_by is None:
            raise ValueError(f"cannot move '{action.target}' because it is not grasped")
        if obj.held_by != self._held_by_name(action.end_effector):
            raise ValueError(
                f"cannot move '{action.target}' because it is held by '{obj.held_by}',"
                f" not '{self._held_by_name(action.end_effector)}'"
            )
        destination = vec3(action.destination or obj.pose.position)
        start = obj.pose.position.copy()
        distance = float(np.linalg.norm(destination - start))
        steps = max(int(distance / max(action.speed * self.dt, 1e-6)), 1)
        for step_index in range(1, steps + 1):
            alpha = step_index / steps
            obj.pose = Pose(position=start * (1 - alpha) + destination * alpha, orientation=obj.pose.orientation)
            self.step(visualize, recording=recording, active_action=active_action)
        return steps

    def _place(
        self,
        action: ActionPrimitive,
        visualize: "Visualizer | None",
        recording: SimulationRecording | None,
        active_action: str | None,
    ) -> int:
        obj = self.world.objects[action.target]
        surface = self.world.surfaces[action.support or ""]
        z = surface.pose.position[2] + surface.size[2] / 2.0 + obj.size[2] / 2.0
        obj.pose = Pose(position=vec3([obj.pose.position[0], obj.pose.position[1], z]), orientation=obj.pose.orientation)
        obj.held_by = None
        self.step(visualize, recording=recording, active_action=active_action)
        return 1

    def _translate_object(
        self,
        action: ActionPrimitive,
        push: bool,
        visualize: "Visualizer | None",
        recording: SimulationRecording | None,
        active_action: str | None,
    ) -> int:
        direction = normalize(vec3(action.axis or [1.0, 0.0, 0.0]))
        if not push:
            direction *= -1.0
        magnitude = max((action.force_newtons or 5.0) / 50.0, 0.05)
        destination = self.world.objects[action.target].pose.position + direction * magnitude
        translated = ActionPrimitive.move(target=action.target, destination=destination.tolist(), end_effector=action.end_effector)
        self.world.objects[action.target].held_by = action.end_effector
        steps = self._move_object(translated, visualize, recording, active_action)
        self.world.objects[action.target].held_by = None
        return steps

    def _rotate_object(
        self,
        action: ActionPrimitive,
        visualize: "Visualizer | None",
        recording: SimulationRecording | None,
        active_action: str | None,
    ) -> int:
        obj = self.world.objects[action.target]
        axis = normalize(vec3(action.axis or [0.0, 0.0, 1.0]))
        half_angle = (action.angle_rad or 0.0) / 2.0
        q = Quaternion(np.cos(half_angle), *(axis * np.sin(half_angle)))
        obj.pose = Pose(position=obj.pose.position, orientation=obj.pose.orientation * q)
        self.step(visualize, recording=recording, active_action=active_action)
        return 1

    def _emit_frame(
        self,
        *,
        visualize: "Visualizer | None",
        recording: SimulationRecording | None,
        active_action: str | None,
        collisions: list[Collision],
    ) -> None:
        if visualize is not None:
            visualize.update_collisions(collisions)
            visualize.render(self.world, self.robot)
        if recording is not None:
            recording.capture_frame(
                self.robot,
                self.world,
                active_action=active_action,
                collisions=collisions,
            )

    def _check_collisions(self) -> list[Collision]:
        collisions: list[Collision] = []
        for obj in self.world.objects.values():
            for surface in self.world.surfaces.values():
                collision = object_surface_collision(obj, surface)
                if collision is not None and obj.held_by is None:
                    collisions.append(collision)
        return collisions

    def _held_by_name(self, end_effector: str) -> str:
        """Resolve the runtime holder identifier for an end effector."""

        canonical_effector = self.robot.end_effectors.get(end_effector, end_effector)
        if self.holder_prefix:
            return f"{self.holder_prefix}:{canonical_effector}"
        return canonical_effector


class Visualizer:
    """Protocol-like visualization interface used by the execution engine."""

    def start_task(self, task: TaskDefinition, world: WorldState, robot: RobotModel) -> None:
        """Initialize visualization state for a new task."""

        return None

    def start_action(self, action: ActionPrimitive, *, index: int, total_actions: int) -> None:
        """Notify the visualizer that a new action is starting."""

        return None

    def update_collisions(self, collisions: list[Collision]) -> None:
        """Provide the visualizer with the latest collision reports."""

        return None

    def render(self, world: WorldState, robot: RobotModel) -> None:
        """Render the current world and robot state."""

        raise NotImplementedError

    def finish(
        self,
        task: TaskDefinition,
        world: WorldState,
        robot: RobotModel,
        collisions: list[Collision],
    ) -> None:
        """Finalize visualization after task completion."""

        return None
