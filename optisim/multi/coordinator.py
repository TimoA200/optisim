"""Round-robin multi-robot task coordination."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from optisim.core import ActionPrimitive, ActionType
from optisim.math3d import Pose, Quaternion, normalize, vec3
from optisim.multi.assignment import TaskAssignment
from optisim.multi.collision import InterRobotCollision, inter_robot_collisions
from optisim.multi.fleet import RobotFleet
from optisim.robot import IKOptions, solve_inverse_kinematics
from optisim.sim import ExecutionEngine, SimulationRecording


@dataclass(slots=True)
class RobotTrace:
    """Per-robot execution trace for a coordinated run."""

    robot_name: str
    executed_actions: list[str] = field(default_factory=list)
    completed_action_count: int = 0
    recording: SimulationRecording | None = None


@dataclass(slots=True)
class MultiRobotRecord:
    """Result of a multi-robot coordinated execution."""

    duration_s: float
    steps: int
    traces: dict[str, RobotTrace]
    collisions: list[InterRobotCollision] = field(default_factory=list)
    completion_order: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class _RobotExecutionState:
    assignment: TaskAssignment
    engine: ExecutionEngine
    trace: RobotTrace
    current_action_index: int = 0
    current_runner: Iterator[None] | None = None

    @property
    def robot_name(self) -> str:
        return self.assignment.robot_name

    @property
    def is_complete(self) -> bool:
        return self.current_action_index >= len(self.assignment.task.actions) and self.current_runner is None


class TaskCoordinator:
    """Coordinate multiple robots sharing one world with dependency-aware scheduling."""

    def __init__(self, fleet: RobotFleet, assignments: list[TaskAssignment]) -> None:
        self.fleet = fleet
        self.assignments = assignments

    def execute(self) -> MultiRobotRecord:
        """Execute assignments in round-robin order and return per-robot traces."""

        from optisim.multi.assignment import AssignmentValidator

        report = AssignmentValidator().validate(self.fleet, self.assignments)
        if not report.is_valid:
            details = "; ".join(issue.message for issue in report.errors)
            raise ValueError(f"assignment validation failed: {details}")

        states = {
            assignment.robot_name: _RobotExecutionState(
                assignment=assignment,
                engine=ExecutionEngine(
                    robot=self.fleet.get_robot(assignment.robot_name),
                    world=self.fleet.world,
                    holder_prefix=assignment.robot_name,
                ),
                trace=RobotTrace(
                    robot_name=assignment.robot_name,
                    recording=SimulationRecording.from_robot(
                        self.fleet.get_robot(assignment.robot_name),
                        task_name=assignment.task.name,
                        dt=0.05,
                        metadata={"mode": "multi_robot"},
                    ),
                ),
            )
            for assignment in self.assignments
        }

        for state in states.values():
            if state.trace.recording is not None:
                state.engine._emit_frame(
                    visualize=None,
                    recording=state.trace.recording,
                    active_action=None,
                    collisions=[],
                )

        all_collisions: list[InterRobotCollision] = []
        completion_order: list[tuple[str, int]] = []
        total_steps = 0

        while not all(state.is_complete for state in states.values()):
            progressed = False
            for state in states.values():
                if state.is_complete or not self._dependencies_satisfied(state.assignment, states):
                    continue
                if state.current_runner is None:
                    if state.current_action_index >= len(state.assignment.task.actions):
                        continue
                    action = state.assignment.task.actions[state.current_action_index]
                    state.trace.executed_actions.append(action.action_type.value)
                    state.current_runner = self._action_runner(state.engine, action, state.trace.recording)
                try:
                    assert state.current_runner is not None
                    next(state.current_runner)
                    progressed = True
                    total_steps += 1
                    all_collisions.extend(inter_robot_collisions(self.fleet.robots))
                except StopIteration:
                    state.current_runner = None
                    completion_order.append((state.robot_name, state.current_action_index))
                    state.current_action_index += 1
                    state.trace.completed_action_count = state.current_action_index
                    progressed = True
                    all_collisions.extend(inter_robot_collisions(self.fleet.robots))
            if not progressed:
                raise ValueError("multi-robot execution deadlocked on unresolved dependencies")

        return MultiRobotRecord(
            duration_s=float(self.fleet.world.time_s),
            steps=total_steps,
            traces={name: state.trace for name, state in states.items()},
            collisions=all_collisions,
            completion_order=completion_order,
        )

    @staticmethod
    def _dependencies_satisfied(
        assignment: TaskAssignment,
        states: dict[str, _RobotExecutionState],
    ) -> bool:
        return all(
            states[dependency.robot_name].trace.completed_action_count > dependency.action_index
            for dependency in assignment.dependencies
        )

    def _action_runner(
        self,
        engine: ExecutionEngine,
        action: ActionPrimitive,
        recording: SimulationRecording | None,
    ) -> Iterator[None]:
        active_action = f"{action.action_type.value} {action.target}"
        if action.action_type is ActionType.REACH:
            yield from self._reach(engine, action, recording, active_action)
            return
        if action.action_type is ActionType.GRASP:
            engine.world.objects[action.target].held_by = engine._held_by_name(action.end_effector)
            engine._emit_frame(visualize=None, recording=recording, active_action=active_action, collisions=[])
            yield
            return
        if action.action_type is ActionType.MOVE:
            yield from self._move_object(engine, action, recording, active_action)
            return
        if action.action_type is ActionType.PLACE:
            obj = engine.world.objects[action.target]
            surface = engine.world.surfaces[action.support or ""]
            z = surface.pose.position[2] + surface.size[2] / 2.0 + obj.size[2] / 2.0
            obj.pose = Pose(
                position=vec3([obj.pose.position[0], obj.pose.position[1], z]),
                orientation=obj.pose.orientation,
            )
            obj.held_by = None
            engine._emit_frame(visualize=None, recording=recording, active_action=active_action, collisions=[])
            yield
            return
        if action.action_type in {ActionType.PUSH, ActionType.PULL}:
            yield from self._translate_object(
                engine,
                action,
                action.action_type is ActionType.PUSH,
                recording,
                active_action,
            )
            return
        if action.action_type is ActionType.ROTATE:
            obj = engine.world.objects[action.target]
            axis = normalize(vec3(action.axis or [0.0, 0.0, 1.0]))
            half_angle = (action.angle_rad or 0.0) / 2.0
            q = Quaternion(np.cos(half_angle), *(axis * np.sin(half_angle)))
            obj.pose = Pose(position=obj.pose.position, orientation=obj.pose.orientation * q)
            engine._emit_frame(visualize=None, recording=recording, active_action=active_action, collisions=[])
            yield
            return
        raise NotImplementedError(action.action_type)

    def _reach(
        self,
        engine: ExecutionEngine,
        action: ActionPrimitive,
        recording: SimulationRecording | None,
        active_action: str,
    ) -> Iterator[None]:
        target_pose = action.pose or engine.world.objects[action.target].pose
        ik_result = solve_inverse_kinematics(
            engine.robot,
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
        while any(abs(engine.robot.joint_positions[name] - value) > 1e-3 for name, value in targets.items()):
            engine.controller.step_towards(targets, engine.dt)
            engine.step(recording=recording, active_action=active_action)
            steps += 1
            yield
            if steps > 200:
                break
        if not engine._ik_result_is_actionable(ik_result, position_only=action.pose is None):
            reason = ik_result.failure_reason or "unknown reason"
            raise ValueError(
                f"ik failed for action targeting '{action.target}' with end effector "
                f"'{action.end_effector}': {reason}"
            )
        if steps == 0:
            engine._emit_frame(visualize=None, recording=recording, active_action=active_action, collisions=[])
            yield

    def _move_object(
        self,
        engine: ExecutionEngine,
        action: ActionPrimitive,
        recording: SimulationRecording | None,
        active_action: str,
    ) -> Iterator[None]:
        obj = engine.world.objects[action.target]
        holder_name = engine._held_by_name(action.end_effector)
        if obj.held_by is None:
            raise ValueError(f"cannot move '{action.target}' because it is not grasped")
        if obj.held_by != holder_name:
            raise ValueError(
                f"cannot move '{action.target}' because it is held by '{obj.held_by}', not '{holder_name}'"
            )
        destination = vec3(action.destination or obj.pose.position)
        start = obj.pose.position.copy()
        distance = float(np.linalg.norm(destination - start))
        steps = max(int(distance / max(action.speed * engine.dt, 1e-6)), 1)
        for step_index in range(1, steps + 1):
            alpha = step_index / steps
            obj.pose = Pose(position=start * (1 - alpha) + destination * alpha, orientation=obj.pose.orientation)
            engine.step(recording=recording, active_action=active_action)
            yield

    def _translate_object(
        self,
        engine: ExecutionEngine,
        action: ActionPrimitive,
        push: bool,
        recording: SimulationRecording | None,
        active_action: str,
    ) -> Iterator[None]:
        direction = normalize(vec3(action.axis or [1.0, 0.0, 0.0]))
        if not push:
            direction *= -1.0
        magnitude = max((action.force_newtons or 5.0) / 50.0, 0.05)
        destination = engine.world.objects[action.target].pose.position + direction * magnitude
        translated = ActionPrimitive.move(
            target=action.target,
            destination=destination.tolist(),
            end_effector=action.end_effector,
        )
        engine.world.objects[action.target].held_by = engine._held_by_name(action.end_effector)
        yield from self._move_object(engine, translated, recording, active_action)
        engine.world.objects[action.target].held_by = None
