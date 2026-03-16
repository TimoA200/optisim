"""Step-based simulator and task execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from optisim.core.action_primitives import ActionPrimitive, ActionType
from optisim.core.task_definition import TaskDefinition
from optisim.core.task_validator import TaskValidator, ValidationReport
from optisim.math3d import Pose, Quaternion, normalize, vec3
from optisim.robot import JointController, RobotModel, build_demo_humanoid
from optisim.sim.collision import Collision, object_surface_collision
from optisim.sim.world import WorldState


@dataclass(slots=True)
class SimulationRecord:
    steps: int
    duration_s: float
    executed_actions: list[str] = field(default_factory=list)
    collisions: list[Collision] = field(default_factory=list)


@dataclass
class ExecutionEngine:
    robot: RobotModel = field(default_factory=build_demo_humanoid)
    world: WorldState = field(default_factory=WorldState.with_defaults)
    dt: float = 0.05

    def __post_init__(self) -> None:
        self.controller = JointController(self.robot)
        self.validator = TaskValidator()

    def validate(self, task: TaskDefinition) -> ValidationReport:
        return self.validator.validate(task=task, world=self.world, robot=self.robot)

    def run(self, task: TaskDefinition, visualize: "Visualizer | None" = None) -> SimulationRecord:
        report = self.validate(task)
        if not report.is_valid:
            details = "; ".join(issue.message for issue in report.errors)
            raise ValueError(f"task validation failed: {details}")

        executed_actions: list[str] = []
        collisions: list[Collision] = []
        steps = 0

        for action in task.actions:
            executed_actions.append(action.action_type.value)
            steps += self._execute_action(action, visualize)
            collisions.extend(self._check_collisions())

        return SimulationRecord(steps=steps, duration_s=self.world.time_s, executed_actions=executed_actions, collisions=collisions)

    def step(self, visualize: "Visualizer | None" = None) -> None:
        self.world.time_s += self.dt
        if visualize is not None:
            visualize.render(self.world, self.robot)

    def _execute_action(self, action: ActionPrimitive, visualize: "Visualizer | None") -> int:
        if action.action_type is ActionType.REACH:
            return self._reach(action, visualize)
        if action.action_type is ActionType.GRASP:
            self.world.objects[action.target].held_by = action.end_effector
            return 1
        if action.action_type is ActionType.MOVE:
            return self._move_object(action, visualize)
        if action.action_type is ActionType.PLACE:
            return self._place(action, visualize)
        if action.action_type in {ActionType.PUSH, ActionType.PULL}:
            return self._translate_object(action, action.action_type is ActionType.PUSH, visualize)
        if action.action_type is ActionType.ROTATE:
            return self._rotate_object(action, visualize)
        raise NotImplementedError(action.action_type)

    def _reach(self, action: ActionPrimitive, visualize: "Visualizer | None") -> int:
        target = self.world.objects[action.target].pose.position
        sign = -1.0 if "right" in action.end_effector else 1.0
        shoulder = self.robot.end_effector_pose(action.end_effector).position
        delta = target - shoulder
        planar = np.linalg.norm(delta[:2])
        elbow = float(np.clip(planar / max(self.robot.max_reach(), 1e-6), 0.0, 1.0) * 1.8)
        shoulder_pitch = float(np.clip(np.arctan2(delta[2], planar + 1e-6), -1.3, 1.3))
        targets = {}
        prefix = "right" if sign < 0 else "left"
        for suffix, value in {
            "shoulder_pitch": shoulder_pitch,
            "elbow_pitch": elbow,
            "wrist_pitch": -0.5 * elbow,
        }.items():
            joint_name = f"{prefix}_{suffix}"
            if joint_name in self.robot.joints:
                targets[joint_name] = value
        steps = 0
        while any(abs(self.robot.joint_positions[name] - value) > 1e-3 for name, value in targets.items()):
            self.controller.step_towards(targets, self.dt)
            self.step(visualize)
            steps += 1
            if steps > 200:
                break
        return max(steps, 1)

    def _move_object(self, action: ActionPrimitive, visualize: "Visualizer | None") -> int:
        obj = self.world.objects[action.target]
        if obj.held_by is None:
            raise ValueError(f"cannot move '{action.target}' because it is not grasped")
        destination = vec3(action.destination or obj.pose.position)
        start = obj.pose.position.copy()
        distance = float(np.linalg.norm(destination - start))
        steps = max(int(distance / max(action.speed * self.dt, 1e-6)), 1)
        for step_index in range(1, steps + 1):
            alpha = step_index / steps
            obj.pose = Pose(position=start * (1 - alpha) + destination * alpha, orientation=obj.pose.orientation)
            self.step(visualize)
        return steps

    def _place(self, action: ActionPrimitive, visualize: "Visualizer | None") -> int:
        obj = self.world.objects[action.target]
        surface = self.world.surfaces[action.support or ""]
        z = surface.pose.position[2] + surface.size[2] / 2.0 + obj.size[2] / 2.0
        obj.pose = Pose(position=vec3([obj.pose.position[0], obj.pose.position[1], z]), orientation=obj.pose.orientation)
        obj.held_by = None
        self.step(visualize)
        return 1

    def _translate_object(self, action: ActionPrimitive, push: bool, visualize: "Visualizer | None") -> int:
        direction = normalize(vec3(action.axis or [1.0, 0.0, 0.0]))
        if not push:
            direction *= -1.0
        magnitude = max((action.force_newtons or 5.0) / 50.0, 0.05)
        destination = self.world.objects[action.target].pose.position + direction * magnitude
        translated = ActionPrimitive.move(target=action.target, destination=destination.tolist(), end_effector=action.end_effector)
        self.world.objects[action.target].held_by = action.end_effector
        steps = self._move_object(translated, visualize)
        self.world.objects[action.target].held_by = None
        return steps

    def _rotate_object(self, action: ActionPrimitive, visualize: "Visualizer | None") -> int:
        obj = self.world.objects[action.target]
        axis = normalize(vec3(action.axis or [0.0, 0.0, 1.0]))
        half_angle = (action.angle_rad or 0.0) / 2.0
        q = Quaternion(np.cos(half_angle), *(axis * np.sin(half_angle)))
        obj.pose = Pose(position=obj.pose.position, orientation=obj.pose.orientation * q)
        self.step(visualize)
        return 1

    def _check_collisions(self) -> list[Collision]:
        collisions: list[Collision] = []
        for obj in self.world.objects.values():
            for surface in self.world.surfaces.values():
                collision = object_surface_collision(obj, surface)
                if collision is not None and obj.held_by is None:
                    collisions.append(collision)
        return collisions


class Visualizer:
    def render(self, world: WorldState, robot: RobotModel) -> None:
        raise NotImplementedError
