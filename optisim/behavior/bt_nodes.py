"""Core behavior tree nodes and execution context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import numpy as np

from optisim.core.action_primitives import ActionPrimitive, ActionType
from optisim.math3d import Pose, Quaternion, normalize, vec3
from optisim.robot import IKOptions, solve_inverse_kinematics
from optisim.sim import ExecutionEngine, SimulationRecording, WorldState

SuccessPolicy = Literal["require_all", "require_one"]
Predicate = Callable[["BehaviorContext"], bool]
ActionFactory = Callable[["BehaviorContext"], ActionPrimitive]


class BTStatus(StrEnum):
    """Status returned by every behavior tree node tick."""

    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


@dataclass(slots=True)
class BehaviorContext:
    """Mutable runtime context shared by all nodes during execution."""

    engine: ExecutionEngine
    blackboard: dict[str, Any] = field(default_factory=dict)
    recording: SimulationRecording | None = None
    visualizer: Any | None = None
    tick_count: int = 0
    tree_name: str = "behavior_tree"

    @property
    def world(self) -> WorldState:
        """Return the active world state from the execution engine."""

        return self.engine.world


@dataclass(slots=True)
class BTNode(ABC):
    """Abstract base class for all behavior tree nodes."""

    name: str | None = None

    @abstractmethod
    def tick(self, context: BehaviorContext) -> BTStatus:
        """Advance the node by one behavior-tree tick."""

    def reset(self) -> None:
        """Clear any node-local runtime state."""

    def children(self) -> tuple["BTNode", ...]:
        """Return the node's direct children."""

        return ()

    def to_lines(self, indent: int = 0) -> list[str]:
        """Return a printable tree rendering for the node hierarchy."""

        label = self.name or self.__class__.__name__
        lines = [f"{'  ' * indent}{label} [{self.__class__.__name__}]"]
        for child in self.children():
            lines.extend(child.to_lines(indent + 1))
        return lines


@dataclass(slots=True)
class Sequence(BTNode):
    """Run children in order until one fails or all succeed."""

    child_nodes: tuple[BTNode, ...] = field(default_factory=tuple)
    _current_index: int = field(default=0, init=False, repr=False)

    def tick(self, context: BehaviorContext) -> BTStatus:
        """Tick the current child and advance while children succeed."""

        if not self.child_nodes:
            return BTStatus.SUCCESS
        while self._current_index < len(self.child_nodes):
            status = self.child_nodes[self._current_index].tick(context)
            if status is BTStatus.SUCCESS:
                self._current_index += 1
                continue
            if status is BTStatus.FAILURE:
                self.reset()
                return BTStatus.FAILURE
            return BTStatus.RUNNING
        self.reset()
        return BTStatus.SUCCESS

    def reset(self) -> None:
        """Reset sequence progress and all descendant state."""

        self._current_index = 0
        for child in self.child_nodes:
            child.reset()

    def children(self) -> tuple[BTNode, ...]:
        """Return the ordered sequence children."""

        return self.child_nodes


@dataclass(slots=True)
class Selector(BTNode):
    """Try children in order until one succeeds."""

    child_nodes: tuple[BTNode, ...] = field(default_factory=tuple)
    _current_index: int = field(default=0, init=False, repr=False)

    def tick(self, context: BehaviorContext) -> BTStatus:
        """Tick the current child until one succeeds or all fail."""

        if not self.child_nodes:
            return BTStatus.FAILURE
        while self._current_index < len(self.child_nodes):
            status = self.child_nodes[self._current_index].tick(context)
            if status is BTStatus.SUCCESS:
                self.reset()
                return BTStatus.SUCCESS
            if status is BTStatus.FAILURE:
                self.child_nodes[self._current_index].reset()
                self._current_index += 1
                continue
            return BTStatus.RUNNING
        self.reset()
        return BTStatus.FAILURE

    def reset(self) -> None:
        """Reset selector progress and all descendant state."""

        self._current_index = 0
        for child in self.child_nodes:
            child.reset()

    def children(self) -> tuple[BTNode, ...]:
        """Return the ordered selector children."""

        return self.child_nodes


Fallback = Selector


@dataclass(slots=True)
class Parallel(BTNode):
    """Tick all children each step and resolve according to a success policy."""

    child_nodes: tuple[BTNode, ...] = field(default_factory=tuple)
    success_policy: SuccessPolicy = "require_all"

    def tick(self, context: BehaviorContext) -> BTStatus:
        """Tick each child and combine child statuses."""

        if not self.child_nodes:
            return BTStatus.SUCCESS if self.success_policy == "require_all" else BTStatus.FAILURE

        statuses = [child.tick(context) for child in self.child_nodes]
        success_count = sum(status is BTStatus.SUCCESS for status in statuses)
        failure_count = sum(status is BTStatus.FAILURE for status in statuses)

        if self.success_policy == "require_all":
            if failure_count:
                self.reset()
                return BTStatus.FAILURE
            if success_count == len(statuses):
                self.reset()
                return BTStatus.SUCCESS
            return BTStatus.RUNNING

        if success_count:
            self.reset()
            return BTStatus.SUCCESS
        if failure_count == len(statuses):
            self.reset()
            return BTStatus.FAILURE
        return BTStatus.RUNNING

    def reset(self) -> None:
        """Reset all parallel children."""

        for child in self.child_nodes:
            child.reset()

    def children(self) -> tuple[BTNode, ...]:
        """Return the parallel children."""

        return self.child_nodes


@dataclass(slots=True)
class Condition(BTNode):
    """Leaf node that evaluates a predicate against the runtime context."""

    predicate: Predicate = field(default=lambda _context: True)

    def tick(self, context: BehaviorContext) -> BTStatus:
        """Return success when the predicate is true."""

        return BTStatus.SUCCESS if self.predicate(context) else BTStatus.FAILURE


class _ActionRunner(ABC):
    """Internal per-action incremental executor."""

    @abstractmethod
    def tick(self, context: BehaviorContext, action: ActionPrimitive, label: str) -> BTStatus:
        """Advance the action by one simulation step."""

    def reset(self) -> None:
        """Clear any internal runner state."""


@dataclass(slots=True)
class _ReachRunner(_ActionRunner):
    targets: dict[str, float] | None = None
    complete: bool = False

    def tick(self, context: BehaviorContext, action: ActionPrimitive, label: str) -> BTStatus:
        if self.complete:
            return BTStatus.SUCCESS
        target_pose = action.pose or context.world.objects[action.target].pose
        if self.targets is None:
            result = solve_inverse_kinematics(
                context.engine.robot,
                action.end_effector,
                target_pose,
                options=IKOptions(
                    max_iterations=120,
                    convergence_threshold=2e-3,
                    damping=0.12,
                    position_only=action.pose is None,
                ),
            )
            if not context.engine._ik_result_is_actionable(result, position_only=action.pose is None):
                return BTStatus.FAILURE
            self.targets = result.joint_positions
        if self.targets is None:
            return BTStatus.FAILURE
        context.engine.controller.step_towards(self.targets, context.engine.dt)
        context.engine.step(context.visualizer, recording=context.recording, active_action=label)
        if all(abs(context.engine.robot.joint_positions[name] - value) <= 1e-3 for name, value in self.targets.items()):
            self.complete = True
            return BTStatus.SUCCESS
        return BTStatus.RUNNING

    def reset(self) -> None:
        self.targets = None
        self.complete = False


@dataclass(slots=True)
class _MoveRunner(_ActionRunner):
    start: np.ndarray | None = None
    destination: np.ndarray | None = None
    steps_total: int = 0
    step_index: int = 0
    release_on_finish: bool = False

    def tick(self, context: BehaviorContext, action: ActionPrimitive, label: str) -> BTStatus:
        obj = context.world.objects[action.target]
        if obj.held_by is None:
            return BTStatus.FAILURE
        if self.start is None:
            self.start = obj.pose.position.copy()
        if self.destination is None:
            self.destination = vec3(action.destination or obj.pose.position.tolist())
        if self.steps_total == 0:
            distance = float(np.linalg.norm(self.destination - self.start))
            self.steps_total = max(int(distance / max(action.speed * context.engine.dt, 1e-6)), 1)
        self.step_index += 1
        alpha = min(self.step_index / self.steps_total, 1.0)
        obj.pose = Pose(
            position=self.start * (1.0 - alpha) + self.destination * alpha,
            orientation=obj.pose.orientation,
        )
        context.engine.step(context.visualizer, recording=context.recording, active_action=label)
        if self.step_index >= self.steps_total:
            if self.release_on_finish:
                obj.held_by = None
            return BTStatus.SUCCESS
        return BTStatus.RUNNING

    def reset(self) -> None:
        self.start = None
        self.destination = None
        self.steps_total = 0
        self.step_index = 0
        self.release_on_finish = False


@dataclass(slots=True)
class _GraspRunner(_ActionRunner):
    complete: bool = False

    def tick(self, context: BehaviorContext, action: ActionPrimitive, label: str) -> BTStatus:
        if self.complete:
            return BTStatus.SUCCESS
        context.world.objects[action.target].held_by = action.end_effector
        context.engine.step(context.visualizer, recording=context.recording, active_action=label)
        self.complete = True
        return BTStatus.SUCCESS

    def reset(self) -> None:
        self.complete = False


@dataclass(slots=True)
class _PlaceRunner(_ActionRunner):
    complete: bool = False

    def tick(self, context: BehaviorContext, action: ActionPrimitive, label: str) -> BTStatus:
        if self.complete:
            return BTStatus.SUCCESS
        obj = context.world.objects[action.target]
        if obj.held_by is None:
            return BTStatus.FAILURE
        surface = context.world.surfaces[action.support or ""]
        z = surface.pose.position[2] + surface.size[2] / 2.0 + obj.size[2] / 2.0
        obj.pose = Pose(position=vec3([obj.pose.position[0], obj.pose.position[1], z]), orientation=obj.pose.orientation)
        obj.held_by = None
        context.engine.step(context.visualizer, recording=context.recording, active_action=label)
        self.complete = True
        return BTStatus.SUCCESS

    def reset(self) -> None:
        self.complete = False


@dataclass(slots=True)
class _TranslateRunner(_MoveRunner):
    push: bool = True
    prepared: bool = False

    def tick(self, context: BehaviorContext, action: ActionPrimitive, label: str) -> BTStatus:
        obj = context.world.objects[action.target]
        if not self.prepared:
            direction = normalize(vec3(action.axis or [1.0, 0.0, 0.0]))
            if not self.push:
                direction *= -1.0
            magnitude = max((action.force_newtons or 5.0) / 50.0, 0.05)
            self.destination = obj.pose.position + direction * magnitude
            obj.held_by = action.end_effector
            self.release_on_finish = True
            self.prepared = True
        return super().tick(context, action, label)

    def reset(self) -> None:
        super().reset()
        self.prepared = False


@dataclass(slots=True)
class _RotateRunner(_ActionRunner):
    complete: bool = False

    def tick(self, context: BehaviorContext, action: ActionPrimitive, label: str) -> BTStatus:
        if self.complete:
            return BTStatus.SUCCESS
        obj = context.world.objects[action.target]
        axis = normalize(vec3(action.axis or [0.0, 0.0, 1.0]))
        half_angle = (action.angle_rad or 0.0) / 2.0
        q = Quaternion(np.cos(half_angle), *(axis * np.sin(half_angle)))
        obj.pose = Pose(position=obj.pose.position, orientation=obj.pose.orientation * q)
        context.engine.step(context.visualizer, recording=context.recording, active_action=label)
        self.complete = True
        return BTStatus.SUCCESS

    def reset(self) -> None:
        self.complete = False


def _make_action_runner(action_type: ActionType) -> _ActionRunner:
    if action_type is ActionType.REACH:
        return _ReachRunner()
    if action_type is ActionType.GRASP:
        return _GraspRunner()
    if action_type is ActionType.MOVE:
        return _MoveRunner()
    if action_type is ActionType.PLACE:
        return _PlaceRunner()
    if action_type is ActionType.PUSH:
        return _TranslateRunner(push=True)
    if action_type is ActionType.PULL:
        return _TranslateRunner(push=False)
    if action_type is ActionType.ROTATE:
        return _RotateRunner()
    raise NotImplementedError(action_type)


@dataclass(slots=True)
class ActionNode(BTNode):
    """Leaf node that incrementally executes an ``ActionPrimitive``."""

    action: ActionPrimitive | None = None
    action_factory: ActionFactory | None = None
    _runner: _ActionRunner | None = field(default=None, init=False, repr=False)
    _active_action: ActionPrimitive | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the node was created with an action source."""

        if self.action is None and self.action_factory is None:
            raise ValueError("ActionNode requires either an action or action_factory")

    def tick(self, context: BehaviorContext) -> BTStatus:
        """Advance the wrapped action by one simulation step."""

        if self._active_action is None:
            self._active_action = self.action_factory(context) if self.action_factory is not None else self.action
            if self._active_action is None:
                return BTStatus.FAILURE
            self._runner = _make_action_runner(self._active_action.action_type)
        if self._runner is None or self._active_action is None:
            return BTStatus.FAILURE
        label = self.name or f"{self._active_action.action_type.value} {self._active_action.target}"
        status = self._runner.tick(context, self._active_action, label)
        if status is not BTStatus.RUNNING:
            self.reset()
        return status

    def reset(self) -> None:
        """Reset the in-flight action runner."""

        if self._runner is not None:
            self._runner.reset()
        self._runner = None
        self._active_action = None


@dataclass(slots=True)
class Inverter(BTNode):
    """Invert success and failure from a single child."""

    child: BTNode = field(default_factory=Condition)

    def tick(self, context: BehaviorContext) -> BTStatus:
        """Invert child terminal statuses while preserving running."""

        status = self.child.tick(context)
        if status is BTStatus.SUCCESS:
            self.child.reset()
            return BTStatus.FAILURE
        if status is BTStatus.FAILURE:
            self.child.reset()
            return BTStatus.SUCCESS
        return BTStatus.RUNNING

    def reset(self) -> None:
        """Reset the wrapped child."""

        self.child.reset()

    def children(self) -> tuple[BTNode, ...]:
        """Return the single wrapped child."""

        return (self.child,)


@dataclass(slots=True)
class Repeat(BTNode):
    """Repeat a child node a fixed number of successful completions."""

    child: BTNode = field(default_factory=Condition)
    count: int = 1
    _completed: int = field(default=0, init=False, repr=False)

    def tick(self, context: BehaviorContext) -> BTStatus:
        """Tick the child until the repeat budget is exhausted."""

        if self.count <= 0:
            return BTStatus.SUCCESS
        status = self.child.tick(context)
        if status is BTStatus.SUCCESS:
            self._completed += 1
            self.child.reset()
            if self._completed >= self.count:
                self.reset()
                return BTStatus.SUCCESS
            return BTStatus.RUNNING
        if status is BTStatus.FAILURE:
            self.reset()
            return BTStatus.FAILURE
        return BTStatus.RUNNING

    def reset(self) -> None:
        """Reset the repetition counter and child state."""

        self._completed = 0
        self.child.reset()

    def children(self) -> tuple[BTNode, ...]:
        """Return the single repeated child."""

        return (self.child,)


@dataclass(slots=True)
class RetryUntilSuccess(BTNode):
    """Retry a child until it succeeds or the attempt budget is exhausted."""

    child: BTNode = field(default_factory=Condition)
    max_attempts: int = 1
    _attempts: int = field(default=0, init=False, repr=False)

    def tick(self, context: BehaviorContext) -> BTStatus:
        """Tick the child, retrying after failures."""

        status = self.child.tick(context)
        if status is BTStatus.SUCCESS:
            self.reset()
            return BTStatus.SUCCESS
        if status is BTStatus.FAILURE:
            self._attempts += 1
            self.child.reset()
            if self._attempts >= self.max_attempts:
                self.reset()
                return BTStatus.FAILURE
            return BTStatus.RUNNING
        return BTStatus.RUNNING

    def reset(self) -> None:
        """Reset retry accounting and child state."""

        self._attempts = 0
        self.child.reset()

    def children(self) -> tuple[BTNode, ...]:
        """Return the single retried child."""

        return (self.child,)


@dataclass(slots=True)
class Timeout(BTNode):
    """Fail a child when it runs longer than a fixed number of ticks."""

    child: BTNode = field(default_factory=Condition)
    steps: int = 1
    _elapsed: int = field(default=0, init=False, repr=False)

    def tick(self, context: BehaviorContext) -> BTStatus:
        """Tick the child while enforcing a maximum running duration."""

        status = self.child.tick(context)
        if status is BTStatus.RUNNING:
            self._elapsed += 1
            if self._elapsed >= self.steps:
                self.reset()
                return BTStatus.FAILURE
            return BTStatus.RUNNING
        self.reset()
        return status

    def reset(self) -> None:
        """Reset timeout state and child state."""

        self._elapsed = 0
        self.child.reset()

    def children(self) -> tuple[BTNode, ...]:
        """Return the single wrapped child."""

        return (self.child,)


def ensure_childless(node_name: str, children: Iterable[BTNode]) -> None:
    """Validate that a leaf node definition did not receive children."""

    if any(True for _ in children):
        raise ValueError(f"{node_name} cannot contain children")
