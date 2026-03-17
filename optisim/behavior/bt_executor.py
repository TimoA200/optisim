"""Tick-based behavior tree execution integrated with the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optisim.core import TaskDefinition
from optisim.robot import build_humanoid_model, load_urdf
from optisim.sim import ExecutionEngine, SimulationRecording, WorldState

from .bt_builder import BehaviorTreeDefinition
from .bt_nodes import BTNode, BTStatus, BehaviorContext


@dataclass(slots=True)
class BehaviorTreeExecutionResult:
    """Summary of a completed behavior tree execution."""

    status: BTStatus
    ticks: int
    duration_s: float
    collisions: list[Any] = field(default_factory=list)
    recording: SimulationRecording | None = None


@dataclass
class BehaviorTreeExecutor:
    """Execute a behavior tree one tick at a time using the existing sim engine."""

    root: BTNode
    engine: ExecutionEngine = field(default_factory=ExecutionEngine)
    blackboard: dict[str, Any] = field(default_factory=dict)
    tree_name: str = "behavior_tree"

    def create_context(
        self,
        *,
        recording: SimulationRecording | None = None,
        visualizer: Any | None = None,
    ) -> BehaviorContext:
        """Create a fresh execution context for the current tree."""

        return BehaviorContext(
            engine=self.engine,
            blackboard=self.blackboard,
            recording=recording,
            visualizer=visualizer,
            tree_name=self.tree_name,
        )

    def tick(self, context: BehaviorContext) -> BTStatus:
        """Advance the tree by one tick."""

        status = self.root.tick(context)
        context.tick_count += 1
        return status

    def run(
        self,
        *,
        max_ticks: int = 1_000,
        visualizer: Any | None = None,
        recording: SimulationRecording | None = None,
    ) -> BehaviorTreeExecutionResult:
        """Run the tree to completion or until ``max_ticks`` is reached."""

        if recording is None:
            recording = SimulationRecording.from_robot(
                self.engine.robot,
                task_name=self.tree_name,
                dt=self.engine.dt,
                metadata={"mode": "behavior_tree", "world_time_start_s": float(self.engine.world.time_s)},
            )
        context = self.create_context(recording=recording, visualizer=visualizer)
        task = TaskDefinition(name=self.tree_name, actions=[], metadata={"mode": "behavior_tree"})

        if visualizer is not None:
            visualizer.start_task(task, self.engine.world, self.engine.robot)
        self.engine._emit_frame(visualize=visualizer, recording=recording, active_action=None, collisions=[])

        status = BTStatus.RUNNING
        collisions: list[Any] = []
        for _ in range(max_ticks):
            frame_count_before = recording.frame_count()
            status = self.tick(context)
            current_collisions = self.engine._check_collisions()
            if recording.frame_count() == frame_count_before:
                self.engine._emit_frame(
                    visualize=visualizer,
                    recording=recording,
                    active_action=self.root.name or self.tree_name,
                    collisions=current_collisions,
                )
            collisions.extend(current_collisions)
            if status is not BTStatus.RUNNING:
                break

        if status is BTStatus.RUNNING:
            status = BTStatus.FAILURE
        if visualizer is not None:
            visualizer.finish(task, self.engine.world, self.engine.robot, collisions)

        self.root.reset()
        return BehaviorTreeExecutionResult(
            status=status,
            ticks=context.tick_count,
            duration_s=float(self.engine.world.time_s),
            collisions=collisions,
            recording=recording,
        )

    @classmethod
    def from_definition(
        cls,
        definition: BehaviorTreeDefinition,
    ) -> "BehaviorTreeExecutor":
        """Build an executor from a loaded behavior tree document."""

        world = WorldState.from_dict(definition.world)
        if not definition.robot:
            robot = build_humanoid_model()
        elif "urdf" in definition.robot:
            robot = load_urdf(definition.robot["urdf"])
        else:
            robot = build_humanoid_model()
        engine = ExecutionEngine(robot=robot, world=world)
        return cls(
            root=definition.root,
            engine=engine,
            blackboard=dict(definition.blackboard),
            tree_name=definition.name,
        )

__all__ = ["BehaviorTreeExecutionResult", "BehaviorTreeExecutor"]
