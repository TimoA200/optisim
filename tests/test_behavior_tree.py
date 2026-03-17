from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from optisim.behavior import (
    ActionNode,
    BTNode,
    BTStatus,
    BehaviorContext,
    BehaviorTreeBuilder,
    BehaviorTreeDefinition,
    BehaviorTreeExecutor,
    Condition,
    Inverter,
    Parallel,
    Repeat,
    RetryUntilSuccess,
    Selector,
    Sequence,
    Timeout,
)
from optisim.core import ActionPrimitive
from optisim.sim import ExecutionEngine, WorldState


@dataclass(slots=True)
class StaticNode(BTNode):
    status: BTStatus = BTStatus.SUCCESS
    tick_calls: int = field(default=0, init=False)

    def tick(self, context: BehaviorContext) -> BTStatus:
        self.tick_calls += 1
        return self.status


@dataclass(slots=True)
class ScriptedNode(BTNode):
    statuses: list[BTStatus] = field(default_factory=list)
    tick_calls: int = field(default=0, init=False)
    _index: int = field(default=0, init=False, repr=False)

    def tick(self, context: BehaviorContext) -> BTStatus:
        self.tick_calls += 1
        status = self.statuses[min(self._index, len(self.statuses) - 1)]
        if self._index < len(self.statuses) - 1:
            self._index += 1
        return status

    def reset(self) -> None:
        self._index = 0


@dataclass(slots=True)
class FailThenSucceedNode(BTNode):
    failures_before_success: int = 1
    tick_calls: int = field(default=0, init=False)

    def tick(self, context: BehaviorContext) -> BTStatus:
        self.tick_calls += 1
        if self.tick_calls <= self.failures_before_success:
            return BTStatus.FAILURE
        return BTStatus.SUCCESS


def _context(world: WorldState | None = None) -> BehaviorContext:
    return BehaviorContext(engine=ExecutionEngine(world=world or WorldState.with_defaults()))


def test_sequence_returns_success_when_all_children_succeed() -> None:
    node = Sequence(child_nodes=(StaticNode(status=BTStatus.SUCCESS), StaticNode(status=BTStatus.SUCCESS)))
    assert node.tick(_context()) is BTStatus.SUCCESS


def test_sequence_returns_failure_on_first_failed_child() -> None:
    first = StaticNode(status=BTStatus.SUCCESS)
    second = StaticNode(status=BTStatus.FAILURE)
    third = StaticNode(status=BTStatus.SUCCESS)
    node = Sequence(child_nodes=(first, second, third))
    assert node.tick(_context()) is BTStatus.FAILURE
    assert third.tick_calls == 0


def test_selector_returns_success_when_any_child_succeeds() -> None:
    first = StaticNode(status=BTStatus.FAILURE)
    second = StaticNode(status=BTStatus.SUCCESS)
    node = Selector(child_nodes=(first, second))
    assert node.tick(_context()) is BTStatus.SUCCESS
    assert first.tick_calls == 1
    assert second.tick_calls == 1


def test_parallel_require_all_waits_for_all_children() -> None:
    node = Parallel(
        child_nodes=(
            ScriptedNode(statuses=[BTStatus.RUNNING, BTStatus.SUCCESS]),
            ScriptedNode(statuses=[BTStatus.SUCCESS]),
        ),
        success_policy="require_all",
    )
    context = _context()
    assert node.tick(context) is BTStatus.RUNNING
    assert node.tick(context) is BTStatus.SUCCESS


def test_parallel_require_one_succeeds_when_one_child_succeeds() -> None:
    node = Parallel(
        child_nodes=(
            ScriptedNode(statuses=[BTStatus.RUNNING, BTStatus.RUNNING]),
            ScriptedNode(statuses=[BTStatus.SUCCESS]),
        ),
        success_policy="require_one",
    )
    assert node.tick(_context()) is BTStatus.SUCCESS


def test_condition_uses_context_blackboard() -> None:
    node = Condition(predicate=lambda context: context.blackboard["ready"])
    context = _context()
    context.blackboard["ready"] = True
    assert node.tick(context) is BTStatus.SUCCESS


def test_inverter_swaps_success_and_failure() -> None:
    assert Inverter(child=StaticNode(status=BTStatus.SUCCESS)).tick(_context()) is BTStatus.FAILURE
    assert Inverter(child=StaticNode(status=BTStatus.FAILURE)).tick(_context()) is BTStatus.SUCCESS


def test_repeat_repeats_child_until_count_reached() -> None:
    node = Repeat(child=StaticNode(status=BTStatus.SUCCESS), count=3)
    context = _context()
    assert node.tick(context) is BTStatus.RUNNING
    assert node.tick(context) is BTStatus.RUNNING
    assert node.tick(context) is BTStatus.SUCCESS


def test_retry_until_success_retries_failed_child() -> None:
    node = RetryUntilSuccess(child=FailThenSucceedNode(failures_before_success=1), max_attempts=3)
    context = _context()
    assert node.tick(context) is BTStatus.RUNNING
    assert node.tick(context) is BTStatus.SUCCESS


def test_timeout_fails_long_running_child() -> None:
    node = Timeout(child=ScriptedNode(statuses=[BTStatus.RUNNING, BTStatus.RUNNING]), steps=2)
    context = _context()
    assert node.tick(context) is BTStatus.RUNNING
    assert node.tick(context) is BTStatus.FAILURE


def test_action_node_grasp_updates_world_state() -> None:
    context = _context()
    node = ActionNode(action=ActionPrimitive.grasp(target="box", gripper="right_palm"))
    assert node.tick(context) is BTStatus.SUCCESS
    assert context.world.objects["box"].held_by == "right_palm"


def test_action_node_move_requires_grasp() -> None:
    context = _context()
    node = ActionNode(action=ActionPrimitive.move(target="box", destination=[0.5, 0.0, 1.0]))
    assert node.tick(context) is BTStatus.FAILURE


def test_builder_creates_nested_tree() -> None:
    root = (
        BehaviorTreeBuilder()
        .sequence(name="root")
        .condition(lambda _context: True, name="ready")
        .selector(name="choose")
        .condition(lambda _context: False, name="nope")
        .condition(lambda _context: True, name="yep")
        .end()
        .end()
        .build()
    )
    assert isinstance(root, Sequence)
    assert len(root.child_nodes) == 2


def test_yaml_loading_builds_expected_tree() -> None:
    definition = BehaviorTreeDefinition.from_file(Path("examples/bt_pickup.yaml"))
    assert definition.name == "bt_pickup"
    assert isinstance(definition.root, Sequence)
    assert definition.blackboard["preferred_hand"] == "right_palm"


def test_blackboard_data_sharing_across_nodes() -> None:
    setter = ActionNode(
        action_factory=lambda context: (
            context.blackboard.__setitem__("chosen_hand", "right_palm") or ActionPrimitive.grasp("box", "right_palm")
        )
    )
    root = Sequence(
        child_nodes=(
            setter,
            Condition(predicate=lambda context: context.blackboard.get("chosen_hand") == "right_palm"),
        )
    )
    result = BehaviorTreeExecutor(root=root, engine=ExecutionEngine(world=WorldState.with_defaults())).run(max_ticks=5)
    assert result.status is BTStatus.SUCCESS


def test_sequence_of_actions_with_conditions_moves_object_to_surface() -> None:
    world = WorldState.with_defaults()
    root = Sequence(
        child_nodes=(
            Condition(predicate=lambda context: "box" in context.world.objects),
            ActionNode(action=ActionPrimitive.grasp("box", "right_palm")),
            ActionNode(action=ActionPrimitive.move("box", [0.60, -0.25, 1.08], end_effector="right_palm")),
            ActionNode(action=ActionPrimitive.place("box", "shelf", end_effector="right_palm")),
            Condition(predicate=lambda context: context.world.objects["box"].held_by is None),
        )
    )
    result = BehaviorTreeExecutor(root=root, engine=ExecutionEngine(world=world)).run(max_ticks=200)
    assert result.status is BTStatus.SUCCESS
    assert world.objects["box"].held_by is None


def test_fallback_tries_second_branch_when_first_fails() -> None:
    world = WorldState.with_defaults()
    failing_branch = Sequence(
        child_nodes=(
            Condition(predicate=lambda _context: False),
            ActionNode(action=ActionPrimitive.grasp("box", "right_palm")),
        )
    )
    fallback = Selector(
        child_nodes=(
            failing_branch,
            ActionNode(action=ActionPrimitive.grasp("box", "left_palm")),
        )
    )
    result = BehaviorTreeExecutor(root=fallback, engine=ExecutionEngine(world=world)).run(max_ticks=10)
    assert result.status is BTStatus.SUCCESS
    assert world.objects["box"].held_by == "left_palm"


def test_executor_records_frames_and_returns_terminal_status() -> None:
    world = WorldState.with_defaults()
    root = Sequence(child_nodes=(ActionNode(action=ActionPrimitive.grasp("box", "right_palm")),))
    result = BehaviorTreeExecutor(root=root, engine=ExecutionEngine(world=world)).run(max_ticks=10)
    assert result.status is BTStatus.SUCCESS
    assert result.recording is not None
    assert result.recording.frame_count() >= 2


def test_executor_from_definition_runs_real_yaml_tree() -> None:
    definition = BehaviorTreeDefinition.from_file(Path("examples/bt_pickup.yaml"))
    executor = BehaviorTreeExecutor.from_definition(definition)
    result = executor.run(max_ticks=400)
    assert result.status is BTStatus.SUCCESS
    assert executor.engine.world.objects["box"].held_by is None


def test_yaml_validation_rejects_invalid_decorator_shape(tmp_path: Path) -> None:
    path = tmp_path / "invalid_bt.yaml"
    path.write_text(
        """
name: invalid
tree:
  type: timeout
  steps: 2
  children: []
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        BehaviorTreeDefinition.from_file(path)


def test_timeout_can_wrap_action_node() -> None:
    world = WorldState.with_defaults()
    node = Timeout(
        child=ActionNode(action=ActionPrimitive.move("box", [0.60, -0.25, 1.08], end_effector="right_palm")),
        steps=1,
    )
    world.objects["box"].held_by = "right_palm"
    result = BehaviorTreeExecutor(root=node, engine=ExecutionEngine(world=world)).run(max_ticks=5)
    assert result.status is BTStatus.FAILURE

__all__ = ["StaticNode", "ScriptedNode", "FailThenSucceedNode", "test_sequence_returns_success_when_all_children_succeed", "test_sequence_returns_failure_on_first_failed_child", "test_selector_returns_success_when_any_child_succeeds", "test_parallel_require_all_waits_for_all_children", "test_parallel_require_one_succeeds_when_one_child_succeeds", "test_condition_uses_context_blackboard", "test_inverter_swaps_success_and_failure", "test_repeat_repeats_child_until_count_reached", "test_retry_until_success_retries_failed_child", "test_timeout_fails_long_running_child", "test_action_node_grasp_updates_world_state", "test_action_node_move_requires_grasp", "test_builder_creates_nested_tree", "test_yaml_loading_builds_expected_tree", "test_blackboard_data_sharing_across_nodes", "test_sequence_of_actions_with_conditions_moves_object_to_surface", "test_fallback_tries_second_branch_when_first_fails", "test_executor_records_frames_and_returns_terminal_status", "test_executor_from_definition_runs_real_yaml_tree", "test_yaml_validation_rejects_invalid_decorator_shape", "test_timeout_can_wrap_action_node"]
