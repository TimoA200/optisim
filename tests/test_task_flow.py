from __future__ import annotations

from pathlib import Path

import pytest

from optisim.core import TaskDefinition
from optisim.robot import build_humanoid_model
from optisim.sim import ExecutionEngine, WorldState


@pytest.mark.parametrize(
    ("task_path", "expected_actions"),
    [
        (Path("examples/pick_and_place.yaml"), ["reach", "grasp", "move", "place"]),
        (Path("examples/pour_water.yaml"), ["reach", "grasp", "move", "rotate", "move", "place"]),
        (Path("examples/open_door.yaml"), ["reach", "grasp", "rotate", "pull"]),
        (
            Path("examples/stack_blocks.yaml"),
            ["reach", "grasp", "move", "place", "reach", "grasp", "move", "place", "reach", "grasp", "move", "place"],
        ),
    ],
)
def test_examples_validate_and_run(task_path: Path, expected_actions: list[str]) -> None:
    task = TaskDefinition.from_file(task_path)
    engine = ExecutionEngine(robot=build_humanoid_model(), world=WorldState.from_dict(task.world))
    report = engine.validate(task)
    assert report.is_valid, report.summary()
    record = engine.run(task)
    assert record.executed_actions == expected_actions


def test_move_without_grasp_is_invalid() -> None:
    task = TaskDefinition.from_dict(
        {
            "name": "invalid",
            "actions": [{"type": "move", "target": "box", "destination": [0.1, 0.0, 1.0]}],
        }
    )
    engine = ExecutionEngine(world=WorldState.with_defaults())
    report = engine.validate(task)
    assert not report.is_valid

__all__ = ["test_examples_validate_and_run", "test_move_without_grasp_is_invalid"]
