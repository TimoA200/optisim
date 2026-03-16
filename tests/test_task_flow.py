from __future__ import annotations

from pathlib import Path

from optisim.core import TaskDefinition
from optisim.sim import ExecutionEngine, WorldState


def test_pick_and_place_example_validates_and_runs() -> None:
    task = TaskDefinition.from_file(Path("examples/pick_and_place.yaml"))
    engine = ExecutionEngine(world=WorldState.from_dict(task.world))
    report = engine.validate(task)
    assert report.is_valid, report.summary()
    record = engine.run(task)
    assert record.executed_actions == ["reach", "grasp", "move", "place"]
    box = engine.world.objects["box"]
    assert box.held_by is None
    assert box.pose.position[2] > 1.0


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
