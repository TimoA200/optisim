"""Run the built-in block stacking example."""

from __future__ import annotations

from pathlib import Path

from optisim.core import TaskDefinition
from optisim.sim import ExecutionEngine, WorldState
from optisim.viz import TerminalVisualizer


def main() -> None:
    task = TaskDefinition.from_file(Path(__file__).with_name("stack_blocks.yaml"))
    engine = ExecutionEngine(world=WorldState.from_dict(task.world))
    report = engine.validate(task)
    if not report.is_valid:
        raise SystemExit(report.summary())
    record = engine.run(task, visualize=TerminalVisualizer())
    print(record)


if __name__ == "__main__":
    main()

__all__ = ["main"]
