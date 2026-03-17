"""Demonstrate optisim task hooks."""

from __future__ import annotations

from pathlib import Path

from optisim.core import TaskDefinition
from optisim.hooks import CollisionRecorder, EventLogger, HookRegistry
from optisim.sim import ExecutionEngine, WorldState


def main() -> None:
    task = TaskDefinition.from_file(Path(__file__).with_name("pick_and_place.yaml"))
    registry = HookRegistry()
    logger = EventLogger(Path("events.jsonl"))
    collisions = CollisionRecorder()
    logger.attach(registry)
    collisions.attach(registry)

    @registry.on("action_end")
    def report_action(event) -> None:
        print(
            "action_end",
            event.data["action_type"],
            f"success={event.data['success']}",
            f"elapsed_ms={event.data['elapsed_ms']:.2f}",
        )

    engine = ExecutionEngine(world=WorldState.from_dict(task.world), hooks=registry)
    engine.run(task)

    print(f"collision events recorded: {len(collisions.events)}")
    print("last 3 log lines:")
    lines = Path("events.jsonl").read_text(encoding="utf-8").splitlines()
    for line in lines[-3:]:
        print(line)


if __name__ == "__main__":
    main()


__all__ = ["main"]
