"""Example behavior tree pickup workflow."""

from __future__ import annotations

from pathlib import Path

from optisim.behavior import BehaviorTreeDefinition, BehaviorTreeExecutor


def main() -> None:
    """Load the pickup tree, print its structure, and execute it."""

    definition = BehaviorTreeDefinition.from_file(Path("examples/bt_pickup.yaml"))
    print("\n".join(definition.root.to_lines()))
    executor = BehaviorTreeExecutor.from_definition(definition)
    result = executor.run()
    print(f"status={result.status.value} ticks={result.ticks} duration={result.duration_s:.2f}s")


if __name__ == "__main__":
    main()

__all__ = ["main"]
