"""Command-line interface for optisim."""

from __future__ import annotations

import argparse
from pathlib import Path

from optisim import __version__
from optisim.core import TaskDefinition
from optisim.robot import RobotModel, build_demo_humanoid, load_urdf
from optisim.sim import ExecutionEngine, WorldState
from optisim.viz import MatplotlibVisualizer, TerminalVisualizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="optisim", description="Humanoid robot task planner and simulator")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a task file")
    run_parser.add_argument("task_file", type=Path)
    run_parser.add_argument("--visualize", action="store_true")
    run_parser.add_argument("--backend", choices=("terminal", "matplotlib"), default="terminal")

    validate_parser = subparsers.add_parser("validate", help="validate a task file")
    validate_parser.add_argument("task_file", type=Path)

    sim_parser = subparsers.add_parser("sim", help="run the simulator with an optional task")
    sim_parser.add_argument("task_file", nargs="?", type=Path, default=Path("examples/pick_and_place.yaml"))
    sim_parser.add_argument("--visualize", action="store_true")
    sim_parser.add_argument("--backend", choices=("terminal", "matplotlib"), default="terminal")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task = TaskDefinition.from_file(args.task_file)
    world = WorldState.from_dict(task.world)
    robot = _load_robot(task.robot)
    engine = ExecutionEngine(robot=robot, world=world)

    if args.command == "validate":
        report = engine.validate(task)
        print(report.summary())
        for issue in report.errors + report.warnings:
            location = f" action[{issue.action_index}]" if issue.action_index is not None else ""
            print(f"{issue.severity}{location}: {issue.message}")
        return 0 if report.is_valid else 1

    visualizer = None
    if getattr(args, "visualize", False):
        visualizer = TerminalVisualizer() if args.backend == "terminal" else MatplotlibVisualizer()

    record = engine.run(task, visualize=visualizer)
    print(
        f"completed '{task.name}' in {record.duration_s:.2f}s over {record.steps} steps; "
        f"actions={record.executed_actions}"
    )
    if record.collisions:
        print("collisions:")
        for collision in record.collisions:
            print(f"  {collision.entity_a} vs {collision.entity_b} depth={collision.penetration_depth:.3f}m")
    return 0


def _load_robot(payload: dict) -> RobotModel:
    if not payload:
        return build_demo_humanoid()
    if "urdf" in payload:
        return load_urdf(payload["urdf"])
    return build_demo_humanoid()
