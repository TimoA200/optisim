"""Command-line interface for optisim."""

from __future__ import annotations

import argparse
from pathlib import Path

from optisim import __version__
from optisim.analytics import analyze_trajectory
from optisim.core import TaskDefinition
from optisim.robot import RobotModel, build_humanoid_model, load_urdf
from optisim.sim import ExecutionEngine, SimulationRecording, WorldState, replay_recording
from optisim.viz import MatplotlibVisualizer, TerminalVisualizer, WebVisualizer


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser for the ``optisim`` CLI."""

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
    sim_parser.add_argument("--web", action="store_true", help="launch the web visualizer")
    sim_parser.add_argument("--recording-out", type=Path, help="export a JSON simulation recording")

    replay_parser = subparsers.add_parser("replay", help="replay a previously exported simulation recording")
    replay_parser.add_argument("recording_file", type=Path)
    replay_parser.add_argument("--backend", choices=("terminal", "matplotlib"), default="terminal")
    replay_parser.add_argument("--web", action="store_true", help="launch the web visualizer for replay")

    analyze_parser = subparsers.add_parser("analyze", help="analyze a simulation recording")
    analyze_parser.add_argument("recording_file", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI with optional argument overrides and return a process exit code."""

    args = build_parser().parse_args(argv)
    if args.command == "validate":
        task = TaskDefinition.from_file(args.task_file)
        world = WorldState.from_dict(task.world)
        robot = _load_robot(task.robot)
        engine = ExecutionEngine(robot=robot, world=world)
        report = engine.validate(task)
        print(report.summary())
        for issue in report.errors + report.warnings:
            location = f" action[{issue.action_index}]" if issue.action_index is not None else ""
            print(f"{issue.severity}{location}: {issue.message}")
        return 0 if report.is_valid else 1

    if args.command == "replay":
        return _run_replay(args)

    if args.command == "analyze":
        return _run_analysis(args)

    task = TaskDefinition.from_file(args.task_file)
    world = WorldState.from_dict(task.world)
    robot = _load_robot(task.robot)
    engine = ExecutionEngine(robot=robot, world=world)
    visualizer = _build_visualizer(args)

    try:
        record = engine.run(task, visualize=visualizer)
        if args.recording_out and record.recording is not None:
            record.recording.dump(args.recording_out)
            print(f"recording saved to {args.recording_out}")
        print(
            f"completed '{task.name}' in {record.duration_s:.2f}s over {record.steps} steps; "
            f"actions={record.executed_actions}"
        )
        if record.collisions:
            print("collisions:")
            for collision in record.collisions:
                print(f"  {collision.entity_a} vs {collision.entity_b} depth={collision.penetration_depth:.3f}m")
        if args.web and isinstance(visualizer, WebVisualizer):
            print(f"web visualizer available at {visualizer.url} (Ctrl+C to exit)")
            visualizer.block()
        return 0
    finally:
        if isinstance(visualizer, WebVisualizer):
            visualizer.close()


def _load_robot(payload: dict) -> RobotModel:
    """Resolve the robot configuration payload into a ``RobotModel`` instance."""

    if not payload:
        return build_humanoid_model()
    if "urdf" in payload:
        return load_urdf(payload["urdf"])
    if payload.get("model") in {None, "humanoid", "demo_humanoid", "optimus_humanoid"}:
        return build_humanoid_model()
    return build_humanoid_model()


def _build_visualizer(args: argparse.Namespace):
    if getattr(args, "web", False):
        return WebVisualizer()
    if getattr(args, "visualize", False) or getattr(args, "command", None) == "replay":
        return TerminalVisualizer() if args.backend == "terminal" else MatplotlibVisualizer()
    return None


def _run_replay(args: argparse.Namespace) -> int:
    recording = SimulationRecording.from_file(args.recording_file)
    robot = build_humanoid_model()
    world = WorldState.with_defaults()
    visualizer = _build_visualizer(args)

    try:
        replay_recording(recording, robot=robot, world=world, visualizer=visualizer, realtime=True)
        print(
            f"replayed '{recording.task_name or args.recording_file.name}' with "
            f"{recording.frame_count()} frames"
        )
        if args.web and isinstance(visualizer, WebVisualizer):
            print(f"web visualizer available at {visualizer.url} (Ctrl+C to exit)")
            visualizer.block()
        return 0
    finally:
        if isinstance(visualizer, WebVisualizer):
            visualizer.close()


def _run_analysis(args: argparse.Namespace) -> int:
    recording = SimulationRecording.from_file(args.recording_file)
    metrics = analyze_trajectory(recording)

    print(f"Trajectory analysis for {args.recording_file}")
    print(f"robot: {recording.robot_name}")
    print(f"task: {recording.task_name or 'unknown'}")
    print(f"frames: {metrics.total_frames}")
    print(f"time: {metrics.total_time_s:.3f}s")
    print(f"idle fraction: {metrics.idle_fraction:.3f}")
    print(f"smoothness: {metrics.smoothness_score:.3f}")
    print(f"collisions: {metrics.collision_count} events over {metrics.collision_time_s:.3f}s")

    print("joint travel:")
    for name, value in sorted(metrics.joint_travel.items()):
        print(f"  {name}: {value:.6f} rad")

    print("peak joint velocity:")
    for name, value in sorted(metrics.peak_joint_velocity.items()):
        print(f"  {name}: {value:.6f} rad/s")

    print("end effector path length:")
    for name, value in sorted(metrics.end_effector_path_length.items()):
        print(f"  {name}: {value:.6f} m")

    print("action durations:")
    if metrics.action_durations:
        for name, value in sorted(metrics.action_durations.items()):
            print(f"  {name}: {value:.6f}s")
    else:
        print("  none")

    return 0
