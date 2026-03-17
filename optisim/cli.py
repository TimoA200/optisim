"""Command-line interface for optisim."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from optisim import __version__
from optisim.analytics import ParameterRange, analyze_trajectory, composite_score, sweep_task
from optisim.behavior import BehaviorTreeDefinition, BehaviorTreeExecutor
from optisim.core import TaskDefinition
from optisim.library import TaskCatalog
from optisim.planning import MotionPlanner
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

    plan_parser = subparsers.add_parser("plan", help="plan task reach motions")
    plan_parser.add_argument("task_file", type=Path)
    plan_parser.add_argument("--visualize", action="store_true")
    plan_parser.add_argument("--backend", choices=("terminal", "matplotlib"), default="terminal")

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

    sweep_parser = subparsers.add_parser("sweep", help="run a task parameter sweep")
    sweep_parser.add_argument("task_file", type=Path)
    sweep_parser.add_argument(
        "--vary",
        dest="vary_specs",
        action="append",
        default=[],
        metavar="ACTION:FIELD:VALUES",
        help="parameter range such as 2:speed:0.2,0.3 or 2:destination:[0.5,0,1.0],[0.6,-0.2,1.1]",
    )

    bt_parser = subparsers.add_parser("bt", help="behavior tree tools")
    bt_subparsers = bt_parser.add_subparsers(dest="bt_command", required=True)

    bt_run_parser = bt_subparsers.add_parser("run", help="run a behavior tree file")
    bt_run_parser.add_argument("tree_file", type=Path)
    bt_run_parser.add_argument("--visualize", action="store_true")
    bt_run_parser.add_argument("--backend", choices=("terminal", "matplotlib"), default="terminal")
    bt_run_parser.add_argument("--web", action="store_true", help="launch the web visualizer")
    bt_run_parser.add_argument("--recording-out", type=Path, help="export a JSON simulation recording")
    bt_run_parser.add_argument("--max-ticks", type=int, default=1_000)

    bt_validate_parser = bt_subparsers.add_parser("validate", help="validate a behavior tree file")
    bt_validate_parser.add_argument("tree_file", type=Path)

    library_parser = subparsers.add_parser("library", help="inspect and run built-in task templates")
    library_subparsers = library_parser.add_subparsers(dest="library_command", required=True)

    library_list_parser = library_subparsers.add_parser("list", help="list available task templates")
    library_list_parser.add_argument("--search", help="filter templates by keyword")

    library_info_parser = library_subparsers.add_parser("info", help="show details for a task template")
    library_info_parser.add_argument("name")

    library_run_parser = library_subparsers.add_parser("run", help="run a task template directly")
    library_run_parser.add_argument("name")
    library_run_parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    library_run_parser.add_argument("--visualize", action="store_true")
    library_run_parser.add_argument("--backend", choices=("terminal", "matplotlib"), default="terminal")
    library_run_parser.add_argument("--web", action="store_true", help="launch the web visualizer")
    library_run_parser.add_argument("--recording-out", type=Path, help="export a JSON simulation recording")

    library_export_parser = library_subparsers.add_parser("export", help="export a task template to YAML or JSON")
    library_export_parser.add_argument("name")
    library_export_parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    library_export_parser.add_argument("--output", type=Path, required=True)

    gym_parser = subparsers.add_parser("gym", help="launch a random-agent Gymnasium demo")
    gym_parser.add_argument("--task-file", type=Path, default=Path("examples/pick_and_place.yaml"))
    gym_parser.add_argument("--episodes", type=int, default=1)
    gym_parser.add_argument("--max-steps", type=int, default=100)
    gym_parser.add_argument("--render", action="store_true")
    gym_parser.add_argument("--backend", choices=("terminal", "web"), default="terminal")
    gym_parser.add_argument("--recording-dir", type=Path, help="directory for optional episode recordings")

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

    if args.command == "plan":
        return _run_plan(args)

    if args.command == "analyze":
        return _run_analysis(args)

    if args.command == "sweep":
        return _run_sweep(args)

    if args.command == "bt":
        return _run_behavior_tree(args)

    if args.command == "library":
        return _run_library(args)

    if args.command == "gym":
        return _run_gym_demo(args)

    task = TaskDefinition.from_file(args.task_file)
    return _execute_task_definition(task, args)


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


def _execute_task_definition(task: TaskDefinition, args: argparse.Namespace) -> int:
    """Run a task definition through the execution engine and CLI presentation layer."""

    world = WorldState.from_dict(task.world)
    robot = _load_robot(task.robot)
    engine = ExecutionEngine(robot=robot, world=world)
    visualizer = _build_visualizer(args)

    try:
        record = engine.run(task, visualize=visualizer)
        recording_out = getattr(args, "recording_out", None)
        if recording_out and record.recording is not None:
            record.recording.dump(recording_out)
            print(f"recording saved to {recording_out}")
        print(
            f"completed '{task.name}' in {record.duration_s:.2f}s over {record.steps} steps; "
            f"actions={record.executed_actions}"
        )
        if record.collisions:
            print("collisions:")
            for collision in record.collisions:
                print(f"  {collision.entity_a} vs {collision.entity_b} depth={collision.penetration_depth:.3f}m")
        if getattr(args, "web", False) and isinstance(visualizer, WebVisualizer):
            print(f"web visualizer available at {visualizer.url} (Ctrl+C to exit)")
            visualizer.block()
        return 0
    finally:
        if isinstance(visualizer, WebVisualizer):
            visualizer.close()


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


def _run_plan(args: argparse.Namespace) -> int:
    task = TaskDefinition.from_file(args.task_file)
    world = WorldState.from_dict(task.world)
    robot = _load_robot(task.robot)
    planner = MotionPlanner(robot=robot, world=world)
    visualizer = _build_visualizer(args)
    current_config = dict(robot.joint_positions)

    try:
        if visualizer is not None:
            visualizer.start_task(task, world, robot)
            visualizer.render(world, robot)

        planned_segments = 0
        total_waypoints = 0
        for index, action in enumerate(task.actions, start=1):
            if action.action_type.value != "reach":
                print(f"skipping action {index}: {action.action_type.value} {action.target}")
                continue

            target_pose = action.pose or world.objects[action.target].pose
            result = planner.plan_to_pose(
                current_config,
                target_pose,
                ik_options={"end_effector": action.end_effector, "position_only": action.pose is None},
            )
            if not result.success:
                print(f"failed action {index}: reach {action.target} iterations={result.iterations}")
                return 1

            planned_segments += 1
            total_waypoints += len(result.path)
            current_config = dict(result.path[-1])
            robot.set_joint_positions(current_config)
            print(
                f"planned action {index}: reach {action.target} "
                f"waypoints={len(result.path)} iterations={result.iterations} time={result.planning_time:.3f}s"
            )

            if visualizer is not None:
                visualizer.start_action(action, index=index, total_actions=len(task.actions))
                for waypoint in result.path:
                    robot.set_joint_positions(waypoint)
                    visualizer.render(world, robot)

        if visualizer is not None:
            visualizer.finish(task, world, robot, collisions=[])
        print(f"planned {planned_segments} reach motions with {total_waypoints} total waypoints")
        return 0
    finally:
        if isinstance(visualizer, WebVisualizer):
            visualizer.close()


def _run_gym_demo(args: argparse.Namespace) -> int:
    try:
        import gymnasium as gym
        from optisim.gym_env import RecordEpisode, register_optisim_env
    except ModuleNotFoundError:
        print("gymnasium support is not installed. Install with `pip install optisim[gym]`.")
        return 1

    env_id = register_optisim_env(
        max_steps=args.max_steps,
        task_definition=args.task_file,
        render_mode="web" if args.render and args.backend == "web" else ("human" if args.render else None),
        render_backend=args.backend,
    )
    env = gym.make(env_id)
    if args.recording_dir is not None:
        env = RecordEpisode(env, output_dir=args.recording_dir)

    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=episode)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
            print(
                f"episode={episode} steps={steps} reward={total_reward:.3f} "
                f"task_complete={info.get('task_complete', False)}"
            )
            del observation
    finally:
        env.close()
    return 0


def _run_sweep(args: argparse.Namespace) -> int:
    task = TaskDefinition.from_file(args.task_file)
    parameter_ranges = [_parse_vary_spec(spec) for spec in args.vary_specs]
    results = sweep_task(task, parameter_ranges)

    print(f"Sweep results for {args.task_file}")
    print(f"task: {task.name}")
    print(f"runs: {len(results)}")

    for index, result in enumerate(results[:3], start=1):
        print(f"#{index} score={composite_score(result):.6f} params={result.parameters or {'baseline': True}}")
        print(
            "  metrics: "
            f"time={result.metrics.total_time_s:.3f}s "
            f"smoothness={result.metrics.smoothness_score:.3f} "
            f"idle={result.metrics.idle_fraction:.3f} "
            f"collisions={result.metrics.collision_count}"
        )

    return 0


def _parse_vary_spec(spec: str) -> ParameterRange:
    action_text, field, values_text = spec.split(":", maxsplit=2)
    values = [_parse_cli_value(token) for token in _split_cli_values(values_text)]
    if not values:
        raise ValueError(f"invalid vary spec '{spec}': no values provided")
    return ParameterRange(action_index=int(action_text), field=field, values=values)


def _split_cli_values(values_text: str) -> list[str]:
    values: list[str] = []
    token: list[str] = []
    depth = 0
    for char in values_text:
        if char == "," and depth == 0:
            candidate = "".join(token).strip()
            if candidate:
                values.append(candidate)
            token = []
            continue
        if char in "[{(":
            depth += 1
        elif char in "]})":
            depth = max(depth - 1, 0)
        token.append(char)
    candidate = "".join(token).strip()
    if candidate:
        values.append(candidate)
    return values


def _parse_cli_value(token: str) -> float | list[float]:
    payload = yaml.safe_load(token)
    if isinstance(payload, list):
        return [float(item) for item in payload]
    return float(payload)


def _parse_template_params(entries: list[str]) -> dict[str, Any]:
    """Parse repeated ``KEY=VALUE`` CLI template parameter assignments."""

    parameters: dict[str, Any] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"invalid template parameter '{entry}': expected KEY=VALUE")
        key, raw_value = entry.split("=", maxsplit=1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid template parameter '{entry}': empty key")
        parameters[key] = yaml.safe_load(raw_value.strip())
    return parameters


def _run_library(args: argparse.Namespace) -> int:
    """Handle task library discovery, export, and execution commands."""

    catalog = TaskCatalog()
    if args.library_command == "list":
        templates = catalog.search(args.search) if args.search else catalog.list()
        for template in templates:
            print(f"{template.name:20} {template.difficulty.value:12} {template.description}")
        return 0

    if args.library_command == "info":
        template = catalog.info(args.name)
        print(f"name: {template.name}")
        print(f"difficulty: {template.difficulty.value}")
        print(f"tags: {', '.join(template.tags)}")
        print(f"description: {template.description}")
        if template.parameters:
            print("parameters:")
            for parameter in template.parameters:
                print(f"  {parameter.name}={parameter.default!r}  {parameter.description}")
        return 0

    parameters = _parse_template_params(args.param)
    task = catalog.get(args.name, **parameters)
    if args.library_command == "export":
        task.dump(args.output)
        print(f"exported '{args.name}' to {args.output}")
        return 0
    if args.library_command == "run":
        return _execute_task_definition(task, args)
    raise ValueError(f"unsupported library command '{args.library_command}'")


def _run_behavior_tree(args: argparse.Namespace) -> int:
    definition = BehaviorTreeDefinition.from_file(args.tree_file)
    if args.bt_command == "validate":
        print(f"valid behavior tree '{definition.name}'")
        return 0

    executor = BehaviorTreeExecutor.from_definition(definition)
    visualizer = _build_visualizer(args)
    try:
        result = executor.run(max_ticks=args.max_ticks, visualizer=visualizer)
        if args.recording_out and result.recording is not None:
            result.recording.dump(args.recording_out)
            print(f"recording saved to {args.recording_out}")
        print(
            f"behavior tree '{definition.name}' finished with status={result.status.value} "
            f"ticks={result.ticks} duration={result.duration_s:.2f}s"
        )
        if args.web and isinstance(visualizer, WebVisualizer):
            print(f"web visualizer available at {visualizer.url} (Ctrl+C to exit)")
            visualizer.block()
        return 0 if result.status.value == "success" else 1
    finally:
        if isinstance(visualizer, WebVisualizer):
            visualizer.close()
