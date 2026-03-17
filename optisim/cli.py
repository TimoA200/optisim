"""Command-line interface for optisim."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from optisim import __version__
from optisim.analytics import ParameterRange, analyze_trajectory, composite_score, sweep_task
from optisim.batch import BatchConfig, BatchTaskResult
from optisim.benchmark import BenchmarkEvaluator, BenchmarkReporter, BenchmarkSuite
from optisim.behavior import BehaviorTreeDefinition, BehaviorTreeExecutor
from optisim.core import TaskDefinition
from optisim.dynamics import ConstraintSet, DynamicsValidator, PayloadConstraint
from optisim.grasp import GraspExecutor, GraspPlanner, Gripper, GripperType, default_parallel_jaw, default_suction, default_three_finger
from optisim.library import TaskCatalog
from optisim.multi import Dependency, RobotFleet, TaskAssignment, TaskCoordinator
from optisim.planning import MotionPlanner
from optisim.robot import RobotModel, build_humanoid_model, load_robot_yaml, load_urdf
from optisim.scenario import ScenarioConfig, ScenarioRunner
from optisim.safety import SafetyConfig
from optisim.sim import ExecutionEngine, SimulationRecording, WorldState, replay_recording
from optisim.sensors import SensorSuite
from optisim.viz import MatplotlibVisualizer, TerminalVisualizer, WebVisualizer
from optisim.batch.runner import _execute_batch


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser for the ``optisim`` CLI."""

    parser = argparse.ArgumentParser(prog="optisim", description="Humanoid robot task planner and simulator")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a task file")
    run_parser.add_argument("task_file", type=Path)
    run_parser.add_argument("--visualize", action="store_true")
    run_parser.add_argument("--backend", choices=("terminal", "matplotlib"), default="terminal")
    _add_robot_spec_argument(run_parser)

    validate_parser = subparsers.add_parser("validate", help="validate a task file")
    validate_parser.add_argument("task_file", type=Path)

    dynamics_parser = subparsers.add_parser("validate-dynamics", help="validate task dynamics and constraints")
    dynamics_parser.add_argument("task_file", type=Path)
    dynamics_parser.add_argument("--max-payload", type=float)
    dynamics_parser.add_argument("--end-effector", default="right_palm")
    dynamics_parser.add_argument("--check-torques", action="store_true")

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
    _add_robot_spec_argument(sim_parser)

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

    multi_parser = subparsers.add_parser("multi", help="run a multi-robot coordination scenario")
    multi_parser.add_argument("scenario_file", nargs="?", type=Path, default=Path("examples/multi_robot_warehouse.yaml"))

    scenario_parser = subparsers.add_parser("scenario", help="run a full scenario with sensors and safety monitoring")
    scenario_parser.add_argument("task_file", type=Path)
    scenario_parser.add_argument("--seed", type=int, default=42, help="RNG seed for sensor noise")
    scenario_parser.add_argument("--dt", type=float, default=0.05, help="simulation timestep in seconds")
    scenario_parser.add_argument("--no-safety", action="store_true", help="disable safety monitoring")
    scenario_parser.add_argument("--no-sensors", action="store_true", help="disable sensor simulation")
    scenario_parser.add_argument("--visualize", action="store_true")
    scenario_parser.add_argument("--backend", choices=("terminal", "matplotlib"), default="terminal")
    scenario_parser.add_argument("--summary-out", type=Path, help="write scenario summary to a text file")
    _add_robot_spec_argument(scenario_parser)

    grasp_parser = subparsers.add_parser("grasp", help="plan grasps for objects in a task file")
    grasp_parser.add_argument("task_file", type=Path)
    grasp_parser.add_argument(
        "--gripper",
        choices=("parallel_jaw", "suction", "three_finger", "multi_finger"),
        default="parallel_jaw",
    )
    grasp_parser.add_argument("--object", dest="object_name", help="limit planning to a single object in the world")
    grasp_parser.add_argument("--top-k", type=int, default=5)
    grasp_parser.add_argument("--visualize", action="store_true")
    grasp_parser.add_argument("--backend", choices=("terminal", "matplotlib"), default="terminal")

    batch_parser = subparsers.add_parser("batch", help="run multiple tasks in parallel")
    batch_parser.add_argument("task_files", nargs="*", type=Path)
    batch_parser.add_argument("--workers", type=int)
    batch_parser.add_argument("--repeat", type=int, default=1)
    batch_parser.add_argument("--output-dir", type=Path)
    batch_parser.add_argument("--csv", type=Path)
    batch_parser.add_argument("--json", type=Path)
    batch_parser.add_argument("--timeout", type=float, default=60.0)
    _add_robot_spec_argument(batch_parser)

    benchmark_parser = subparsers.add_parser("benchmark", help="run standardized manipulation benchmarks")
    benchmark_parser.add_argument("--task", help="run a single benchmark task by name")
    benchmark_parser.add_argument("--all", action="store_true", help="run all benchmark tasks")
    benchmark_parser.add_argument("--list", action="store_true", help="list available benchmark tasks")
    benchmark_parser.add_argument("--format", choices=("table", "json", "csv"), default="table")

    return parser


def _add_robot_spec_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--robot-spec",
        default="builtin",
        help="robot spec path (.yaml/.yml or .urdf) or 'builtin' to use the default task/built-in robot selection",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI with optional argument overrides and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
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

        if args.command == "validate-dynamics":
            task = TaskDefinition.from_file(args.task_file)
            world = WorldState.from_dict(task.world)
            robot = _load_robot(task.robot)
            engine = ExecutionEngine(robot=robot, world=world)
            constraints = ConstraintSet()
            if args.max_payload is not None:
                constraints.payload_constraints.append(
                    PayloadConstraint(max_payload_kg=args.max_payload, end_effector=args.end_effector)
                )
            if args.check_torques:
                constraints.joint_torque_limits = DynamicsValidator.default_torque_limits(robot)
            report = engine.validate_dynamics(task, constraint_set=constraints)
            status = "feasible" if report.feasible else "infeasible"
            print(
                f"{status} total_energy={report.energy_profile.total_energy:.3f}J "
                f"peak_power={report.energy_profile.peak_power:.3f}W"
            )
            for violation in report.violations:
                joint = f" joint={violation.joint_name}" if violation.joint_name else ""
                print(f"{violation.severity}{joint}: {violation.message}")
            for warning in report.warnings:
                print(f"warning: {warning}")
            return 0 if report.feasible else 1

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

        if args.command == "multi":
            return _run_multi(args)

        if args.command == "scenario":
            return _run_scenario(args)

        if args.command == "grasp":
            return _run_grasp(args)

        if args.command == "batch":
            return _run_batch(args)

        if args.command == "benchmark":
            return _run_benchmark(args)

        task = TaskDefinition.from_file(args.task_file)
        return _execute_task_definition(task, args)
    except (KeyError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1


def _load_robot(payload: dict, robot_spec: str | None = None) -> RobotModel:
    """Resolve the robot configuration payload into a ``RobotModel`` instance."""

    if robot_spec not in {None, "builtin"}:
        return _load_robot_override(robot_spec)
    if not payload:
        return build_humanoid_model()
    if "urdf" in payload:
        return load_urdf(payload["urdf"])
    if "yaml_spec" in payload:
        return load_robot_yaml(payload["yaml_spec"])
    if payload.get("model") in {None, "humanoid", "demo_humanoid", "optimus_humanoid"}:
        return build_humanoid_model()
    return build_humanoid_model()


def _load_robot_override(robot_spec: str) -> RobotModel:
    source = Path(robot_spec)
    if source.suffix.lower() in {".yaml", ".yml"}:
        return load_robot_yaml(source)
    if source.suffix.lower() == ".urdf":
        return load_urdf(source)
    raise ValueError(f"unsupported robot spec '{robot_spec}': expected .yaml, .yml, .urdf, or 'builtin'")


def _build_visualizer(args: argparse.Namespace):
    if getattr(args, "web", False):
        return WebVisualizer()
    if getattr(args, "visualize", False) or getattr(args, "command", None) == "replay":
        return TerminalVisualizer() if args.backend == "terminal" else MatplotlibVisualizer()
    return None


def _execute_task_definition(task: TaskDefinition, args: argparse.Namespace) -> int:
    """Run a task definition through the execution engine and CLI presentation layer."""

    world = WorldState.from_dict(task.world)
    robot = _load_robot(task.robot, getattr(args, "robot_spec", None))
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
        print("gymnasium support is not installed. Install with `pip install optisim[rl]`.")
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


def _run_benchmark(args: argparse.Namespace) -> int:
    suite = BenchmarkSuite.DEFAULT
    if args.list:
        for name in suite.list_tasks():
            task = suite.get(name)
            print(f"{name}\t{task.difficulty}\t{','.join(task.tags)}")
        return 0

    task_names = None if args.all or not args.task else [args.task]
    evaluator = BenchmarkEvaluator()
    reporter = BenchmarkReporter()
    results = evaluator.run_suite(suite, task_names=task_names)
    if args.format == "json":
        print(reporter.to_json(results))
    elif args.format == "csv":
        print(reporter.to_csv(results), end="")
    else:
        print(reporter.format_table(results))
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


def _run_multi(args: argparse.Namespace) -> int:
    payload = yaml.safe_load(args.scenario_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"scenario file {args.scenario_file} does not contain a mapping")

    fleet = RobotFleet.from_dict(
        {
            "world": payload.get("world", {}),
            "robots": payload.get("robots", []),
        }
    )
    assignments = [
        TaskAssignment(
            robot_name=str(item["robot_name"]),
            task=TaskDefinition.from_dict(
                {
                    "name": item.get("task", {}).get("name", f"{item['robot_name']}_task"),
                    "actions": item.get("task", {}).get("actions", []),
                }
            ),
            dependencies=[
                Dependency(robot_name=str(dep["robot_name"]), action_index=int(dep["action_index"]))
                for dep in item.get("dependencies", [])
            ],
        )
        for item in payload.get("assignments", [])
    ]
    coordinator = TaskCoordinator(fleet, assignments)
    record = coordinator.execute()

    print(
        f"completed multi-robot scenario '{payload.get('name', args.scenario_file.stem)}' "
        f"in {record.duration_s:.2f}s over {record.steps} steps"
    )
    for robot_name, trace in record.traces.items():
        print(
            f"  {robot_name}: actions={trace.executed_actions} "
            f"completed={trace.completed_action_count}"
        )
    if record.collisions:
        print("inter-robot collisions:")
        for collision in record.collisions[:10]:
            print(
                f"  {collision.robot_a}.{collision.link_a} vs "
                f"{collision.robot_b}.{collision.link_b} distance={collision.distance:.3f}m"
            )
    return 0


def _run_scenario(args: argparse.Namespace) -> int:
    task = TaskDefinition.from_file(args.task_file)
    config = ScenarioConfig(
        name=task.name,
        task=task,
        robot=_load_robot(task.robot, args.robot_spec),
        sensor_suite=None if args.no_sensors else SensorSuite.default_humanoid_suite(),
        safety_config=None if args.no_safety else SafetyConfig.default_humanoid(),
        dt=args.dt,
        rng_seed=args.seed,
    )
    runner = ScenarioRunner(config)
    result = runner.run()
    summary = result.summary()
    print(summary)
    if args.summary_out is not None:
        args.summary_out.write_text(summary, encoding="utf-8")
    return 0


def _run_grasp(args: argparse.Namespace) -> int:
    task = TaskDefinition.from_file(args.task_file)
    world = WorldState.from_dict(task.world)
    robot = _load_robot(task.robot)
    planner = GraspPlanner(robot=robot)
    gripper = _gripper_from_name(args.gripper)
    object_names = [args.object_name] if args.object_name else sorted(world.objects)
    if not object_names:
        print("no objects available for grasp planning")
        return 1

    best_object = None
    best_grasp = None
    best_score = float("-inf")
    for object_name in object_names:
        if object_name not in world.objects:
            print(f"unknown object '{object_name}'")
            return 1
        obj = world.objects[object_name]
        grasps = planner.plan_grasps(obj, gripper, n_candidates=args.top_k)
        print(f"object: {object_name}")
        if not grasps:
            print("  no feasible grasps")
            continue
        for index, grasp in enumerate(grasps, start=1):
            print(
                f"  {index}. score={grasp.quality_score:.4f} "
                f"pos={grasp.position.round(3).tolist()} "
                f"aperture={grasp.aperture:.3f} "
                f"contacts={len(grasp.contact_points)}"
            )
        if grasps[0].quality_score > best_score:
            best_object = obj
            best_grasp = grasps[0]
            best_score = grasps[0].quality_score

    if args.visualize and best_object is not None and best_grasp is not None:
        visualizer = _build_visualizer(args)
        engine = ExecutionEngine(robot=robot, world=world)
        executor = GraspExecutor()
        demo_task = TaskDefinition(name=f"grasp_plan_{best_object.name}", actions=[], world=task.world, robot=task.robot)
        try:
            if visualizer is not None:
                visualizer.start_task(demo_task, world, robot)
            result = executor.execute_grasp(engine, robot, best_grasp, best_object)
            if visualizer is not None:
                visualizer.render(world, robot)
                visualizer.finish(demo_task, world, robot, [])
            print(
                f"executed best grasp on '{best_object.name}': "
                f"success={result.success} stability={result.stability_score:.4f} slip={result.slip_detected}"
            )
        finally:
            if isinstance(visualizer, WebVisualizer):
                visualizer.close()

    return 0


def _gripper_from_name(name: str) -> Gripper:
    normalized = name.lower()
    if normalized == GripperType.PARALLEL_JAW.value:
        return default_parallel_jaw()
    if normalized == GripperType.SUCTION.value:
        return default_suction()
    if normalized in {GripperType.THREE_FINGER.value, "multi_finger"}:
        return default_three_finger()
    raise ValueError(f"unsupported gripper '{name}'")


def _run_batch(args: argparse.Namespace) -> int:
    catalog = TaskCatalog()
    tasks: list[Path | TaskDefinition]
    if args.task_files:
        tasks = list(args.task_files)
    else:
        tasks = [catalog.get(template.name) for template in catalog.list()]

    config = BatchConfig(
        tasks=tasks,
        n_workers=args.workers if args.workers is not None else min(4, len(tasks) or 1),
        timeout_per_task=args.timeout,
        repeat=args.repeat,
        robot_spec=args.robot_spec,
        output_dir=args.output_dir,
    )

    total_runs = len(tasks) * config.repeat
    print(f"running batch with {total_runs} run(s) across {config.n_workers} worker(s)")

    def _progress(done: int, total: int, result: BatchTaskResult) -> None:
        status = "ok" if result.success else "failed"
        print(f"[{done}/{total}] {result.task_name} run={result.run_index} {status}")

    result = _execute_batch(config, progress_callback=_progress)
    print(result.summary_table())
    if args.csv is not None:
        result.to_csv(args.csv)
        print(f"csv saved to {args.csv}")
    if args.json is not None:
        result.to_json(args.json)
        print(f"json saved to {args.json}")
    return 0 if result.n_failed == 0 else 1
