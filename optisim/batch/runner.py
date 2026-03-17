"""Parallel batch execution helpers for optisim tasks."""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from optisim.core import TaskDefinition


def _default_workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


@dataclass(slots=True)
class BatchConfig:
    """Configuration for running a set of tasks in parallel."""

    tasks: list[str | Path | TaskDefinition]
    n_workers: int = field(default_factory=_default_workers)
    timeout_per_task: float = 60.0
    repeat: int = 1
    seed: int | None = None
    robot_spec: str | None = None
    output_dir: Path | None = None
    include_dynamics: bool = True
    include_grasp: bool = False

    def __post_init__(self) -> None:
        self.tasks = list(self.tasks)
        self.n_workers = max(1, int(self.n_workers))
        self.timeout_per_task = float(self.timeout_per_task)
        self.repeat = max(1, int(self.repeat))
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)


@dataclass(slots=True)
class BatchTaskResult:
    """Structured result for a single task execution attempt."""

    task_name: str
    run_index: int
    success: bool
    elapsed_seconds: float
    step_count: int
    collision_count: int
    total_energy_j: float | None
    validation_ok: bool
    error: str | None = None
    recording_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["recording_path"] = None if self.recording_path is None else str(self.recording_path)
        return payload


@dataclass(slots=True)
class BatchResult:
    """Aggregate result across a batch run."""

    results: list[BatchTaskResult]
    total_elapsed: float
    n_success: int = field(init=False)
    n_failed: int = field(init=False)

    def __post_init__(self) -> None:
        self.results = sorted(self.results, key=lambda item: (item.task_name, item.run_index))
        self.n_success = sum(1 for item in self.results if item.success)
        self.n_failed = len(self.results) - self.n_success

    def summary_table(self) -> str:
        """Render a compact plain-text summary table."""

        headers = ("task", "run", "ok", "time_s", "steps", "collisions", "energy_j", "validation", "error")
        rows = [
            (
                item.task_name,
                str(item.run_index),
                "yes" if item.success else "no",
                f"{item.elapsed_seconds:.3f}",
                str(item.step_count),
                str(item.collision_count),
                "-" if item.total_energy_j is None else f"{item.total_energy_j:.3f}",
                "ok" if item.validation_ok else "failed",
                item.error or "",
            )
            for item in self.results
        ]
        widths = [len(header) for header in headers]
        for row in rows:
            for index, value in enumerate(row):
                widths[index] = max(widths[index], len(value))
        lines = [
            " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
            "-+-".join("-" * width for width in widths),
        ]
        lines.extend(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows)
        lines.append(
            f"completed {len(self.results)} runs in {self.total_elapsed:.3f}s; "
            f"success={self.n_success} failed={self.n_failed}"
        )
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """Convert results to a pandas DataFrame if pandas is installed."""

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for BatchResult.to_dataframe()") from exc
        return pd.DataFrame([item.to_dict() for item in self.results])

    def to_csv(self, path: str | Path) -> None:
        """Write results as CSV without requiring pandas."""

        destination = Path(path)
        fieldnames = [
            "task_name",
            "run_index",
            "success",
            "elapsed_seconds",
            "step_count",
            "collision_count",
            "total_energy_j",
            "validation_ok",
            "error",
            "recording_path",
        ]
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in self.results:
                writer.writerow(item.to_dict())

    def to_json(self, path: str | Path) -> None:
        """Write results as JSON."""

        destination = Path(path)
        payload = {
            "total_elapsed": self.total_elapsed,
            "n_success": self.n_success,
            "n_failed": self.n_failed,
            "results": [item.to_dict() for item in self.results],
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(slots=True)
class _TaskInvocation:
    task_source: str | Path | TaskDefinition
    run_index: int
    seed: int | None
    submitted_at: float


def _task_name_from_source(task_source: str | Path | TaskDefinition) -> str:
    if isinstance(task_source, TaskDefinition):
        return task_source.name
    return Path(task_source).stem


def _task_name_from_serialized_source(task_source: Any) -> str:
    source_type, payload = task_source
    if source_type == "task_dict" and isinstance(payload, dict):
        name = payload.get("name")
        if isinstance(name, str) and name.strip():
            return name
        return "task"
    return Path(payload).stem


def _recording_destination(output_dir: Path | None, task_name: str, run_index: int) -> Path | None:
    if output_dir is None:
        return None
    safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in task_name).strip("_")
    return output_dir / f"{safe_name or 'task'}_run{run_index}.json"


def _serialize_task_source(task_source: str | Path | TaskDefinition) -> tuple[str, Any]:
    if isinstance(task_source, TaskDefinition):
        return ("task_dict", task_source.to_dict())
    return ("task_path", str(task_source))


def _load_task_from_source(task_source: Any) -> TaskDefinition:
    from optisim.core import TaskDefinition

    source_type, payload = task_source
    if source_type == "task_dict":
        return TaskDefinition.from_dict(payload)
    return TaskDefinition.from_file(payload)


def _load_robot_for_batch(task: TaskDefinition, robot_spec: str | None) -> Any:
    from optisim.robot import build_humanoid_model, load_robot_yaml, load_urdf

    if robot_spec:
        source = Path(robot_spec)
        if robot_spec != "builtin":
            if source.suffix.lower() in {".yaml", ".yml"}:
                return load_robot_yaml(source)
            if source.suffix.lower() == ".urdf":
                return load_urdf(source)
            raise ValueError(f"unsupported robot spec '{robot_spec}': expected .yaml, .yml, .urdf, or 'builtin'")
    if task.robot.get("urdf"):
        return load_urdf(task.robot["urdf"])
    if task.robot.get("yaml_spec"):
        return load_robot_yaml(task.robot["yaml_spec"])
    if task.robot.get("model") in {None, "humanoid", "demo_humanoid", "optimus_humanoid"}:
        return build_humanoid_model()
    return build_humanoid_model()


def _maybe_run_grasp(task: TaskDefinition, world: Any, robot: Any) -> None:
    from optisim.grasp import GraspPlanner, default_parallel_jaw

    planner = GraspPlanner(robot=robot)
    gripper = default_parallel_jaw()
    object_names = {action.target for action in task.actions if action.target in world.objects}
    if not object_names:
        object_names = set(world.objects)
    for object_name in sorted(object_names):
        planner.plan_grasps(world.objects[object_name], gripper, n_candidates=1)


def _config_to_worker_payload(config: BatchConfig) -> dict[str, Any]:
    return {
        "robot_spec": config.robot_spec,
        "output_dir": None if config.output_dir is None else str(config.output_dir),
        "include_dynamics": config.include_dynamics,
        "include_grasp": config.include_grasp,
    }


def _error_result(task_name: str, run_index: int, error: str) -> BatchTaskResult:
    return BatchTaskResult(
        task_name=task_name,
        run_index=run_index,
        success=False,
        elapsed_seconds=0.0,
        step_count=0,
        collision_count=0,
        total_energy_j=None,
        validation_ok=False,
        error=error,
        recording_path=None,
    )


def _execute_batch(
    config: BatchConfig,
    *,
    progress_callback: Callable[[int, int, BatchTaskResult], None] | None = None,
) -> BatchResult:
    start_time = time.perf_counter()
    config_payload = _config_to_worker_payload(config)
    invocations: list[_TaskInvocation] = []
    for task_index, task_source in enumerate(config.tasks):
        for run_index in range(config.repeat):
            seed = None if config.seed is None else config.seed + task_index
            invocations.append(
                _TaskInvocation(
                    task_source=task_source,
                    run_index=run_index,
                    seed=seed,
                    submitted_at=0.0,
                )
            )

    futures: dict[Future[BatchTaskResult], _TaskInvocation] = {}
    results: list[BatchTaskResult] = []
    # Use "spawn" start method to avoid fork-related deadlocks when called
    # from a multi-threaded parent (e.g. pytest, web servers, async runtimes).
    import multiprocessing
    mp_ctx = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=config.n_workers, mp_context=mp_ctx)
    try:
        for invocation in invocations:
            invocation.submitted_at = time.perf_counter()
            future = executor.submit(
                _run_single,
                (
                    _serialize_task_source(invocation.task_source),
                    invocation.run_index,
                    config_payload,
                    invocation.seed,
                ),
            )
            futures[future] = invocation

        completed = 0
        for future, invocation in futures.items():
            task_name = _task_name_from_source(invocation.task_source)
            remaining = config.timeout_per_task - (time.perf_counter() - invocation.submitted_at)
            try:
                result = future.result(timeout=max(remaining, 0.0))
            except TimeoutError:
                future.cancel()
                result = _error_result(task_name, invocation.run_index, "task timed out")
            except Exception as exc:  # pragma: no cover - defensive aggregation
                result = _error_result(task_name, invocation.run_index, str(exc))
            results.append(result)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(invocations), result)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return BatchResult(results=results, total_elapsed=time.perf_counter() - start_time)


@dataclass(slots=True)
class BatchRunner:
    """Convenience wrapper exposing a ``run`` method around ``BatchConfig``."""

    config: BatchConfig

    def run(self) -> BatchResult:
        return run_batch(self.config)


def run_batch(config: BatchConfig) -> BatchResult:
    """Run the configured task suite in parallel and return the aggregate results."""

    return _execute_batch(config)


def _run_single(args: tuple[Any, int, dict[str, Any], int | None]) -> BatchTaskResult:
    """Run one task inside a process-pool worker."""

    task_source, run_index, config_dict, seed = args
    from random import seed as set_random_seed

    from optisim.dynamics import DynamicsValidator
    from optisim.sim import ExecutionEngine, WorldState

    start_time = time.perf_counter()
    try:
        task = _load_task_from_source(task_source)
        if seed is not None:
            set_random_seed(seed)

        world = WorldState.from_dict(task.world)
        robot = _load_robot_for_batch(task, config_dict["robot_spec"])
        engine = ExecutionEngine(robot=robot, world=world)

        total_energy_j: float | None = None
        validation_ok = True
        if config_dict["include_dynamics"]:
            dynamics = DynamicsValidator().validate_task(task=task, robot=robot, world=world)
            total_energy_j = dynamics.energy_profile.total_energy
            validation_ok = dynamics.feasible
            if not dynamics.feasible:
                messages = ", ".join(violation.message for violation in dynamics.violations) or "dynamics validation failed"
                raise ValueError(messages)

        if config_dict["include_grasp"]:
            _maybe_run_grasp(task, world, robot)

        record = engine.run(task)
        output_dir = None if config_dict["output_dir"] is None else Path(config_dict["output_dir"])
        recording_path = _recording_destination(output_dir, task.name, run_index)
        if recording_path is not None:
            recording_path.parent.mkdir(parents=True, exist_ok=True)
            if record.recording is not None:
                record.recording.dump(recording_path)

        return BatchTaskResult(
            task_name=task.name,
            run_index=run_index,
            success=True,
            elapsed_seconds=time.perf_counter() - start_time,
            step_count=record.steps,
            collision_count=len(record.collisions),
            total_energy_j=total_energy_j,
            validation_ok=validation_ok,
            error=None,
            recording_path=recording_path,
        )
    except Exception as exc:
        task_name = _task_name_from_serialized_source(task_source)
        return BatchTaskResult(
            task_name=task_name,
            run_index=run_index,
            success=False,
            elapsed_seconds=time.perf_counter() - start_time,
            step_count=0,
            collision_count=0,
            total_energy_j=None,
            validation_ok=False,
            error=str(exc),
            recording_path=None,
        )
