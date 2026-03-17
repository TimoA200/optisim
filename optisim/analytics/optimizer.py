"""Task parameter sweep and optimization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from optisim.analytics.trajectory import TrajectoryMetrics, analyze_trajectory
from optisim.core import ActionPrimitive, TaskDefinition
from optisim.robot import RobotModel, build_humanoid_model, load_urdf
from optisim.sim import ExecutionEngine, SimulationRecording, WorldState


@dataclass(slots=True)
class ParameterRange:
    """Discrete values to try for one action field."""

    action_index: int
    field: str
    values: list[float] | list[list[float]]


@dataclass(slots=True)
class SweepResult:
    """Result for a single parameter combination."""

    parameters: dict[str, Any]
    metrics: TrajectoryMetrics
    recording: SimulationRecording


def sweep_task(
    task: TaskDefinition,
    parameter_ranges: list[ParameterRange],
    world_config: dict | None = None,
    robot_config: dict | None = None,
) -> list[SweepResult]:
    """Run a discrete parameter sweep and return results sorted by score."""

    if not parameter_ranges:
        return [_run_combination(task, {}, world_config=world_config, robot_config=robot_config)]

    combinations = product(*(parameter_range.values for parameter_range in parameter_ranges))
    results = [
        _run_combination(
            task,
            {
                _parameter_key(parameter_range): _normalize_parameter_value(value)
                for parameter_range, value in zip(parameter_ranges, combination)
            },
            world_config=world_config,
            robot_config=robot_config,
        )
        for combination in combinations
    ]
    return sorted(results, key=_composite_score, reverse=True)


def find_best(results: list[SweepResult]) -> SweepResult:
    """Return the best sweep result by composite score."""

    if not results:
        raise ValueError("results must not be empty")
    return max(results, key=_composite_score)


def composite_score(result: SweepResult) -> float:
    """Expose the composite score used for ranking sweep results."""

    return _composite_score(result)


def _run_combination(
    task: TaskDefinition,
    parameters: dict[str, Any],
    *,
    world_config: dict | None,
    robot_config: dict | None,
) -> SweepResult:
    swept_task = TaskDefinition.from_dict(task.to_dict())
    for parameter_name, value in parameters.items():
        action_index, field = _parse_parameter_key(parameter_name)
        _set_action_field(swept_task.actions[action_index], field, value)

    world_payload = dict(world_config if world_config is not None else task.world)
    robot_payload = dict(robot_config if robot_config is not None else task.robot)
    engine = ExecutionEngine(
        robot=_load_robot(robot_payload),
        world=WorldState.from_dict(world_payload),
    )
    record = engine.run(swept_task)
    if record.recording is None:
        raise RuntimeError("simulation did not produce a recording")
    metrics = analyze_trajectory(record.recording)
    return SweepResult(
        parameters=parameters,
        metrics=metrics,
        recording=record.recording,
    )


def _set_action_field(action: ActionPrimitive, field: str, value: Any) -> None:
    if not hasattr(action, field):
        raise AttributeError(f"action does not have field '{field}'")
    normalized = tuple(float(item) for item in value) if isinstance(value, list) else value
    setattr(action, field, normalized)


def _parameter_key(parameter_range: ParameterRange) -> str:
    return f"action[{parameter_range.action_index}].{parameter_range.field}"


def _parse_parameter_key(parameter_name: str) -> tuple[int, str]:
    prefix, field = parameter_name.split(".", maxsplit=1)
    action_index = int(prefix.removeprefix("action[").removesuffix("]"))
    return action_index, field


def _normalize_parameter_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [float(item) for item in value]
    return float(value)


def _composite_score(result: SweepResult) -> float:
    metrics = result.metrics
    return (
        0.3 * metrics.smoothness_score
        + 0.3 * (1.0 - metrics.idle_fraction)
        + 0.2 * (1.0 / (1.0 + metrics.collision_count))
        + 0.2 * (1.0 / (1.0 + metrics.total_time_s))
    )


def _load_robot(payload: dict[str, Any]) -> RobotModel:
    if not payload:
        return build_humanoid_model()
    if "urdf" in payload:
        return load_urdf(payload["urdf"])
    if payload.get("model") in {None, "humanoid", "demo_humanoid", "optimus_humanoid"}:
        return build_humanoid_model()
    return build_humanoid_model()

__all__ = ["ParameterRange", "SweepResult", "sweep_task", "find_best", "composite_score"]
