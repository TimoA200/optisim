from __future__ import annotations

from pathlib import Path

from optisim.analytics import ParameterRange, SweepResult, TrajectoryMetrics, find_best, sweep_task
from optisim.core import TaskDefinition
from optisim.sim import SimulationRecording


def test_sweep_task_speed_variation_on_pick_and_place() -> None:
    task = TaskDefinition.from_file(Path("examples/pick_and_place.yaml"))

    results = sweep_task(
        task,
        [ParameterRange(action_index=2, field="speed", values=[0.14, 0.28])],
    )

    assert len(results) == 2
    assert results[0].parameters == {"action[2].speed": 0.28}
    assert results[1].parameters == {"action[2].speed": 0.14}
    assert results[0].metrics.total_time_s < results[1].metrics.total_time_s


def test_find_best_returns_expected_winner() -> None:
    recording = SimulationRecording(robot_name="demo")
    faster = SweepResult(
        parameters={"action[0].speed": 0.3},
        metrics=TrajectoryMetrics(
            total_time_s=1.0,
            total_frames=10,
            joint_travel={"joint_a": 0.5},
            peak_joint_velocity={"joint_a": 0.8},
            end_effector_path_length={"hand": 0.4},
            idle_fraction=0.1,
            action_durations={"move box": 1.0},
            smoothness_score=0.9,
            collision_count=0,
            collision_time_s=0.0,
        ),
        recording=recording,
    )
    slower = SweepResult(
        parameters={"action[0].speed": 0.1},
        metrics=TrajectoryMetrics(
            total_time_s=2.0,
            total_frames=20,
            joint_travel={"joint_a": 0.5},
            peak_joint_velocity={"joint_a": 0.4},
            end_effector_path_length={"hand": 0.4},
            idle_fraction=0.3,
            action_durations={"move box": 2.0},
            smoothness_score=0.9,
            collision_count=0,
            collision_time_s=0.0,
        ),
        recording=recording,
    )

    assert find_best([slower, faster]) is faster


def test_sweep_task_empty_parameter_ranges_returns_baseline() -> None:
    task = TaskDefinition.from_file(Path("examples/pick_and_place.yaml"))

    results = sweep_task(task, [])

    assert len(results) == 1
    assert results[0].parameters == {}
    assert results[0].recording.task_name == task.name
    assert results[0].metrics.total_frames > 0

__all__ = ["test_sweep_task_speed_variation_on_pick_and_place", "test_find_best_returns_expected_winner", "test_sweep_task_empty_parameter_ranges_returns_baseline"]
