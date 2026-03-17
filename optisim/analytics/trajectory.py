"""Trajectory analytics for simulation recordings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from optisim.sim import SimulationRecording


@dataclass(slots=True)
class TrajectoryMetrics:
    """Aggregated metrics derived from a ``SimulationRecording``."""

    total_time_s: float
    total_frames: int
    joint_travel: dict[str, float]
    peak_joint_velocity: dict[str, float]
    end_effector_path_length: dict[str, float]
    idle_fraction: float
    action_durations: dict[str, float]
    smoothness_score: float
    collision_count: int
    collision_time_s: float


def analyze_trajectory(recording: SimulationRecording) -> TrajectoryMetrics:
    """Compute optimization-oriented metrics from a simulation recording."""

    frames = recording.frames
    joint_names = list(recording.joint_names) if recording.joint_names else _joint_names_from_frames(recording)
    end_effectors = dict(recording.end_effectors)

    if not frames:
        return TrajectoryMetrics(
            total_time_s=0.0,
            total_frames=0,
            joint_travel={name: 0.0 for name in joint_names},
            peak_joint_velocity={name: 0.0 for name in joint_names},
            end_effector_path_length={name: 0.0 for name in end_effectors},
            idle_fraction=0.0,
            action_durations={},
            smoothness_score=1.0,
            collision_count=0,
            collision_time_s=0.0,
        )

    times = np.asarray([float(frame.time_s) for frame in frames], dtype=np.float64)
    total_time_s = float(max(times[-1] - times[0], 0.0))

    joint_travel = _compute_joint_travel(frames, joint_names)
    peak_joint_velocity = _compute_peak_joint_velocity(frames, joint_names)
    end_effector_path_length = _compute_end_effector_path(frames, end_effectors)
    idle_fraction = _compute_idle_fraction(frames)
    action_durations = _compute_action_durations(recording)
    smoothness_score = _compute_smoothness_score(frames, joint_names)
    collision_count = int(sum(len(frame.collisions) for frame in frames))
    collision_time_s = _compute_collision_time(recording)

    return TrajectoryMetrics(
        total_time_s=total_time_s,
        total_frames=len(frames),
        joint_travel=joint_travel,
        peak_joint_velocity=peak_joint_velocity,
        end_effector_path_length=end_effector_path_length,
        idle_fraction=idle_fraction,
        action_durations=action_durations,
        smoothness_score=smoothness_score,
        collision_count=collision_count,
        collision_time_s=collision_time_s,
    )


def compare_trajectories(a: TrajectoryMetrics, b: TrajectoryMetrics) -> dict[str, Any]:
    """Compare two trajectory metric sets and identify the preferred result."""

    comparison = {
        "total_time_s": _compare_scalar(a.total_time_s, b.total_time_s, lower_is_better=True),
        "total_frames": _compare_scalar(a.total_frames, b.total_frames, lower_is_better=True),
        "joint_travel": _compare_scalar(sum(a.joint_travel.values()), sum(b.joint_travel.values()), lower_is_better=True),
        "peak_joint_velocity": _compare_scalar(
            max(a.peak_joint_velocity.values(), default=0.0),
            max(b.peak_joint_velocity.values(), default=0.0),
            lower_is_better=True,
        ),
        "end_effector_path_length": _compare_scalar(
            sum(a.end_effector_path_length.values()),
            sum(b.end_effector_path_length.values()),
            lower_is_better=True,
        ),
        "idle_fraction": _compare_scalar(a.idle_fraction, b.idle_fraction, lower_is_better=True),
        "smoothness_score": _compare_scalar(a.smoothness_score, b.smoothness_score, lower_is_better=False),
        "collision_count": _compare_scalar(a.collision_count, b.collision_count, lower_is_better=True),
        "collision_time_s": _compare_scalar(a.collision_time_s, b.collision_time_s, lower_is_better=True),
        "action_durations": _compare_action_durations(a.action_durations, b.action_durations),
    }
    wins_a = sum(1 for value in comparison.values() if isinstance(value, dict) and value.get("better") == "a")
    wins_b = sum(1 for value in comparison.values() if isinstance(value, dict) and value.get("better") == "b")
    comparison["overall"] = {
        "a_wins": wins_a,
        "b_wins": wins_b,
        "better": "a" if wins_a > wins_b else "b" if wins_b > wins_a else "tie",
    }
    return comparison


def _joint_names_from_frames(recording: SimulationRecording) -> list[str]:
    names: set[str] = set()
    for frame in recording.frames:
        names.update(frame.joint_positions)
    return sorted(names)


def _compute_joint_travel(frames: list[Any], joint_names: list[str]) -> dict[str, float]:
    if len(frames) < 2:
        return {name: 0.0 for name in joint_names}
    travel = {}
    for name in joint_names:
        series = np.asarray([frame.joint_positions.get(name, 0.0) for frame in frames], dtype=np.float64)
        travel[name] = float(np.abs(np.diff(series)).sum())
    return travel


def _compute_peak_joint_velocity(frames: list[Any], joint_names: list[str]) -> dict[str, float]:
    if len(frames) < 2:
        return {name: 0.0 for name in joint_names}
    times = np.asarray([frame.time_s for frame in frames], dtype=np.float64)
    delta_t = np.diff(times)
    velocity = {}
    for name in joint_names:
        series = np.asarray([frame.joint_positions.get(name, 0.0) for frame in frames], dtype=np.float64)
        delta_q = np.abs(np.diff(series))
        safe_delta_t = np.where(delta_t > 0.0, delta_t, np.nan)
        step_velocity = np.divide(delta_q, safe_delta_t, out=np.zeros_like(delta_q), where=~np.isnan(safe_delta_t))
        velocity[name] = float(np.max(step_velocity, initial=0.0))
    return velocity


def _compute_end_effector_path(frames: list[Any], end_effectors: dict[str, str]) -> dict[str, float]:
    if len(frames) < 2:
        return {name: 0.0 for name in end_effectors}
    path_lengths: dict[str, float] = {}
    for effector, link_name in end_effectors.items():
        points = np.asarray(
            [frame.link_positions.get(link_name, [0.0, 0.0, 0.0]) for frame in frames],
            dtype=np.float64,
        )
        path_lengths[effector] = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return path_lengths


def _compute_idle_fraction(frames: list[Any]) -> float:
    if not frames:
        return 0.0
    idle_frames = 0
    previous = None
    for frame in frames:
        if frame.moving_joints:
            previous = frame
            continue
        if previous is None:
            idle_frames += 1
        else:
            stationary = all(
                np.isclose(frame.joint_positions.get(name, 0.0), previous.joint_positions.get(name, 0.0))
                for name in set(previous.joint_positions) | set(frame.joint_positions)
            )
            if stationary:
                idle_frames += 1
        previous = frame
    return float(idle_frames / len(frames))


def _compute_action_durations(recording: SimulationRecording) -> dict[str, float]:
    frames = recording.frames
    if len(frames) < 2:
        return {}
    durations: dict[str, float] = {}
    for previous, current in zip(frames[:-1], frames[1:]):
        if current.active_action is None:
            continue
        delta_t = max(float(current.time_s - previous.time_s), 0.0)
        durations[current.active_action] = durations.get(current.active_action, 0.0) + delta_t
    return durations


def _compute_smoothness_score(frames: list[Any], joint_names: list[str]) -> float:
    if len(frames) < 4 or not joint_names:
        return 1.0
    times = np.asarray([frame.time_s for frame in frames], dtype=np.float64)
    if np.any(np.diff(times) <= 0.0):
        return 1.0

    positions = np.asarray(
        [[frame.joint_positions.get(name, 0.0) for name in joint_names] for frame in frames],
        dtype=np.float64,
    )
    velocities = np.gradient(positions, times, axis=0)
    accelerations = np.gradient(velocities, times, axis=0)
    jerks = np.gradient(accelerations, times, axis=0)
    mean_jerk = float(np.mean(np.abs(jerks)))
    return float(1.0 / (1.0 + mean_jerk))


def _compute_collision_time(recording: SimulationRecording) -> float:
    frames = recording.frames
    if len(frames) < 2:
        return 0.0
    collision_time = 0.0
    for previous, current in zip(frames[:-1], frames[1:]):
        if not current.collisions:
            continue
        collision_time += max(float(current.time_s - previous.time_s), 0.0)
    return float(collision_time)


def _compare_scalar(a_value: float, b_value: float, *, lower_is_better: bool) -> dict[str, Any]:
    if np.isclose(a_value, b_value):
        better = "tie"
    elif lower_is_better:
        better = "a" if a_value < b_value else "b"
    else:
        better = "a" if a_value > b_value else "b"
    return {"a": a_value, "b": b_value, "better": better}


def _compare_action_durations(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    labels = sorted(set(a) | set(b))
    per_action = {
        label: _compare_scalar(a.get(label, 0.0), b.get(label, 0.0), lower_is_better=True) for label in labels
    }
    totals = _compare_scalar(sum(a.values()), sum(b.values()), lower_is_better=True)
    return {"per_action": per_action, "total": totals, "better": totals["better"]}

__all__ = ["TrajectoryMetrics", "analyze_trajectory", "compare_trajectories"]
