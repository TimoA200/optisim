"""Tests for optisim.trajopt."""

from __future__ import annotations

import math

from optisim.math3d import Pose
from optisim.robot import JointSpec, LinkSpec, RobotModel
from optisim.trajopt import TrajOptConfig, TrajOptResult, TrajectoryOptimizer, optimize_path


def _build_test_robot() -> RobotModel:
    return RobotModel(
        name="trajopt-test",
        links={
            "base": LinkSpec(name="base"),
            "link1": LinkSpec(name="link1", parent_joint="joint_a"),
            "link2": LinkSpec(name="link2", parent_joint="joint_b"),
        },
        joints={
            "joint_a": JointSpec(
                name="joint_a",
                parent="base",
                child="link1",
                origin=Pose.identity(),
                limit_lower=-1.5,
                limit_upper=1.5,
                velocity_limit=2.0,
            ),
            "joint_b": JointSpec(
                name="joint_b",
                parent="link1",
                child="link2",
                origin=Pose.identity(),
                limit_lower=-2.0,
                limit_upper=2.0,
                velocity_limit=1.5,
            ),
        },
        root_link="base",
        end_effectors={"tool": "link2"},
        joint_positions={"joint_a": 0.0, "joint_b": 0.0},
    )


def _simple_waypoints() -> list[dict[str, float]]:
    return [
        {"joint_a": 0.0, "joint_b": 0.0},
        {"joint_a": 0.5, "joint_b": -0.25},
    ]


def _multi_waypoints() -> list[dict[str, float]]:
    return [
        {"joint_a": 0.0, "joint_b": 0.0},
        {"joint_a": 0.15, "joint_b": -0.1},
        {"joint_a": 0.35, "joint_b": -0.05},
        {"joint_a": 0.45, "joint_b": 0.2},
        {"joint_a": 0.25, "joint_b": 0.35},
        {"joint_a": 0.55, "joint_b": 0.1},
    ]


def test_trajopt_config_defaults() -> None:
    config = TrajOptConfig()
    assert config.max_velocity_scale == 1.0
    assert config.max_acceleration_scale == 0.5
    assert config.min_segment_time == 0.01
    assert config.num_waypoints == 50
    assert config.smoothing_window == 5
    assert config.optimize_for == "time"


def test_trajopt_result_dataclass_construction() -> None:
    result = TrajOptResult(
        waypoints=[{"joint_a": 0.0}],
        times=[0.0],
        joint_velocities=[{"joint_a": 0.0}],
        joint_accelerations=[{"joint_a": 0.0}],
        total_time_s=0.0,
        energy_estimate=0.0,
        feasible=True,
        constraint_violations=[],
    )
    assert result.feasible is True
    assert result.energy_estimate == 0.0
    assert result.times == [0.0]


def test_optimize_two_waypoint_path_returns_feasible_result() -> None:
    robot = _build_test_robot()
    optimizer = TrajectoryOptimizer()

    result = optimizer.optimize(robot, _simple_waypoints())

    assert result.feasible is True
    assert optimizer.is_feasible(result) is True


def test_optimize_multi_waypoint_path_runs() -> None:
    robot = _build_test_robot()
    optimizer = TrajectoryOptimizer(TrajOptConfig(num_waypoints=60))

    result = optimizer.optimize(robot, _multi_waypoints())

    assert len(result.waypoints) == 60
    assert result.feasible is True


def test_total_time_is_positive_and_finite() -> None:
    robot = _build_test_robot()
    result = TrajectoryOptimizer().optimize(robot, _simple_waypoints())

    assert result.total_time_s > 0.0
    assert math.isfinite(result.total_time_s)


def test_waypoints_list_length_at_least_two() -> None:
    robot = _build_test_robot()
    result = TrajectoryOptimizer().optimize(robot, _simple_waypoints())

    assert len(result.waypoints) >= 2


def test_joint_velocities_respect_limits() -> None:
    robot = _build_test_robot()
    result = TrajectoryOptimizer().optimize(robot, _multi_waypoints())

    for sample in result.joint_velocities:
        assert abs(sample["joint_a"]) <= robot.joints["joint_a"].velocity_limit + 1e-6
        assert abs(sample["joint_b"]) <= robot.joints["joint_b"].velocity_limit + 1e-6


def test_resample_returns_requested_number_of_samples() -> None:
    robot = _build_test_robot()
    optimizer = TrajectoryOptimizer()
    result = optimizer.optimize(robot, _multi_waypoints())

    resampled = optimizer.resample(result, 17)

    assert len(resampled.times) == 17
    assert len(resampled.waypoints) == 17


def test_compute_energy_returns_non_negative_float() -> None:
    robot = _build_test_robot()
    optimizer = TrajectoryOptimizer()
    result = optimizer.optimize(robot, _multi_waypoints())

    energy = optimizer.compute_energy(result, robot)

    assert isinstance(energy, float)
    assert energy >= 0.0


def test_is_feasible_returns_bool() -> None:
    robot = _build_test_robot()
    optimizer = TrajectoryOptimizer()
    result = optimizer.optimize(robot, _multi_waypoints())

    feasible = optimizer.is_feasible(result)

    assert isinstance(feasible, bool)


def test_energy_mode_produces_longer_trajectory_than_time_mode() -> None:
    robot = _build_test_robot()
    time_result = TrajectoryOptimizer(TrajOptConfig(optimize_for="time")).optimize(robot, _multi_waypoints())
    energy_result = TrajectoryOptimizer(TrajOptConfig(optimize_for="energy")).optimize(robot, _multi_waypoints())

    assert energy_result.total_time_s > time_result.total_time_s


def test_smoothness_mode_runs_without_error() -> None:
    robot = _build_test_robot()
    optimizer = TrajectoryOptimizer(TrajOptConfig(optimize_for="smoothness", num_waypoints=40))

    result = optimizer.optimize(robot, _multi_waypoints())

    assert len(result.waypoints) == 40
    assert isinstance(result.feasible, bool)


def test_constraint_violations_listed_when_waypoint_exceeds_limits() -> None:
    robot = _build_test_robot()
    optimizer = TrajectoryOptimizer()

    result = optimizer.optimize(
        robot,
        [
            {"joint_a": 0.0, "joint_b": 0.0},
            {"joint_a": 2.5, "joint_b": 0.0},
        ],
    )

    assert result.feasible is False
    assert any("joint_a exceeds position limit" in violation for violation in result.constraint_violations)


def test_optimize_path_helper_function() -> None:
    robot = _build_test_robot()

    result = optimize_path(robot, _simple_waypoints())

    assert isinstance(result, TrajOptResult)
    assert result.total_time_s > 0.0


def test_resample_preserves_total_time() -> None:
    robot = _build_test_robot()
    optimizer = TrajectoryOptimizer()
    result = optimizer.optimize(robot, _multi_waypoints())

    resampled = optimizer.resample(result, 25)

    assert math.isclose(resampled.total_time_s, result.total_time_s)


def test_result_energy_estimate_is_populated() -> None:
    robot = _build_test_robot()
    result = TrajectoryOptimizer().optimize(robot, _simple_waypoints())

    assert result.energy_estimate >= 0.0


__all__ = [
    "test_trajopt_config_defaults",
    "test_trajopt_result_dataclass_construction",
    "test_optimize_two_waypoint_path_returns_feasible_result",
    "test_optimize_multi_waypoint_path_runs",
    "test_total_time_is_positive_and_finite",
    "test_waypoints_list_length_at_least_two",
    "test_joint_velocities_respect_limits",
    "test_resample_returns_requested_number_of_samples",
    "test_compute_energy_returns_non_negative_float",
    "test_is_feasible_returns_bool",
    "test_energy_mode_produces_longer_trajectory_than_time_mode",
    "test_smoothness_mode_runs_without_error",
    "test_constraint_violations_listed_when_waypoint_exceeds_limits",
    "test_optimize_path_helper_function",
    "test_resample_preserves_total_time",
    "test_result_energy_estimate_is_populated",
]
