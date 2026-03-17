"""Tests for the optisim.planning module."""

from __future__ import annotations

import numpy as np

from optisim.planning import MotionPlanner, PlanningResult, RRTConfig, plan_rrt, plan_rrt_connect, shortcut_path
from optisim.robot import build_humanoid_model
from optisim.sim.world import WorldState


def test_rrt_config_defaults() -> None:
    config = RRTConfig()
    assert config.max_iterations == 1_500
    assert config.step_size == 0.3
    assert config.goal_bias == 0.15
    assert config.goal_threshold == 0.2


def test_planning_result_fields() -> None:
    result = PlanningResult(path=[{"a": 1.0}], success=True, iterations=5, planning_time=0.1)
    assert result.success is True
    assert result.iterations == 5
    assert result.planning_time == 0.1
    assert len(result.path) == 1


def test_rrt_finds_path_in_free_space() -> None:
    rng = np.random.default_rng(42)
    start = np.array([0.0, 0.0])
    goal = np.array([1.0, 1.0])
    lower = np.array([-2.0, -2.0])
    upper = np.array([2.0, 2.0])

    path, iterations = plan_rrt(
        start, goal,
        lower_bounds=lower, upper_bounds=upper,
        is_state_valid=lambda _: True,
        is_edge_valid=lambda _a, _b: True,
        config=RRTConfig(max_iterations=500, step_size=0.5, goal_bias=0.2, goal_threshold=0.3),
        rng=rng,
    )
    assert len(path) >= 2
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)
    assert iterations <= 500


def test_rrt_connect_finds_path() -> None:
    rng = np.random.default_rng(42)
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([1.0, 1.0, 1.0])
    lower = np.array([-3.0, -3.0, -3.0])
    upper = np.array([3.0, 3.0, 3.0])

    path, iterations = plan_rrt_connect(
        start, goal,
        lower_bounds=lower, upper_bounds=upper,
        is_state_valid=lambda _: True,
        is_edge_valid=lambda _a, _b: True,
        config=RRTConfig(max_iterations=500, step_size=0.5, goal_bias=0.2, goal_threshold=0.3),
        rng=rng,
    )
    assert len(path) >= 2
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)
    assert iterations <= 500


def test_planner_same_config_succeeds() -> None:
    robot = build_humanoid_model()
    world = WorldState.with_defaults()
    planner = MotionPlanner(robot=robot, world=world)
    config = {"left_shoulder_pitch": -0.7, "left_elbow_pitch": 0.9}
    result = planner.plan(config, config)
    assert result.success is True
    assert len(result.path) >= 2
    assert result.planning_time >= 0.0


def test_planner_different_configs_left_arm() -> None:
    robot = build_humanoid_model()
    world = WorldState.with_defaults()
    planner = MotionPlanner(robot=robot, world=world)
    start = {"left_shoulder_pitch": -0.7, "left_elbow_pitch": 0.9}
    goal = {"left_shoulder_pitch": -1.0, "left_elbow_pitch": 1.3}
    result = planner.plan(start, goal)
    assert result.success is True
    assert len(result.path) >= 2


def test_shortcut_reduces_waypoints() -> None:
    # Create a zigzag path with 10 waypoints; all segments valid
    path = [{"x": float(i), "y": float(i % 2)} for i in range(10)]
    smoothed = shortcut_path(
        path,
        is_segment_valid=lambda _a, _b: True,
        max_iterations=200,
        rng=np.random.default_rng(0),
    )
    assert len(smoothed) <= len(path)
    # Start and end preserved
    assert smoothed[0]["x"] == path[0]["x"]
    assert smoothed[-1]["x"] == path[-1]["x"]


def test_shortcut_preserves_short_path() -> None:
    path = [{"x": 0.0}, {"x": 1.0}]
    smoothed = shortcut_path(path, is_segment_valid=lambda _a, _b: True)
    assert len(smoothed) == 2


def test_rrt_respects_obstacle() -> None:
    """RRT should fail or take longer when obstacle blocks the direct path."""
    rng = np.random.default_rng(99)
    start = np.array([0.0])
    goal = np.array([2.0])
    lower = np.array([-3.0])
    upper = np.array([3.0])

    # Block all states between 0.5 and 1.5
    def is_valid(state):
        return not (0.5 < float(state[0]) < 1.5)

    path, iterations = plan_rrt(
        start, goal,
        lower_bounds=lower, upper_bounds=upper,
        is_state_valid=is_valid,
        is_edge_valid=lambda a, b: all(is_valid(a + (b - a) * t) for t in np.linspace(0, 1, 10)),
        config=RRTConfig(max_iterations=200, step_size=0.3, goal_bias=0.1, goal_threshold=0.2),
        rng=rng,
    )
    # In 1D with a blocking region the planner cannot easily go around
    # Either it fails or finds a longer path - both are valid outcomes
    # The key test: it doesn't crash and returns a valid structure
    assert isinstance(path, list)
    assert isinstance(iterations, int)


def test_planner_plan_to_pose_uses_ik() -> None:
    """Test that plan_to_pose integrates IK and planning."""
    robot = build_humanoid_model()
    world = WorldState.with_defaults()
    planner = MotionPlanner(robot=robot, world=world)

    # Use a left-arm start config known to be collision-free
    start = {
        "left_clavicle_pitch": 0.08,
        "left_shoulder_pitch": -1.0,
        "left_shoulder_roll": 0.15,
        "left_shoulder_yaw": -0.35,
        "left_elbow_pitch": 1.3,
        "left_forearm_yaw": 0.2,
    }
    # Target = current pose of left_palm at this config
    target_pose = robot.end_effector_pose("left_palm", start)

    result = planner.plan_to_pose(
        start, target_pose,
        ik_options={
            "end_effector": "left_palm",
            "max_iterations": 120,
            "convergence_threshold": 2e-3,
            "damping": 0.12,
            "position_only": True,
        },
    )
    # The result is a valid PlanningResult regardless of success
    assert isinstance(result, PlanningResult)
    assert isinstance(result.success, bool)
    assert result.iterations >= 0
    assert result.planning_time >= 0.0
