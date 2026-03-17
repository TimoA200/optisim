from __future__ import annotations

from math import isclose

from optisim.analytics import TrajectoryMetrics, analyze_trajectory, compare_trajectories
from optisim.sim import RecordingFrame, SimulationRecording


def test_analyze_trajectory_minimal_recording() -> None:
    recording = SimulationRecording(
        robot_name="demo",
        task_name="reach",
        dt=0.1,
        joint_names=["joint_a", "joint_b"],
        end_effectors={"hand": "hand_link"},
        frames=[
            RecordingFrame(
                index=0,
                time_s=0.0,
                active_action=None,
                joint_positions={"joint_a": 0.0, "joint_b": 0.0},
                link_positions={"hand_link": [0.0, 0.0, 0.0]},
                objects={},
                surfaces={},
                moving_joints=[],
                collisions=[],
            ),
            RecordingFrame(
                index=1,
                time_s=0.1,
                active_action="reach box",
                joint_positions={"joint_a": 0.1, "joint_b": 0.0},
                link_positions={"hand_link": [0.1, 0.0, 0.0]},
                objects={},
                surfaces={},
                moving_joints=["joint_a"],
                collisions=[],
            ),
            RecordingFrame(
                index=2,
                time_s=0.2,
                active_action="reach box",
                joint_positions={"joint_a": 0.3, "joint_b": 0.0},
                link_positions={"hand_link": [0.3, 0.0, 0.0]},
                objects={},
                surfaces={},
                moving_joints=["joint_a"],
                collisions=[{"entity_a": "box", "entity_b": "table", "penetration_depth": 0.01}],
            ),
            RecordingFrame(
                index=3,
                time_s=0.3,
                active_action="hold",
                joint_positions={"joint_a": 0.3, "joint_b": 0.0},
                link_positions={"hand_link": [0.3, 0.0, 0.0]},
                objects={},
                surfaces={},
                moving_joints=[],
                collisions=[],
            ),
        ],
    )

    metrics = analyze_trajectory(recording)

    assert isclose(metrics.total_time_s, 0.3)
    assert metrics.total_frames == 4
    assert isclose(metrics.joint_travel["joint_a"], 0.3)
    assert isclose(metrics.joint_travel["joint_b"], 0.0)
    assert isclose(metrics.peak_joint_velocity["joint_a"], 2.0)
    assert isclose(metrics.peak_joint_velocity["joint_b"], 0.0)
    assert isclose(metrics.end_effector_path_length["hand"], 0.3)
    assert isclose(metrics.idle_fraction, 0.5)
    assert isclose(metrics.action_durations["reach box"], 0.2)
    assert isclose(metrics.action_durations["hold"], 0.1)
    assert 0.0 < metrics.smoothness_score < 1.0
    assert metrics.collision_count == 1
    assert isclose(metrics.collision_time_s, 0.1)


def test_compare_trajectories_prefers_lower_cost_and_higher_smoothness() -> None:
    a = TrajectoryMetrics(
        total_time_s=1.0,
        total_frames=10,
        joint_travel={"joint_a": 0.8},
        peak_joint_velocity={"joint_a": 1.0},
        end_effector_path_length={"hand": 0.5},
        idle_fraction=0.1,
        action_durations={"reach": 0.6},
        smoothness_score=0.9,
        collision_count=0,
        collision_time_s=0.0,
    )
    b = TrajectoryMetrics(
        total_time_s=1.5,
        total_frames=12,
        joint_travel={"joint_a": 1.2},
        peak_joint_velocity={"joint_a": 1.6},
        end_effector_path_length={"hand": 0.7},
        idle_fraction=0.25,
        action_durations={"reach": 0.8},
        smoothness_score=0.6,
        collision_count=2,
        collision_time_s=0.2,
    )

    comparison = compare_trajectories(a, b)

    assert comparison["total_time_s"]["better"] == "a"
    assert comparison["joint_travel"]["better"] == "a"
    assert comparison["smoothness_score"]["better"] == "a"
    assert comparison["collision_count"]["better"] == "a"
    assert comparison["action_durations"]["total"]["better"] == "a"
    assert comparison["overall"]["better"] == "a"


def test_analyze_trajectory_empty_recording() -> None:
    recording = SimulationRecording(
        robot_name="demo",
        dt=0.1,
        joint_names=["joint_a"],
        end_effectors={"hand": "hand_link"},
    )

    metrics = analyze_trajectory(recording)

    assert metrics.total_time_s == 0.0
    assert metrics.total_frames == 0
    assert metrics.joint_travel == {"joint_a": 0.0}
    assert metrics.peak_joint_velocity == {"joint_a": 0.0}
    assert metrics.end_effector_path_length == {"hand": 0.0}
    assert metrics.idle_fraction == 0.0
    assert metrics.action_durations == {}
    assert metrics.smoothness_score == 1.0
    assert metrics.collision_count == 0
    assert metrics.collision_time_s == 0.0


def test_analyze_trajectory_single_frame() -> None:
    recording = SimulationRecording(
        robot_name="demo",
        dt=0.1,
        joint_names=["joint_a"],
        end_effectors={"hand": "hand_link"},
        frames=[
            RecordingFrame(
                index=0,
                time_s=0.0,
                active_action="reach box",
                joint_positions={"joint_a": 0.2},
                link_positions={"hand_link": [0.2, 0.0, 0.0]},
                objects={},
                surfaces={},
                moving_joints=[],
                collisions=[],
            )
        ],
    )

    metrics = analyze_trajectory(recording)

    assert metrics.total_time_s == 0.0
    assert metrics.total_frames == 1
    assert metrics.joint_travel == {"joint_a": 0.0}
    assert metrics.peak_joint_velocity == {"joint_a": 0.0}
    assert metrics.end_effector_path_length == {"hand": 0.0}
    assert metrics.idle_fraction == 1.0
    assert metrics.action_durations == {}
    assert metrics.smoothness_score == 1.0
    assert metrics.collision_count == 0
    assert metrics.collision_time_s == 0.0
