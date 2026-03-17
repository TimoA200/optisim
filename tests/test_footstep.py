from __future__ import annotations

import numpy as np

import optisim
from optisim.footstep import (
    Footstep,
    FootstepAnalyzer,
    FootstepPlan,
    FootstepPlanner,
    FootstepSide,
    GaitPhase,
    GaitSchedule,
    SwingTrajectory,
)


def make_plan(n_steps: int = 4) -> FootstepPlan:
    return FootstepPlanner().plan_straight(n_steps=n_steps)


def test_footstep_side_enum_values() -> None:
    assert FootstepSide.LEFT.value == "left"
    assert FootstepSide.RIGHT.value == "right"


def test_footstep_creation() -> None:
    step = Footstep(position=np.array([0.2, 0.1]), heading=0.3, side=FootstepSide.LEFT)
    assert step.position.shape == (2,)
    assert step.heading == 0.3
    assert step.side is FootstepSide.LEFT


def test_footstep_plan_total_duration() -> None:
    plan = make_plan(3)
    assert plan.total_duration == 3.0


def test_footstep_plan_n_steps() -> None:
    plan = make_plan(5)
    assert plan.n_steps == 5


def test_footstep_planner_initialization() -> None:
    planner = FootstepPlanner(step_length=0.25, step_width=0.2, max_step_angle=0.35)
    assert planner.step_length == 0.25
    assert planner.step_width == 0.2
    assert planner.max_step_angle == 0.35


def test_plan_straight_returns_footstep_plan() -> None:
    assert isinstance(make_plan(), FootstepPlan)


def test_plan_straight_n_steps_correct() -> None:
    assert make_plan(6).n_steps == 6


def test_plan_straight_alternates_left_right() -> None:
    sides = [step.side for step in make_plan(5).steps]
    assert sides == [
        FootstepSide.LEFT,
        FootstepSide.RIGHT,
        FootstepSide.LEFT,
        FootstepSide.RIGHT,
        FootstepSide.LEFT,
    ]


def test_plan_straight_positions_advance() -> None:
    positions_x = [step.position[0] for step in make_plan(4).steps]
    assert positions_x[0] < positions_x[-1]


def test_plan_turn_returns_footstep_plan() -> None:
    planner = FootstepPlanner()
    assert isinstance(planner.plan_turn(4, turn_rate=0.1), FootstepPlan)


def test_plan_turn_headings_change() -> None:
    planner = FootstepPlanner()
    headings = [step.heading for step in planner.plan_turn(4, turn_rate=0.1).steps]
    assert headings[0] != headings[-1]


def test_plan_sidestep_returns_footstep_plan() -> None:
    planner = FootstepPlanner()
    assert isinstance(planner.plan_sidestep(4, side="left"), FootstepPlan)


def test_plan_sidestep_moves_laterally() -> None:
    planner = FootstepPlanner()
    plan = planner.plan_sidestep(3, side="left")
    assert plan.steps[-1].position[1] > plan.steps[0].position[1]


def test_plan_to_target_returns_footstep_plan() -> None:
    planner = FootstepPlanner()
    assert isinstance(planner.plan_to_target(np.array([1.0, 0.0])), FootstepPlan)


def test_plan_to_target_last_step_near_target() -> None:
    planner = FootstepPlanner()
    target = np.array([0.9, 0.1])
    plan = planner.plan_to_target(target)
    assert np.linalg.norm(plan.steps[-1].position - target) < planner.step_length


def test_plan_to_target_respects_max_steps() -> None:
    planner = FootstepPlanner(step_length=0.1)
    plan = planner.plan_to_target(np.array([5.0, 0.0]), max_steps=3)
    assert plan.n_steps == 3


def test_swing_trajectory_position_at_zero_near_start() -> None:
    traj = SwingTrajectory(np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.1, 0.0]))
    np.testing.assert_allclose(traj.position_at(0.0), np.array([0.0, 0.0, 0.0]))


def test_swing_trajectory_position_at_duration_near_end() -> None:
    traj = SwingTrajectory(np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.1, 0.0]))
    np.testing.assert_allclose(traj.position_at(traj.duration), np.array([0.3, 0.1, 0.0]), atol=1e-9)


def test_swing_trajectory_midpoint_has_positive_z() -> None:
    traj = SwingTrajectory(np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.1, 0.0]))
    assert traj.position_at(traj.duration * 0.5)[2] > 0.0


def test_swing_trajectory_velocity_at_returns_shape_three() -> None:
    traj = SwingTrajectory(np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.1, 0.0]))
    assert traj.velocity_at(0.2).shape == (3,)


def test_gait_schedule_initialization() -> None:
    schedule = GaitSchedule(make_plan())
    assert isinstance(schedule, GaitSchedule)


def test_gait_schedule_cop_position_returns_shape_two() -> None:
    schedule = GaitSchedule(make_plan())
    assert schedule.cop_position(0.1).shape == (2,)


def test_gait_schedule_feet_positions_returns_left_and_right() -> None:
    schedule = GaitSchedule(make_plan())
    positions = schedule.feet_positions(0.2)
    assert set(positions) == {"left", "right"}


def test_gait_schedule_current_phase_returns_gait_phase() -> None:
    schedule = GaitSchedule(make_plan())
    assert isinstance(schedule.current_phase(0.1), GaitPhase)


def test_gait_schedule_active_swing_returns_tuple_during_swing() -> None:
    schedule = GaitSchedule(make_plan())
    active = schedule.active_swing(0.8)
    assert active is not None
    assert isinstance(active[0], int)


def test_gait_schedule_current_phase_enters_swing() -> None:
    schedule = GaitSchedule(make_plan())
    assert schedule.current_phase(0.8) is GaitPhase.LEFT_SWING


def test_footstep_analyzer_step_lengths_returns_list() -> None:
    analyzer = FootstepAnalyzer()
    assert isinstance(analyzer.step_lengths(make_plan(6)), list)


def test_footstep_analyzer_step_widths_returns_list() -> None:
    analyzer = FootstepAnalyzer()
    assert isinstance(analyzer.step_widths(make_plan(6)), list)


def test_footstep_analyzer_average_cadence_returns_positive_float() -> None:
    analyzer = FootstepAnalyzer()
    assert analyzer.average_cadence(make_plan(4)) > 0.0


def test_footstep_analyzer_path_length_returns_positive_float() -> None:
    analyzer = FootstepAnalyzer()
    assert analyzer.path_length(make_plan(4)) > 0.0


def test_heading_changes_length_matches_plan() -> None:
    analyzer = FootstepAnalyzer()
    plan = FootstepPlanner().plan_turn(5, turn_rate=0.1)
    assert len(analyzer.heading_changes(plan)) == plan.n_steps


def test_root_package_exports_footstep_module() -> None:
    assert hasattr(optisim, "footstep")


def test_gait_schedule_feet_positions_are_three_dimensional() -> None:
    schedule = GaitSchedule(make_plan())
    positions = schedule.feet_positions(0.8)
    assert positions["left"].shape == (3,)
    assert positions["right"].shape == (3,)


def test_plan_to_target_zero_distance_returns_empty_plan() -> None:
    planner = FootstepPlanner()
    plan = planner.plan_to_target(np.array([0.0, 0.0]))
    assert plan.n_steps == 0


__all__ = [name for name in globals() if name.startswith("test_")]
