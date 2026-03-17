from __future__ import annotations

import numpy as np

from optisim.mpc import (
    FootstepPlan,
    FootstepPlanner,
    HumanoidMPC,
    LinearMPC,
    MPCConfig,
    MPCSolution,
    build_humanoid_mpc,
)
from optisim.robot import build_humanoid_model


def test_mpc_config_defaults_match_expected_values() -> None:
    config = MPCConfig()

    assert config.horizon_steps == 20
    assert config.dt == 0.05
    assert config.com_height_z == 0.9
    assert config.Q == 10.0
    assert config.R == 1.0
    assert config.max_zmp_offset == 0.1
    assert config.max_com_velocity == 1.5


def test_mpc_config_allows_custom_values() -> None:
    config = MPCConfig(horizon_steps=12, dt=0.1, com_height_z=0.82, Q=4.0, R=0.5, max_zmp_offset=0.08)

    assert config.horizon_steps == 12
    assert config.dt == 0.1
    assert config.com_height_z == 0.82
    assert config.Q == 4.0
    assert config.R == 0.5
    assert config.max_zmp_offset == 0.08


def test_linear_mpc_initialization_exposes_dimensions_and_horizons() -> None:
    controller = LinearMPC()

    assert controller.horizon == 20
    assert controller.control_horizon == 20
    assert controller.state_dim == 5
    assert controller.input_dim == 2
    assert controller.A.shape == (5, 5)
    assert controller.B.shape == (5, 2)


def test_linear_mpc_solve_returns_solution_with_expected_shapes() -> None:
    controller = LinearMPC()

    solution = controller.solve(
        current_state=np.array([0.0, 0.0, 0.0, 0.0, 0.9]),
        target_state=np.array([0.05, 0.0, -0.02, 0.0, 0.9]),
    )

    assert isinstance(solution, MPCSolution)
    assert solution.optimal_states.shape == (controller.horizon, 5)
    assert solution.optimal_inputs.shape == (controller.horizon, 2)
    assert solution.horizon == controller.horizon


def test_mpc_solution_dataclass_fields_are_accessible() -> None:
    solution = MPCSolution(
        optimal_states=np.zeros((2, 5), dtype=np.float64),
        optimal_inputs=np.zeros((2, 2), dtype=np.float64),
        cost=1.25,
        solve_time_ms=2.5,
        horizon=2,
    )

    assert solution.optimal_states.shape == (2, 5)
    assert solution.optimal_inputs.shape == (2, 2)
    assert solution.cost == 1.25
    assert solution.solve_time_ms == 2.5
    assert solution.horizon == 2


def test_humanoid_mpc_instantiation_works() -> None:
    controller = HumanoidMPC()

    assert isinstance(controller.linear_mpc, LinearMPC)
    assert controller.config.horizon_steps == 20


def test_humanoid_mpc_step_returns_valid_solution() -> None:
    controller = HumanoidMPC()

    solution = controller.step(current_state=np.array([0.0, 0.0, 0.0, 0.0, 0.9]))

    assert isinstance(solution, MPCSolution)
    assert solution.optimal_states.shape[1] == 5
    assert solution.optimal_inputs.shape[1] == 2


def test_humanoid_mpc_tracks_target_position_directionally() -> None:
    controller = HumanoidMPC()

    solution = controller.step(
        current_state=np.array([0.0, 0.0, 0.0, 0.0, 0.9]),
        target_position=np.array([0.08, -0.04]),
        target_velocity=np.array([0.0, 0.0]),
    )

    assert solution.optimal_states[0, 0] > -1e-6
    assert solution.optimal_states[-1, 0] > 0.0
    assert solution.optimal_states[-1, 2] < 0.0


def test_humanoid_mpc_warm_start_does_not_raise() -> None:
    controller = HumanoidMPC()

    controller.warm_start(np.array([0.02, 0.0, -0.01, 0.0, 0.9]))


def test_footstep_planner_returns_valid_plan() -> None:
    planner = FootstepPlanner()

    plan = planner.plan_walk(direction=np.array([1.0, 0.0, 0.0]), steps=4)

    assert isinstance(plan, FootstepPlan)
    assert len(plan.left_foot_positions) == 5
    assert len(plan.right_foot_positions) == 5
    assert len(plan.timing) == 5


def test_footstep_plan_contains_left_and_right_foot_positions() -> None:
    plan = FootstepPlanner().plan_walk(direction=np.array([1.0, 0.2, 0.0]), steps=3)

    assert all(position.shape == (3,) for position in plan.left_foot_positions)
    assert all(position.shape == (3,) for position in plan.right_foot_positions)


def test_mpc_respects_custom_horizon_and_dt() -> None:
    controller = LinearMPC(MPCConfig(horizon_steps=6, dt=0.1))

    solution = controller.solve(
        current_state=np.array([0.0, 0.0, 0.0, 0.0, 0.9]),
        target_state=np.array([0.03, 0.0, 0.02, 0.0, 0.9]),
    )

    assert controller.horizon == 6
    assert solution.horizon == 6
    assert controller.config.dt == 0.1


def test_mpc_solve_time_is_reasonable() -> None:
    controller = LinearMPC()

    solution = controller.solve(
        current_state=np.array([0.0, 0.0, 0.0, 0.0, 0.9]),
        target_state=np.array([0.05, 0.1, 0.01, 0.0, 0.9]),
    )

    assert solution.solve_time_ms < 5000.0


def test_optimal_zmp_stays_within_reference_bounds() -> None:
    controller = LinearMPC()
    zmp_reference = np.tile(np.array([0.02, -0.01], dtype=np.float64), (controller.horizon, 1))

    solution = controller.solve(
        current_state=np.array([0.0, 0.0, 0.0, 0.0, 0.9]),
        target_state=np.array([0.05, 0.0, 0.0, 0.0, 0.9]),
        zmp_reference=zmp_reference,
    )

    lower = zmp_reference - controller.config.max_zmp_offset - 1e-9
    upper = zmp_reference + controller.config.max_zmp_offset + 1e-9
    assert np.all(solution.optimal_inputs >= lower)
    assert np.all(solution.optimal_inputs <= upper)


def test_multiple_consecutive_mpc_steps_support_receding_horizon_usage() -> None:
    controller = HumanoidMPC()
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.9])

    for _ in range(4):
        solution = controller.step(current_state=state, target_position=np.array([0.1, 0.0]))
        state = solution.optimal_states[0]

    assert state[0] > 0.0
    assert abs(state[1]) <= controller.config.max_com_velocity + 1e-9


def test_humanoid_mpc_integrates_with_builtin_humanoid_robot() -> None:
    robot = build_humanoid_model()
    controller = HumanoidMPC(robot=robot)

    solution = controller.step(target_position=np.array([0.25, -0.2]), target_velocity=np.array([0.0, 0.0]))

    assert isinstance(solution, MPCSolution)
    assert np.isfinite(solution.optimal_states).all()
    assert solution.optimal_states[0, 4] == controller.config.com_height_z


def test_single_step_horizon_edge_case() -> None:
    controller = LinearMPC(MPCConfig(horizon_steps=1))

    solution = controller.solve(
        current_state=np.array([0.0, 0.0, 0.0, 0.0, 0.9]),
        target_state=np.array([0.01, 0.0, 0.0, 0.0, 0.9]),
    )

    assert solution.optimal_states.shape == (1, 5)
    assert solution.optimal_inputs.shape == (1, 2)


def test_zero_velocity_target_remains_bounded() -> None:
    controller = HumanoidMPC()

    solution = controller.step(
        current_state=np.array([0.01, 0.2, -0.01, -0.2, 0.9]),
        target_position=np.array([0.01, -0.01]),
        target_velocity=np.array([0.0, 0.0]),
    )

    assert abs(solution.optimal_states[-1, 1]) <= controller.config.max_com_velocity
    assert abs(solution.optimal_states[-1, 3]) <= controller.config.max_com_velocity


def test_build_humanoid_mpc_helper_returns_wrapper() -> None:
    controller = build_humanoid_mpc()

    assert isinstance(controller, HumanoidMPC)


def test_humanoid_mpc_accepts_footstep_plan_in_step() -> None:
    planner = FootstepPlanner()
    plan = planner.plan_walk(np.array([1.0, 0.0, 0.0]), steps=2)
    controller = HumanoidMPC()

    solution = controller.step(
        current_state=np.array([0.0, 0.0, 0.0, 0.0, 0.9]),
        target_position=np.array([0.1, 0.0]),
        footstep_plan=plan,
    )

    assert solution.optimal_inputs.shape == (controller.config.horizon_steps, 2)
    assert np.isfinite(solution.cost)


__all__ = [name for name in globals() if name.startswith("test_")]
