from __future__ import annotations

from math import pi

import pytest

from optisim import (
    ContactPhase,
    ManipulationFSM,
    ReactiveConfig,
    ReactiveController,
    ReactiveExecutionResult,
    ReactiveState,
    RobotModel,
    WorldState,
    run_reactive_manipulation,
)
from optisim.robot import JointSpec, LinkSpec
from optisim.wbc import PostureTask, WBCController


class ScriptedSensorSuite:
    def __init__(self, readings: list[dict[str, object]]) -> None:
        self._readings = list(readings)
        self._index = 0

    def read(self) -> dict[str, object]:
        if self._index < len(self._readings):
            reading = self._readings[self._index]
            self._index += 1
            return reading
        return self._readings[-1] if self._readings else {}


def test_reactive_config_default_values() -> None:
    config = ReactiveConfig()

    assert config.control_freq_hz == pytest.approx(50.0)
    assert config.safety_stop_on_ft_threshold == pytest.approx(50.0)
    assert config.proximity_slowdown_distance == pytest.approx(0.15)
    assert config.proximity_stop_distance == pytest.approx(0.05)
    assert config.max_step_count == 500
    assert config.verbose is False


def test_contact_phase_enum_has_expected_values() -> None:
    assert [phase.name for phase in ContactPhase] == [
        "PRE_APPROACH",
        "APPROACH",
        "CONTACT",
        "GRASP",
        "LIFT",
        "TRANSPORT",
        "PLACE",
        "RETRACT",
        "DONE",
    ]


def test_reactive_state_construction() -> None:
    state = ReactiveState(
        phase=ContactPhase.CONTACT,
        step=3,
        ft_reading=2.5,
        proximity_reading=0.12,
        velocity_scale=0.3,
        triggered_safety_stop=False,
        phase_history=[ContactPhase.PRE_APPROACH, ContactPhase.APPROACH, ContactPhase.CONTACT],
    )

    assert state.phase is ContactPhase.CONTACT
    assert state.step == 3
    assert state.ft_reading == pytest.approx(2.5)
    assert state.proximity_reading == pytest.approx(0.12)
    assert state.velocity_scale == pytest.approx(0.3)
    assert state.phase_history[-1] is ContactPhase.CONTACT


def test_manipulation_fsm_reset_returns_pre_approach() -> None:
    state = ManipulationFSM(ReactiveConfig()).reset()

    assert state.phase is ContactPhase.PRE_APPROACH
    assert state.step == 0
    assert state.phase_history == [ContactPhase.PRE_APPROACH]


def test_fsm_transition_pre_approach_to_approach() -> None:
    fsm = ManipulationFSM(ReactiveConfig())
    state = fsm.reset()

    next_phase = fsm.transition(state, {"at_target": False})

    assert next_phase is ContactPhase.APPROACH


def test_fsm_transition_approach_to_contact_on_touch() -> None:
    fsm = ManipulationFSM(ReactiveConfig())
    state = ReactiveState(ContactPhase.APPROACH, 1, 0.0, 0.3, 1.0, False, [ContactPhase.PRE_APPROACH, ContactPhase.APPROACH])

    next_phase = fsm.transition(state, {"ft_magnitude": 1.2, "proximity_min": 0.2})

    assert next_phase is ContactPhase.CONTACT


def test_contact_to_grasp_transition() -> None:
    fsm = ManipulationFSM(ReactiveConfig())
    state = ReactiveState(ContactPhase.CONTACT, 2, 1.5, 0.08, 0.3, False, [ContactPhase.CONTACT])

    next_phase = fsm.transition(state, {"gripper_closed": True})

    assert next_phase is ContactPhase.GRASP


def test_grasp_to_lift_to_transport_transition_chain() -> None:
    fsm = ManipulationFSM(ReactiveConfig())
    grasp_state = ReactiveState(ContactPhase.GRASP, 3, 1.5, 0.08, 0.3, False, [ContactPhase.GRASP])

    lift_phase = fsm.transition(grasp_state, {"object_held": True})
    lift_state = ReactiveState(lift_phase, 4, 5.0, 0.3, 1.0, False, [ContactPhase.GRASP, ContactPhase.LIFT])
    transport_phase = fsm.transition(lift_state, {"object_held": True, "ft_magnitude": 4.0})

    assert lift_phase is ContactPhase.LIFT
    assert transport_phase is ContactPhase.TRANSPORT


def test_velocity_scale_reduces_near_obstacles() -> None:
    fsm = ManipulationFSM(ReactiveConfig())
    state = ReactiveState(ContactPhase.APPROACH, 1, 0.0, 0.3, 1.0, False, [ContactPhase.APPROACH])

    updated = fsm.step(state, {"ft_magnitude": 0.0, "proximity_min": 0.1})

    assert updated.velocity_scale == pytest.approx(0.3)


def test_safety_stop_triggers_on_high_ft() -> None:
    fsm = ManipulationFSM(ReactiveConfig())
    state = ReactiveState(ContactPhase.APPROACH, 1, 0.0, 0.3, 1.0, False, [ContactPhase.APPROACH])

    updated = fsm.step(state, {"ft_magnitude": 55.0, "proximity_min": 0.2})

    assert updated.triggered_safety_stop is True
    assert updated.velocity_scale == pytest.approx(0.0)


def test_reactive_controller_run_step_runs_without_error() -> None:
    robot = _build_test_robot()
    world = WorldState()
    controller = ReactiveController()
    controller.attach_sensor_suite(ScriptedSensorSuite([{"ft_magnitude": 0.0, "proximity_min": 0.5}]))
    controller.attach_wbc(WBCController([PostureTask({"joint1": 0.2})]))

    state = controller.run_step(robot, world, dt=0.1)

    assert state.step == 1
    assert state.phase is ContactPhase.APPROACH
    assert robot.joint_positions["joint1"] > 0.0


def test_reactive_controller_run_loop_returns_list_of_states() -> None:
    robot = _build_test_robot()
    world = WorldState()
    controller = ReactiveController()
    controller.attach_sensor_suite(ScriptedSensorSuite(_completion_readings()))

    states = controller.run_loop(robot, world, max_steps=20)

    assert isinstance(states, list)
    assert states
    assert states[-1].phase is ContactPhase.DONE


def test_run_reactive_manipulation_helper_runs_end_to_end() -> None:
    robot = _build_test_robot()
    world = WorldState()

    result = run_reactive_manipulation(robot, world, config=ReactiveConfig(max_step_count=3))

    assert isinstance(result, ReactiveExecutionResult)
    assert result.steps_taken == 3
    assert len(result.states) == 3
    assert result.final_phase is ContactPhase.APPROACH


def test_reactive_module_public_exports_are_available_from_optisim() -> None:
    import optisim

    assert "ReactiveController" in optisim.__all__
    assert "run_reactive_manipulation" in optisim.__all__


def _completion_readings() -> list[dict[str, object]]:
    return [
        {"ft_magnitude": 0.0, "proximity_min": 0.5, "at_target": False},
        {"ft_magnitude": 2.0, "proximity_min": 0.2},
        {"gripper_closed": True, "proximity_min": 0.2},
        {"object_held": True, "ft_magnitude": 2.0, "proximity_min": 0.2},
        {"object_held": True, "ft_magnitude": 0.0, "proximity_min": 0.2},
        {"object_held": True, "at_target": True, "proximity_min": 0.2},
        {"object_held": False, "at_target": True, "proximity_min": 0.2},
        {"proximity_min": 0.2},
        {"proximity_min": 0.2},
        {"proximity_min": 0.2},
        {"proximity_min": 0.2},
        {"proximity_min": 0.2},
        {"proximity_min": 0.2},
        {"proximity_min": 0.2},
        {"proximity_min": 0.2},
        {"proximity_min": 0.2},
        {"proximity_min": 0.2},
    ]


def _build_test_robot() -> RobotModel:
    links = {
        "base": LinkSpec("base"),
        "link1": LinkSpec("link1", parent_joint="joint1"),
        "link2": LinkSpec("link2", parent_joint="joint2"),
        "tool": LinkSpec("tool", parent_joint="joint3"),
    }
    joints = {
        "joint1": JointSpec(
            name="joint1",
            parent="base",
            child="link1",
            joint_type="revolute",
            limit_lower=-pi,
            limit_upper=pi,
            velocity_limit=2.0,
            dh_a=0.45,
        ),
        "joint2": JointSpec(
            name="joint2",
            parent="link1",
            child="link2",
            joint_type="revolute",
            limit_lower=-pi / 2.0,
            limit_upper=pi / 2.0,
            velocity_limit=2.0,
            dh_a=0.35,
        ),
        "joint3": JointSpec(
            name="joint3",
            parent="link2",
            child="tool",
            joint_type="revolute",
            limit_lower=-pi / 2.0,
            limit_upper=pi / 2.0,
            velocity_limit=2.0,
            dh_a=0.2,
        ),
    }
    return RobotModel(
        name="test_reactive_robot",
        links=links,
        joints=joints,
        root_link="base",
        end_effectors={"tool": "tool"},
    )


__all__ = [
    "test_reactive_config_default_values",
    "test_contact_phase_enum_has_expected_values",
    "test_reactive_state_construction",
    "test_manipulation_fsm_reset_returns_pre_approach",
    "test_fsm_transition_pre_approach_to_approach",
    "test_fsm_transition_approach_to_contact_on_touch",
    "test_contact_to_grasp_transition",
    "test_grasp_to_lift_to_transport_transition_chain",
    "test_velocity_scale_reduces_near_obstacles",
    "test_safety_stop_triggers_on_high_ft",
    "test_reactive_controller_run_step_runs_without_error",
    "test_reactive_controller_run_loop_returns_list_of_states",
    "test_run_reactive_manipulation_helper_runs_end_to_end",
    "test_reactive_module_public_exports_are_available_from_optisim",
]
