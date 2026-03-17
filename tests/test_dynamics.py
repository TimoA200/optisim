from __future__ import annotations

from math import isclose
from pathlib import Path

import numpy as np
import pytest

from optisim.cli import main
from optisim.core import TaskDefinition
from optisim.dynamics import (
    ConstraintSet,
    DynamicsValidator,
    JointTorqueLimit,
    PayloadConstraint,
    RigidBodyState,
    WorkspaceConstraint,
    check_joint_torques,
    check_payload,
    check_workspace_bounds,
    compute_inertia_box,
    compute_inertia_cylinder,
    gravitational_force,
    joint_power,
    kinetic_energy,
    potential_energy,
    step_dynamics,
    total_mechanical_energy,
)
from optisim.math3d import vec3
from optisim.robot import build_humanoid_model
from optisim.sim import ExecutionEngine, WorldState
from optisim.sim.world import ObjectState


def test_rigid_body_state_creation() -> None:
    state = RigidBodyState(
        mass=2.0,
        inertia_tensor=np.eye(3),
        position=[1.0, 2.0, 3.0],
        velocity=[0.1, 0.2, 0.3],
        angular_velocity=[0.0, 0.0, 1.0],
    )

    assert state.mass == 2.0
    assert np.allclose(state.position, [1.0, 2.0, 3.0])
    assert np.allclose(state.velocity, [0.1, 0.2, 0.3])


def test_rigid_body_state_rejects_negative_mass() -> None:
    with pytest.raises(ValueError, match="mass must be non-negative"):
        RigidBodyState(mass=-1.0, inertia_tensor=np.eye(3))


def test_step_dynamics_integrates_linear_motion() -> None:
    state = RigidBodyState(
        mass=2.0,
        inertia_tensor=np.eye(3),
        position=[0.0, 0.0, 0.0],
        velocity=[1.0, 0.0, 0.0],
        forces=[4.0, 0.0, 0.0],
    )

    updated = step_dynamics(state, 0.5)

    assert np.allclose(updated.position, [0.5, 0.0, 0.0])
    assert np.allclose(updated.velocity, [2.0, 0.0, 0.0])
    assert np.allclose(updated.forces, [0.0, 0.0, 0.0])


def test_step_dynamics_integrates_angular_motion() -> None:
    state = RigidBodyState(
        mass=1.0,
        inertia_tensor=np.diag([2.0, 2.0, 2.0]),
        angular_velocity=[0.0, 0.0, 1.0],
        torques=[0.0, 0.0, 4.0],
    )

    updated = step_dynamics(state, 0.5)

    assert np.allclose(updated.angular_velocity, [0.0, 0.0, 2.0])


def test_step_dynamics_handles_zero_mass_without_linear_acceleration() -> None:
    state = RigidBodyState(
        mass=0.0,
        inertia_tensor=np.eye(3),
        velocity=[0.0, 1.0, 0.0],
        forces=[10.0, 0.0, 0.0],
    )

    updated = step_dynamics(state, 0.25)

    assert np.allclose(updated.velocity, [0.0, 1.0, 0.0])


def test_compute_inertia_box() -> None:
    inertia = compute_inertia_box(12.0, (2.0, 4.0, 6.0))

    assert np.allclose(np.diag(inertia), [52.0, 40.0, 20.0])


def test_compute_inertia_cylinder() -> None:
    inertia = compute_inertia_cylinder(3.0, radius=2.0, height=4.0)

    assert np.allclose(np.diag(inertia), [7.0, 7.0, 6.0])


def test_gravitational_force_default() -> None:
    assert np.allclose(gravitational_force(2.0), [0.0, 0.0, -19.62])


def test_kinetic_energy_combines_linear_and_rotational_terms() -> None:
    state = RigidBodyState(
        mass=2.0,
        inertia_tensor=np.diag([1.0, 2.0, 3.0]),
        velocity=[3.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 2.0],
    )

    assert isclose(kinetic_energy(state), 15.0)


def test_potential_energy() -> None:
    assert isclose(potential_energy(2.0, 3.0), 58.86)


def test_total_mechanical_energy() -> None:
    state = RigidBodyState(
        mass=1.0,
        inertia_tensor=np.eye(3),
        velocity=[2.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 2.0],
    )

    assert isclose(total_mechanical_energy(state, height=1.0), 13.81)


def test_joint_power() -> None:
    assert isclose(joint_power(4.0, -2.0), -8.0)


def test_check_workspace_bounds_inside_returns_none() -> None:
    bounds = WorkspaceConstraint(bounds_min=[0.0, 0.0, 0.0], bounds_max=[1.0, 1.0, 1.0])

    assert check_workspace_bounds([0.5, 0.5, 0.5], bounds) is None


def test_check_workspace_bounds_on_boundary_returns_none() -> None:
    bounds = WorkspaceConstraint(bounds_min=[0.0, 0.0, 0.0], bounds_max=[1.0, 1.0, 1.0])

    assert check_workspace_bounds([1.0, 0.0, 0.5], bounds) is None


def test_check_workspace_bounds_outside_returns_violation() -> None:
    bounds = WorkspaceConstraint(bounds_min=[0.0, 0.0, 0.0], bounds_max=[1.0, 1.0, 1.0])

    violation = check_workspace_bounds([1.2, 0.5, 0.5], bounds)

    assert violation is not None
    assert violation.constraint_type == "workspace"
    assert isclose(violation.value or 0.0, 0.2)


def test_check_payload_within_limit_returns_none() -> None:
    robot = build_humanoid_model()
    constraint = PayloadConstraint(max_payload_kg=2.0, end_effector="right_palm")

    assert check_payload(robot, 1.5, "right_gripper", constraint) is None


def test_check_payload_exceeds_limit_returns_violation() -> None:
    robot = build_humanoid_model()
    constraint = PayloadConstraint(max_payload_kg=1.0, end_effector="right_palm")

    violation = check_payload(robot, 2.0, "right_gripper", constraint)

    assert violation is not None
    assert violation.constraint_type == "payload"
    assert violation.limit == 1.0


def test_check_payload_unknown_effector_returns_warning_violation() -> None:
    robot = build_humanoid_model()

    violation = check_payload(robot, 1.0, "tool0")

    assert violation is not None
    assert violation.severity == "warning"


def test_check_joint_torques_returns_empty_without_limits() -> None:
    robot = build_humanoid_model()

    violations = check_joint_torques(robot, robot.joint_positions, 2.0)

    assert violations == []


def test_check_joint_torques_detects_limit_violation() -> None:
    robot = build_humanoid_model()
    limits = [JointTorqueLimit("right_shoulder_pitch", 1.0)]

    violations = check_joint_torques(
        robot,
        robot.joint_positions,
        5.0,
        torque_limits=limits,
        end_effector="right_palm",
    )

    assert len(violations) == 1
    assert violations[0].joint_name == "right_shoulder_pitch"


def test_check_joint_torques_respects_limit() -> None:
    robot = build_humanoid_model()
    limits = [JointTorqueLimit("right_shoulder_pitch", 1000.0)]

    violations = check_joint_torques(
        robot,
        robot.joint_positions,
        0.5,
        torque_limits=limits,
        end_effector="right_palm",
    )

    assert violations == []


def test_check_joint_torques_accepts_object_state_payload() -> None:
    robot = build_humanoid_model()
    payload = ObjectState(
        name="crate",
        pose=robot.base_pose,
        size=(0.1, 0.1, 0.1),
        mass_kg=4.0,
    )
    limits = [JointTorqueLimit("right_shoulder_pitch", 1.0)]

    violations = check_joint_torques(
        robot,
        robot.joint_positions,
        payload,
        torque_limits=limits,
        end_effector="right_palm",
    )

    assert violations


def test_dynamics_validator_feasible_task() -> None:
    task = TaskDefinition.from_file(Path("examples/pick_and_place.yaml"))
    robot = build_humanoid_model()
    world = WorldState.from_dict(task.world)
    validator = DynamicsValidator()
    constraints = ConstraintSet(
        payload_constraints=[PayloadConstraint(max_payload_kg=5.0, end_effector="right_palm")],
        workspace_constraints=[
            WorkspaceConstraint(bounds_min=[-1.0, -1.0, 0.0], bounds_max=[2.0, 2.0, 2.0])
        ],
    )

    report = validator.validate_task(task, robot, world, constraints)

    assert report.feasible
    assert report.violations == []
    assert report.energy_profile.total_energy > 0.0
    assert "move box" in report.energy_profile.energy_per_action


def test_dynamics_validator_detects_payload_violation() -> None:
    task = TaskDefinition.from_file(Path("examples/pick_and_place.yaml"))
    robot = build_humanoid_model()
    world = WorldState.from_dict(task.world)
    validator = DynamicsValidator()
    constraints = ConstraintSet(
        payload_constraints=[PayloadConstraint(max_payload_kg=0.5, end_effector="right_palm")]
    )

    report = validator.validate_task(task, robot, world, constraints)

    assert not report.feasible
    assert any(item.constraint_type == "payload" for item in report.violations)


def test_dynamics_validator_detects_workspace_violation() -> None:
    task = TaskDefinition.from_file(Path("examples/pick_and_place.yaml"))
    robot = build_humanoid_model()
    world = WorldState.from_dict(task.world)
    validator = DynamicsValidator()
    constraints = ConstraintSet(
        workspace_constraints=[
            WorkspaceConstraint(bounds_min=[0.45, -0.05, 0.8], bounds_max=[0.5, 0.05, 0.9])
        ]
    )

    report = validator.validate_task(task, robot, world, constraints)

    assert not report.feasible
    assert any(item.constraint_type == "workspace" for item in report.violations)


def test_dynamics_validator_detects_torque_violation() -> None:
    task = TaskDefinition.from_file(Path("examples/pick_and_place.yaml"))
    robot = build_humanoid_model()
    world = WorldState.from_dict(task.world)
    validator = DynamicsValidator()
    constraints = ConstraintSet(
        joint_torque_limits=[JointTorqueLimit("right_shoulder_pitch", 0.5)]
    )

    report = validator.validate_task(task, robot, world, constraints)

    assert not report.feasible
    assert any(item.constraint_type == "joint_torque" for item in report.violations)


def test_dynamics_validator_marks_semantically_invalid_task_infeasible() -> None:
    task = TaskDefinition.from_dict(
        {
            "name": "invalid",
            "actions": [{"type": "move", "target": "box", "destination": [0.1, 0.0, 1.0]}],
        }
    )
    validator = DynamicsValidator()

    report = validator.validate_task(
        task,
        build_humanoid_model(),
        WorldState.with_defaults(),
        ConstraintSet(),
    )

    assert not report.feasible
    assert any(item.constraint_type == "task_validation" for item in report.violations)


def test_execution_engine_validate_dynamics() -> None:
    task = TaskDefinition.from_file(Path("examples/pick_and_place.yaml"))
    engine = ExecutionEngine(robot=build_humanoid_model(), world=WorldState.from_dict(task.world))

    report = engine.validate_dynamics(
        task,
        ConstraintSet(
            payload_constraints=[PayloadConstraint(max_payload_kg=5.0, end_effector="right_palm")]
        ),
    )

    assert report.feasible


def test_dynamics_validator_default_torque_limits_cover_robot_joints() -> None:
    robot = build_humanoid_model()

    limits = DynamicsValidator.default_torque_limits(robot)

    assert len(limits) == len(robot.joints)
    assert all(limit.max_torque >= 10.0 for limit in limits)


def test_cli_validate_dynamics_returns_success(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "validate-dynamics",
            "examples/pick_and_place.yaml",
            "--max-payload",
            "5.0",
            "--check-torques",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "feasible" in captured.out


def test_cli_validate_dynamics_returns_failure_for_low_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "validate-dynamics",
            "examples/pick_and_place.yaml",
            "--max-payload",
            "0.5",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "payload" in captured.out

__all__ = ["test_rigid_body_state_creation", "test_rigid_body_state_rejects_negative_mass", "test_step_dynamics_integrates_linear_motion", "test_step_dynamics_integrates_angular_motion", "test_step_dynamics_handles_zero_mass_without_linear_acceleration", "test_compute_inertia_box", "test_compute_inertia_cylinder", "test_gravitational_force_default", "test_kinetic_energy_combines_linear_and_rotational_terms", "test_potential_energy", "test_total_mechanical_energy", "test_joint_power", "test_check_workspace_bounds_inside_returns_none", "test_check_workspace_bounds_on_boundary_returns_none", "test_check_workspace_bounds_outside_returns_violation", "test_check_payload_within_limit_returns_none", "test_check_payload_exceeds_limit_returns_violation", "test_check_payload_unknown_effector_returns_warning_violation", "test_check_joint_torques_returns_empty_without_limits", "test_check_joint_torques_detects_limit_violation", "test_check_joint_torques_respects_limit", "test_check_joint_torques_accepts_object_state_payload", "test_dynamics_validator_feasible_task", "test_dynamics_validator_detects_payload_violation", "test_dynamics_validator_detects_workspace_violation", "test_dynamics_validator_detects_torque_violation", "test_dynamics_validator_marks_semantically_invalid_task_infeasible", "test_execution_engine_validate_dynamics", "test_dynamics_validator_default_torque_limits_cover_robot_joints", "test_cli_validate_dynamics_returns_success", "test_cli_validate_dynamics_returns_failure_for_low_payload"]
