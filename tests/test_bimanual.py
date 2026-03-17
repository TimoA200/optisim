from __future__ import annotations

import numpy as np
import pytest

from optisim import BimanualCoordinator, BimanualTask, CooperativeManipulation, TaskPresets
from optisim.bimanual import BimanualConstraint, BimanualPlan, GraspFrame


def _relative_transform(offset: tuple[float, float, float] = (0.0, 0.4, 0.0)) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(offset, dtype=np.float64)
    return transform


def _task() -> BimanualTask:
    return BimanualTask(
        left_waypoints=[
            np.asarray([0.4, -0.2, 0.9], dtype=np.float64),
            np.asarray([0.6, -0.2, 1.0], dtype=np.float64),
        ],
        right_waypoints=[
            np.asarray([0.4, 0.2, 0.9], dtype=np.float64),
            np.asarray([0.6, 0.2, 1.0], dtype=np.float64),
            np.asarray([0.8, 0.2, 1.1], dtype=np.float64),
        ],
        constraint=BimanualConstraint("rigid", _relative_transform()),
        task_name="test_task",
    )


def test_grasp_frame_construction() -> None:
    frame = GraspFrame(hand="left", position=[0.1, 0.2, 0.3], orientation=[2.0, 0.0, 0.0, 0.0])
    assert frame.hand == "left"
    assert np.allclose(frame.position, [0.1, 0.2, 0.3])
    assert np.allclose(frame.orientation, [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(frame.contact_normal, [0.0, 0.0, 1.0])


def test_bimanual_constraint_defaults() -> None:
    constraint = BimanualConstraint("rigid", _relative_transform())
    assert constraint.compliance == 0.0
    assert constraint.max_force_error == 10.0
    assert constraint.relative_transform.shape == (4, 4)


def test_bimanual_task_construction() -> None:
    task = _task()
    assert task.task_name == "test_task"
    assert len(task.left_waypoints) == 2
    assert len(task.right_waypoints) == 3


def test_bimanual_coordinator_initialization() -> None:
    coordinator = BimanualCoordinator(robot_model="demo")
    assert coordinator.robot_model == "demo"


def test_bimanual_coordinator_plan_returns_bimanual_plan() -> None:
    plan = BimanualCoordinator().plan(_task())
    assert isinstance(plan, BimanualPlan)


def test_bimanual_plan_has_synchronized_trajectories() -> None:
    plan = BimanualCoordinator().plan(_task())
    assert len(plan.left_trajectory) == len(plan.right_trajectory) == len(plan.timestamps)
    assert plan.is_synchronized is True


def test_execute_step_returns_valid_grasp_frame_pair() -> None:
    plan = BimanualCoordinator().plan(_task())
    left, right = BimanualCoordinator().execute_step(plan, 1)
    assert isinstance(left, GraspFrame)
    assert isinstance(right, GraspFrame)
    assert left.hand == "left"
    assert right.hand == "right"


def test_check_constraint_violation_returns_zero_for_perfect_constraint() -> None:
    constraint = BimanualConstraint("rigid", _relative_transform())
    left = GraspFrame(hand="left", position=[0.4, -0.2, 0.9], orientation=[1.0, 0.0, 0.0, 0.0])
    right = GraspFrame(hand="right", position=[0.4, 0.2, 0.9], orientation=[1.0, 0.0, 0.0, 0.0])
    violation = BimanualCoordinator().check_constraint_violation(left, right, constraint)
    assert violation == pytest.approx(0.0)


def test_check_constraint_violation_returns_positive_for_violation() -> None:
    constraint = BimanualConstraint("rigid", _relative_transform())
    left = GraspFrame(hand="left", position=[0.4, -0.2, 0.9], orientation=[1.0, 0.0, 0.0, 0.0])
    right = GraspFrame(hand="right", position=[0.5, 0.4, 0.9], orientation=[1.0, 0.0, 0.0, 0.0])
    violation = BimanualCoordinator().check_constraint_violation(left, right, constraint)
    assert violation > 0.0


def test_cooperative_manipulation_initialization() -> None:
    coop = CooperativeManipulation(object_mass=2.5)
    assert coop.object_mass == 2.5
    assert np.allclose(coop.object_inertia, np.eye(3))


def test_distribute_wrench_returns_two_wrenches() -> None:
    coop = CooperativeManipulation()
    left, right = coop.distribute_wrench(
        np.asarray([2.0, 0.0, 4.0, 0.0, 1.0, 0.0], dtype=np.float64),
        GraspFrame(hand="left", position=[0.5, -0.2, 0.9], orientation=[1.0, 0.0, 0.0, 0.0]),
        GraspFrame(hand="right", position=[0.5, 0.2, 0.9], orientation=[1.0, 0.0, 0.0, 0.0]),
    )
    assert left.shape == (6,)
    assert right.shape == (6,)


def test_wrenches_sum_to_total_wrench() -> None:
    coop = CooperativeManipulation()
    total = np.asarray([2.0, 1.0, 4.0, 0.2, 1.0, 0.4], dtype=np.float64)
    left, right = coop.distribute_wrench(
        total,
        GraspFrame(hand="left", position=[0.5, -0.2, 0.9], orientation=[1.0, 0.0, 0.0, 0.0]),
        GraspFrame(hand="right", position=[0.5, 0.2, 0.9], orientation=[1.0, 0.0, 0.0, 0.0]),
    )
    assert np.allclose(left + right, total)


def test_compute_internal_force_returns_non_negative_float() -> None:
    coop = CooperativeManipulation(object_mass=2.0)
    force = coop.compute_internal_force(
        GraspFrame(
            hand="left",
            position=[0.5, -0.2, 0.9],
            orientation=[1.0, 0.0, 0.0, 0.0],
            contact_normal=[0.0, 1.0, 0.0],
        ),
        GraspFrame(
            hand="right",
            position=[0.5, 0.2, 0.9],
            orientation=[1.0, 0.0, 0.0, 0.0],
            contact_normal=[0.0, -1.0, 0.0],
        ),
    )
    assert isinstance(force, float)
    assert force >= 0.0


def test_is_stable_grasp_returns_bool() -> None:
    stable = CooperativeManipulation().is_stable_grasp(
        GraspFrame(hand="left", position=[0.5, -0.2, 0.9], orientation=[1.0, 0.0, 0.0, 0.0]),
        GraspFrame(hand="right", position=[0.5, 0.2, 0.9], orientation=[1.0, 0.0, 0.0, 0.0]),
        np.asarray([2.0, 0.0, 8.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    assert isinstance(stable, bool)


def test_task_presets_pick_large_box_returns_valid_bimanual_task() -> None:
    task = TaskPresets.pick_large_box()
    assert isinstance(task, BimanualTask)
    assert task.task_name == "pick_large_box"
    assert task.object_pose is not None


def test_task_presets_carry_tray_returns_valid_bimanual_task() -> None:
    task = TaskPresets.carry_tray()
    assert isinstance(task, BimanualTask)
    assert task.task_name == "carry_tray"
    assert len(task.left_waypoints) == len(task.right_waypoints)


def test_task_presets_handoff_returns_valid_bimanual_task() -> None:
    task = TaskPresets.handoff()
    assert isinstance(task, BimanualTask)
    assert "handoff" in task.task_name


def test_task_presets_assembly_insert_returns_valid_bimanual_task() -> None:
    task = TaskPresets.assembly_insert()
    assert isinstance(task, BimanualTask)
    assert task.constraint.constraint_type == "relative_pose"


def test_top_level_exports_include_bimanual_types() -> None:
    coordinator = BimanualCoordinator()
    task = TaskPresets.carry_tray()
    plan = coordinator.plan(task)
    assert isinstance(task, BimanualTask)
    assert isinstance(plan, BimanualPlan)


def test_full_pipeline_plan_execute_five_steps_and_check_constraints() -> None:
    task = TaskPresets.carry_tray(
        waypoints=[
            np.asarray([0.45, 0.0, 0.92], dtype=np.float64),
            np.asarray([0.55, 0.0, 0.95], dtype=np.float64),
            np.asarray([0.65, 0.0, 0.98], dtype=np.float64),
            np.asarray([0.75, 0.0, 1.00], dtype=np.float64),
            np.asarray([0.85, 0.0, 1.02], dtype=np.float64),
        ]
    )
    coordinator = BimanualCoordinator()
    plan = coordinator.plan(task)

    assert len(plan.left_trajectory) == 5
    for step_idx in range(5):
        left, right = coordinator.execute_step(plan, step_idx)
        violation = coordinator.check_constraint_violation(left, right, task.constraint)
        assert violation == pytest.approx(0.0)
