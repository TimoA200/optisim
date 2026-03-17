from __future__ import annotations

from math import pi

import numpy as np

from optisim.math3d import Pose
from optisim.robot import JointSpec, LinkSpec, RobotModel, build_humanoid_model
from optisim.wbc import (
    BalanceTask,
    EndEffectorTask,
    JointLimitTask,
    PostureTask,
    WBCController,
    WBCSolution,
    WBCTask,
    build_wbc_controller,
)


def test_posture_task_converges_to_desired_joint_positions() -> None:
    robot = _build_test_robot()
    controller = WBCController([PostureTask({"joint1": 0.4, "joint2": -0.3, "joint3": 0.2})])

    result = controller.solve(robot, dt=0.1, max_iterations=40, tolerance=1e-3)

    assert result.converged
    assert abs(robot.joint_positions["joint1"] - 0.4) < 1e-3
    assert abs(robot.joint_positions["joint2"] + 0.3) < 1e-3
    assert abs(robot.joint_positions["joint3"] - 0.2) < 1e-3


def test_end_effector_task_reduces_position_error_over_iterations() -> None:
    robot = _build_test_robot()
    target_pose = robot.end_effector_pose("tool", {"joint1": 0.35, "joint2": -0.45, "joint3": 0.25})
    task = EndEffectorTask("tool", target_pose, position_only=True)
    controller = WBCController([task])

    initial_error = np.linalg.norm(task.compute_jacobian_and_error(robot)[1])
    controller.solve(robot, dt=0.1, max_iterations=30, tolerance=1e-4)
    final_error = np.linalg.norm(task.compute_jacobian_and_error(robot)[1])

    assert final_error < initial_error


def test_wbc_controller_stacks_multiple_tasks_without_crashing() -> None:
    robot = _build_test_robot()
    target_pose = robot.end_effector_pose("tool", {"joint1": 0.2, "joint2": -0.2, "joint3": 0.1})
    controller = WBCController(
        [
            EndEffectorTask("tool", target_pose, position_only=True, priority=0),
            PostureTask({"joint3": 0.1}, priority=1),
            JointLimitTask(priority=2),
        ]
    )

    velocities = controller.compute_joint_velocities(robot, dt=0.1)

    assert set(velocities) == set(robot.joints)
    assert np.isfinite(np.asarray(list(velocities.values()), dtype=np.float64)).all()


def test_nullspace_projection_preserves_higher_priority_task_when_compatible() -> None:
    robot = _build_test_robot()
    high_priority = _SingleJointTask("joint1", 0.15, priority=0)
    low_priority = _SingleJointTask("joint2", -0.1, priority=1)
    controller = WBCController(
        [
            high_priority,
            low_priority,
        ]
    )

    result = controller.solve(robot, dt=0.1, max_iterations=20, tolerance=1e-4)

    assert result.converged
    assert abs(robot.joint_positions["joint1"] - 0.15) < 1e-4
    assert abs(robot.joint_positions["joint2"] + 0.1) < 1e-4


def test_joint_velocity_clamping() -> None:
    robot = _build_test_robot(velocity_limit=0.05)
    controller = WBCController([PostureTask({"joint1": 1.0, "joint2": -1.0, "joint3": 0.8})])

    velocities = controller.compute_joint_velocities(robot, dt=0.1)

    assert all(abs(value) <= 0.05 + 1e-9 for value in velocities.values())


def test_wbc_solution_dataclass_fields() -> None:
    solution = WBCSolution(
        iterations=3,
        converged=True,
        task_errors={"PostureTask": 1e-5},
        joint_positions={"joint1": 0.1},
    )

    assert solution.iterations == 3
    assert solution.converged is True
    assert solution.task_errors["PostureTask"] == 1e-5
    assert solution.joint_positions["joint1"] == 0.1


def test_balance_task_runs_without_error() -> None:
    robot = build_humanoid_model()
    task = BalanceTask()

    jacobian, error = task.compute_jacobian_and_error(robot)

    assert jacobian.shape == (3, len(robot.joints))
    assert error.shape == (3,)
    assert np.isfinite(jacobian).all()
    assert np.isfinite(error).all()


def test_joint_limit_task_pushes_joints_away_from_limits() -> None:
    robot = _build_test_robot()
    upper = robot.joints["joint1"].limit_upper
    robot.set_joint_positions({"joint1": upper - 0.01})
    controller = WBCController([JointLimitTask()])

    velocities = controller.compute_joint_velocities(robot, dt=0.1)

    assert velocities["joint1"] < 0.0


def test_solve_returns_converged_true_for_reachable_targets() -> None:
    robot = _build_test_robot()
    target_pose = robot.end_effector_pose("tool", {"joint1": 0.25, "joint2": -0.2, "joint3": 0.15})
    controller = WBCController([EndEffectorTask("tool", target_pose, position_only=True)])

    result = controller.solve(robot, dt=0.1, max_iterations=80, tolerance=1e-3)

    assert result.converged
    assert result.iterations <= 80
    assert result.task_errors["EndEffectorTask[tool]"] < 1e-3


def test_step_modifies_robot_joint_positions() -> None:
    robot = _build_test_robot()
    controller = WBCController([PostureTask({"joint1": 0.2})])

    before = dict(robot.joint_positions)
    after = controller.step(robot, dt=0.1)

    assert after["joint1"] != before["joint1"]


def test_build_wbc_controller_helper() -> None:
    task = PostureTask({"joint1": 0.1})

    controller = build_wbc_controller([task])

    assert isinstance(controller, WBCController)
    assert controller.tasks == [task]


def test_adding_tasks_by_priority_sorts_them_correctly() -> None:
    controller = WBCController()
    low = PostureTask({"joint1": 0.1}, priority=5)
    high = JointLimitTask(priority=1)
    mid = BalanceTask(priority=3)

    controller.add_task(low)
    controller.add_task(high)
    controller.add_task(mid)

    assert controller.tasks == [high, mid, low]


def test_end_effector_task_returns_full_pose_error_when_requested() -> None:
    robot = _build_test_robot()
    target_pose = Pose.from_xyz_rpy([0.8, 0.2, 0.0], [0.0, 0.0, 0.5])
    task = EndEffectorTask("tool", target_pose, position_only=False)

    jacobian, error = task.compute_jacobian_and_error(robot)

    assert jacobian.shape == (6, len(robot.joints))
    assert error.shape == (6,)


def _build_test_robot(velocity_limit: float = 2.0) -> RobotModel:
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
            velocity_limit=velocity_limit,
            dh_a=0.45,
        ),
        "joint2": JointSpec(
            name="joint2",
            parent="link1",
            child="link2",
            joint_type="revolute",
            limit_lower=-pi / 2.0,
            limit_upper=pi / 2.0,
            velocity_limit=velocity_limit,
            dh_a=0.35,
        ),
        "joint3": JointSpec(
            name="joint3",
            parent="link2",
            child="tool",
            joint_type="revolute",
            limit_lower=-pi / 2.0,
            limit_upper=pi / 2.0,
            velocity_limit=velocity_limit,
            dh_a=0.2,
        ),
    }
    return RobotModel(
        name="test_wbc_robot",
        links=links,
        joints=joints,
        root_link="base",
        end_effectors={"tool": "tool"},
    )


class _SingleJointTask(WBCTask):
    def __init__(self, joint_name: str, target: float, priority: int) -> None:
        super().__init__(priority=priority, weight=1.0)
        self.joint_name = joint_name
        self.target = target

    def compute_jacobian_and_error(self, robot: RobotModel) -> tuple[np.ndarray, np.ndarray]:
        joint_names = list(robot.joints)
        jacobian = np.zeros((1, len(joint_names)), dtype=np.float64)
        jacobian[0, joint_names.index(self.joint_name)] = 1.0
        error = np.asarray([self.target - robot.joint_positions[self.joint_name]], dtype=np.float64)
        return jacobian, error


__all__ = [
    "test_posture_task_converges_to_desired_joint_positions",
    "test_end_effector_task_reduces_position_error_over_iterations",
    "test_wbc_controller_stacks_multiple_tasks_without_crashing",
    "test_nullspace_projection_preserves_higher_priority_task_when_compatible",
    "test_joint_velocity_clamping",
    "test_wbc_solution_dataclass_fields",
    "test_balance_task_runs_without_error",
    "test_joint_limit_task_pushes_joints_away_from_limits",
    "test_solve_returns_converged_true_for_reachable_targets",
    "test_step_modifies_robot_joint_positions",
    "test_build_wbc_controller_helper",
    "test_adding_tasks_by_priority_sorts_them_correctly",
    "test_end_effector_task_returns_full_pose_error_when_requested",
]
