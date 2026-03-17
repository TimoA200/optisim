"""Run a reach task on a bundled URDF robot and animate it in the terminal."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optisim.core import ActionPrimitive, TaskDefinition
from optisim.robot import IKOptions, load_urdf, solve_inverse_kinematics
from optisim.sim import WorldState
from optisim.viz import TerminalVisualizer

__all__ = ["main"]


def main() -> None:
    """Load a bundled URDF arm, solve a reach target, and animate the result."""

    robot_path = Path(__file__).resolve().parent / "robots" / "rrbot.urdf"
    robot = load_urdf(robot_path)
    effector = next(iter(robot.end_effectors))

    target_joints = {"joint1": 0.7, "joint2": -1.05}
    target_pose = robot.end_effector_pose(effector, target_joints)
    result = solve_inverse_kinematics(
        robot,
        effector,
        target_pose,
        options=IKOptions(max_iterations=120, convergence_threshold=1e-4, damping=0.05, position_only=True),
    )
    if not result.success:
        raise RuntimeError(result.failure_reason or "URDF reach demo failed to converge")

    task = TaskDefinition(
        name="urdf_reach_demo",
        actions=[ActionPrimitive.reach(target="pose_target", end_effector=effector, pose=target_pose)],
        metadata={"urdf": str(robot_path.name)},
    )
    world = WorldState()
    visualizer = TerminalVisualizer(width=60, height=18)
    visualizer.start_task(task, world, robot)
    visualizer.start_action(task.actions[0], index=1, total_actions=1)

    start_positions = dict(robot.joint_positions)
    steps = 36
    for step_index in range(1, steps + 1):
        alpha = step_index / steps
        robot.set_joint_positions(
            {
                joint_name: start_positions.get(joint_name, 0.0)
                + (target_value - start_positions.get(joint_name, 0.0)) * alpha
                for joint_name, target_value in result.joint_positions.items()
            }
        )
        world.time_s += 0.05
        visualizer.render(world, robot)
        time.sleep(0.04)

    visualizer.finish(task, world, robot, collisions=[])
    solved_pose = robot.end_effector_pose(effector)
    print(f"Loaded URDF: {robot_path}")
    print(f"End effector: {effector}")
    print(f"IK iterations: {result.iterations}")
    print(
        "Solved pose: "
        f"x={solved_pose.position[0]:+.3f} "
        f"y={solved_pose.position[1]:+.3f} "
        f"z={solved_pose.position[2]:+.3f}"
    )


if __name__ == "__main__":
    main()
