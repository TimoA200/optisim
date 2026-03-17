from __future__ import annotations

import numpy as np

from optisim.bimanual import BimanualCoordinator, CooperativeManipulation, TaskPresets


def main() -> None:
    task = TaskPresets.pick_large_box()
    coordinator = BimanualCoordinator()
    plan = coordinator.plan(task)

    print(f"task={task.task_name} steps={min(5, len(plan.left_trajectory))}")
    for step_idx in range(min(5, len(plan.left_trajectory))):
        left, right = coordinator.execute_step(plan, step_idx)
        print(f"step {step_idx}: left={left.position.round(3)} right={right.position.round(3)}")

    cooperative = CooperativeManipulation(object_mass=4.0)
    total_wrench = np.asarray([0.0, 0.0, 39.24, 0.0, 0.0, 0.0], dtype=np.float64)
    left_wrench, right_wrench = cooperative.distribute_wrench(
        total_wrench,
        plan.left_trajectory[0],
        plan.right_trajectory[0],
    )
    print(f"left_wrench={left_wrench.round(3)}")
    print(f"right_wrench={right_wrench.round(3)}")

    violation = coordinator.check_constraint_violation(
        plan.left_trajectory[0],
        plan.right_trajectory[0],
        plan.constraint,
    )
    print(f"constraint_violation={violation:.6f}")


if __name__ == "__main__":
    main()
