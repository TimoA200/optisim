"""Minimal humanoid MPC balance example."""

from __future__ import annotations

import numpy as np

from optisim.mpc import FootstepPlanner, build_humanoid_mpc


def main() -> None:
    controller = build_humanoid_mpc()
    planner = FootstepPlanner()
    plan = planner.plan_walk(direction=np.array([1.0, 0.0, 0.0]), steps=4)
    state = np.asarray([0.0, 0.0, 0.0, 0.0, controller.config.com_height_z], dtype=np.float64)

    print("planned footsteps:")
    for index, (left, right, timing) in enumerate(
        zip(plan.left_foot_positions, plan.right_foot_positions, plan.timing, strict=False)
    ):
        print(f"  step {index}: t={timing:.2f}s left={left} right={right}")

    print("\nmpc rollout:")
    for step in range(10):
        solution = controller.step(
            current_state=state,
            target_position=np.array([0.08, 0.0], dtype=np.float64),
            target_velocity=np.array([0.0, 0.0], dtype=np.float64),
            footstep_plan=plan,
        )
        state = solution.optimal_states[0]
        print(f"step {step}: state={state} zmp={solution.optimal_inputs[0]}")


if __name__ == "__main__":
    main()
