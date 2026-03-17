from __future__ import annotations

import math

from optisim.lfd import DemonstrationRecorder, DynamicMovementPrimitive


def main() -> None:
    recorder = DemonstrationRecorder(
        task_name="reach_target",
        metadata={"robot": "simple_arm", "description": "Synthetic reaching demonstration"},
    )

    for step in range(50):
        phase = step / 49.0
        joint_positions = [
            0.1 + 0.7 * phase + 0.03 * math.sin(math.pi * phase),
            -0.2 + 0.5 * phase + 0.02 * math.sin(2.0 * math.pi * phase),
            0.3 - 0.4 * phase + 0.015 * math.sin(math.pi * phase),
        ]
        ee_pose = (
            0.35 + 0.25 * phase,
            -0.10 + 0.18 * phase,
            0.45 + 0.08 * math.sin(math.pi * phase),
            0.0,
            0.0,
            0.0,
            1.0,
        )
        recorder.record(step=step, joint_positions=joint_positions, ee_pose=ee_pose, gripper_open=True)

    demo = recorder.demonstration
    dmp = DynamicMovementPrimitive()
    dmp.train(demo)

    new_goal = [0.95, 0.45, -0.25]
    generated = dmp.generate(goal=new_goal)

    indices = [0, generated.num_steps // 2, generated.num_steps - 1]
    labels = ["start", "mid", "end"]

    print("Generated trajectory summary:")
    for label, index in zip(labels, indices, strict=True):
        joints = [round(value, 4) for value in generated.steps[index].joint_positions]
        print(f"  {label}: {joints}")


if __name__ == "__main__":
    main()
