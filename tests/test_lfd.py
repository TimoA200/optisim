from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optisim.lfd import (
    DemoStep,
    Demonstration,
    DemonstrationLibrary,
    DemonstrationPlayer,
    DemonstrationRecorder,
    DynamicMovementPrimitive,
)


def build_demo(
    *,
    task_name: str = "reach",
    n_steps: int = 20,
    joint_dim: int = 3,
    start_time: float = 0.0,
    dt: float = 0.1,
    offset: float = 0.0,
) -> Demonstration:
    steps: list[DemoStep] = []
    for index in range(n_steps):
        phase = 0.0 if n_steps == 1 else index / (n_steps - 1)
        joints = [
            offset + 0.15 * axis + (axis + 1) * 0.4 * phase + 0.02 * np.sin(np.pi * phase)
            for axis in range(joint_dim)
        ]
        ee_pose = (
            0.4 + 0.2 * phase,
            0.1 * phase,
            0.5 + 0.03 * np.sin(np.pi * phase),
            0.0,
            0.0,
            0.0,
            1.0,
        )
        steps.append(
            DemoStep(
                timestep=start_time + index * dt,
                joint_positions=joints,
                ee_pose=ee_pose,
                gripper_open=index < max(n_steps - 3, 1),
                extras={"phase": float(phase)} if index % 2 == 0 else {},
            )
        )
    return Demonstration(task_name=task_name, steps=steps, metadata={"source": "test"})


def test_demo_step_roundtrip() -> None:
    step = DemoStep(1.0, [0.1, 0.2], (0.0, 0.0, 1.0), True, {"tag": "a"})

    restored = DemoStep.from_dict(step.to_dict())

    assert restored == step


def test_demonstration_properties_for_multi_step_demo() -> None:
    demo = build_demo(n_steps=5, joint_dim=4, start_time=2.0, dt=0.25)

    assert demo.num_steps == 5
    assert demo.joint_dim == 4
    assert demo.duration == pytest.approx(1.0)


def test_demonstration_properties_for_empty_demo() -> None:
    demo = Demonstration(task_name="empty")

    assert demo.num_steps == 0
    assert demo.joint_dim == 0
    assert demo.duration == 0.0


def test_demonstration_json_roundtrip(tmp_path: Path) -> None:
    demo = build_demo(n_steps=8)
    path = tmp_path / "demo.json"

    demo.save(str(path))
    restored = Demonstration.load(str(path))

    assert restored.task_name == demo.task_name
    assert restored.metadata == demo.metadata
    assert restored.num_steps == demo.num_steps
    assert restored.steps[-1].joint_positions == pytest.approx(demo.steps[-1].joint_positions)


def test_recorder_records_multiple_steps() -> None:
    recorder = DemonstrationRecorder(task_name="recorded", metadata={"robot": "arm"})

    recorder.record(0, [0.0, 0.1], (0.1, 0.2, 0.3), True, {"frame": 0})
    recorder.record(1, [0.2, 0.3], (0.4, 0.5, 0.6), False)

    assert recorder.num_steps == 2
    assert recorder.joint_dim == 2
    assert recorder.duration == 1.0
    assert recorder.demonstration.steps[1].extras == {}


def test_recorder_save_and_load_roundtrip(tmp_path: Path) -> None:
    recorder = DemonstrationRecorder(task_name="recorded")
    recorder.record(0, [0.1, 0.2, 0.3], (0.0, 0.1, 0.2), True)
    recorder.record(1, [0.4, 0.5, 0.6], (0.3, 0.4, 0.5), False, {"closed": True})
    path = tmp_path / "recorded.json"

    recorder.save(str(path))
    restored = DemonstrationRecorder.load(str(path))

    assert restored.demonstration.task_name == "recorded"
    assert restored.demonstration.steps[1].extras == {"closed": True}


def test_recorder_keeps_empty_extras_as_empty_dict() -> None:
    recorder = DemonstrationRecorder(task_name="empty_extras")

    recorder.record(0, [0.0], (0.0,), True, None)

    assert recorder.demonstration.steps[0].extras == {}


def test_dmp_train_requires_at_least_two_steps() -> None:
    dmp = DynamicMovementPrimitive()

    with pytest.raises(ValueError):
        dmp.train(build_demo(n_steps=1))


def test_dmp_generate_requires_training() -> None:
    dmp = DynamicMovementPrimitive()

    with pytest.raises(ValueError):
        dmp.generate(goal=[1.0, 2.0, 3.0])


def test_dmp_train_populates_weights_per_joint() -> None:
    demo = build_demo(n_steps=30, joint_dim=4)
    dmp = DynamicMovementPrimitive(n_basis=15)

    dmp.train(demo)

    assert dmp.weights is not None
    assert dmp.weights.shape == (4, 15)


def test_dmp_generate_matches_demo_length_by_default() -> None:
    demo = build_demo(n_steps=18, joint_dim=3)
    dmp = DynamicMovementPrimitive()
    dmp.train(demo)

    generated = dmp.generate(goal=[0.8, 1.0, 1.2])

    assert generated.num_steps == demo.num_steps
    assert generated.joint_dim == demo.joint_dim


def test_dmp_generated_trajectory_converges_to_goal() -> None:
    demo = build_demo(n_steps=40, joint_dim=3)
    dmp = DynamicMovementPrimitive()
    dmp.train(demo)
    goal = [1.0, 1.2, 1.4]

    generated = dmp.generate(goal=goal)

    assert generated.steps[-1].joint_positions == pytest.approx(goal, abs=1e-6)


def test_dmp_generated_endpoint_is_near_goal_with_custom_duration() -> None:
    demo = build_demo(n_steps=35, joint_dim=3)
    dmp = DynamicMovementPrimitive(n_basis=20)
    dmp.train(demo)
    goal = [0.9, 1.05, 1.2]

    generated = dmp.generate(goal=goal, duration=6.0)

    assert generated.steps[-1].joint_positions == pytest.approx(goal, abs=1e-2)
    assert generated.duration == pytest.approx(6.0)


def test_dmp_generate_supports_custom_start() -> None:
    demo = build_demo(n_steps=25, joint_dim=2)
    dmp = DynamicMovementPrimitive()
    dmp.train(demo)

    generated = dmp.generate(goal=[0.9, 1.1], start=[-0.2, 0.1])

    assert generated.steps[0].joint_positions == pytest.approx([-0.2, 0.1])


def test_dmp_generate_rejects_wrong_goal_dimension() -> None:
    demo = build_demo(n_steps=10, joint_dim=3)
    dmp = DynamicMovementPrimitive()
    dmp.train(demo)

    with pytest.raises(ValueError):
        dmp.generate(goal=[1.0, 2.0])


def test_dmp_generate_rejects_wrong_start_dimension() -> None:
    demo = build_demo(n_steps=10, joint_dim=3)
    dmp = DynamicMovementPrimitive()
    dmp.train(demo)

    with pytest.raises(ValueError):
        dmp.generate(goal=[1.0, 2.0, 3.0], start=[0.0, 0.1])


def test_dmp_preserves_task_name_and_marks_generated_metadata() -> None:
    demo = build_demo(task_name="pour", n_steps=12, joint_dim=2)
    dmp = DynamicMovementPrimitive()
    dmp.train(demo)

    generated = dmp.generate(goal=[0.8, 1.2])

    assert generated.task_name == "pour"
    assert generated.metadata["generated_by"] == "DynamicMovementPrimitive"


def test_library_add_get_and_len() -> None:
    library = DemonstrationLibrary()
    demo_a = build_demo(task_name="pick", n_steps=10)
    demo_b = build_demo(task_name="pick", n_steps=12, offset=0.2)

    library.add(demo_a)
    library.add(demo_b)

    assert library.len == 2
    assert len(library.get("pick")) == 2


def test_library_best_returns_longest_demo() -> None:
    library = DemonstrationLibrary()
    short_demo = build_demo(task_name="stack", n_steps=8)
    long_demo = build_demo(task_name="stack", n_steps=14)
    library.add(short_demo)
    library.add(long_demo)

    assert library.best("stack") is long_demo


def test_library_best_returns_none_for_missing_task() -> None:
    library = DemonstrationLibrary()

    assert library.best("missing") is None


def test_library_all_tasks_returns_sorted_names() -> None:
    library = DemonstrationLibrary()
    library.add(build_demo(task_name="z_task"))
    library.add(build_demo(task_name="a_task"))

    assert library.all_tasks() == ["a_task", "z_task"]


def test_library_save_all_and_load_all(tmp_path: Path) -> None:
    library = DemonstrationLibrary()
    library.add(build_demo(task_name="reach", n_steps=6))
    library.add(build_demo(task_name="place", n_steps=7))

    library.save_all(str(tmp_path))
    restored = DemonstrationLibrary.load_all(str(tmp_path))

    assert restored.len == 2
    assert restored.all_tasks() == ["place", "reach"]


def test_library_load_all_missing_directory_returns_empty_library(tmp_path: Path) -> None:
    restored = DemonstrationLibrary.load_all(str(tmp_path / "missing"))

    assert restored.len == 0
    assert restored.all_tasks() == []


def test_player_play_returns_frames_with_scaled_time() -> None:
    demo = build_demo(n_steps=4, dt=0.5)
    player = DemonstrationPlayer()

    frames = player.play(demo, time_scale=2.0)

    assert len(frames) == 4
    assert frames[-1]["timestep"] == pytest.approx(3.0)
    assert frames[0]["joint_positions"] == pytest.approx(demo.steps[0].joint_positions)


def test_player_interpolate_resamples_to_requested_steps() -> None:
    demo = build_demo(n_steps=5, joint_dim=3, dt=0.25)
    player = DemonstrationPlayer()

    resampled = player.interpolate(demo, n_steps=11)

    assert resampled.num_steps == 11
    assert resampled.steps[0].joint_positions == pytest.approx(demo.steps[0].joint_positions)
    assert resampled.steps[-1].joint_positions == pytest.approx(demo.steps[-1].joint_positions)


def test_player_interpolate_single_step_demo_repeats_step() -> None:
    demo = build_demo(n_steps=1, joint_dim=2)
    player = DemonstrationPlayer()

    resampled = player.interpolate(demo, n_steps=4)

    assert resampled.num_steps == 4
    assert all(step.joint_positions == pytest.approx(demo.steps[0].joint_positions) for step in resampled.steps)


def test_player_interpolate_rejects_non_positive_step_count() -> None:
    player = DemonstrationPlayer()

    with pytest.raises(ValueError):
        player.interpolate(build_demo(), n_steps=0)


def test_adapt_to_goal_shifts_endpoint_correctly() -> None:
    demo = build_demo(n_steps=6, joint_dim=3)
    player = DemonstrationPlayer()
    new_goal = [1.5, 1.6, 1.7]

    adapted = player.adapt_to_goal(demo, new_goal)

    assert adapted.steps[0].joint_positions == pytest.approx(demo.steps[0].joint_positions)
    assert adapted.steps[-1].joint_positions == pytest.approx(new_goal)


def test_adapt_to_goal_rejects_wrong_dimension() -> None:
    player = DemonstrationPlayer()

    with pytest.raises(ValueError):
        player.adapt_to_goal(build_demo(joint_dim=3), [0.1, 0.2])


def test_adapt_to_goal_handles_empty_demo() -> None:
    player = DemonstrationPlayer()
    demo = Demonstration(task_name="empty")

    adapted = player.adapt_to_goal(demo, [])

    assert adapted.num_steps == 0


def test_different_joint_dimensions_are_supported() -> None:
    demo_2d = build_demo(task_name="two_joint", joint_dim=2)
    demo_5d = build_demo(task_name="five_joint", joint_dim=5)

    assert demo_2d.joint_dim == 2
    assert demo_5d.joint_dim == 5


__all__ = [name for name in globals() if name.startswith("test_")]
