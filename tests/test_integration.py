from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optisim.cli import main
from optisim.core import TaskComposer, TaskDefinition
from optisim.math3d import Pose, Quaternion, vec3
from optisim.robot import IKOptions, build_humanoid_model, solve_inverse_kinematics
from optisim.sim import ExecutionEngine, SimulationRecording, WorldState, replay_recording
from optisim.viz import WebVisualizer


def _pick_and_place_task() -> TaskDefinition:
    composer = TaskComposer("integration_pick_and_place").pick_and_place(
        target="box",
        pickup_effector="right_palm",
        destination=(0.58, -0.20, 1.08),
        support="shelf",
    )
    return TaskDefinition.from_composer(
        composer,
        world={
            "gravity": [0.0, 0.0, -9.81],
            "surfaces": [
                {
                    "name": "table",
                    "pose": {"position": [0.55, 0.0, 0.74], "rpy": [0.0, 0.0, 0.0]},
                    "size": [0.90, 0.60, 0.05],
                },
                {
                    "name": "shelf",
                    "pose": {"position": [0.60, -0.25, 1.02], "rpy": [0.0, 0.0, 0.0]},
                    "size": [0.35, 0.25, 0.04],
                },
            ],
            "objects": [
                {
                    "name": "box",
                    "pose": {"position": [0.42, -0.12, 0.81], "rpy": [0.0, 0.0, 0.0]},
                    "size": [0.08, 0.08, 0.12],
                    "mass_kg": 0.75,
                }
            ],
        },
        robot={"model": "optimus_humanoid"},
    )


class _ReplayProbe:
    def __init__(self) -> None:
        self.frame_count = 0
        self.started = False
        self.finished = False

    def start_task(self, task: TaskDefinition, world: WorldState, robot: object) -> None:
        del task, world, robot
        self.started = True

    def update_collisions(self, collisions: list[object]) -> None:
        del collisions

    def render(self, world: WorldState, robot: object) -> None:
        del world, robot
        self.frame_count += 1

    def finish(self, task: TaskDefinition, world: WorldState, robot: object, collisions: list[object]) -> None:
        del task, world, robot, collisions
        self.finished = True


def test_end_to_end_pick_and_place_simulation_moves_object_to_shelf() -> None:
    task = _pick_and_place_task()
    robot = build_humanoid_model()
    engine = ExecutionEngine(robot=robot, world=WorldState.from_dict(task.world))

    record = engine.run(task)

    box = engine.world.objects["box"]
    shelf = engine.world.surfaces["shelf"]
    expected_z = shelf.pose.position[2] + shelf.size[2] / 2.0 + box.size[2] / 2.0

    assert record.executed_actions == ["reach", "grasp", "move", "place"]
    assert record.steps > 0
    assert box.held_by is None
    assert np.isclose(box.pose.position[2], expected_z)
    assert np.allclose(box.pose.position[:2], [0.58, -0.20], atol=1e-6)


def test_ik_then_fk_returns_same_pose() -> None:
    robot = build_humanoid_model()
    desired_joints = {
        "torso_yaw": -0.1,
        "torso_pitch": 0.12,
        "right_clavicle_pitch": 0.1,
        "right_shoulder_pitch": -0.9,
        "right_shoulder_roll": -0.25,
        "right_shoulder_yaw": 0.6,
        "right_elbow_pitch": 1.2,
        "right_forearm_yaw": -0.35,
        "right_wrist_pitch": 0.2,
    }
    target = robot.end_effector_pose("right_palm", desired_joints)

    result = solve_inverse_kinematics(
        robot,
        "right_palm",
        target,
        options=IKOptions(max_iterations=150, convergence_threshold=1e-3, damping=0.12),
    )
    solved_pose = robot.end_effector_pose("right_palm", result.joint_positions)

    assert result.success
    assert np.linalg.norm(solved_pose.position - target.position) < 2e-3
    assert result.orientation_error < 2e-3


def test_recording_round_trip_replay_preserves_frame_count(tmp_path: Path) -> None:
    task = TaskDefinition.from_file(Path("examples/pick_and_place.yaml"))
    engine = ExecutionEngine(robot=build_humanoid_model(), world=WorldState.from_dict(task.world))

    record = engine.run(task)
    assert record.recording is not None
    recording_path = tmp_path / "pick_and_place_recording.json"
    record.recording.dump(recording_path)

    reloaded = SimulationRecording.from_file(recording_path)
    probe = _ReplayProbe()
    replay_recording(
        reloaded,
        robot=build_humanoid_model(),
        world=WorldState.with_defaults(),
        visualizer=probe,
        realtime=False,
    )

    assert probe.started is True
    assert probe.finished is True
    assert probe.frame_count == reloaded.frame_count()


def test_cli_validate_run_and_sim_commands_emit_expected_output(capsys: pytest.CaptureFixture[str]) -> None:
    task_path = Path("examples/pick_and_place.yaml")

    validate_exit = main(["validate", str(task_path)])
    validate_output = capsys.readouterr().out
    run_exit = main(["run", str(task_path)])
    run_output = capsys.readouterr().out
    sim_exit = main(["sim", str(task_path)])
    sim_output = capsys.readouterr().out

    assert validate_exit == 0
    assert "valid" in validate_output
    assert run_exit == 0
    assert "completed 'humanoid_pick_and_place'" in run_output
    assert "actions=['reach', 'grasp', 'move', 'place']" in run_output
    assert sim_exit == 0
    assert "completed 'humanoid_pick_and_place'" in sim_output


def test_web_visualizer_can_be_instantiated_without_crashing() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")

    visualizer = WebVisualizer(open_browser=False)

    assert visualizer.host == "127.0.0.1"
    assert visualizer.url.startswith("http://127.0.0.1:")
    visualizer.close()


def test_invalid_yaml_task_file_reports_clear_schema_error(tmp_path: Path) -> None:
    task_path = tmp_path / "invalid_task.yaml"
    task_path.write_text("actions:\n  - type: reach\n    target: box\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid task file .* field 'name'"):
        TaskDefinition.from_file(task_path)


def test_invalid_yaml_syntax_reports_source_file(tmp_path: Path) -> None:
    task_path = tmp_path / "broken_task.yaml"
    task_path.write_text("name: broken\nactions: [\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid YAML in task file"):
        TaskDefinition.from_file(task_path)


def test_ik_failure_reports_out_of_reach_reason() -> None:
    robot = build_humanoid_model()
    target = Pose(position=vec3([4.0, 0.0, 1.0]), orientation=Quaternion.identity())

    result = solve_inverse_kinematics(
        robot,
        "right_palm",
        target,
        options=IKOptions(max_iterations=40, convergence_threshold=1e-3, position_only=True),
    )

    assert result.success is False
    assert result.failure_reason is not None
    assert "out of reach" in result.failure_reason


def test_ik_failure_reports_joint_limit_reason() -> None:
    robot = build_humanoid_model()
    target = Pose(
        position=vec3([0.5854072347027225, -0.801903841332337, 0.5715040668116087]),
        orientation=Quaternion.identity(),
    )

    result = solve_inverse_kinematics(
        robot,
        "right_palm",
        target,
        options=IKOptions(max_iterations=80, convergence_threshold=1e-3, damping=0.08),
    )

    assert result.success is False
    assert result.failure_reason is not None
    assert "joint limits" in result.failure_reason


def test_cli_reports_missing_optional_visualizer_dependencies_with_install_hint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from optisim import cli

    def _missing_web_visualizer() -> object:
        raise ModuleNotFoundError("Install with `pip install optisim[web]`.")

    monkeypatch.setattr(cli, "WebVisualizer", _missing_web_visualizer)

    exit_code = main(["sim", "examples/pick_and_place.yaml", "--web"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "pip install optisim[web]" in output

__all__ = ["test_end_to_end_pick_and_place_simulation_moves_object_to_shelf", "test_ik_then_fk_returns_same_pose", "test_recording_round_trip_replay_preserves_frame_count", "test_cli_validate_run_and_sim_commands_emit_expected_output", "test_web_visualizer_can_be_instantiated_without_crashing", "test_invalid_yaml_task_file_reports_clear_schema_error", "test_invalid_yaml_syntax_reports_source_file", "test_ik_failure_reports_out_of_reach_reason", "test_ik_failure_reports_joint_limit_reason", "test_cli_reports_missing_optional_visualizer_dependencies_with_install_hint"]
