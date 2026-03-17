from __future__ import annotations

import json

import numpy as np
import pytest

from optisim import (
    BenchmarkExporter,
    BenchmarkResult,
    ExportFormat,
    PrimitiveResult,
    PrimitiveStatus,
    SceneExport,
    SceneExporter,
    SceneGraph,
    SceneNode,
    SceneRelation,
    TrajectoryExport,
    TrajectoryExporter,
)
from optisim.cli import build_parser, main


def make_pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def make_primitive_result(offset: float, frames: int = 2, dof: int = 31) -> PrimitiveResult:
    trajectory = [np.linspace(offset + index, offset + index + 0.3, dof, dtype=float) for index in range(frames)]
    return PrimitiveResult(
        status=PrimitiveStatus.SUCCESS,
        message="ok",
        joint_trajectory=trajectory,
        duration_s=frames * 0.05,
        metadata={"offset": offset},
    )


def make_scene_graph() -> SceneGraph:
    graph = SceneGraph()
    graph.add_node(SceneNode("robot", "robot", "robot", pose=make_pose(0.0, 0.0, 0.0)))
    graph.add_node(SceneNode("table", "table", "surface", pose=make_pose(0.8, 0.0, 0.75)))
    graph.add_node(SceneNode("cup", "cup", "container", pose=make_pose(0.7, 0.1, 0.82)))
    graph.add_relation(SceneRelation("cup", "on", "table", confidence=0.9))
    graph.add_relation(SceneRelation("robot", "near", "cup", confidence=0.8))
    return graph


def make_trajectory_export() -> TrajectoryExport:
    return TrajectoryExporter.from_primitive_results([make_primitive_result(0.0), make_primitive_result(2.0)], dt=0.1)


def make_scene_export() -> SceneExport:
    return SceneExport(
        scene_graph=make_scene_graph(),
        robot_trajectory=make_trajectory_export(),
        task_name="pick_cup",
        metadata={"source": "test"},
    )


def make_benchmark_result(
    task_name: str = "pick_cup",
    *,
    success: bool = True,
    partial_success: float = 1.0,
    difficulty: str = "easy",
) -> BenchmarkResult:
    return BenchmarkResult(
        task_name=task_name,
        success=success,
        partial_success=partial_success,
        steps_completed=2,
        steps_total=3,
        primitive_results=[make_primitive_result(0.0, frames=1, dof=3)],
        elapsed_steps=4,
        error_message=None,
        metadata={"difficulty": difficulty, "tags": ["demo"]},
    )


def test_trajectory_exporter_from_primitive_results_single_result() -> None:
    export = TrajectoryExporter.from_primitive_results([make_primitive_result(0.0)])
    assert len(export.frames) == 2


def test_trajectory_exporter_from_primitive_results_two_results() -> None:
    export = TrajectoryExporter.from_primitive_results([make_primitive_result(0.0), make_primitive_result(1.0)])
    assert len(export.frames) == 4


def test_trajectory_exporter_from_primitive_results_three_results() -> None:
    export = TrajectoryExporter.from_primitive_results(
        [make_primitive_result(0.0), make_primitive_result(1.0), make_primitive_result(2.0)]
    )
    assert len(export.frames) == 6


def test_trajectory_exporter_concatenates_correct_number_of_frames() -> None:
    export = TrajectoryExporter.from_primitive_results([make_primitive_result(0.0, 1), make_primitive_result(1.0, 3)])
    assert len(export.frames) == 4


def test_trajectory_exporter_timestamps_length_matches_frames() -> None:
    export = TrajectoryExporter.from_primitive_results([make_primitive_result(0.0), make_primitive_result(1.0)], dt=0.2)
    assert len(export.timestamps) == len(export.frames)


def test_trajectory_exporter_default_joint_names_are_31_items() -> None:
    export = TrajectoryExporter.from_primitive_results([make_primitive_result(0.0)])
    assert len(export.joint_names) == 31


def test_trajectory_exporter_custom_joint_names_are_preserved() -> None:
    export = TrajectoryExporter.from_primitive_results(
        [make_primitive_result(0.0, dof=3)],
        joint_names=["shoulder", "elbow", "wrist"],
    )
    assert export.joint_names == ["shoulder", "elbow", "wrist"]


def test_trajectory_exporter_timestamps_increase_monotonically() -> None:
    export = TrajectoryExporter.from_primitive_results([make_primitive_result(0.0), make_primitive_result(1.0)], dt=0.05)
    assert all(b > a for a, b in zip(export.timestamps, export.timestamps[1:]))


def test_trajectory_exporter_json_round_trip_preserves_frame_count() -> None:
    export = make_trajectory_export()
    restored = TrajectoryExporter.from_json(TrajectoryExporter.to_json(export))
    assert len(restored.frames) == len(export.frames)


def test_trajectory_exporter_json_round_trip_preserves_frames() -> None:
    export = make_trajectory_export()
    restored = TrajectoryExporter.from_json(TrajectoryExporter.to_json(export))
    for actual, expected in zip(restored.frames, export.frames):
        np.testing.assert_allclose(actual, expected)


def test_trajectory_exporter_json_round_trip_preserves_joint_names() -> None:
    export = make_trajectory_export()
    restored = TrajectoryExporter.from_json(TrajectoryExporter.to_json(export))
    assert restored.joint_names == export.joint_names


def test_trajectory_exporter_to_csv_has_header_row_with_joint_names() -> None:
    csv_text = TrajectoryExporter.to_csv(make_trajectory_export())
    assert "timestamp,j0,j1,j2" in csv_text


def test_trajectory_exporter_to_csv_has_correct_number_of_data_rows() -> None:
    export = make_trajectory_export()
    csv_text = TrajectoryExporter.to_csv(export)
    assert len(csv_text.strip().splitlines()) == len(export.frames) + 1


def test_trajectory_exporter_from_csv_reconstructs_correct_frame_count() -> None:
    export = make_trajectory_export()
    restored = TrajectoryExporter.from_csv(TrajectoryExporter.to_csv(export))
    assert len(restored.frames) == len(export.frames)


def test_trajectory_exporter_from_csv_reconstructs_correct_timestamps() -> None:
    export = make_trajectory_export()
    restored = TrajectoryExporter.from_csv(TrajectoryExporter.to_csv(export))
    np.testing.assert_allclose(restored.timestamps, export.timestamps)


def test_trajectory_exporter_from_csv_reconstructs_joint_names() -> None:
    export = make_trajectory_export()
    restored = TrajectoryExporter.from_csv(TrajectoryExporter.to_csv(export))
    assert restored.joint_names == export.joint_names


def test_trajectory_exporter_to_mocap_csv_header_contains_frame_and_time() -> None:
    header = TrajectoryExporter.to_mocap_csv(make_trajectory_export()).splitlines()[0]
    assert "frame" in header and "time" in header


def test_trajectory_exporter_to_mocap_csv_contains_xyz_columns() -> None:
    header = TrajectoryExporter.to_mocap_csv(make_trajectory_export()).splitlines()[0]
    assert "j0_x" in header and "j0_y" in header and "j0_z" in header


def test_trajectory_exporter_to_ros2_bag_json_is_valid_json_list() -> None:
    payload = json.loads(TrajectoryExporter.to_ros2_bag_json(make_trajectory_export()))
    assert isinstance(payload, list)


def test_trajectory_exporter_ros2_messages_have_required_keys() -> None:
    payload = json.loads(TrajectoryExporter.to_ros2_bag_json(make_trajectory_export()))
    assert {"topic", "sec", "nanosec", "data"} <= set(payload[0])


def test_trajectory_exporter_ros2_messages_include_joint_state_fields() -> None:
    payload = json.loads(TrajectoryExporter.to_ros2_bag_json(make_trajectory_export()))
    assert {"name", "position", "velocity", "effort"} <= set(payload[0]["data"])


def test_trajectory_exporter_empty_results_handled_gracefully() -> None:
    export = TrajectoryExporter.from_primitive_results([])
    assert export.frames == []
    assert export.timestamps == []


def test_trajectory_exporter_empty_csv_round_trip_is_empty() -> None:
    restored = TrajectoryExporter.from_csv(TrajectoryExporter.to_csv(TrajectoryExporter.from_primitive_results([])))
    assert restored.frames == []


def test_export_format_enum_has_expected_members() -> None:
    assert [member.name for member in ExportFormat] == [
        "JSON_TRAJ",
        "CSV_TRAJ",
        "MOCAP_CSV",
        "ROS2_BAG_JSON",
        "SCENE_JSON",
        "URDF_ANNOTATION",
    ]


def test_scene_exporter_to_json_is_valid_json() -> None:
    payload = json.loads(SceneExporter.to_json(make_scene_export()))
    assert payload["task_name"] == "pick_cup"


def test_scene_exporter_json_includes_trajectory_payload_when_present() -> None:
    payload = json.loads(SceneExporter.to_json(make_scene_export()))
    assert payload["trajectory"] is not None


def test_scene_exporter_from_json_round_trip_preserves_node_count() -> None:
    export = make_scene_export()
    restored = SceneExporter.from_json(SceneExporter.to_json(export))
    assert len(restored.scene_graph.nodes) == len(export.scene_graph.nodes)


def test_scene_exporter_from_json_round_trip_preserves_relation_count() -> None:
    export = make_scene_export()
    restored = SceneExporter.from_json(SceneExporter.to_json(export))
    assert len(restored.scene_graph.relations) == len(export.scene_graph.relations)


def test_scene_exporter_from_json_round_trip_preserves_task_name() -> None:
    export = make_scene_export()
    restored = SceneExporter.from_json(SceneExporter.to_json(export))
    assert restored.task_name == export.task_name


def test_scene_exporter_to_urdf_annotation_contains_xml_root_tag() -> None:
    xml_text = SceneExporter.to_urdf_annotation(make_scene_export())
    assert "<task_annotation" in xml_text


def test_scene_exporter_to_urdf_annotation_contains_object_elements() -> None:
    xml_text = SceneExporter.to_urdf_annotation(make_scene_export())
    assert "<object " in xml_text


def test_scene_exporter_to_urdf_annotation_contains_relation_elements() -> None:
    xml_text = SceneExporter.to_urdf_annotation(make_scene_export())
    assert "<relation " in xml_text


def test_scene_exporter_empty_scene_handled_gracefully() -> None:
    export = SceneExport(scene_graph=SceneGraph(), robot_trajectory=None, task_name="empty", metadata={})
    payload = json.loads(SceneExporter.to_json(export))
    assert payload["scene"]["nodes"] == {}


def test_benchmark_exporter_to_json_is_valid_json() -> None:
    payload = json.loads(BenchmarkExporter.to_json([make_benchmark_result()]))
    assert payload[0]["task_name"] == "pick_cup"


def test_benchmark_exporter_to_csv_has_header() -> None:
    csv_text = BenchmarkExporter.to_csv([make_benchmark_result()])
    assert csv_text.startswith("task_name,difficulty,success,steps,partial")


def test_benchmark_exporter_to_markdown_contains_pipe_chars() -> None:
    markdown = BenchmarkExporter.to_markdown([make_benchmark_result()])
    assert "|" in markdown


def test_benchmark_exporter_markdown_summary_line_present() -> None:
    markdown = BenchmarkExporter.to_markdown([make_benchmark_result()])
    assert "Success rate:" in markdown


def test_benchmark_exporter_empty_results_handled_gracefully_in_json() -> None:
    assert json.loads(BenchmarkExporter.to_json([])) == []


def test_benchmark_exporter_empty_results_handled_gracefully_in_csv() -> None:
    assert "task_name" in BenchmarkExporter.to_csv([])


def test_benchmark_exporter_empty_results_handled_gracefully_in_markdown() -> None:
    markdown = BenchmarkExporter.to_markdown([])
    assert "Success rate: 0/0 (0.0%)" in markdown


def test_export_cli_parser_accepts_run_benchmark() -> None:
    args = build_parser().parse_args(["export", "run-benchmark", "--format", "json"])
    assert args.command == "export"
    assert args.export_command == "run-benchmark"


def test_export_cli_run_benchmark_json_runs_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["export", "run-benchmark", "--format", "json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)


def test_export_cli_run_benchmark_csv_runs_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["export", "run-benchmark", "--format", "csv"])
    assert exit_code == 0
    assert "task_name,difficulty,success,steps,partial" in capsys.readouterr().out


def test_export_cli_run_benchmark_markdown_runs_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["export", "run-benchmark", "--format", "markdown"])
    assert exit_code == 0
    assert "| Task | Difficulty | Success | Steps | Partial |" in capsys.readouterr().out


def test_export_cli_help_shows_usage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["export", "--help"])
    assert exc_info.value.code == 0
    assert "usage:" in capsys.readouterr().out
