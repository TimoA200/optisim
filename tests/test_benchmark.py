from __future__ import annotations

import json

import numpy as np
import pytest

from optisim import (
    BenchmarkEvaluator,
    BenchmarkReporter,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkTask,
    PrimitiveResult,
    PrimitiveStatus,
    SceneBuilder,
    SceneGraph,
    SceneNode,
    SceneRelation,
)
from optisim.cli import build_parser, main


def make_pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def build_simple_scene() -> SceneGraph:
    scene = SceneGraph()
    scene.add_node(
        SceneNode(
            "humanoid",
            "humanoid",
            "robot",
            pose=make_pose(0.0, 0.0, 0.0),
            properties={"mobile": True, "manipulator": True},
        )
    )
    scene.add_node(
        SceneNode(
            "cup",
            "cup",
            "container",
            pose=make_pose(0.4, 0.0, 0.2),
            properties={"graspable": True},
        )
    )
    scene.add_node(SceneNode("table", "table", "surface", pose=make_pose(0.5, 0.0, 0.2)))
    scene.add_node(SceneNode("shelf", "shelf", "surface", pose=make_pose(0.6, 0.1, 0.4)))
    scene.add_relation(SceneRelation("cup", "on", "table"))
    return scene


def build_navigation_scene() -> SceneGraph:
    scene = build_simple_scene()
    scene.add_node(
        SceneNode(
            "obstacle",
            "obstacle",
            "obstacle",
            pose=make_pose(0.45, -0.1, 0.2),
            properties={"movable": True},
        )
    )
    return scene


def make_pick_task() -> BenchmarkTask:
    return BenchmarkTask(
        name="pick_test",
        description="Pick the cup.",
        build_scene=build_simple_scene,
        primitive_sequence=[
            {"primitive": "reach", "params": {"target_id": "cup", "end_effector": "right"}},
            {"primitive": "grasp", "params": {"target_id": "cup", "end_effector": "right"}},
        ],
        success_predicates=[{"subject_id": "cup", "predicate": "held_by", "object_id": "humanoid"}],
        difficulty="easy",
        tags=["kitchen", "pick-and-place"],
    )


def make_place_task() -> BenchmarkTask:
    return BenchmarkTask(
        name="place_test",
        description="Pick and place the cup on the shelf.",
        build_scene=build_simple_scene,
        primitive_sequence=[
            {"primitive": "reach", "params": {"target_id": "cup", "end_effector": "right"}},
            {"primitive": "grasp", "params": {"target_id": "cup", "end_effector": "right"}},
            {"primitive": "place", "params": {"object_id": "cup", "surface_id": "shelf"}},
        ],
        success_predicates=[{"subject_id": "cup", "predicate": "on", "object_id": "shelf"}],
        difficulty="medium",
        tags=["kitchen", "pick-and-place"],
    )


def make_push_task() -> BenchmarkTask:
    return BenchmarkTask(
        name="push_test",
        description="Navigate to and push an obstacle.",
        build_scene=build_navigation_scene,
        primitive_sequence=[
            {"primitive": "navigate", "params": {"target_id": "obstacle"}},
            {"primitive": "push", "params": {"target_id": "obstacle", "direction": [1.0, 0.0, 0.0]}},
        ],
        success_predicates=[{"subject_id": "obstacle", "predicate": "displaced"}],
        difficulty="hard",
        tags=["navigation"],
    )


def make_suite() -> BenchmarkSuite:
    suite = BenchmarkSuite()
    suite.register(make_pick_task())
    suite.register(make_place_task())
    suite.register(make_push_task())
    return suite


def make_result(
    task_name: str = "pick_test",
    *,
    success: bool = True,
    partial_success: float = 1.0,
    difficulty: str = "easy",
) -> BenchmarkResult:
    primitive = PrimitiveResult(status=PrimitiveStatus.SUCCESS, message="ok", joint_trajectory=[np.zeros(3)], duration_s=0.1)
    return BenchmarkResult(
        task_name=task_name,
        success=success,
        partial_success=partial_success,
        steps_completed=1,
        steps_total=1,
        primitive_results=[primitive],
        elapsed_steps=1,
        error_message=None,
        metadata={"difficulty": difficulty, "tags": ["tag"]},
    )


def test_benchmark_task_creation() -> None:
    task = make_pick_task()
    assert task.name == "pick_test"


def test_benchmark_task_preserves_description() -> None:
    task = make_pick_task()
    assert "Pick" in task.description


def test_benchmark_task_preserves_tags() -> None:
    task = make_pick_task()
    assert "kitchen" in task.tags


def test_benchmark_task_rejects_unknown_difficulty() -> None:
    with pytest.raises(ValueError):
        BenchmarkTask(
            name="bad",
            description="bad",
            build_scene=build_simple_scene,
            primitive_sequence=[],
            success_predicates=[],
            difficulty="expert",
            tags=[],
        )


def test_benchmark_suite_register_stores_task() -> None:
    suite = BenchmarkSuite()
    task = make_pick_task()
    suite.register(task)
    assert suite.tasks["pick_test"] is task


def test_benchmark_suite_register_duplicate_raises() -> None:
    suite = BenchmarkSuite()
    suite.register(make_pick_task())
    with pytest.raises(ValueError):
        suite.register(make_pick_task())


def test_benchmark_suite_get_returns_task() -> None:
    suite = make_suite()
    assert suite.get("place_test").name == "place_test"


def test_benchmark_suite_get_missing_raises_key_error() -> None:
    with pytest.raises(KeyError):
        make_suite().get("missing")


def test_benchmark_suite_list_tasks_returns_sorted_names() -> None:
    assert make_suite().list_tasks() == ["pick_test", "place_test", "push_test"]


def test_benchmark_suite_list_tasks_filters_by_difficulty() -> None:
    assert make_suite().list_tasks(difficulty="medium") == ["place_test"]


def test_benchmark_suite_list_tasks_filters_by_tag() -> None:
    assert make_suite().list_tasks(tag="navigation") == ["push_test"]


def test_benchmark_suite_list_tasks_filters_by_difficulty_and_tag() -> None:
    assert make_suite().list_tasks(difficulty="hard", tag="navigation") == ["push_test"]


def test_benchmark_suite_default_has_eight_tasks() -> None:
    assert len(BenchmarkSuite.DEFAULT.tasks) == 8


def test_benchmark_suite_default_includes_pick_cup() -> None:
    assert "pick_cup" in BenchmarkSuite.DEFAULT.tasks


def test_benchmark_suite_default_includes_warehouse_pick() -> None:
    assert "warehouse_pick" in BenchmarkSuite.DEFAULT.tasks


def test_benchmark_suite_default_easy_filter_returns_expected_tasks() -> None:
    assert BenchmarkSuite.DEFAULT.list_tasks(difficulty="easy") == ["pick_cup", "place_cup_on_shelf", "push_obstacle"]


def test_benchmark_suite_default_tag_filter_returns_kitchen_tasks() -> None:
    assert "pick_cup" in BenchmarkSuite.DEFAULT.list_tasks(tag="kitchen")


def test_benchmark_evaluator_run_task_returns_benchmark_result() -> None:
    result = BenchmarkEvaluator().run_task(make_pick_task())
    assert isinstance(result, BenchmarkResult)


def test_benchmark_evaluator_run_task_sets_success_field() -> None:
    result = BenchmarkEvaluator().run_task(make_pick_task())
    assert isinstance(result.success, bool)


def test_benchmark_evaluator_partial_success_between_zero_and_one() -> None:
    result = BenchmarkEvaluator().run_task(make_pick_task())
    assert 0.0 <= result.partial_success <= 1.0


def test_benchmark_evaluator_steps_completed_not_greater_than_total() -> None:
    result = BenchmarkEvaluator().run_task(make_place_task())
    assert result.steps_completed <= result.steps_total


def test_benchmark_evaluator_elapsed_steps_non_negative() -> None:
    result = BenchmarkEvaluator().run_task(make_place_task())
    assert result.elapsed_steps >= 0


def test_benchmark_evaluator_run_suite_returns_results_list() -> None:
    results = BenchmarkEvaluator().run_suite(make_suite())
    assert isinstance(results, list)


def test_benchmark_evaluator_run_suite_respects_task_names_subset() -> None:
    results = BenchmarkEvaluator().run_suite(make_suite(), task_names=["push_test"])
    assert [result.task_name for result in results] == ["push_test"]


def test_benchmark_evaluator_check_predicates_true_when_all_hold() -> None:
    scene = build_simple_scene()
    scene.add_relation(SceneRelation("cup", "held_by", "humanoid"))
    ok, failing = BenchmarkEvaluator().check_predicates(
        scene,
        [{"subject_id": "cup", "predicate": "held_by", "object_id": "humanoid"}],
    )
    assert ok is True
    assert failing == []


def test_benchmark_evaluator_check_predicates_false_with_explanation() -> None:
    ok, failing = BenchmarkEvaluator().check_predicates(
        build_simple_scene(),
        [{"subject_id": "cup", "predicate": "held_by", "object_id": "humanoid"}],
    )
    assert ok is False
    assert failing


def test_benchmark_evaluator_check_predicates_supports_unary_predicate() -> None:
    scene = build_navigation_scene()
    scene.add_relation(SceneRelation("obstacle", "displaced", "obstacle"))
    ok, failing = BenchmarkEvaluator().check_predicates(scene, [{"subject_id": "obstacle", "predicate": "displaced"}])
    assert ok is True
    assert failing == []


def test_benchmark_evaluator_run_task_failure_reports_error_message() -> None:
    task = BenchmarkTask(
        name="bad_step",
        description="Fails immediately.",
        build_scene=build_simple_scene,
        primitive_sequence=[{"primitive": "unknown", "params": {}}],
        success_predicates=[],
        difficulty="easy",
        tags=[],
    )
    result = BenchmarkEvaluator().run_task(task)
    assert result.success is False
    assert result.error_message is not None


def test_benchmark_evaluator_run_task_accepts_custom_joint_vector() -> None:
    joints = np.zeros(31, dtype=float)
    result = BenchmarkEvaluator().run_task(make_pick_task(), robot_joints=joints)
    assert result.steps_total == 2


def test_benchmark_evaluator_run_task_rejects_non_vector_joint_input() -> None:
    with pytest.raises(ValueError):
        BenchmarkEvaluator().run_task(make_pick_task(), robot_joints=np.zeros((2, 2), dtype=float))


def test_benchmark_reporter_summary_has_expected_keys() -> None:
    summary = BenchmarkReporter().summary([make_result()])
    assert {"total", "succeeded", "success_rate", "avg_partial_success", "by_difficulty", "by_task"} <= set(summary)


def test_benchmark_reporter_success_rate_between_zero_and_one() -> None:
    summary = BenchmarkReporter().summary([make_result(success=True), make_result(task_name="b", success=False, partial_success=0.5)])
    assert 0.0 <= summary["success_rate"] <= 1.0


def test_benchmark_reporter_summary_by_difficulty_present() -> None:
    summary = BenchmarkReporter().summary([make_result(difficulty="easy"), make_result(task_name="b", difficulty="hard")])
    assert {"easy", "hard"} <= set(summary["by_difficulty"])


def test_benchmark_reporter_format_table_contains_task_names() -> None:
    table = BenchmarkReporter().format_table([make_result(task_name="alpha"), make_result(task_name="beta")])
    assert "alpha" in table and "beta" in table


def test_benchmark_reporter_format_table_contains_header() -> None:
    table = BenchmarkReporter().format_table([make_result()])
    assert "| Task" in table


def test_benchmark_reporter_to_json_is_valid_json() -> None:
    payload = json.loads(BenchmarkReporter().to_json([make_result()]))
    assert "results" in payload


def test_benchmark_reporter_to_json_includes_summary() -> None:
    payload = json.loads(BenchmarkReporter().to_json([make_result()]))
    assert "summary" in payload


def test_benchmark_reporter_to_csv_has_header_row() -> None:
    csv_text = BenchmarkReporter().to_csv([make_result()])
    assert csv_text.splitlines()[0].startswith("task_name,")


def test_benchmark_reporter_to_csv_contains_task_name() -> None:
    csv_text = BenchmarkReporter().to_csv([make_result(task_name="alpha")])
    assert "alpha" in csv_text


def test_benchmark_reporter_empty_results_handled_gracefully() -> None:
    summary = BenchmarkReporter().summary([])
    assert summary["total"] == 0
    assert summary["success_rate"] == 0.0


def test_benchmark_cli_parser_accepts_subcommand() -> None:
    args = build_parser().parse_args(["benchmark", "--all"])
    assert args.command == "benchmark"


def test_benchmark_cli_list_runs_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["benchmark", "--list"])
    assert exit_code == 0
    assert "pick_cup" in capsys.readouterr().out


def test_benchmark_cli_single_task_json_runs_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["benchmark", "--task", "pick_cup", "--format", "json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["task_name"] == "pick_cup"


def test_benchmark_cli_default_runs_table(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["benchmark"])
    assert exit_code == 0
    assert "| Task" in capsys.readouterr().out


def test_default_pick_cup_runs_without_exception() -> None:
    result = BenchmarkEvaluator().run_task(BenchmarkSuite.DEFAULT.get("pick_cup"))
    assert result.task_name == "pick_cup"


def test_default_place_cup_on_shelf_runs_without_exception() -> None:
    result = BenchmarkEvaluator().run_task(BenchmarkSuite.DEFAULT.get("place_cup_on_shelf"))
    assert result.task_name == "place_cup_on_shelf"


def test_default_push_obstacle_runs_without_exception() -> None:
    result = BenchmarkEvaluator().run_task(BenchmarkSuite.DEFAULT.get("push_obstacle"))
    assert result.task_name == "push_obstacle"


def test_default_handover_tool_runs_without_exception() -> None:
    result = BenchmarkEvaluator().run_task(BenchmarkSuite.DEFAULT.get("handover_tool"))
    assert result.task_name == "handover_tool"


def test_default_navigate_and_grasp_runs_without_exception() -> None:
    result = BenchmarkEvaluator().run_task(BenchmarkSuite.DEFAULT.get("navigate_and_grasp"))
    assert result.task_name == "navigate_and_grasp"


def test_default_multi_step_kitchen_runs_without_exception() -> None:
    result = BenchmarkEvaluator().run_task(BenchmarkSuite.DEFAULT.get("multi_step_kitchen"))
    assert result.task_name == "multi_step_kitchen"


def test_default_warehouse_pick_runs_without_exception() -> None:
    result = BenchmarkEvaluator().run_task(BenchmarkSuite.DEFAULT.get("warehouse_pick"))
    assert result.task_name == "warehouse_pick"


def test_default_precision_place_runs_without_exception() -> None:
    result = BenchmarkEvaluator().run_task(BenchmarkSuite.DEFAULT.get("precision_place"))
    assert result.task_name == "precision_place"


def test_benchmark_task_can_use_scene_builder() -> None:
    scene = SceneBuilder.build_kitchen()
    assert "cup" in scene.nodes
