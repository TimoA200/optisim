from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from optisim.cli import main
from optisim.core import TaskDefinition
from optisim.multi import (
    AssignmentValidator,
    Dependency,
    RobotFleet,
    TaskAssignment,
    TaskCoordinator,
    inter_robot_collisions,
)


def _world_payload() -> dict[str, object]:
    return {
        "objects": [
            {"name": "box_a", "pose": {"position": [0.4, -0.2, 0.82]}, "size": [0.08, 0.08, 0.12]},
            {"name": "box_b", "pose": {"position": [0.4, 0.2, 0.82]}, "size": [0.08, 0.08, 0.12]},
        ],
        "surfaces": [
            {"name": "table_a", "pose": {"position": [0.8, -0.2, 0.74]}, "size": [0.4, 0.4, 0.05]},
            {"name": "table_b", "pose": {"position": [0.8, 0.2, 0.74]}, "size": [0.4, 0.4, 0.05]},
        ],
    }


def _move_task(name: str, target: str, destination: list[float], support: str) -> TaskDefinition:
    return TaskDefinition.from_dict(
        {
            "name": name,
            "actions": [
                {"type": "grasp", "target": target},
                {"type": "move", "target": target, "destination": destination, "speed": 1.0},
                {"type": "place", "target": target, "support": support},
            ],
        }
    )


def _grasp_task(name: str, target: str) -> TaskDefinition:
    return TaskDefinition.from_dict({"name": name, "actions": [{"type": "grasp", "target": target}]})


def _build_fleet() -> RobotFleet:
    fleet = RobotFleet.from_dict({"world": _world_payload()})
    fleet.add_robot("alpha", [0.0, -0.8, 0.0])
    fleet.add_robot("beta", [0.0, 0.8, 0.0])
    return fleet


def test_fleet_creation_with_multiple_robots() -> None:
    fleet = _build_fleet()

    assert set(fleet.robots) == {"alpha", "beta"}


def test_base_offset_positioning_changes_robot_pose() -> None:
    fleet = _build_fleet()

    assert fleet.get_robot("alpha").base_pose.position.tolist() == [0.0, -0.8, 0.0]
    assert fleet.get_robot("beta").base_pose.position.tolist() == [0.0, 0.8, 0.0]


def test_fleet_shares_single_world_state() -> None:
    fleet = _build_fleet()

    fleet.get_robot("alpha").base_pose.position[0] = 1.0
    fleet.world.objects["box_a"].held_by = "alpha:right_palm"

    assert fleet.get_robot("beta") is fleet.robots["beta"]
    assert fleet.world.objects["box_a"].held_by == "alpha:right_palm"


def test_fleet_rejects_duplicate_robot_names() -> None:
    fleet = _build_fleet()

    with pytest.raises(ValueError):
        fleet.add_robot("alpha", [1.0, 0.0, 0.0])


def test_assignment_validator_accepts_feasible_assignments() -> None:
    fleet = _build_fleet()
    assignments = [
        TaskAssignment("alpha", _move_task("a_task", "box_a", [0.9, -0.2, 0.9], "table_a")),
        TaskAssignment("beta", _move_task("b_task", "box_b", [0.9, 0.2, 0.9], "table_b")),
    ]

    report = AssignmentValidator().validate(fleet, assignments)

    assert report.is_valid, report.summary()


def test_assignment_validator_rejects_unknown_robot() -> None:
    fleet = _build_fleet()

    report = AssignmentValidator().validate(fleet, [TaskAssignment("gamma", _grasp_task("missing", "box_a"))])

    assert not report.is_valid


def test_assignment_validator_rejects_invalid_dependency_robot() -> None:
    fleet = _build_fleet()
    assignments = [
        TaskAssignment("alpha", _grasp_task("a_task", "box_a")),
        TaskAssignment("beta", _grasp_task("b_task", "box_b"), dependencies=[Dependency("gamma", 0)]),
    ]

    report = AssignmentValidator().validate(fleet, assignments)

    assert not report.is_valid


def test_assignment_validator_rejects_invalid_dependency_action_index() -> None:
    fleet = _build_fleet()
    assignments = [
        TaskAssignment("alpha", _grasp_task("a_task", "box_a")),
        TaskAssignment("beta", _grasp_task("b_task", "box_b"), dependencies=[Dependency("alpha", 3)]),
    ]

    report = AssignmentValidator().validate(fleet, assignments)

    assert not report.is_valid


def test_assignment_validator_rejects_dependency_cycles() -> None:
    fleet = _build_fleet()
    assignments = [
        TaskAssignment("alpha", _grasp_task("a_task", "box_a"), dependencies=[Dependency("beta", 0)]),
        TaskAssignment("beta", _grasp_task("b_task", "box_b"), dependencies=[Dependency("alpha", 0)]),
    ]

    report = AssignmentValidator().validate(fleet, assignments)

    assert not report.is_valid


def test_dependency_resolution_waits_for_upstream_completion() -> None:
    fleet = _build_fleet()
    assignments = [
        TaskAssignment("alpha", _grasp_task("a_task", "box_a")),
        TaskAssignment("beta", _grasp_task("b_task", "box_b"), dependencies=[Dependency("alpha", 0)]),
    ]

    record = TaskCoordinator(fleet, assignments).execute()

    assert record.completion_order[:2] == [("alpha", 0), ("beta", 0)]


def test_round_robin_execution_interleaves_long_running_actions() -> None:
    fleet = _build_fleet()
    assignments = [
        TaskAssignment("alpha", _move_task("a_task", "box_a", [0.9, -0.2, 0.9], "table_a")),
        TaskAssignment("beta", _move_task("b_task", "box_b", [0.9, 0.2, 0.9], "table_b")),
    ]

    record = TaskCoordinator(fleet, assignments).execute()
    alpha_times = [frame.time_s for frame in record.traces["alpha"].recording.frames]
    beta_times = [frame.time_s for frame in record.traces["beta"].recording.frames]

    assert record.steps > 2
    assert alpha_times[0] == 0.0
    assert beta_times[0] == 0.0
    assert alpha_times[2] < beta_times[-1]
    assert beta_times[2] > alpha_times[1]


def test_inter_robot_collision_detection_reports_close_robots() -> None:
    fleet = RobotFleet.from_dict({"world": _world_payload()})
    fleet.add_robot("alpha", [0.0, 0.0, 0.0])
    fleet.add_robot("beta", [0.02, 0.0, 0.0])

    collisions = inter_robot_collisions(fleet.robots, threshold=0.05)

    assert collisions


def test_inter_robot_collision_detection_ignores_separated_robots() -> None:
    fleet = _build_fleet()

    collisions = inter_robot_collisions(fleet.robots, threshold=0.05)

    assert collisions == []


def test_multi_robot_pick_and_place_scenario_updates_shared_world() -> None:
    fleet = _build_fleet()
    assignments = [
        TaskAssignment("alpha", _move_task("sort_a", "box_a", [0.95, -0.2, 0.9], "table_a")),
        TaskAssignment("beta", _move_task("sort_b", "box_b", [0.95, 0.2, 0.9], "table_b")),
    ]

    record = TaskCoordinator(fleet, assignments).execute()

    assert record.traces["alpha"].completed_action_count == 3
    assert record.traces["beta"].completed_action_count == 3
    assert fleet.world.objects["box_a"].held_by is None
    assert fleet.world.objects["box_b"].held_by is None
    assert fleet.world.objects["box_a"].pose.position[0] == pytest.approx(0.95)
    assert fleet.world.objects["box_b"].pose.position[0] == pytest.approx(0.95)


def test_coordinator_returns_per_robot_traces() -> None:
    fleet = _build_fleet()
    assignments = [
        TaskAssignment("alpha", _grasp_task("a_task", "box_a")),
        TaskAssignment("beta", _grasp_task("b_task", "box_b")),
    ]

    record = TaskCoordinator(fleet, assignments).execute()

    assert set(record.traces) == {"alpha", "beta"}
    assert record.traces["alpha"].recording is not None
    assert record.traces["beta"].recording is not None


def test_fleet_serialization_round_trip() -> None:
    fleet = _build_fleet()

    reloaded = RobotFleet.from_dict(fleet.to_dict())

    assert set(reloaded.robots) == {"alpha", "beta"}
    assert reloaded.get_robot("beta").base_pose.position.tolist() == [0.0, 0.8, 0.0]
    assert sorted(reloaded.world.objects) == ["box_a", "box_b"]


def test_cli_multi_subcommand_runs_example(tmp_path: Path) -> None:
    scenario = {
        "name": "warehouse_pair",
        "world": _world_payload(),
        "robots": [
            {"name": "alpha", "base_offset": [0.0, -0.8, 0.0]},
            {"name": "beta", "base_offset": [0.0, 0.8, 0.0]},
        ],
        "assignments": [
            {"robot_name": "alpha", "task": _move_task("sort_a", "box_a", [0.95, -0.2, 0.9], "table_a").to_dict()},
            {"robot_name": "beta", "task": _move_task("sort_b", "box_b", [0.95, 0.2, 0.9], "table_b").to_dict()},
        ],
    }
    scenario_path = tmp_path / "multi.yaml"
    scenario_path.write_text(yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8")

    exit_code = main(["multi", str(scenario_path)])

    assert exit_code == 0


def test_bundled_multi_robot_example_is_valid_yaml() -> None:
    payload = yaml.safe_load(Path("examples/multi_robot_warehouse.yaml").read_text(encoding="utf-8"))

    assert payload["robots"]
    assert payload["assignments"]

__all__ = ["test_fleet_creation_with_multiple_robots", "test_base_offset_positioning_changes_robot_pose", "test_fleet_shares_single_world_state", "test_fleet_rejects_duplicate_robot_names", "test_assignment_validator_accepts_feasible_assignments", "test_assignment_validator_rejects_unknown_robot", "test_assignment_validator_rejects_invalid_dependency_robot", "test_assignment_validator_rejects_invalid_dependency_action_index", "test_assignment_validator_rejects_dependency_cycles", "test_dependency_resolution_waits_for_upstream_completion", "test_round_robin_execution_interleaves_long_running_actions", "test_inter_robot_collision_detection_reports_close_robots", "test_inter_robot_collision_detection_ignores_separated_robots", "test_multi_robot_pick_and_place_scenario_updates_shared_world", "test_coordinator_returns_per_robot_traces", "test_fleet_serialization_round_trip", "test_cli_multi_subcommand_runs_example", "test_bundled_multi_robot_example_is_valid_yaml"]
