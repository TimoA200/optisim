from __future__ import annotations

from pathlib import Path

import numpy as np

from optisim.cli import build_parser, main
from optisim.grasp import (
    ContactPatch,
    ContactPoint,
    GraspExecutor,
    GraspPlanner,
    GripperType,
    default_parallel_jaw,
    default_suction,
    default_three_finger,
    force_closure,
    friction_cone_check,
    grasp_wrench_space,
    min_resisted_wrench,
    slip_margin,
    surface_contacts,
)
from optisim.math3d import Pose, Quaternion, vec3
from optisim.robot import build_humanoid_model
from optisim.sim import ExecutionEngine, WorldState
from optisim.sim.world import ObjectState


def _box(
    *,
    name: str = "box",
    position: tuple[float, float, float] = (0.42, -0.12, 0.81),
    size: tuple[float, float, float] = (0.08, 0.08, 0.12),
    mass_kg: float = 0.75,
) -> ObjectState:
    return ObjectState(
        name=name,
        pose=Pose(position=vec3(position), orientation=Quaternion.identity()),
        size=size,
        mass_kg=mass_kg,
    )


def test_contact_point_normalizes_inputs() -> None:
    contact = ContactPoint(position=[1.0, 2.0, 3.0], normal=[0.0, 0.0, 2.0], force=[0.0, 0.0, -1.0])
    assert np.allclose(contact.position, [1.0, 2.0, 3.0])
    assert np.allclose(contact.normal, [0.0, 0.0, 1.0])
    assert np.allclose(contact.force, [0.0, 0.0, -1.0])


def test_contact_patch_computes_centroid_and_force() -> None:
    patch = ContactPatch(
        [
            ContactPoint(position=[0.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0], force=[1.0, 0.0, 0.0]),
            ContactPoint(position=[2.0, 0.0, 0.0], normal=[-1.0, 0.0, 0.0], force=[-0.5, 0.0, 0.0]),
        ]
    )
    assert np.allclose(patch.centroid, [1.0, 0.0, 0.0])
    assert np.allclose(patch.total_force, [0.5, 0.0, 0.0])


def test_friction_cone_check_accepts_force_inside_cone() -> None:
    contact = ContactPoint(position=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], friction_coeff=0.5)
    assert friction_cone_check(contact, np.asarray([0.2, 0.0, -1.0], dtype=np.float64))


def test_friction_cone_check_rejects_force_outside_cone() -> None:
    contact = ContactPoint(position=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], friction_coeff=0.3)
    assert not friction_cone_check(contact, np.asarray([0.5, 0.0, -1.0], dtype=np.float64))


def test_surface_contacts_parallel_jaw_returns_opposed_pair() -> None:
    contacts = surface_contacts(
        _box(),
        {
            "position": [0.42, -0.12, 0.81],
            "orientation": Quaternion.identity(),
            "aperture": 0.2,
            "gripper_type": "parallel_jaw",
        },
    )
    assert len(contacts) == 2
    assert np.isclose(np.dot(contacts[0].normal, contacts[1].normal), -1.0)


def test_surface_contacts_suction_returns_single_face_contact() -> None:
    contacts = surface_contacts(
        _box(),
        {
            "position": [0.42, -0.12, 0.87],
            "orientation": Quaternion.identity(),
            "aperture": 0.08,
            "gripper_type": "suction",
        },
    )
    assert len(contacts) == 1
    assert np.allclose(contacts[0].position, [0.42, -0.12, 0.87])


def test_surface_contacts_three_finger_returns_triplet() -> None:
    contacts = surface_contacts(
        _box(),
        {
            "position": [0.42, -0.12, 0.81],
            "orientation": Quaternion.identity(),
            "aperture": 0.2,
            "gripper_type": "three_finger",
        },
    )
    assert len(contacts) == 3


def test_surface_contacts_rejects_aperture_too_small() -> None:
    contacts = surface_contacts(
        _box(size=(0.08, 0.08, 0.12)),
        {
            "position": [0.42, -0.12, 0.81],
            "orientation": Quaternion.from_euler(0.0, -np.pi / 2.0, 0.0),
            "aperture": 0.05,
            "gripper_type": "parallel_jaw",
        },
    )
    assert contacts == []


def test_force_closure_positive_for_antipodal_contacts() -> None:
    contacts = [
        ContactPoint(position=[0.04, 0.0, 0.0], normal=[1.0, 0.0, 0.0], friction_coeff=0.5),
        ContactPoint(position=[-0.04, 0.0, 0.0], normal=[-1.0, 0.0, 0.0], friction_coeff=0.5),
    ]
    assert force_closure(contacts)


def test_force_closure_negative_for_single_contact() -> None:
    contacts = [ContactPoint(position=[0.0, 0.0, 0.06], normal=[0.0, 0.0, 1.0], friction_coeff=0.8)]
    assert not force_closure(contacts)


def test_grasp_wrench_space_has_expected_shape() -> None:
    contacts = [
        ContactPoint(position=[0.04, 0.0, 0.0], normal=[1.0, 0.0, 0.0], friction_coeff=0.5),
        ContactPoint(position=[-0.04, 0.0, 0.0], normal=[-1.0, 0.0, 0.0], friction_coeff=0.5),
    ]
    wrench_space = grasp_wrench_space(contacts)
    assert wrench_space.shape == (6, 10)


def test_min_resisted_wrench_positive_for_stable_antipodal_grasp() -> None:
    contacts = [
        ContactPoint(position=[0.04, 0.0, 0.0], normal=[1.0, 0.0, 0.0], friction_coeff=0.6),
        ContactPoint(position=[-0.04, 0.0, 0.0], normal=[-1.0, 0.0, 0.0], friction_coeff=0.6),
    ]
    assert min_resisted_wrench(contacts) > 0.0


def test_min_resisted_wrench_is_zero_for_empty_contacts() -> None:
    assert min_resisted_wrench([]) == 0.0


def test_slip_margin_positive_when_load_is_safe() -> None:
    contact = ContactPoint(position=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], friction_coeff=0.8)
    assert slip_margin(contact, np.asarray([0.1, 0.0, -1.0], dtype=np.float64)) > 0.0


def test_slip_margin_negative_when_tangential_load_is_high() -> None:
    contact = ContactPoint(position=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], friction_coeff=0.2)
    assert slip_margin(contact, np.asarray([1.0, 0.0, -1.0], dtype=np.float64)) < 0.0


def test_parallel_jaw_planner_returns_ranked_grasps() -> None:
    planner = GraspPlanner()
    grasps = planner.plan_grasps(_box(), default_parallel_jaw(), n_candidates=3)
    assert 1 <= len(grasps) <= 3
    assert all(grasp.gripper_type == GripperType.PARALLEL_JAW.value for grasp in grasps)
    assert grasps == sorted(grasps, key=lambda grasp: grasp.quality_score, reverse=True)


def test_suction_planner_returns_face_grasps() -> None:
    planner = GraspPlanner()
    grasps = planner.plan_grasps(_box(), default_suction(), n_candidates=6)
    assert grasps
    assert all(len(grasp.contact_points) == 1 for grasp in grasps)


def test_three_finger_planner_returns_tripod_contacts() -> None:
    planner = GraspPlanner()
    grasps = planner.plan_grasps(_box(), default_three_finger(), n_candidates=3)
    assert grasps
    assert all(len(grasp.contact_points) == 3 for grasp in grasps)


def test_antipodal_grasps_returns_axis_aligned_candidates() -> None:
    planner = GraspPlanner()
    grasps = planner.antipodal_grasps(_box(), default_parallel_jaw())
    assert len(grasps) == 3
    assert all(len(grasp.contact_points) >= 2 for grasp in grasps)


def test_planner_rejects_oversized_object_for_parallel_jaw() -> None:
    planner = GraspPlanner()
    grasps = planner.plan_grasps(_box(size=(0.40, 0.30, 0.20)), default_parallel_jaw(), n_candidates=5)
    assert grasps == []


def test_planner_rejects_unreachable_grasps_when_robot_is_provided() -> None:
    robot = build_humanoid_model()
    planner = GraspPlanner(robot=robot)
    far_object = _box(position=(5.0, 0.0, 0.8))
    grasps = planner.plan_grasps(far_object, default_parallel_jaw(), n_candidates=5)
    assert grasps == []


def test_evaluate_grasp_is_deterministic() -> None:
    planner = GraspPlanner()
    grasp = planner.plan_grasps(_box(), default_parallel_jaw(), n_candidates=1)[0]
    score_a = planner.evaluate_grasp(grasp, _box())
    score_b = planner.evaluate_grasp(grasp, _box())
    assert score_a == score_b


def test_grasp_executor_returns_result_with_contacts() -> None:
    world = WorldState.with_defaults()
    robot = build_humanoid_model()
    planner = GraspPlanner(robot=robot)
    grasp = planner.plan_grasps(world.objects["box"], default_parallel_jaw(), n_candidates=1)[0]
    engine = ExecutionEngine(robot=robot, world=world)
    result = GraspExecutor().execute_grasp(engine, robot, grasp, world.objects["box"])
    assert len(result.contact_points) >= 2
    assert result.stability_score > 0.0
    assert isinstance(result.slip_detected, bool)


def test_grasp_executor_reports_failure_for_unstable_zero_friction_grasp() -> None:
    world = WorldState.with_defaults()
    robot = build_humanoid_model()
    grasp = GraspPlanner(robot=robot).plan_grasps(world.objects["box"], default_parallel_jaw(), n_candidates=1)[0]
    for contact in grasp.contact_points:
        contact.friction_coeff = 0.0
    engine = ExecutionEngine(robot=robot, world=world)
    result = GraspExecutor().execute_grasp(engine, robot, grasp, world.objects["box"])
    assert result.success is False


def test_grasp_cli_parser_accepts_subcommand() -> None:
    args = build_parser().parse_args(["grasp", "examples/pick_and_place.yaml", "--gripper", "parallel_jaw"])
    assert args.command == "grasp"
    assert args.gripper == "parallel_jaw"


def test_grasp_cli_prints_ranked_results(capsys) -> None:
    exit_code = main(["grasp", "examples/pick_and_place.yaml", "--gripper", "parallel_jaw", "--top-k", "2"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "object: box" in captured.out
    assert "score=" in captured.out


def test_task_example_path_exists_for_cli_smoke() -> None:
    assert Path("examples/pick_and_place.yaml").exists()
