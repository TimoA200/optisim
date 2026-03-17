from __future__ import annotations

import numpy as np

from optisim import (
    GraspPrimitive,
    HandoverPrimitive,
    MotionPrimitive,
    NavigatePrimitive,
    PlacePrimitive,
    PrimitiveExecutor,
    PrimitiveResult,
    PrimitiveStatus,
    PushPrimitive,
    ReachPrimitive,
    SceneGraph,
    SceneNode,
    SceneQuery,
    SceneRelation,
    apply_effects,
)


def make_pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def make_scene() -> SceneGraph:
    scene = SceneGraph()
    scene.add_node(
        SceneNode(
            "robot",
            "robot",
            "robot",
            pose=make_pose(0.0, 0.0, 0.0),
            properties={"mobile": True},
        )
    )
    scene.add_node(
        SceneNode(
            "cup",
            "cup",
            "container",
            pose=make_pose(0.5, 0.0, 0.2),
            properties={"graspable": True},
        )
    )
    scene.add_node(
        SceneNode(
            "book",
            "book",
            "object",
            pose=make_pose(0.4, 0.2, 0.2),
            properties={"graspable": False},
        )
    )
    scene.add_node(SceneNode("table", "table", "surface", pose=make_pose(0.7, 0.0, 0.3)))
    scene.add_node(SceneNode("shelf", "shelf", "surface", pose=make_pose(0.9, 0.3, 0.5)))
    scene.add_node(SceneNode("far_box", "far_box", "object", pose=make_pose(2.0, 0.0, 0.8)))
    return scene


def joint_state() -> np.ndarray:
    return np.zeros(31, dtype=float)


def test_primitive_result_creation() -> None:
    result = PrimitiveResult(status=PrimitiveStatus.SUCCESS, message="ok", duration_s=1.2)
    assert result.status is PrimitiveStatus.SUCCESS
    assert result.message == "ok"
    assert result.joint_trajectory is None
    assert result.duration_s == 1.2
    assert result.metadata == {}


def test_reach_primitive_instantiation() -> None:
    primitive = ReachPrimitive({"target_id": "cup", "end_effector": "left"})
    assert isinstance(primitive, MotionPrimitive)
    assert primitive.name == "reach"


def test_grasp_primitive_instantiation() -> None:
    primitive = GraspPrimitive({"target_id": "cup", "end_effector": "right"})
    assert primitive.name == "grasp"


def test_place_primitive_instantiation() -> None:
    primitive = PlacePrimitive({"object_id": "cup", "surface_id": "table"})
    assert primitive.name == "place"


def test_push_primitive_instantiation() -> None:
    primitive = PushPrimitive({"target_id": "cup", "direction": np.array([1.0, 0.0, 0.0])})
    assert primitive.name == "push"


def test_handover_primitive_instantiation() -> None:
    primitive = HandoverPrimitive({"object_id": "cup", "from_arm": "left", "to_arm": "right"})
    assert primitive.name == "handover"


def test_navigate_primitive_instantiation() -> None:
    primitive = NavigatePrimitive({"target_id": "table"})
    assert primitive.name == "navigate"


def test_reach_preconditions_succeed() -> None:
    ok, reason = ReachPrimitive({"target_id": "cup", "end_effector": "left"}).check_preconditions(make_scene(), "robot")
    assert ok is True
    assert reason == ""


def test_reach_preconditions_fail_for_missing_target() -> None:
    ok, reason = ReachPrimitive({"target_id": "missing", "end_effector": "left"}).check_preconditions(make_scene(), "robot")
    assert ok is False
    assert "does not exist" in reason


def test_reach_preconditions_fail_for_missing_robot() -> None:
    ok, reason = ReachPrimitive({"target_id": "cup", "end_effector": "left"}).check_preconditions(make_scene(), "missing")
    assert ok is False
    assert "robot" in reason


def test_grasp_preconditions_succeed_when_near() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("robot", "near", "cup"))
    ok, reason = GraspPrimitive({"target_id": "cup", "end_effector": "right"}).check_preconditions(scene, "robot")
    assert ok is True
    assert reason == ""


def test_grasp_preconditions_fail_when_not_graspable() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("robot", "near", "book"))
    ok, reason = GraspPrimitive({"target_id": "book", "end_effector": "right"}).check_preconditions(scene, "robot")
    assert ok is False
    assert "not graspable" in reason


def test_grasp_preconditions_fail_when_not_near() -> None:
    ok, reason = GraspPrimitive({"target_id": "cup", "end_effector": "right"}).check_preconditions(make_scene(), "robot")
    assert ok is False
    assert "not near" in reason


def test_place_preconditions_succeed_when_object_held() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("cup", "held_by", "robot"))
    ok, reason = PlacePrimitive({"object_id": "cup", "surface_id": "table"}).check_preconditions(scene, "robot")
    assert ok is True
    assert reason == ""


def test_place_preconditions_fail_when_object_not_held() -> None:
    ok, reason = PlacePrimitive({"object_id": "cup", "surface_id": "table"}).check_preconditions(make_scene(), "robot")
    assert ok is False
    assert "not held" in reason


def test_push_preconditions_succeed_when_near() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("robot", "near", "cup"))
    ok, reason = PushPrimitive({"target_id": "cup", "direction": np.array([1.0, 0.0, 0.0])}).check_preconditions(scene, "robot")
    assert ok is True
    assert reason == ""


def test_push_preconditions_fail_when_not_near() -> None:
    ok, reason = PushPrimitive({"target_id": "cup", "direction": np.array([1.0, 0.0, 0.0])}).check_preconditions(make_scene(), "robot")
    assert ok is False
    assert "not near" in reason


def test_handover_preconditions_succeed() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("cup", "held_by", "robot"))
    ok, reason = HandoverPrimitive({"object_id": "cup", "from_arm": "left", "to_arm": "right"}).check_preconditions(scene, "robot")
    assert ok is True
    assert reason == ""


def test_handover_preconditions_fail_when_same_arm() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("cup", "held_by", "robot"))
    ok, reason = HandoverPrimitive({"object_id": "cup", "from_arm": "left", "to_arm": "left"}).check_preconditions(scene, "robot")
    assert ok is False
    assert "different" in reason


def test_navigate_preconditions_succeed() -> None:
    ok, reason = NavigatePrimitive({"target_id": "table"}).check_preconditions(make_scene(), "robot")
    assert ok is True
    assert reason == ""


def test_navigate_preconditions_fail_for_missing_target() -> None:
    ok, reason = NavigatePrimitive({"target_id": "missing"}).check_preconditions(make_scene(), "robot")
    assert ok is False
    assert "does not exist" in reason


def test_reach_execute_returns_success_when_target_reachable() -> None:
    result = ReachPrimitive({"target_id": "cup", "end_effector": "left"}).execute(make_scene(), "robot", joint_state())
    assert result.status is PrimitiveStatus.SUCCESS


def test_reach_execute_returns_failure_when_target_unreachable() -> None:
    result = ReachPrimitive({"target_id": "far_box", "end_effector": "left"}).execute(make_scene(), "robot", joint_state())
    assert result.status is PrimitiveStatus.FAILURE


def test_grasp_execute_returns_success() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("robot", "near", "cup"))
    result = GraspPrimitive({"target_id": "cup", "end_effector": "right"}).execute(scene, "robot", joint_state())
    assert result.status is PrimitiveStatus.SUCCESS


def test_grasp_primitive_fails_when_not_graspable() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("robot", "near", "book"))
    result = GraspPrimitive({"target_id": "book", "end_effector": "right"}).execute(scene, "robot", joint_state())
    assert result.status is PrimitiveStatus.FAILURE


def test_place_execute_returns_success() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("cup", "held_by", "robot"))
    result = PlacePrimitive({"object_id": "cup", "surface_id": "table"}).execute(scene, "robot", joint_state())
    assert result.status is PrimitiveStatus.SUCCESS


def test_place_primitive_fails_when_object_not_held() -> None:
    result = PlacePrimitive({"object_id": "cup", "surface_id": "table"}).execute(make_scene(), "robot", joint_state())
    assert result.status is PrimitiveStatus.FAILURE


def test_push_execute_returns_success() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("robot", "near", "cup"))
    result = PushPrimitive({"target_id": "cup", "direction": np.array([1.0, 0.0, 0.0])}).execute(scene, "robot", joint_state())
    assert result.status is PrimitiveStatus.SUCCESS


def test_handover_execute_returns_success() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("cup", "held_by", "robot"))
    result = HandoverPrimitive({"object_id": "cup", "from_arm": "left", "to_arm": "right"}).execute(scene, "robot", joint_state())
    assert result.status is PrimitiveStatus.SUCCESS


def test_handover_primitive_fails_when_from_arm_equals_to_arm() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("cup", "held_by", "robot"))
    result = HandoverPrimitive({"object_id": "cup", "from_arm": "left", "to_arm": "left"}).execute(scene, "robot", joint_state())
    assert result.status is PrimitiveStatus.FAILURE


def test_navigate_execute_returns_success() -> None:
    result = NavigatePrimitive({"target_id": "table"}).execute(make_scene(), "robot", joint_state())
    assert result.status is PrimitiveStatus.SUCCESS


def test_reach_trajectory_shape() -> None:
    result = ReachPrimitive({"target_id": "cup", "end_effector": "left"}).execute(make_scene(), "robot", joint_state())
    assert result.joint_trajectory is not None
    assert len(result.joint_trajectory) == 10
    assert all(isinstance(step, np.ndarray) and step.shape == (31,) for step in result.joint_trajectory)


def test_grasp_trajectory_shape() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("robot", "near", "cup"))
    result = GraspPrimitive({"target_id": "cup", "end_effector": "right"}).execute(scene, "robot", joint_state())
    assert result.joint_trajectory is not None
    assert len(result.joint_trajectory) == 5
    assert all(step.shape == (31,) for step in result.joint_trajectory)


def test_place_trajectory_shape() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("cup", "held_by", "robot"))
    result = PlacePrimitive({"object_id": "cup", "surface_id": "table"}).execute(scene, "robot", joint_state())
    assert result.joint_trajectory is not None
    assert len(result.joint_trajectory) == 8


def test_push_trajectory_shape() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("robot", "near", "cup"))
    result = PushPrimitive({"target_id": "cup", "direction": np.array([1.0, 0.0, 0.0])}).execute(scene, "robot", joint_state())
    assert result.joint_trajectory is not None
    assert len(result.joint_trajectory) == 6


def test_handover_trajectory_shape() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("cup", "held_by", "robot"))
    result = HandoverPrimitive({"object_id": "cup", "from_arm": "left", "to_arm": "right"}).execute(scene, "robot", joint_state())
    assert result.joint_trajectory is not None
    assert len(result.joint_trajectory) == 12


def test_navigate_trajectory_shape() -> None:
    result = NavigatePrimitive({"target_id": "table"}).execute(make_scene(), "robot", joint_state())
    assert result.joint_trajectory is not None
    assert len(result.joint_trajectory) == 15


def test_apply_effects_adds_relation() -> None:
    scene = make_scene()
    apply_effects(scene, [{"name": "near", "args": ["robot", "cup"], "value": True}])
    assert scene.get_relations(subject_id="robot", predicate="near", object_id="cup")


def test_apply_effects_removes_relation() -> None:
    scene = make_scene()
    scene.add_relation(SceneRelation("robot", "near", "cup"))
    apply_effects(scene, [{"name": "near", "args": ["robot", "cup"], "value": False}])
    assert scene.get_relations(subject_id="robot", predicate="near", object_id="cup") == []


def test_apply_effects_supports_unary_relation_convention() -> None:
    scene = make_scene()
    apply_effects(scene, [{"name": "displaced", "args": ["cup"], "value": True}])
    assert scene.get_relations(subject_id="cup", predicate="displaced", object_id="cup")


def test_primitive_executor_get_returns_correct_type() -> None:
    executor = PrimitiveExecutor()
    primitive = executor.get("reach", {"target_id": "cup", "end_effector": "left"})
    assert isinstance(primitive, ReachPrimitive)


def test_primitive_executor_execute_sequence_happy_path() -> None:
    scene = make_scene()
    executor = PrimitiveExecutor()
    results = executor.execute_sequence(
        scene,
        "robot",
        joint_state(),
        [
            {"primitive": "reach", "params": {"target_id": "cup", "end_effector": "left"}},
            {"primitive": "grasp", "params": {"target_id": "cup", "end_effector": "left"}},
            {"primitive": "place", "params": {"object_id": "cup", "surface_id": "table"}},
        ],
    )
    assert [result.status for result in results] == [PrimitiveStatus.SUCCESS] * 3
    assert scene.get_relations(subject_id="cup", predicate="on", object_id="table")
    assert scene.get_relations(subject_id="cup", predicate="held_by", object_id="robot") == []


def test_primitive_executor_execute_sequence_stops_on_failure() -> None:
    scene = make_scene()
    executor = PrimitiveExecutor()
    results = executor.execute_sequence(
        scene,
        "robot",
        joint_state(),
        [
            {"primitive": "reach", "params": {"target_id": "cup", "end_effector": "left"}},
            {"primitive": "grasp", "params": {"target_id": "book", "end_effector": "left"}},
            {"primitive": "place", "params": {"object_id": "cup", "surface_id": "table"}},
        ],
    )
    assert len(results) == 2
    assert results[-1].status is PrimitiveStatus.FAILURE
    assert scene.get_relations(subject_id="cup", predicate="on", object_id="table") == []


def test_available_primitives_returns_all_six() -> None:
    executor = PrimitiveExecutor()
    assert executor.available_primitives() == ["grasp", "handover", "navigate", "place", "push", "reach"]


def test_reach_effects_match_expected_predicate() -> None:
    effects = ReachPrimitive({"target_id": "cup", "end_effector": "left"}).get_effects(make_scene(), "robot")
    assert effects == [{"name": "near", "args": ["robot", "cup"], "value": True}]


def test_grasp_effects_match_expected_predicates() -> None:
    effects = GraspPrimitive({"target_id": "cup", "end_effector": "left"}).get_effects(make_scene(), "robot")
    assert effects[0]["name"] == "held_by"
    assert effects[1]["value"] is False


def test_navigate_makes_target_scene_node_reachable_by_query() -> None:
    scene = make_scene()
    reachable = {node.id for node in SceneQuery.find_reachable(scene, "robot")}
    assert "cup" in reachable
