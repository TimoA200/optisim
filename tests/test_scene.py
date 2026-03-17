from __future__ import annotations

import numpy as np
import pytest

import optisim
from optisim.scene import SceneBuilder, SceneGraph, SceneNode, SceneQuery, SceneRelation


def make_pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def make_graph() -> SceneGraph:
    graph = SceneGraph()
    graph.add_node(
        SceneNode("robot", "robot", "robot", pose=make_pose(0.0, 0.0, 0.0), bbox=(0.3, 0.3, 0.9))
    )
    graph.add_node(
        SceneNode("table", "table", "surface", pose=make_pose(0.8, 0.0, 0.75), bbox=(0.5, 0.4, 0.05))
    )
    graph.add_node(
        SceneNode(
            "cup",
            "cup",
            "container",
            pose=make_pose(0.6, 0.0, 0.8),
            bbox=(0.04, 0.04, 0.06),
            properties={"graspable": True, "mass": 0.3},
        )
    )
    graph.add_node(
        SceneNode(
            "knife",
            "knife",
            "tool",
            pose=make_pose(1.4, 0.0, 0.8),
            bbox=(0.1, 0.01, 0.01),
            properties={"graspable": True},
        )
    )
    graph.add_node(
        SceneNode(
            "shelf",
            "shelf",
            "surface",
            pose=make_pose(1.6, 0.5, 1.2),
            bbox=(0.6, 0.2, 1.0),
        )
    )
    graph.add_relation(SceneRelation("cup", "on", "table"))
    graph.add_relation(SceneRelation("knife", "on", "table"))
    graph.add_relation(SceneRelation("table", "near", "shelf"))
    return graph


def test_scene_node_defaults_identity_pose() -> None:
    node = SceneNode(id="cup", label="cup", category="container")

    np.testing.assert_allclose(node.pose, np.eye(4, dtype=float))
    assert node.bbox == (0.0, 0.0, 0.0)
    assert node.properties == {}
    assert node.parent_id is None


def test_scene_node_preserves_properties() -> None:
    node = SceneNode(
        id="cup",
        label="cup",
        category="container",
        bbox=(0.1, 0.2, 0.3),
        properties={"graspable": True, "mass": 0.3},
        parent_id="table",
    )

    assert node.bbox == (0.1, 0.2, 0.3)
    assert node.properties["graspable"] is True
    assert node.parent_id == "table"


def test_scene_node_invalid_pose_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SceneNode(id="cup", label="cup", category="container", pose=np.eye(3, dtype=float))


def test_scene_node_to_dict_roundtrip() -> None:
    node = SceneNode(
        id="bowl",
        label="bowl",
        category="container",
        pose=make_pose(0.1, 0.2, 0.3),
        bbox=(0.2, 0.2, 0.1),
        properties={"graspable": True},
        parent_id="table",
    )

    restored = SceneNode.from_dict(node.to_dict())

    assert restored.id == "bowl"
    np.testing.assert_allclose(restored.pose, node.pose)
    assert restored.bbox == node.bbox
    assert restored.properties == node.properties
    assert restored.parent_id == "table"


def test_scene_relation_defaults_confidence_to_one() -> None:
    relation = SceneRelation(subject_id="cup", predicate="on", object_id="table")

    assert relation.confidence == 1.0


def test_scene_relation_to_dict_roundtrip() -> None:
    relation = SceneRelation("cup", "near", "table", confidence=0.7)

    restored = SceneRelation.from_dict(relation.to_dict())

    assert restored == relation


def test_scene_graph_add_node_and_get_node() -> None:
    graph = SceneGraph()
    node = SceneNode("table", "table", "surface")

    graph.add_node(node)

    assert graph.get_node("table").label == "table"


def test_scene_graph_add_node_duplicate_raises_value_error() -> None:
    graph = SceneGraph()
    graph.add_node(SceneNode("table", "table", "surface"))

    with pytest.raises(ValueError):
        graph.add_node(SceneNode("table", "other", "surface"))


def test_scene_graph_add_node_with_missing_parent_raises_key_error() -> None:
    graph = SceneGraph()

    with pytest.raises(KeyError):
        graph.add_node(SceneNode("cup", "cup", "container", parent_id="table"))


def test_scene_graph_remove_node_removes_relations() -> None:
    graph = make_graph()

    graph.remove_node("cup")

    assert "cup" not in graph.nodes
    assert graph.get_relations(subject_id="cup") == []


def test_scene_graph_remove_node_clears_child_parent_ids() -> None:
    graph = SceneGraph()
    graph.add_node(SceneNode("table", "table", "surface"))
    graph.add_node(SceneNode("cup", "cup", "container", parent_id="table"))

    graph.remove_node("table")

    assert graph.get_node("cup").parent_id is None


def test_scene_graph_remove_missing_node_raises_key_error() -> None:
    graph = SceneGraph()

    with pytest.raises(KeyError):
        graph.remove_node("missing")


def test_scene_graph_update_pose_replaces_pose() -> None:
    graph = SceneGraph()
    graph.add_node(SceneNode("robot", "robot", "robot"))
    pose = make_pose(1.0, 2.0, 3.0)

    graph.update_pose("robot", pose)

    np.testing.assert_allclose(graph.get_node("robot").pose, pose)


def test_scene_graph_update_pose_invalid_shape_raises_value_error() -> None:
    graph = SceneGraph()
    graph.add_node(SceneNode("robot", "robot", "robot"))

    with pytest.raises(ValueError):
        graph.update_pose("robot", np.eye(3, dtype=float))


def test_scene_graph_add_relation_requires_existing_nodes() -> None:
    graph = SceneGraph()
    graph.add_node(SceneNode("cup", "cup", "container"))

    with pytest.raises(KeyError):
        graph.add_relation(SceneRelation("cup", "on", "table"))


def test_scene_graph_add_relation_replaces_duplicate_triple() -> None:
    graph = SceneGraph()
    graph.add_node(SceneNode("cup", "cup", "container"))
    graph.add_node(SceneNode("table", "table", "surface"))
    graph.add_relation(SceneRelation("cup", "on", "table", confidence=0.2))

    graph.add_relation(SceneRelation("cup", "on", "table", confidence=0.9))

    assert len(graph.relations) == 1
    assert graph.relations[0].confidence == 0.9


def test_scene_graph_remove_relation_removes_matching_relation() -> None:
    graph = make_graph()

    graph.remove_relation("cup", "on", "table")

    assert graph.get_relations(subject_id="cup", predicate="on", object_id="table") == []


def test_scene_graph_remove_missing_relation_raises_key_error() -> None:
    graph = make_graph()

    with pytest.raises(KeyError):
        graph.remove_relation("cup", "in", "table")


def test_scene_graph_get_relations_filters_by_subject() -> None:
    graph = make_graph()

    relations = graph.get_relations(subject_id="cup")

    assert len(relations) == 1
    assert relations[0].predicate == "on"


def test_scene_graph_get_relations_filters_by_predicate() -> None:
    graph = make_graph()

    relations = graph.get_relations(predicate="on")

    assert len(relations) == 2


def test_scene_graph_get_relations_filters_by_object() -> None:
    graph = make_graph()

    relations = graph.get_relations(object_id="table")

    assert len(relations) == 2


def test_scene_graph_neighbors_returns_all_related_nodes() -> None:
    graph = make_graph()

    neighbors = graph.neighbors("table")

    assert neighbors == ["cup", "knife", "shelf"]


def test_scene_graph_subgraph_depth_zero_contains_only_root() -> None:
    graph = make_graph()

    subgraph = graph.subgraph("table", depth=0)

    assert list(subgraph.nodes) == ["table"]
    assert subgraph.relations == []


def test_scene_graph_subgraph_depth_one_collects_adjacent_nodes() -> None:
    graph = make_graph()

    subgraph = graph.subgraph("table", depth=1)

    assert set(subgraph.nodes) == {"cup", "knife", "shelf", "table"}
    assert len(subgraph.relations) == 3


def test_scene_graph_subgraph_missing_root_raises_key_error() -> None:
    graph = make_graph()

    with pytest.raises(KeyError):
        graph.subgraph("missing")


def test_scene_graph_subgraph_negative_depth_raises_value_error() -> None:
    graph = make_graph()

    with pytest.raises(ValueError):
        graph.subgraph("table", depth=-1)


def test_scene_graph_serialization_roundtrip() -> None:
    graph = make_graph()

    restored = SceneGraph.from_dict(graph.to_dict())

    assert set(restored.nodes) == set(graph.nodes)
    assert len(restored.relations) == len(graph.relations)
    np.testing.assert_allclose(restored.get_node("cup").pose, graph.get_node("cup").pose)


def test_scene_graph_from_dict_supports_embedded_node_ids() -> None:
    graph = SceneGraph.from_dict(
        {
            "nodes": {
                "table": {"label": "table", "category": "surface"},
                "cup": {"label": "cup", "category": "container"},
            },
            "relations": [{"subject_id": "cup", "predicate": "on", "object_id": "table"}],
        }
    )

    assert set(graph.nodes) == {"cup", "table"}
    assert graph.get_relations(predicate="on")[0].object_id == "table"


def test_scene_graph_from_dict_supports_child_before_parent() -> None:
    graph = SceneGraph.from_dict(
        {
            "nodes": {
                "cup": {"label": "cup", "category": "container", "parent_id": "table"},
                "table": {"label": "table", "category": "surface"},
            }
        }
    )

    assert graph.get_node("cup").parent_id == "table"


def test_scene_builder_build_kitchen_contains_expected_nodes() -> None:
    graph = SceneBuilder.build_kitchen()

    assert {"kitchen_table", "countertop", "cup", "bowl", "knife", "sink"} <= set(graph.nodes)


def test_scene_builder_build_kitchen_relations_reference_existing_nodes() -> None:
    graph = SceneBuilder.build_kitchen()

    assert graph.relations
    for relation in graph.relations:
        assert relation.subject_id in graph.nodes
        assert relation.object_id in graph.nodes


def test_scene_builder_build_warehouse_contains_robot() -> None:
    graph = SceneBuilder.build_warehouse()

    assert "humanoid" in graph.nodes
    assert graph.get_node("humanoid").category == "robot"


def test_scene_builder_add_robot_uses_custom_pose() -> None:
    graph = SceneGraph()
    pose = make_pose(1.0, 1.0, 0.0)

    SceneBuilder.add_robot(graph, robot_id="atlas", pose=pose)

    np.testing.assert_allclose(graph.get_node("atlas").pose, pose)


def test_scene_query_find_by_category_returns_matching_nodes() -> None:
    graph = make_graph()

    nodes = SceneQuery.find_by_category(graph, "surface")

    assert {node.id for node in nodes} == {"shelf", "table"}


def test_scene_query_find_graspable_returns_only_graspable_nodes() -> None:
    graph = make_graph()

    nodes = SceneQuery.find_graspable(graph)

    assert {node.id for node in nodes} == {"cup", "knife"}


def test_scene_query_find_on_surface_returns_objects_on_surface() -> None:
    graph = make_graph()

    nodes = SceneQuery.find_on_surface(graph, "table")

    assert {node.id for node in nodes} == {"cup", "knife"}


def test_scene_query_find_on_surface_missing_surface_raises_key_error() -> None:
    graph = make_graph()

    with pytest.raises(KeyError):
        SceneQuery.find_on_surface(graph, "missing")


def test_scene_query_find_path_returns_empty_when_unconnected() -> None:
    graph = make_graph()

    path = SceneQuery.find_path(graph, "robot", "cup")

    assert path == []


def test_scene_query_find_path_returns_connected_route() -> None:
    graph = make_graph()
    graph.add_relation(SceneRelation("robot", "near", "table"))

    path = SceneQuery.find_path(graph, "robot", "shelf")

    assert path == ["robot", "table", "shelf"]


def test_scene_query_find_path_same_start_and_goal_returns_singleton() -> None:
    graph = make_graph()

    assert SceneQuery.find_path(graph, "table", "table") == ["table"]


def test_scene_query_find_reachable_returns_nodes_within_radius() -> None:
    graph = make_graph()

    nodes = SceneQuery.find_reachable(graph, "robot", reach_radius=1.1)

    assert {node.id for node in nodes} == {"cup", "table"}


def test_scene_query_find_reachable_empty_graph_returns_no_nodes() -> None:
    graph = SceneGraph()
    graph.add_node(SceneNode("robot", "robot", "robot"))

    assert SceneQuery.find_reachable(graph, "robot") == []


def test_scene_query_find_reachable_negative_radius_raises_value_error() -> None:
    graph = make_graph()

    with pytest.raises(ValueError):
        SceneQuery.find_reachable(graph, "robot", reach_radius=-0.1)


def test_scene_query_to_tamp_predicates_format() -> None:
    graph = make_graph()

    predicates = SceneQuery.to_tamp_predicates(graph)

    assert predicates
    assert set(predicates[0]) == {"name", "args", "value"}
    assert isinstance(predicates[0]["args"], list)
    assert isinstance(predicates[0]["value"], bool)


def test_scene_query_to_tamp_predicates_empty_graph_returns_empty_list() -> None:
    assert SceneQuery.to_tamp_predicates(SceneGraph()) == []


def test_scene_module_is_available_from_top_level_package() -> None:
    assert optisim.scene.SceneGraph is SceneGraph
    assert optisim.SceneGraph is SceneGraph
