"""Built-in benchmark task definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from optisim.scene import SceneBuilder, SceneGraph, SceneNode, SceneRelation

_DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass(slots=True)
class BenchmarkTask:
    """Definition of a benchmark manipulation scenario."""

    name: str
    description: str
    build_scene: Callable[[], SceneGraph]
    primitive_sequence: list[dict]
    success_predicates: list[dict]
    difficulty: str
    tags: list[str]

    def __post_init__(self) -> None:
        if self.difficulty not in _DIFFICULTIES:
            raise ValueError(f"unsupported difficulty {self.difficulty!r}")
        self.primitive_sequence = [dict(step) for step in self.primitive_sequence]
        self.success_predicates = [dict(predicate) for predicate in self.success_predicates]
        self.tags = [str(tag) for tag in self.tags]


class BenchmarkSuite:
    """Registry for benchmark manipulation tasks."""

    DEFAULT: "BenchmarkSuite"

    def __init__(self) -> None:
        self.tasks: dict[str, BenchmarkTask] = {}

    def register(self, task: BenchmarkTask) -> None:
        """Register a task by name."""

        if task.name in self.tasks:
            raise ValueError(f"benchmark task {task.name!r} is already registered")
        self.tasks[task.name] = task

    def get(self, name: str) -> BenchmarkTask:
        """Return a task by name."""

        try:
            return self.tasks[name]
        except KeyError as exc:
            raise KeyError(f"unknown benchmark task {name!r}") from exc

    def list_tasks(self, difficulty: str | None = None, tag: str | None = None) -> list[str]:
        """List task names with optional difficulty and tag filters."""

        names: list[str] = []
        for name, task in sorted(self.tasks.items()):
            if difficulty is not None and task.difficulty != difficulty:
                continue
            if tag is not None and tag not in task.tags:
                continue
            names.append(name)
        return names


def _make_pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def _build_kitchen_scene() -> SceneGraph:
    graph = SceneBuilder.build_kitchen()
    return SceneBuilder.add_robot(graph, pose=_make_pose(0.2, 0.0, 0.75))


def _build_kitchen_with_shelf() -> SceneGraph:
    graph = _build_kitchen_scene()
    graph.add_node(
        SceneNode(
            id="kitchen_shelf",
            label="kitchen shelf",
            category="surface",
            pose=_make_pose(0.95, 0.25, 1.1),
            bbox=(0.4, 0.15, 0.2),
            properties={"supports_objects": True},
        )
    )
    graph.add_relation(SceneRelation("kitchen_shelf", "near", "kitchen_table", confidence=0.85))
    return graph


def _build_navigation_scene() -> SceneGraph:
    graph = SceneGraph()
    graph.add_node(
        SceneNode(
            id="humanoid",
            label="humanoid robot",
            category="robot",
            pose=_make_pose(0.0, 0.0, 0.0),
            bbox=(0.3, 0.3, 0.9),
            properties={"mobile": True, "manipulator": True},
        )
    )
    graph.add_node(
        SceneNode(
            id="obstacle",
            label="crate",
            category="obstacle",
            pose=_make_pose(0.45, 0.05, 0.2),
            bbox=(0.2, 0.2, 0.2),
            properties={"movable": True},
        )
    )
    graph.add_node(
        SceneNode(
            id="pickup_bin",
            label="pickup bin",
            category="surface",
            pose=_make_pose(0.6, -0.1, 0.2),
            bbox=(0.25, 0.2, 0.1),
            properties={"supports_objects": True},
        )
    )
    graph.add_node(
        SceneNode(
            id="parcel",
            label="parcel",
            category="container",
            pose=_make_pose(0.55, -0.1, 0.28),
            bbox=(0.07, 0.07, 0.07),
            properties={"graspable": True},
        )
    )
    graph.add_node(
        SceneNode(
            id="precision_pad",
            label="precision pad",
            category="surface",
            pose=_make_pose(0.72, -0.2, 0.42),
            bbox=(0.08, 0.08, 0.01),
            properties={"supports_objects": True},
        )
    )
    graph.add_relation(SceneRelation("parcel", "on", "pickup_bin"))
    return graph


def _build_warehouse_scene() -> SceneGraph:
    graph = SceneBuilder.build_warehouse()
    graph.update_pose("humanoid", _make_pose(0.7, -0.2, 0.2))
    return graph


def _default_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            name="pick_cup",
            description="Pick a cup from the kitchen table.",
            build_scene=_build_kitchen_scene,
            primitive_sequence=[
                {"primitive": "reach", "params": {"target_id": "cup", "end_effector": "right"}},
                {"primitive": "grasp", "params": {"target_id": "cup", "end_effector": "right"}},
            ],
            success_predicates=[{"subject_id": "cup", "predicate": "held_by", "object_id": "humanoid"}],
            difficulty="easy",
            tags=["kitchen", "pick-and-place"],
        ),
        BenchmarkTask(
            name="place_cup_on_shelf",
            description="Pick a cup from the table and place it on a shelf.",
            build_scene=_build_kitchen_with_shelf,
            primitive_sequence=[
                {"primitive": "reach", "params": {"target_id": "cup", "end_effector": "right"}},
                {"primitive": "grasp", "params": {"target_id": "cup", "end_effector": "right"}},
                {"primitive": "place", "params": {"object_id": "cup", "surface_id": "kitchen_shelf"}},
            ],
            success_predicates=[{"subject_id": "cup", "predicate": "on", "object_id": "kitchen_shelf"}],
            difficulty="easy",
            tags=["kitchen", "pick-and-place"],
        ),
        BenchmarkTask(
            name="push_obstacle",
            description="Push a movable obstacle out of the robot path.",
            build_scene=_build_navigation_scene,
            primitive_sequence=[
                {"primitive": "navigate", "params": {"target_id": "obstacle"}},
                {"primitive": "push", "params": {"target_id": "obstacle", "direction": [1.0, 0.0, 0.0]}},
            ],
            success_predicates=[{"subject_id": "obstacle", "predicate": "displaced"}],
            difficulty="easy",
            tags=["navigation", "manipulation"],
        ),
        BenchmarkTask(
            name="handover_tool",
            description="Pick a tool and hand it from one arm to the other.",
            build_scene=_build_kitchen_scene,
            primitive_sequence=[
                {"primitive": "reach", "params": {"target_id": "knife", "end_effector": "left"}},
                {"primitive": "grasp", "params": {"target_id": "knife", "end_effector": "left"}},
                {"primitive": "handover", "params": {"object_id": "knife", "from_arm": "left", "to_arm": "right"}},
            ],
            success_predicates=[{"subject_id": "knife", "predicate": "held_by", "object_id": "humanoid"}],
            difficulty="medium",
            tags=["bimanual", "kitchen", "tool-use"],
        ),
        BenchmarkTask(
            name="navigate_and_grasp",
            description="Navigate near an object and then grasp it.",
            build_scene=_build_navigation_scene,
            primitive_sequence=[
                {"primitive": "navigate", "params": {"target_id": "parcel"}},
                {"primitive": "grasp", "params": {"target_id": "parcel", "end_effector": "right"}},
            ],
            success_predicates=[{"subject_id": "parcel", "predicate": "held_by", "object_id": "humanoid"}],
            difficulty="medium",
            tags=["navigation", "pick-and-place"],
        ),
        BenchmarkTask(
            name="multi_step_kitchen",
            description="Navigate, reach, grasp, and place an object in a kitchen workflow.",
            build_scene=_build_kitchen_with_shelf,
            primitive_sequence=[
                {"primitive": "navigate", "params": {"target_id": "cup"}},
                {"primitive": "reach", "params": {"target_id": "cup", "end_effector": "right"}},
                {"primitive": "grasp", "params": {"target_id": "cup", "end_effector": "right"}},
                {"primitive": "place", "params": {"object_id": "cup", "surface_id": "kitchen_shelf"}},
            ],
            success_predicates=[{"subject_id": "cup", "predicate": "on", "object_id": "kitchen_shelf"}],
            difficulty="hard",
            tags=["kitchen", "navigation", "pick-and-place"],
        ),
        BenchmarkTask(
            name="warehouse_pick",
            description="Navigate to a warehouse pallet and grasp a box.",
            build_scene=_build_warehouse_scene,
            primitive_sequence=[
                {"primitive": "navigate", "params": {"target_id": "box"}},
                {"primitive": "grasp", "params": {"target_id": "box", "end_effector": "right"}},
            ],
            success_predicates=[{"subject_id": "box", "predicate": "held_by", "object_id": "humanoid"}],
            difficulty="medium",
            tags=["warehouse", "navigation", "pick-and-place"],
        ),
        BenchmarkTask(
            name="precision_place",
            description="Place a small object onto a precise elevated surface.",
            build_scene=_build_navigation_scene,
            primitive_sequence=[
                {"primitive": "navigate", "params": {"target_id": "parcel"}},
                {"primitive": "grasp", "params": {"target_id": "parcel", "end_effector": "right"}},
                {"primitive": "place", "params": {"object_id": "parcel", "surface_id": "precision_pad"}},
            ],
            success_predicates=[{"subject_id": "parcel", "predicate": "on", "object_id": "precision_pad"}],
            difficulty="hard",
            tags=["manipulation", "precision", "pick-and-place"],
        ),
    ]


def _build_default_suite() -> BenchmarkSuite:
    suite = BenchmarkSuite()
    for task in _default_tasks():
        suite.register(task)
    return suite


BenchmarkSuite.DEFAULT = _build_default_suite()

__all__ = ["BenchmarkTask", "BenchmarkSuite"]
