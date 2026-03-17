"""Benchmark execution and predicate evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from optisim.benchmark.suite import BenchmarkSuite, BenchmarkTask
from optisim.primitives import PrimitiveExecutor, PrimitiveResult, PrimitiveStatus
from optisim.scene import SceneGraph


@dataclass(slots=True)
class BenchmarkResult:
    """Outcome of a benchmark task execution."""

    task_name: str
    success: bool
    partial_success: float
    steps_completed: int
    steps_total: int
    primitive_results: list[PrimitiveResult]
    elapsed_steps: int
    error_message: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkEvaluator:
    """Execute benchmark tasks with motion primitives."""

    def __init__(self, executor: PrimitiveExecutor | None = None) -> None:
        self.executor = PrimitiveExecutor() if executor is None else executor

    def run_task(self, task: BenchmarkTask, robot_joints: np.ndarray | None = None) -> BenchmarkResult:
        """Run a benchmark task against a fresh scene."""

        scene = task.build_scene()
        robot_id = _find_robot_id(scene)
        joint_state = _normalize_joints(robot_joints)
        error_message: str | None = None

        try:
            primitive_results = self.executor.execute_sequence(
                scene=scene,
                robot_id=robot_id,
                robot_joints=joint_state,
                sequence=task.primitive_sequence,
            )
        except Exception as exc:
            primitive_results = []
            error_message = str(exc)

        steps_completed = sum(1 for result in primitive_results if result.status is PrimitiveStatus.SUCCESS)
        steps_total = len(task.primitive_sequence)
        partial_success = float(steps_completed / steps_total) if steps_total else 1.0
        elapsed_steps = int(
            sum(len(result.joint_trajectory or []) for result in primitive_results if result.status is PrimitiveStatus.SUCCESS)
        )
        sequence_success = steps_completed == steps_total
        predicates_ok, failing = self.check_predicates(scene, task.success_predicates)
        success = bool(sequence_success and predicates_ok and error_message is None)
        if error_message is None and not success:
            if primitive_results and primitive_results[-1].status is PrimitiveStatus.FAILURE:
                error_message = primitive_results[-1].message
            elif failing:
                error_message = "; ".join(failing)

        metadata = {
            "difficulty": task.difficulty,
            "tags": list(task.tags),
            "failing_predicates": failing,
            "robot_id": robot_id,
        }
        return BenchmarkResult(
            task_name=task.name,
            success=success,
            partial_success=partial_success,
            steps_completed=steps_completed,
            steps_total=steps_total,
            primitive_results=primitive_results,
            elapsed_steps=elapsed_steps,
            error_message=error_message,
            metadata=metadata,
        )

    def run_suite(
        self,
        suite: BenchmarkSuite,
        task_names: list[str] | None = None,
        robot_joints: np.ndarray | None = None,
    ) -> list[BenchmarkResult]:
        """Run all tasks in a benchmark suite or a named subset."""

        names = suite.list_tasks() if task_names is None else list(task_names)
        return [self.run_task(suite.get(name), robot_joints=robot_joints) for name in names]

    def check_predicates(self, scene: SceneGraph, predicates: list[dict]) -> tuple[bool, list[str]]:
        """Check scene predicates against the final relation set."""

        failing: list[str] = []
        for predicate in predicates:
            subject_id = str(
                predicate.get("subject_id", predicate.get("subject", predicate.get("node_id", "")))
            )
            relation_name = str(predicate["predicate"])
            object_id = predicate.get("object_id", predicate.get("object"))
            expected = bool(predicate.get("value", True))
            if object_id is None:
                actual = bool(scene.get_relations(subject_id=subject_id, predicate=relation_name, object_id=subject_id))
                object_text = subject_id
            else:
                object_text = str(object_id)
                actual = bool(scene.get_relations(subject_id=subject_id, predicate=relation_name, object_id=object_text))
            if actual == expected:
                continue
            description = predicate.get("description")
            if description:
                failing.append(str(description))
            else:
                verb = "missing" if expected else "unexpected"
                failing.append(f"{verb} predicate: ({subject_id}, {relation_name}, {object_text})")
        return not failing, failing


def _find_robot_id(scene: SceneGraph) -> str:
    for node in scene.nodes.values():
        if node.category == "robot":
            return node.id
    raise ValueError("benchmark scene does not contain a robot node")


def _normalize_joints(robot_joints: np.ndarray | None) -> np.ndarray:
    if robot_joints is None:
        return np.zeros(31, dtype=float)
    joints = np.asarray(robot_joints, dtype=float)
    if joints.ndim != 1:
        raise ValueError("robot_joints must be a one-dimensional vector")
    return joints.copy()


__all__ = ["BenchmarkResult", "BenchmarkEvaluator"]
