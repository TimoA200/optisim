"""Joint-space RRT and RRT-Connect planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
StateValidator = Callable[[Vector], bool]
EdgeValidator = Callable[[Vector, Vector], bool]


@dataclass(slots=True)
class RRTConfig:
    """Numerical settings for the joint-space RRT planner."""

    max_iterations: int = 1_500
    step_size: float = 0.3
    goal_bias: float = 0.15
    goal_threshold: float = 0.2


@dataclass(slots=True)
class _Tree:
    vertices: list[Vector]
    parents: list[int]

    def add(self, vertex: Vector, parent: int) -> int:
        self.vertices.append(vertex)
        self.parents.append(parent)
        return len(self.vertices) - 1

    def nearest_index(self, sample: Vector) -> int:
        distances = [float(np.linalg.norm(vertex - sample)) for vertex in self.vertices]
        return int(np.argmin(distances))

    def path_to_root(self, index: int) -> list[Vector]:
        path: list[Vector] = []
        current = index
        while current >= 0:
            path.append(self.vertices[current])
            current = self.parents[current]
        path.reverse()
        return path


def plan_rrt(
    start: Vector,
    goal: Vector,
    *,
    lower_bounds: Vector,
    upper_bounds: Vector,
    is_state_valid: StateValidator,
    is_edge_valid: EdgeValidator,
    config: RRTConfig,
    rng: np.random.Generator,
) -> tuple[list[Vector], int]:
    """Plan a joint-space path with a single-tree RRT."""

    tree = _Tree(vertices=[start.copy()], parents=[-1])
    for iteration in range(1, config.max_iterations + 1):
        sample = goal if rng.random() < config.goal_bias else rng.uniform(lower_bounds, upper_bounds)
        nearest_index = tree.nearest_index(sample)
        candidate = _steer(tree.vertices[nearest_index], sample, config.step_size, lower_bounds, upper_bounds)
        if not is_state_valid(candidate) or not is_edge_valid(tree.vertices[nearest_index], candidate):
            continue
        new_index = tree.add(candidate, nearest_index)
        if np.linalg.norm(candidate - goal) <= config.goal_threshold and is_edge_valid(candidate, goal):
            goal_index = tree.add(goal.copy(), new_index)
            return tree.path_to_root(goal_index), iteration
    return [], config.max_iterations


def plan_rrt_connect(
    start: Vector,
    goal: Vector,
    *,
    lower_bounds: Vector,
    upper_bounds: Vector,
    is_state_valid: StateValidator,
    is_edge_valid: EdgeValidator,
    config: RRTConfig,
    rng: np.random.Generator,
) -> tuple[list[Vector], int]:
    """Plan a joint-space path with bidirectional RRT-Connect."""

    tree_a = _Tree(vertices=[start.copy()], parents=[-1])
    tree_b = _Tree(vertices=[goal.copy()], parents=[-1])

    for iteration in range(1, config.max_iterations + 1):
        sample = goal if rng.random() < config.goal_bias else rng.uniform(lower_bounds, upper_bounds)
        extend_result = _extend_tree(
            tree=tree_a,
            sample=sample,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            is_state_valid=is_state_valid,
            is_edge_valid=is_edge_valid,
            step_size=config.step_size,
        )
        if extend_result is None:
            tree_a, tree_b = tree_b, tree_a
            continue

        new_index, new_vertex = extend_result
        connect_index = _connect_trees(
            tree=tree_b,
            target=new_vertex,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            is_state_valid=is_state_valid,
            is_edge_valid=is_edge_valid,
            step_size=config.step_size,
            goal_threshold=config.goal_threshold,
        )
        if connect_index is not None:
            path_a = tree_a.path_to_root(new_index)
            path_b = tree_b.path_to_root(connect_index)
            path_b.reverse()
            return _deduplicate_vertices(path_a + path_b), iteration
        tree_a, tree_b = tree_b, tree_a

    return [], config.max_iterations


def _extend_tree(
    *,
    tree: _Tree,
    sample: Vector,
    lower_bounds: Vector,
    upper_bounds: Vector,
    is_state_valid: StateValidator,
    is_edge_valid: EdgeValidator,
    step_size: float,
) -> tuple[int, Vector] | None:
    nearest_index = tree.nearest_index(sample)
    candidate = _steer(tree.vertices[nearest_index], sample, step_size, lower_bounds, upper_bounds)
    if not is_state_valid(candidate) or not is_edge_valid(tree.vertices[nearest_index], candidate):
        return None
    return tree.add(candidate, nearest_index), candidate


def _connect_trees(
    *,
    tree: _Tree,
    target: Vector,
    lower_bounds: Vector,
    upper_bounds: Vector,
    is_state_valid: StateValidator,
    is_edge_valid: EdgeValidator,
    step_size: float,
    goal_threshold: float,
) -> int | None:
    while True:
        nearest_index = tree.nearest_index(target)
        nearest = tree.vertices[nearest_index]
        distance = float(np.linalg.norm(target - nearest))
        if distance <= goal_threshold:
            if is_edge_valid(nearest, target):
                return tree.add(target.copy(), nearest_index)
            return None

        candidate = _steer(nearest, target, step_size, lower_bounds, upper_bounds)
        if not is_state_valid(candidate) or not is_edge_valid(nearest, candidate):
            return None
        new_index = tree.add(candidate, nearest_index)
        if np.linalg.norm(candidate - target) <= goal_threshold and is_edge_valid(candidate, target):
            return tree.add(target.copy(), new_index)


def _steer(
    source: Vector,
    target: Vector,
    step_size: float,
    lower_bounds: Vector,
    upper_bounds: Vector,
) -> Vector:
    direction = target - source
    distance = float(np.linalg.norm(direction))
    if distance <= step_size:
        candidate = target
    elif distance == 0.0:
        candidate = source
    else:
        candidate = source + (direction / distance) * step_size
    return np.clip(candidate, lower_bounds, upper_bounds)


def _deduplicate_vertices(vertices: list[Vector]) -> list[Vector]:
    path: list[Vector] = []
    for vertex in vertices:
        if path and np.allclose(path[-1], vertex):
            continue
        path.append(vertex)
    return path
