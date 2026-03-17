"""Grid-based and graph-based path planning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count

import numpy as np

from optisim.occupancy import CollisionChecker, VoxelGrid

WorldPoint = tuple[float, float, float]
GridIndex = tuple[int, int, int]


def _world_point(value: tuple[float, float, float] | list[float] | np.ndarray, *, name: str) -> WorldPoint:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3D point")
    return (float(array[0]), float(array[1]), float(array[2]))


def _grid_distance(a: GridIndex, b: GridIndex) -> float:
    return float(np.linalg.norm(np.subtract(a, b, dtype=np.float64)))


@dataclass(slots=True)
class GridNode:
    """Node used for voxel-grid search."""

    i: int
    j: int
    k: int
    parent: GridNode | None = None


class AStarPlanner:
    """A* path planner on a 3D occupancy grid."""

    def __init__(self, grid: VoxelGrid, allow_diagonal: bool = False) -> None:
        self.grid = grid
        self.allow_diagonal = bool(allow_diagonal)

    def plan(self, start_world: tuple[float, float, float], goal_world: tuple[float, float, float]) -> list[WorldPoint] | None:
        start = self.grid.world_to_voxel(*_world_point(start_world, name="start_world"))
        goal = self.grid.world_to_voxel(*_world_point(goal_world, name="goal_world"))
        if not self._is_free_index(start) or not self._is_free_index(goal):
            return None
        if start == goal:
            return [self.grid.voxel_to_world(*start)]

        frontier: list[tuple[float, int, GridIndex]] = []
        token = count()
        heappush(frontier, (self.heuristic(start, goal), next(token), start))

        nodes: dict[GridIndex, GridNode] = {start: GridNode(*start)}
        cost_so_far: dict[GridIndex, float] = {start: 0.0}
        visited: set[GridIndex] = set()

        while frontier:
            _, _, current = heappop(frontier)
            if current in visited:
                continue
            if current == goal:
                return [self.grid.voxel_to_world(*voxel) for voxel in self.reconstruct_path(nodes[current])]
            visited.add(current)

            parent = nodes[current]
            for neighbor in self._neighbors(current):
                if neighbor in visited:
                    continue
                new_cost = cost_so_far[current] + _grid_distance(current, neighbor)
                best = cost_so_far.get(neighbor)
                if best is not None and new_cost >= best:
                    continue
                cost_so_far[neighbor] = new_cost
                nodes[neighbor] = GridNode(*neighbor, parent=parent)
                priority = new_cost + self.heuristic(neighbor, goal)
                heappush(frontier, (priority, next(token), neighbor))

        return None

    def heuristic(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        delta = np.abs(np.subtract(a, b, dtype=np.int64))
        if self.allow_diagonal:
            return float(np.max(delta))
        return float(np.sum(delta))

    def reconstruct_path(self, node: GridNode) -> list[tuple[int, int, int]]:
        path: list[GridIndex] = []
        current: GridNode | None = node
        while current is not None:
            path.append((current.i, current.j, current.k))
            current = current.parent
        path.reverse()
        return path

    def _neighbors(self, index: GridIndex) -> list[GridIndex]:
        if self.allow_diagonal:
            offsets = [
                (di, dj, dk)
                for di in (-1, 0, 1)
                for dj in (-1, 0, 1)
                for dk in (-1, 0, 1)
                if (di, dj, dk) != (0, 0, 0)
            ]
        else:
            offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        neighbors: list[GridIndex] = []
        for di, dj, dk in offsets:
            neighbor = (index[0] + di, index[1] + dj, index[2] + dk)
            if self._is_free_index(neighbor):
                neighbors.append(neighbor)
        return neighbors

    def _is_free_index(self, index: GridIndex) -> bool:
        return bool(self.grid.is_in_bounds(*index) and self.grid.data[index] == 0)


class WaypointSmoother:
    """Greedy waypoint pruning using occupancy-grid line-of-sight checks."""

    def __init__(self, grid: VoxelGrid, robot_radius: float = 0.0) -> None:
        self.grid = grid
        self.robot_radius = float(robot_radius)
        if self.robot_radius < 0.0:
            raise ValueError("robot_radius must be non-negative")
        self._checker = CollisionChecker(grid)

    def smooth(self, waypoints: list[tuple[float, float, float]]) -> list[WorldPoint]:
        if len(waypoints) <= 2:
            return [_world_point(waypoint, name="waypoint") for waypoint in waypoints]

        points = [_world_point(waypoint, name="waypoint") for waypoint in waypoints]
        smoothed = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            next_index = anchor + 1
            for candidate in range(len(points) - 1, anchor, -1):
                if self.has_line_of_sight(points[anchor], points[candidate]):
                    next_index = candidate
                    break
            smoothed.append(points[next_index])
            anchor = next_index
        return smoothed

    def has_line_of_sight(self, a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
        start = _world_point(a, name="a")
        end = _world_point(b, name="b")
        start_voxel = self.grid.world_to_voxel(*start)
        end_voxel = self.grid.world_to_voxel(*end)
        if not self.grid.is_in_bounds(*start_voxel) or not self.grid.is_in_bounds(*end_voxel):
            return False

        start_array = np.asarray(start, dtype=np.float64)
        delta = np.asarray(end, dtype=np.float64) - start_array
        distance = float(np.linalg.norm(delta))
        steps = max(int(np.ceil(distance / max(self.grid.resolution * 0.5, 1e-9))), 1)
        for alpha in np.linspace(0.0, 1.0, steps + 1, dtype=np.float64):
            point = tuple(float(value) for value in (start_array + alpha * delta))
            voxel = self.grid.world_to_voxel(*point)
            if not self.grid.is_in_bounds(*voxel):
                return False
            if self.robot_radius > 0.0:
                if self._checker.check_sphere(point, self.robot_radius):
                    return False
            elif self.grid.data[voxel] == 255:
                return False
        return True


class RoadmapPlanner:
    """Probabilistic roadmap planner over random free-space samples."""

    def __init__(self, grid: VoxelGrid, n_nodes: int = 100, seed: int = 0, robot_radius: float = 0.0) -> None:
        self.grid = grid
        self.n_nodes = int(n_nodes)
        self.seed = int(seed)
        self.robot_radius = float(robot_radius)
        if self.n_nodes < 0:
            raise ValueError("n_nodes must be non-negative")
        if self.robot_radius < 0.0:
            raise ValueError("robot_radius must be non-negative")
        self.nodes: list[WorldPoint] = []
        self.edges: list[tuple[int, int]] = []
        self.max_connect_dist = max(2.5 * self.grid.resolution, 0.3 * max(self.grid.shape) * self.grid.resolution)
        self._smoother = WaypointSmoother(grid, robot_radius=self.robot_radius)
        self._checker = CollisionChecker(grid)

    def build(self) -> "RoadmapPlanner":
        rng = np.random.default_rng(self.seed)
        free_indices = np.argwhere(self.grid.data == 0)
        if free_indices.size == 0 or self.n_nodes == 0:
            self.nodes = []
            self.edges = []
            return self

        count_nodes = min(self.n_nodes, int(free_indices.shape[0]))
        chosen = rng.choice(free_indices.shape[0], size=count_nodes, replace=False)
        sampled = free_indices[chosen]
        self.nodes = [self.grid.voxel_to_world(int(i), int(j), int(k)) for i, j, k in sampled]
        self.edges = []
        for source in range(len(self.nodes)):
            for target in range(source + 1, len(self.nodes)):
                if self._distance(self.nodes[source], self.nodes[target]) > self.max_connect_dist:
                    continue
                if self._smoother.has_line_of_sight(self.nodes[source], self.nodes[target]):
                    self.edges.append((source, target))
        return self

    def plan(self, start: tuple[float, float, float], goal: tuple[float, float, float]) -> list[WorldPoint] | None:
        start_point = _world_point(start, name="start")
        goal_point = _world_point(goal, name="goal")
        if not self._is_free_world(start_point) or not self._is_free_world(goal_point):
            return None
        if not self.nodes:
            self.build()
        if self._smoother.has_line_of_sight(start_point, goal_point):
            return [start_point, goal_point]

        nodes = list(self.nodes) + [start_point, goal_point]
        start_index = len(nodes) - 2
        goal_index = len(nodes) - 1
        adjacency = self._adjacency_for(nodes, start_index, goal_index)
        if not adjacency[start_index] or not adjacency[goal_index]:
            return None

        queue: list[tuple[float, int]] = [(0.0, start_index)]
        previous: dict[int, int | None] = {start_index: None}
        distances: dict[int, float] = {start_index: 0.0}

        while queue:
            current_cost, current = heappop(queue)
            if current == goal_index:
                break
            if current_cost > distances.get(current, float("inf")):
                continue
            for neighbor, edge_cost in adjacency[current]:
                next_cost = current_cost + edge_cost
                if next_cost >= distances.get(neighbor, float("inf")):
                    continue
                distances[neighbor] = next_cost
                previous[neighbor] = current
                heappush(queue, (next_cost, neighbor))

        if goal_index not in previous:
            return None

        path_indices: list[int] = []
        current: int | None = goal_index
        while current is not None:
            path_indices.append(current)
            current = previous[current]
        path_indices.reverse()
        return [nodes[index] for index in path_indices]

    def _adjacency_for(self, nodes: list[WorldPoint], start_index: int, goal_index: int) -> list[list[tuple[int, float]]]:
        adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
        for source, target in self.edges:
            if source >= len(self.nodes) or target >= len(self.nodes):
                continue
            distance = self._distance(nodes[source], nodes[target])
            adjacency[source].append((target, distance))
            adjacency[target].append((source, distance))

        for extra_index in (start_index, goal_index):
            for neighbor in range(len(nodes) - 2):
                distance = self._distance(nodes[extra_index], nodes[neighbor])
                if distance > self.max_connect_dist:
                    continue
                if self._smoother.has_line_of_sight(nodes[extra_index], nodes[neighbor]):
                    adjacency[extra_index].append((neighbor, distance))
                    adjacency[neighbor].append((extra_index, distance))

        direct_distance = self._distance(nodes[start_index], nodes[goal_index])
        if direct_distance <= self.max_connect_dist and self._smoother.has_line_of_sight(nodes[start_index], nodes[goal_index]):
            adjacency[start_index].append((goal_index, direct_distance))
            adjacency[goal_index].append((start_index, direct_distance))
        return adjacency

    def _is_free_world(self, point: WorldPoint) -> bool:
        voxel = self.grid.world_to_voxel(*point)
        if not self.grid.is_in_bounds(*voxel):
            return False
        if self.robot_radius > 0.0:
            return not self._checker.check_sphere(point, self.robot_radius)
        return bool(self.grid.data[voxel] == 0)

    @staticmethod
    def _distance(a: WorldPoint, b: WorldPoint) -> float:
        return float(np.linalg.norm(np.subtract(a, b, dtype=np.float64)))


__all__ = [
    "GridNode",
    "AStarPlanner",
    "WaypointSmoother",
    "RoadmapPlanner",
]
