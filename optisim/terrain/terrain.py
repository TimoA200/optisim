"""Terrain perception and heightmap-aware locomotion planning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np


def _as_height_grid(heights: np.ndarray | list[list[float]], expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(heights, dtype=np.float64)
    if array.shape != expected_shape:
        raise ValueError(f"heights must have shape {expected_shape}, received {array.shape}")
    return array


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / norm


@dataclass(slots=True)
class HeightMap:
    """Grid-based terrain representation."""

    resolution: float
    width: int
    height: int
    heights: np.ndarray

    def __post_init__(self) -> None:
        self.resolution = float(self.resolution)
        self.width = int(self.width)
        self.height = int(self.height)
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        self.heights = _as_height_grid(self.heights, (self.height, self.width))

    @classmethod
    def from_flat(
        cls,
        width: int = 32,
        height: int = 32,
        resolution: float = 0.1,
        height_value: float = 0.0,
    ) -> "HeightMap":
        return cls(
            resolution=resolution,
            width=width,
            height=height,
            heights=np.full((height, width), float(height_value), dtype=np.float64),
        )

    @classmethod
    def from_random(
        cls,
        seed: int,
        width: int = 32,
        height: int = 32,
        resolution: float = 0.1,
        height_scale: float = 0.2,
    ) -> "HeightMap":
        rng = np.random.default_rng(seed)
        heights = rng.normal(0.0, float(height_scale), size=(height, width))
        for _ in range(3):
            padded = np.pad(heights, 1, mode="edge")
            heights = (
                padded[1:-1, 1:-1]
                + padded[:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
            ) / 5.0
        return cls(resolution=resolution, width=width, height=height, heights=heights)

    @classmethod
    def from_heightfn(
        cls,
        fn: Callable[[float, float], float],
        width: int = 32,
        height: int = 32,
        resolution: float = 0.1,
    ) -> "HeightMap":
        heights = np.zeros((height, width), dtype=np.float64)
        for row in range(height):
            for col in range(width):
                x = float(col * resolution)
                y = float(row * resolution)
                heights[row, col] = float(fn(x, y))
        return cls(resolution=resolution, width=width, height=height, heights=heights)

    def grid_x(self, col: int) -> float:
        return float(col * self.resolution)

    def grid_y(self, row: int) -> float:
        return float(row * self.resolution)

    def _clamped_indices(self, x: float, y: float) -> tuple[int, int, int, int, float, float]:
        gx = np.clip(float(x) / self.resolution, 0.0, self.width - 1.0)
        gy = np.clip(float(y) / self.resolution, 0.0, self.height - 1.0)
        x0 = int(np.floor(gx))
        y0 = int(np.floor(gy))
        x1 = min(x0 + 1, self.width - 1)
        y1 = min(y0 + 1, self.height - 1)
        tx = float(gx - x0)
        ty = float(gy - y0)
        return x0, x1, y0, y1, tx, ty

    def get_height(self, x: float, y: float) -> float:
        x0, x1, y0, y1, tx, ty = self._clamped_indices(x, y)
        z00 = self.heights[y0, x0]
        z10 = self.heights[y0, x1]
        z01 = self.heights[y1, x0]
        z11 = self.heights[y1, x1]
        z0 = (1.0 - tx) * z00 + tx * z10
        z1 = (1.0 - tx) * z01 + tx * z11
        return float((1.0 - ty) * z0 + ty * z1)

    def get_normal(self, x: float, y: float) -> np.ndarray:
        delta = self.resolution
        dzdx = (self.get_height(x + delta, y) - self.get_height(x - delta, y)) / (2.0 * delta)
        dzdy = (self.get_height(x, y + delta) - self.get_height(x, y - delta)) / (2.0 * delta)
        return _normalize(np.array([-dzdx, -dzdy, 1.0], dtype=np.float64))

    def get_slope(self, x: float, y: float) -> float:
        normal = self.get_normal(x, y)
        return float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))

    def is_traversable(self, x: float, y: float, max_slope_deg: float = 30.0) -> bool:
        return bool(self.get_slope(x, y) <= float(max_slope_deg))


@dataclass(slots=True)
class TerrainPatch:
    """Named region of terrain."""

    center: tuple[float, float]
    radius: float
    label: str

    def __post_init__(self) -> None:
        if len(self.center) != 2:
            raise ValueError("center must contain x and y")
        self.center = (float(self.center[0]), float(self.center[1]))
        self.radius = float(self.radius)
        self.label = str(self.label)
        if self.radius < 0.0:
            raise ValueError("radius must be non-negative")


def _component_patches(mask: np.ndarray, heightmap: HeightMap, label: str) -> list[TerrainPatch]:
    visited = np.zeros(mask.shape, dtype=bool)
    patches: list[TerrainPatch] = []
    rows, cols = mask.shape
    for row in range(rows):
        for col in range(cols):
            if not mask[row, col] or visited[row, col]:
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cell_row, cell_col = queue.popleft()
                cells.append((cell_row, cell_col))
                for next_row, next_col in (
                    (cell_row - 1, cell_col),
                    (cell_row + 1, cell_col),
                    (cell_row, cell_col - 1),
                    (cell_row, cell_col + 1),
                ):
                    if 0 <= next_row < rows and 0 <= next_col < cols and mask[next_row, next_col] and not visited[next_row, next_col]:
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
            xs = np.asarray([heightmap.grid_x(col_idx) for _, col_idx in cells], dtype=np.float64)
            ys = np.asarray([heightmap.grid_y(row_idx) for row_idx, _ in cells], dtype=np.float64)
            center = (float(xs.mean()), float(ys.mean()))
            points = np.column_stack((xs, ys))
            radius = float(np.max(np.linalg.norm(points - np.asarray(center), axis=1))) if len(points) else 0.0
            patches.append(TerrainPatch(center=center, radius=radius, label=label))
    return patches


class TerrainAnalyzer:
    """Analyze a heightmap for locomotion-relevant structure."""

    def __init__(self, heightmap: HeightMap, flat_slope_deg: float = 5.0, step_height_threshold: float = 0.15) -> None:
        self.heightmap = heightmap
        self.flat_slope_deg = float(flat_slope_deg)
        self.step_height_threshold = float(step_height_threshold)
        self.traversability_map = np.zeros((heightmap.height, heightmap.width), dtype=bool)
        self._flat_regions: list[TerrainPatch] = []
        self._step_regions: list[TerrainPatch] = []

    def analyze(self) -> "TerrainAnalyzer":
        slope_map = np.zeros((self.heightmap.height, self.heightmap.width), dtype=np.float64)
        step_mask = np.zeros((self.heightmap.height, self.heightmap.width), dtype=bool)
        for row in range(self.heightmap.height):
            for col in range(self.heightmap.width):
                x = self.heightmap.grid_x(col)
                y = self.heightmap.grid_y(row)
                slope_map[row, col] = self.heightmap.get_slope(x, y)
                current = self.heightmap.heights[row, col]
                neighbor_diffs = []
                if row > 0:
                    neighbor_diffs.append(abs(current - self.heightmap.heights[row - 1, col]))
                if row + 1 < self.heightmap.height:
                    neighbor_diffs.append(abs(current - self.heightmap.heights[row + 1, col]))
                if col > 0:
                    neighbor_diffs.append(abs(current - self.heightmap.heights[row, col - 1]))
                if col + 1 < self.heightmap.width:
                    neighbor_diffs.append(abs(current - self.heightmap.heights[row, col + 1]))
                step_mask[row, col] = bool(neighbor_diffs and max(neighbor_diffs) >= self.step_height_threshold)

        flat_mask = slope_map < self.flat_slope_deg
        self.traversability_map = flat_mask & (~step_mask)
        self._flat_regions = _component_patches(flat_mask, self.heightmap, "flat")
        self._step_regions = _component_patches(step_mask, self.heightmap, "step")
        return self

    def get_flat_regions(self) -> list[TerrainPatch]:
        return list(self._flat_regions)

    def get_step_regions(self) -> list[TerrainPatch]:
        return list(self._step_regions)


class TerrainAdaptiveFootstep:
    """Project footstep targets onto a terrain surface."""

    def __init__(self, heightmap: HeightMap) -> None:
        self.heightmap = heightmap

    def adjust_footsteps(self, footsteps: list[tuple[float, float]] | list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        adjusted: list[tuple[float, float, float]] = []
        for step in footsteps:
            if len(step) < 2:
                raise ValueError("each footstep must contain at least x and y")
            x = float(step[0])
            y = float(step[1])
            adjusted.append((x, y, self.heightmap.get_height(x, y)))
        return adjusted


class TerrainCostMap:
    """Cost grid combining slope and step penalties."""

    def __init__(
        self,
        heightmap: HeightMap,
        slope_weight: float = 1.0,
        step_weight: float = 10.0,
        step_height_threshold: float = 0.15,
    ) -> None:
        self.heightmap = heightmap
        self.slope_weight = float(slope_weight)
        self.step_weight = float(step_weight)
        self.step_height_threshold = float(step_height_threshold)
        self.cost_map = np.zeros((heightmap.height, heightmap.width), dtype=np.float64)

    def build_costmap(self) -> np.ndarray:
        cost_map = np.zeros((self.heightmap.height, self.heightmap.width), dtype=np.float64)
        for row in range(self.heightmap.height):
            for col in range(self.heightmap.width):
                x = self.heightmap.grid_x(col)
                y = self.heightmap.grid_y(row)
                slope_cost = self.slope_weight * (self.heightmap.get_slope(x, y) / 90.0)
                current = self.heightmap.heights[row, col]
                neighbor_diff = 0.0
                for next_row, next_col in (
                    (max(row - 1, 0), col),
                    (min(row + 1, self.heightmap.height - 1), col),
                    (row, max(col - 1, 0)),
                    (row, min(col + 1, self.heightmap.width - 1)),
                ):
                    neighbor_diff = max(neighbor_diff, abs(current - self.heightmap.heights[next_row, next_col]))
                step_cost = self.step_weight if neighbor_diff >= self.step_height_threshold else 0.0
                cost_map[row, col] = slope_cost + step_cost
        self.cost_map = cost_map
        return cost_map

    def get_cost(self, x: float, y: float) -> float:
        if not np.any(self.cost_map):
            self.build_costmap()
        x0, x1, y0, y1, tx, ty = self.heightmap._clamped_indices(x, y)
        c00 = self.cost_map[y0, x0]
        c10 = self.cost_map[y0, x1]
        c01 = self.cost_map[y1, x0]
        c11 = self.cost_map[y1, x1]
        c0 = (1.0 - tx) * c00 + tx * c10
        c1 = (1.0 - tx) * c01 + tx * c11
        return float((1.0 - ty) * c0 + ty * c1)
