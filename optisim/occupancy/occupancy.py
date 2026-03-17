"""3D voxel occupancy grid utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _vec3(value: tuple[float, float, float] | list[float] | np.ndarray, *, name: str = "value") -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3D vector")
    return (float(array[0]), float(array[1]), float(array[2]))


def _shape3(value: tuple[int, int, int] | list[int] | np.ndarray) -> tuple[int, int, int]:
    array = np.asarray(value, dtype=np.int64)
    if array.shape != (3,):
        raise ValueError("shape must contain three dimensions")
    shape = (int(array[0]), int(array[1]), int(array[2]))
    if any(size <= 0 for size in shape):
        raise ValueError("shape dimensions must be positive")
    return shape


def _voxel_centers(ii: np.ndarray, jj: np.ndarray, kk: np.ndarray, grid: "VoxelGrid") -> np.ndarray:
    centers = [grid.voxel_to_world(int(i), int(j), int(k)) for i, j, k in zip(ii.ravel(), jj.ravel(), kk.ravel())]
    if not centers:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(centers, dtype=np.float64)


def _indices_for_sphere(grid: "VoxelGrid", center: tuple[float, float, float], radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = float(radius)
    if radius < 0.0:
        raise ValueError("radius must be non-negative")

    center_array = np.asarray(center, dtype=np.float64)
    lower_world = center_array - radius
    upper_world = center_array + radius

    lower = np.floor((lower_world - np.asarray(grid.origin, dtype=np.float64)) / grid.resolution).astype(np.int64)
    upper = np.floor((upper_world - np.asarray(grid.origin, dtype=np.float64)) / grid.resolution).astype(np.int64)

    imin = max(int(lower[0]), 0)
    jmin = max(int(lower[1]), 0)
    kmin = max(int(lower[2]), 0)
    imax = min(int(upper[0]), grid.shape[0] - 1)
    jmax = min(int(upper[1]), grid.shape[1] - 1)
    kmax = min(int(upper[2]), grid.shape[2] - 1)

    if imin > imax or jmin > jmax or kmin > kmax:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty, empty

    return np.meshgrid(
        np.arange(imin, imax + 1, dtype=np.int64),
        np.arange(jmin, jmax + 1, dtype=np.int64),
        np.arange(kmin, kmax + 1, dtype=np.int64),
        indexing="ij",
    )


@dataclass(slots=True)
class VoxelGrid:
    """3D occupancy grid with fixed voxel resolution."""

    resolution: float
    origin: tuple[float, float, float]
    shape: tuple[int, int, int]
    data: np.ndarray

    def __post_init__(self) -> None:
        self.resolution = float(self.resolution)
        self.origin = _vec3(self.origin, name="origin")
        self.shape = _shape3(self.shape)
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        self.data = np.asarray(self.data, dtype=np.uint8)
        if self.data.shape != self.shape:
            raise ValueError(f"data must have shape {self.shape}, received {self.data.shape}")

    @classmethod
    def from_shape(
        cls,
        resolution: float,
        origin: tuple[float, float, float],
        shape: tuple[int, int, int],
    ) -> "VoxelGrid":
        shape3 = _shape3(shape)
        return cls(
            resolution=float(resolution),
            origin=_vec3(origin, name="origin"),
            shape=shape3,
            data=np.zeros(shape3, dtype=np.uint8),
        )

    def is_in_bounds(self, i: int, j: int, k: int) -> bool:
        return bool(0 <= int(i) < self.shape[0] and 0 <= int(j) < self.shape[1] and 0 <= int(k) < self.shape[2])

    def world_to_voxel(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        coords = np.floor((np.asarray([x, y, z], dtype=np.float64) - np.asarray(self.origin, dtype=np.float64)) / self.resolution)
        return (int(coords[0]), int(coords[1]), int(coords[2]))

    def voxel_to_world(self, i: int, j: int, k: int) -> tuple[float, float, float]:
        return (
            self.origin[0] + (int(i) + 0.5) * self.resolution,
            self.origin[1] + (int(j) + 0.5) * self.resolution,
            self.origin[2] + (int(k) + 0.5) * self.resolution,
        )

    def set_occupied(self, x: float, y: float, z: float) -> None:
        voxel = self.world_to_voxel(x, y, z)
        if self.is_in_bounds(*voxel):
            self.data[voxel] = np.uint8(255)

    def set_free(self, x: float, y: float, z: float) -> None:
        voxel = self.world_to_voxel(x, y, z)
        if self.is_in_bounds(*voxel):
            self.data[voxel] = np.uint8(0)

    def is_occupied(self, x: float, y: float, z: float) -> bool:
        voxel = self.world_to_voxel(x, y, z)
        return bool(self.is_in_bounds(*voxel) and self.data[voxel] == 255)

    def is_free(self, x: float, y: float, z: float) -> bool:
        voxel = self.world_to_voxel(x, y, z)
        return bool(self.is_in_bounds(*voxel) and self.data[voxel] == 0)

    def count_occupied(self) -> int:
        return int(np.count_nonzero(self.data == 255))

    def count_free(self) -> int:
        return int(np.count_nonzero(self.data == 0))

    def to_point_cloud(self) -> np.ndarray:
        occupied = np.argwhere(self.data == 255)
        if occupied.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        points = np.empty((occupied.shape[0], 3), dtype=np.float64)
        for index, (i, j, k) in enumerate(occupied):
            points[index] = self.voxel_to_world(int(i), int(j), int(k))
        return points


class OccupancyUpdater:
    """Apply simple occupancy updates from point and ray observations."""

    def __init__(self, grid: VoxelGrid) -> None:
        self.grid = grid

    def mark_occupied(self, points: np.ndarray) -> None:
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        for point in array:
            self.grid.set_occupied(float(point[0]), float(point[1]), float(point[2]))

    def mark_free_ray(
        self,
        origin: tuple[float, float, float],
        endpoint: tuple[float, float, float],
        steps: int = 20,
    ) -> None:
        origin_vec = np.asarray(_vec3(origin, name="origin"), dtype=np.float64)
        endpoint_vec = np.asarray(_vec3(endpoint, name="endpoint"), dtype=np.float64)
        steps = int(steps)
        if steps <= 0:
            raise ValueError("steps must be positive")
        endpoint_voxel = self.grid.world_to_voxel(float(endpoint_vec[0]), float(endpoint_vec[1]), float(endpoint_vec[2]))
        for alpha in np.linspace(0.0, 1.0, steps + 1, endpoint=False, dtype=np.float64):
            point = origin_vec + alpha * (endpoint_vec - origin_vec)
            voxel = self.grid.world_to_voxel(float(point[0]), float(point[1]), float(point[2]))
            if voxel != endpoint_voxel:
                self.grid.set_free(float(point[0]), float(point[1]), float(point[2]))

    def clear_region(self, center: tuple[float, float, float], radius: float) -> None:
        center3 = _vec3(center, name="center")
        ii, jj, kk = _indices_for_sphere(self.grid, center3, radius)
        if ii.size == 0:
            return
        centers = _voxel_centers(ii, jj, kk, self.grid)
        distances = np.linalg.norm(centers - np.asarray(center3, dtype=np.float64), axis=1)
        mask = distances <= float(radius)
        if np.any(mask):
            self.grid.data[ii.ravel()[mask], jj.ravel()[mask], kk.ravel()[mask]] = np.uint8(0)


class CollisionChecker:
    """Occupancy-based collision checks for simple robot geometry."""

    def __init__(self, grid: VoxelGrid) -> None:
        self.grid = grid

    def check_sphere(self, center: tuple[float, float, float], radius: float) -> bool:
        center3 = _vec3(center, name="center")
        ii, jj, kk = _indices_for_sphere(self.grid, center3, radius)
        if ii.size == 0:
            return False
        centers = _voxel_centers(ii, jj, kk, self.grid)
        distances = np.linalg.norm(centers - np.asarray(center3, dtype=np.float64), axis=1)
        mask = distances <= float(radius)
        if not np.any(mask):
            return False
        return bool(np.any(self.grid.data[ii.ravel()[mask], jj.ravel()[mask], kk.ravel()[mask]] == 255))

    def check_capsule(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        radius: float,
    ) -> bool:
        start_vec = np.asarray(_vec3(start, name="start"), dtype=np.float64)
        end_vec = np.asarray(_vec3(end, name="end"), dtype=np.float64)
        radius = float(radius)
        if radius < 0.0:
            raise ValueError("radius must be non-negative")
        length = float(np.linalg.norm(end_vec - start_vec))
        samples = max(int(np.ceil(length / max(self.grid.resolution * 0.5, 1e-9))), 1)
        for alpha in np.linspace(0.0, 1.0, samples + 1, dtype=np.float64):
            point = start_vec + alpha * (end_vec - start_vec)
            if self.check_sphere((float(point[0]), float(point[1]), float(point[2])), radius):
                return True
        return False

    def check_path(self, waypoints: np.ndarray, radius: float) -> bool:
        array = np.asarray(waypoints, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("waypoints must have shape (N, 3)")
        return any(self.check_sphere((float(point[0]), float(point[1]), float(point[2])), radius) for point in array)


class OccupancyStats:
    """Derived statistics for a voxel occupancy grid."""

    def __init__(self, grid: VoxelGrid) -> None:
        self.grid = grid

    def occupancy_ratio(self) -> float:
        return float(self.grid.count_occupied() / self.grid.data.size)

    def bounding_box_occupied(self) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        occupied = np.argwhere(self.grid.data == 255)
        if occupied.size == 0:
            return None
        mins = occupied.min(axis=0)
        maxs = occupied.max(axis=0)
        return (
            self.grid.voxel_to_world(int(mins[0]), int(mins[1]), int(mins[2])),
            self.grid.voxel_to_world(int(maxs[0]), int(maxs[1]), int(maxs[2])),
        )

    def density_by_layer(self, axis: int = 2) -> np.ndarray:
        axis = int(axis)
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        occupied = self.grid.data == 255
        other_axes = tuple(idx for idx in range(3) if idx != axis)
        return occupied.mean(axis=other_axes, dtype=np.float64)


__all__ = [
    "VoxelGrid",
    "OccupancyUpdater",
    "CollisionChecker",
    "OccupancyStats",
]
