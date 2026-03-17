"""Whole-body balance control and ZMP stability analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from time import time

import numpy as np


def _as_vec3(value: np.ndarray | list[float] | tuple[float, float, float], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3D vector")
    return array


def _as_vec2(value: np.ndarray | list[float] | tuple[float, float], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"{name} must be a 2D vector")
    return array


def _cross_2d(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    oa = a - origin
    ob = b - origin
    return float(oa[0] * ob[1] - oa[1] * ob[0])


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def _signed_distance_to_edge(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    edge = end - start
    length = float(np.linalg.norm(edge))
    if length <= 1e-12:
        return -float(np.linalg.norm(point - start))
    normal = np.array([-edge[1], edge[0]], dtype=np.float64) / length
    return float(np.dot(point - start, normal))


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = np.unique(points, axis=0)
    if len(unique) <= 1:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]
    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=np.float64)


@dataclass(slots=True)
class COMState:
    """Center-of-mass kinematics."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    def __post_init__(self) -> None:
        self.position = _as_vec3(self.position, name="position")
        self.velocity = _as_vec3(self.velocity, name="velocity")
        self.acceleration = _as_vec3(self.acceleration, name="acceleration")

    @classmethod
    def from_zeros(cls) -> "COMState":
        zeros = np.zeros(3, dtype=np.float64)
        return cls(position=zeros.copy(), velocity=zeros.copy(), acceleration=zeros.copy())


@dataclass(slots=True)
class BalanceReport:
    """Single balance assessment sample."""

    zmp: np.ndarray
    stable: bool
    margin: float
    support_area: float
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.zmp = _as_vec2(self.zmp, name="zmp")
        self.stable = bool(self.stable)
        self.margin = float(self.margin)
        self.support_area = float(self.support_area)
        self.timestamp = float(self.timestamp)


@dataclass(slots=True)
class SupportPolygon:
    """Convex support polygon derived from contact points."""

    vertices: np.ndarray

    def __post_init__(self) -> None:
        array = np.asarray(self.vertices, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("vertices must have shape (N, 2)")
        if len(array) == 0:
            raise ValueError("vertices must contain at least one point")
        self.vertices = _convex_hull(array)

    @classmethod
    def from_contacts(cls, contacts: list[tuple[float, float]]) -> "SupportPolygon":
        if not contacts:
            raise ValueError("contacts must contain at least one point")
        return cls(vertices=np.asarray(contacts, dtype=np.float64))

    def contains_point(self, point: np.ndarray) -> bool:
        query = _as_vec2(point, name="point")
        vertices = self.vertices
        if len(vertices) == 1:
            return bool(np.linalg.norm(query - vertices[0]) <= 1e-9)
        if len(vertices) == 2:
            distance = _point_segment_distance(query, vertices[0], vertices[1])
            projection = np.dot(query - vertices[0], vertices[1] - vertices[0])
            length_sq = float(np.dot(vertices[1] - vertices[0], vertices[1] - vertices[0]))
            return bool(distance <= 1e-9 and 0.0 <= projection <= length_sq)
        signs = []
        for index in range(len(vertices)):
            start = vertices[index]
            end = vertices[(index + 1) % len(vertices)]
            signs.append(_cross_2d(start, end, query))
        return bool(np.all(np.asarray(signs) >= -1e-9))

    def area(self) -> float:
        vertices = self.vertices
        if len(vertices) < 3:
            return 0.0
        x = vertices[:, 0]
        y = vertices[:, 1]
        area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return abs(area2) * 0.5

    def centroid(self) -> np.ndarray:
        vertices = self.vertices
        if len(vertices) == 1:
            return vertices[0].copy()
        if len(vertices) == 2:
            return vertices.mean(axis=0)
        x = vertices[:, 0]
        y = vertices[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area2 = float(np.sum(cross))
        if abs(area2) <= 1e-12:
            return vertices.mean(axis=0)
        cx = float(np.sum((x + np.roll(x, -1)) * cross) / (3.0 * area2))
        cy = float(np.sum((y + np.roll(y, -1)) * cross) / (3.0 * area2))
        return np.array([cx, cy], dtype=np.float64)


class ZMPCalculator:
    """Zero Moment Point helper methods."""

    @staticmethod
    def compute_zmp(com: COMState, total_mass: float, g: float = 9.81) -> np.ndarray:
        total_mass = float(total_mass)
        g = float(g)
        if total_mass <= 0.0:
            raise ValueError("total_mass must be positive")
        if g <= 0.0:
            raise ValueError("g must be positive")
        denom = g + float(com.acceleration[2])
        if abs(denom) <= 1e-12:
            raise ValueError("g + vertical acceleration must be non-zero")
        zmp = com.position[:2] - (float(com.position[2]) / denom) * com.acceleration[:2]
        return np.asarray(zmp, dtype=np.float64)

    @staticmethod
    def is_stable(zmp: np.ndarray, support_polygon: np.ndarray) -> bool:
        polygon = SupportPolygon(np.asarray(support_polygon, dtype=np.float64))
        return polygon.contains_point(zmp)

    @staticmethod
    def stability_margin(zmp: np.ndarray, support_polygon: np.ndarray) -> float:
        point = _as_vec2(zmp, name="zmp")
        polygon = SupportPolygon(np.asarray(support_polygon, dtype=np.float64))
        vertices = polygon.vertices
        if len(vertices) == 1:
            distance = float(np.linalg.norm(point - vertices[0]))
            return 0.0 if distance <= 1e-9 else -distance
        if len(vertices) == 2:
            distance = _point_segment_distance(point, vertices[0], vertices[1])
            return 0.0 if distance <= 1e-9 else -distance
        distances = [
            _signed_distance_to_edge(point, vertices[index], vertices[(index + 1) % len(vertices)])
            for index in range(len(vertices))
        ]
        return float(min(distances))


class BalanceMonitor:
    """Track recent balance quality from CoM and contact geometry."""

    def __init__(self, total_mass: float, g: float = 9.81) -> None:
        self.total_mass = float(total_mass)
        self.g = float(g)
        if self.total_mass <= 0.0:
            raise ValueError("total_mass must be positive")
        if self.g <= 0.0:
            raise ValueError("g must be positive")
        self.max_history = 100
        self.history: list[BalanceReport] = []

    def update(self, com: COMState, contacts: list[tuple[float, float]]) -> BalanceReport:
        polygon = SupportPolygon.from_contacts(contacts)
        zmp = ZMPCalculator.compute_zmp(com, self.total_mass, g=self.g)
        stable = polygon.contains_point(zmp)
        margin = ZMPCalculator.stability_margin(zmp, polygon.vertices)
        report = BalanceReport(
            zmp=zmp,
            stable=stable,
            margin=margin,
            support_area=polygon.area(),
            timestamp=time(),
        )
        self.history.append(report)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]
        return report

    def is_balanced(self, threshold_margin: float = 0.0) -> bool:
        if not self.history:
            return False
        last = self.history[-1]
        return bool(last.stable and last.margin >= float(threshold_margin))
