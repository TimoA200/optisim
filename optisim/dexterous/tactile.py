"""Tactile fingertip sensing primitives."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optisim.contact import ContactPoint


@dataclass(slots=True)
class TactileCell:
    """Single tactile sample on a fingertip surface."""

    position: np.ndarray
    normal: np.ndarray
    pressure: float = 0.0
    shear: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.normal = np.asarray(self.normal, dtype=np.float64)
        self.pressure = float(max(self.pressure, 0.0))
        self.shear = np.asarray(self.shear, dtype=np.float64)


class TactileSensor:
    """Hemisphere-based tactile sensor model."""

    def __init__(self, n_cells: int = 16, radius: float = 0.012) -> None:
        self.radius = float(max(radius, 1e-6))
        self.cells = self._distribute_cells(int(max(n_cells, 1)), self.radius)
        self._contact_threshold = 1e-5
        self._cell_area = 2.0 * np.pi * self.radius**2 / len(self.cells)

    cells: list[TactileCell]

    @staticmethod
    def _distribute_cells(n_cells: int, radius: float) -> list[TactileCell]:
        cells: list[TactileCell] = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for index in range(n_cells):
            z = index / max(n_cells - 1, 1)
            radial = np.sqrt(max(1.0 - z * z, 0.0))
            theta = index * golden_angle
            normal = np.asarray(
                [radial * np.cos(theta), radial * np.sin(theta), z],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-12:
                normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                normal = normal / norm
            cells.append(TactileCell(position=radius * normal, normal=normal))
        return cells

    def update(self, contact_point: ContactPoint | None, contact_force: float = 0.0) -> None:
        """Update pressures from an optional contact."""

        for cell in self.cells:
            cell.pressure = 0.0
            cell.shear = np.zeros(3, dtype=np.float64)
        if contact_point is None:
            return

        force = float(max(contact_force, 0.0))
        sigma = max(self.radius * 0.35, 1e-6)
        surface_point = np.asarray(contact_point.position, dtype=np.float64)
        normal = np.asarray(contact_point.normal, dtype=np.float64)
        tangential = surface_point - np.dot(surface_point, normal) * normal
        tangential_norm = float(np.linalg.norm(tangential))
        shear_direction = (
            tangential / tangential_norm
            if tangential_norm > 1e-12
            else np.zeros(3, dtype=np.float64)
        )

        weights = []
        for cell in self.cells:
            distance = float(np.linalg.norm(cell.position - surface_point))
            weights.append(np.exp(-(distance**2) / (2.0 * sigma**2)))
        weight_sum = sum(weights)
        if weight_sum <= 1e-12:
            return

        for cell, weight in zip(self.cells, weights, strict=True):
            normalized_weight = weight / weight_sum
            cell.pressure = force * normalized_weight / self._cell_area
            cell.shear = shear_direction * (0.15 * force * normalized_weight)

    def total_force(self) -> float:
        """Return integrated normal force."""

        return float(sum(cell.pressure * self._cell_area for cell in self.cells))

    def contact_centroid(self) -> np.ndarray | None:
        """Return the pressure-weighted centroid on the fingertip."""

        weights = np.asarray([cell.pressure for cell in self.cells], dtype=np.float64)
        total = float(np.sum(weights))
        if total <= self._contact_threshold:
            return None
        positions = np.asarray([cell.position for cell in self.cells], dtype=np.float64)
        return np.sum(positions * weights[:, None], axis=0) / total

    def pressure_map(self) -> np.ndarray:
        """Return per-cell pressure values."""

        return np.asarray([cell.pressure for cell in self.cells], dtype=np.float64)

    @property
    def in_contact(self) -> bool:
        """Return whether any cell exceeds the pressure threshold."""

        return bool(np.any(self.pressure_map() > self._contact_threshold))


__all__ = ["TactileCell", "TactileSensor"]
