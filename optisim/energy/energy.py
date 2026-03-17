"""Robot joint energy consumption helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_float_array(values: np.ndarray | list[float], *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    return array


def _as_time_joint_array(values: np.ndarray | list[list[float]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (T, N_joints)")
    return array


@dataclass(slots=True)
class JointPowerModel:
    """Compute instantaneous joint power with optional viscous damping losses."""

    damping: float = 0.0

    def __post_init__(self) -> None:
        self.damping = float(self.damping)
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative")

    def compute_power(self, torque: float, velocity: float) -> float:
        mechanical = float(torque) * float(velocity)
        damping_loss = self.damping * float(velocity) ** 2
        return float(mechanical + damping_loss)

    def compute_power_array(self, torques: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        torque_array = np.asarray(torques, dtype=np.float64)
        velocity_array = np.asarray(velocities, dtype=np.float64)
        if torque_array.shape != velocity_array.shape:
            raise ValueError("torques and velocities must have matching shapes")
        return torque_array * velocity_array + self.damping * np.square(velocity_array)


@dataclass(slots=True)
class MotorEfficiencyModel:
    """Torque-speed efficiency lookup with bilinear interpolation."""

    torque_grid: np.ndarray
    speed_grid: np.ndarray
    efficiency_map: np.ndarray

    def __post_init__(self) -> None:
        self.torque_grid = _as_float_array(self.torque_grid, ndim=1, name="torque_grid")
        self.speed_grid = _as_float_array(self.speed_grid, ndim=1, name="speed_grid")
        self.efficiency_map = _as_float_array(self.efficiency_map, ndim=2, name="efficiency_map")

        if self.torque_grid.size < 2 or self.speed_grid.size < 2:
            raise ValueError("torque_grid and speed_grid must each contain at least two points")
        if np.any(np.diff(self.torque_grid) <= 0.0):
            raise ValueError("torque_grid must be strictly increasing")
        if np.any(np.diff(self.speed_grid) <= 0.0):
            raise ValueError("speed_grid must be strictly increasing")
        expected_shape = (self.torque_grid.size, self.speed_grid.size)
        if self.efficiency_map.shape != expected_shape:
            raise ValueError(f"efficiency_map must have shape {expected_shape}")
        self.efficiency_map = np.clip(self.efficiency_map, 0.0, 1.0)

    @classmethod
    def from_constant(cls, eta: float) -> "MotorEfficiencyModel":
        efficiency = float(np.clip(eta, 0.0, 1.0))
        grid = np.array([0.0, 1.0], dtype=np.float64)
        table = np.full((2, 2), efficiency, dtype=np.float64)
        return cls(torque_grid=grid, speed_grid=grid, efficiency_map=table)

    def get_efficiency(self, torque: float, velocity: float) -> float:
        torque_value = float(np.clip(abs(float(torque)), self.torque_grid[0], self.torque_grid[-1]))
        speed_value = float(np.clip(abs(float(velocity)), self.speed_grid[0], self.speed_grid[-1]))

        torque_index = int(np.searchsorted(self.torque_grid, torque_value, side="right") - 1)
        speed_index = int(np.searchsorted(self.speed_grid, speed_value, side="right") - 1)
        torque_index = min(max(torque_index, 0), self.torque_grid.size - 2)
        speed_index = min(max(speed_index, 0), self.speed_grid.size - 2)

        t0 = self.torque_grid[torque_index]
        t1 = self.torque_grid[torque_index + 1]
        s0 = self.speed_grid[speed_index]
        s1 = self.speed_grid[speed_index + 1]

        q11 = self.efficiency_map[torque_index, speed_index]
        q21 = self.efficiency_map[torque_index + 1, speed_index]
        q12 = self.efficiency_map[torque_index, speed_index + 1]
        q22 = self.efficiency_map[torque_index + 1, speed_index + 1]

        wt = 0.0 if t1 == t0 else (torque_value - t0) / (t1 - t0)
        ws = 0.0 if s1 == s0 else (speed_value - s0) / (s1 - s0)
        lower = (1.0 - wt) * q11 + wt * q21
        upper = (1.0 - wt) * q12 + wt * q22
        return float(np.clip((1.0 - ws) * lower + ws * upper, 0.0, 1.0))


@dataclass(slots=True)
class EnergyEstimator:
    """Estimate electrical energy consumption from torque and velocity traces."""

    power_model: JointPowerModel
    efficiency_model: MotorEfficiencyModel

    def _validate_inputs(self, torques: np.ndarray, velocities: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        torque_array = _as_time_joint_array(torques, "torques")
        velocity_array = _as_time_joint_array(velocities, "velocities")
        if torque_array.shape != velocity_array.shape:
            raise ValueError("torques and velocities must have matching shapes")
        if float(dt) <= 0.0:
            raise ValueError("dt must be positive")
        return torque_array, velocity_array

    def _electrical_power(self, torques: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        mechanical_power = self.power_model.compute_power_array(torques, velocities)
        electrical_power = np.zeros_like(mechanical_power)
        for time_index in range(mechanical_power.shape[0]):
            for joint_index in range(mechanical_power.shape[1]):
                demand = max(float(mechanical_power[time_index, joint_index]), 0.0)
                if demand <= 0.0:
                    continue
                efficiency = self.efficiency_model.get_efficiency(
                    torques[time_index, joint_index],
                    velocities[time_index, joint_index],
                )
                if efficiency <= 0.0:
                    raise ValueError("efficiency must be positive when power demand is positive")
                electrical_power[time_index, joint_index] = demand / efficiency
        return electrical_power

    def estimate_per_joint(self, torques: np.ndarray, velocities: np.ndarray, dt: float) -> np.ndarray:
        torque_array, velocity_array = self._validate_inputs(torques, velocities, dt)
        electrical_power = self._electrical_power(torque_array, velocity_array)
        return np.trapezoid(electrical_power, dx=float(dt), axis=0)

    def estimate_energy(self, torques: np.ndarray, velocities: np.ndarray, dt: float) -> float:
        return float(self.estimate_per_joint(torques, velocities, dt).sum())


@dataclass(slots=True)
class EnergyBudget:
    """Track cumulative energy usage against a fixed budget."""

    budget: float
    used: float = 0.0

    def __post_init__(self) -> None:
        self.budget = float(self.budget)
        self.used = float(self.used)
        if self.budget <= 0.0:
            raise ValueError("budget must be positive")

    def consume(self, joules: float) -> None:
        self.used += float(joules)

    def remaining(self) -> float:
        return float(self.budget - self.used)

    def is_exhausted(self) -> bool:
        return bool(self.remaining() <= 0.0)

    def reset(self) -> None:
        self.used = 0.0


@dataclass(slots=True)
class TaskEnergyProfile:
    """Named power trace for a task."""

    name: str
    timestamps: np.ndarray
    energy_rates: np.ndarray

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.timestamps = _as_float_array(self.timestamps, ndim=1, name="timestamps")
        self.energy_rates = _as_float_array(self.energy_rates, ndim=1, name="energy_rates")
        if self.timestamps.shape != self.energy_rates.shape:
            raise ValueError("timestamps and energy_rates must have matching shapes")
        if self.timestamps.size == 0:
            raise ValueError("at least one sample is required")
        if np.any(np.diff(self.timestamps) < 0.0):
            raise ValueError("timestamps must be nondecreasing")

    @classmethod
    def from_estimator(
        cls,
        name: str,
        torques: np.ndarray,
        velocities: np.ndarray,
        dt: float,
        estimator: EnergyEstimator,
    ) -> "TaskEnergyProfile":
        torque_array, velocity_array = estimator._validate_inputs(torques, velocities, dt)
        total_power = estimator._electrical_power(torque_array, velocity_array).sum(axis=1)
        timestamps = np.arange(total_power.size, dtype=np.float64) * float(dt)
        return cls(name=name, timestamps=timestamps, energy_rates=total_power)

    def total_energy(self) -> float:
        if self.timestamps.size == 1:
            return 0.0
        return float(np.trapezoid(self.energy_rates, self.timestamps))

    def peak_power(self) -> float:
        return float(np.max(self.energy_rates))

    def mean_power(self) -> float:
        if self.timestamps.size == 1:
            return float(self.energy_rates[0])
        duration = float(self.timestamps[-1] - self.timestamps[0])
        if duration <= 0.0:
            return float(np.mean(self.energy_rates))
        return self.total_energy() / duration


__all__ = [
    "JointPowerModel",
    "MotorEfficiencyModel",
    "EnergyEstimator",
    "EnergyBudget",
    "TaskEnergyProfile",
]
