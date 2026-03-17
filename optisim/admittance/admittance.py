"""Admittance control primitives for compliant contact reaction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_vec6(value: np.ndarray | list[float] | tuple[float, ...], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (6,):
        raise ValueError(f"{name} must be a 6D vector")
    return array


@dataclass(slots=True)
class AdmittanceParams:
    """Virtual mass-spring-damper parameters."""

    mass: float
    damping: float
    stiffness: float

    def __post_init__(self) -> None:
        self.mass = float(self.mass)
        self.damping = float(self.damping)
        self.stiffness = float(self.stiffness)

    def is_valid(self) -> bool:
        return self.mass > 0.0 and self.damping > 0.0 and self.stiffness > 0.0

    @classmethod
    def from_defaults(cls) -> "AdmittanceParams":
        return cls(mass=1.0, damping=10.0, stiffness=100.0)


class AdmittanceController1D:
    """Single-axis admittance controller using Euler integration."""

    def __init__(self, params: AdmittanceParams, dt: float) -> None:
        self.params = params
        self.dt = float(dt)
        if not self.params.is_valid():
            raise ValueError("params must contain positive mass, damping, and stiffness")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0

    def step(self, force_external: float) -> float:
        force_external = float(force_external)
        self.acceleration = (
            force_external
            - self.params.damping * self.velocity
            - self.params.stiffness * self.position
        ) / self.params.mass
        self.velocity += self.acceleration * self.dt
        self.position += self.velocity * self.dt
        return self.position

    def reset(self) -> None:
        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0


class AdmittanceController6D:
    """Independent 1D admittance controllers for 6D wrench response."""

    def __init__(self, params_linear: AdmittanceParams, params_angular: AdmittanceParams, dt: float) -> None:
        self.params_linear = params_linear
        self.params_angular = params_angular
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        self._controllers = [
            AdmittanceController1D(params_linear, self.dt),
            AdmittanceController1D(params_linear, self.dt),
            AdmittanceController1D(params_linear, self.dt),
            AdmittanceController1D(params_angular, self.dt),
            AdmittanceController1D(params_angular, self.dt),
            AdmittanceController1D(params_angular, self.dt),
        ]
        self.state = np.zeros(6, dtype=np.float64)

    def step(self, wrench: np.ndarray) -> np.ndarray:
        wrench_array = _as_vec6(wrench, name="wrench")
        self.state = np.array(
            [controller.step(float(force)) for controller, force in zip(self._controllers, wrench_array)],
            dtype=np.float64,
        )
        return self.state.copy()

    def reset(self) -> None:
        for controller in self._controllers:
            controller.reset()
        self.state = np.zeros(6, dtype=np.float64)


class ContactCompliantMotion:
    """Blend a desired pose with admittance-driven contact deviation."""

    def __init__(self, controller: AdmittanceController6D, max_deviation: float = 0.05) -> None:
        self.controller = controller
        self.max_deviation = float(max_deviation)
        if self.max_deviation < 0.0:
            raise ValueError("max_deviation must be non-negative")

    def step(self, desired_pose: np.ndarray, wrench: np.ndarray) -> np.ndarray:
        desired = _as_vec6(desired_pose, name="desired_pose")
        deviation = self.controller.step(wrench)
        deviation_norm = float(np.linalg.norm(deviation))
        if deviation_norm > self.max_deviation > 0.0:
            deviation = deviation * (self.max_deviation / deviation_norm)
        elif self.max_deviation == 0.0:
            deviation = np.zeros(6, dtype=np.float64)
        return desired + deviation

    def deviation_norm(self) -> float:
        return float(np.linalg.norm(self.controller.state))

    def is_compliant(self, threshold: float = 0.001) -> bool:
        return self.deviation_norm() > float(threshold)


class AdmittanceLogger:
    """Fixed-length history of wrench and deviation samples."""

    def __init__(self, max_steps: int = 1000) -> None:
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.step_history: list[int] = []
        self.wrench_history: list[np.ndarray] = []
        self.deviation_history: list[np.ndarray] = []

    def log(self, step: int, wrench: np.ndarray, deviation: np.ndarray) -> None:
        self.step_history.append(int(step))
        self.wrench_history.append(_as_vec6(wrench, name="wrench").copy())
        self.deviation_history.append(_as_vec6(deviation, name="deviation").copy())
        if len(self.step_history) > self.max_steps:
            self.step_history = self.step_history[-self.max_steps :]
            self.wrench_history = self.wrench_history[-self.max_steps :]
            self.deviation_history = self.deviation_history[-self.max_steps :]

    def mean_wrench(self) -> np.ndarray:
        if not self.wrench_history:
            return np.zeros(6, dtype=np.float64)
        return np.mean(np.stack(self.wrench_history, axis=0), axis=0)

    def peak_force(self) -> float:
        if not self.wrench_history:
            return 0.0
        return float(max(np.linalg.norm(wrench[:3]) for wrench in self.wrench_history))

    def clear(self) -> None:
        self.step_history.clear()
        self.wrench_history.clear()
        self.deviation_history.clear()


__all__ = [
    "AdmittanceParams",
    "AdmittanceController1D",
    "AdmittanceController6D",
    "ContactCompliantMotion",
    "AdmittanceLogger",
]
