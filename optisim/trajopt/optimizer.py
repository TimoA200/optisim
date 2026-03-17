"""Time-parameterized trajectory optimization for joint-space waypoint paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from optisim.robot.model import RobotModel


@dataclass(slots=True)
class TrajOptConfig:
    """Configuration for waypoint-to-trajectory optimization."""

    max_velocity_scale: float = 1.0
    max_acceleration_scale: float = 0.5
    min_segment_time: float = 0.01
    num_waypoints: int = 50
    smoothing_window: int = 5
    optimize_for: Literal["time", "energy", "smoothness"] = "time"


@dataclass(slots=True)
class TrajOptResult:
    """Optimized and time-parameterized joint-space trajectory."""

    waypoints: list[dict[str, float]]
    times: list[float]
    joint_velocities: list[dict[str, float]]
    joint_accelerations: list[dict[str, float]]
    total_time_s: float
    energy_estimate: float
    feasible: bool
    constraint_violations: list[str]


class TrajectoryOptimizer:
    """Convert joint-space waypoints into a constrained, sampled trajectory."""

    def __init__(self, config: TrajOptConfig | None = None) -> None:
        self.config = config or TrajOptConfig()

    def optimize(self, robot: RobotModel, waypoints: list[dict[str, float]]) -> TrajOptResult:
        """Build a cubic-spline trajectory through waypoint knots."""

        normalized_waypoints, joint_names = self._normalize_waypoints(robot, waypoints)
        if len(normalized_waypoints) == 1:
            normalized_waypoints = [normalized_waypoints[0], dict(normalized_waypoints[0])]

        segment_times = self._compute_segment_times(robot, normalized_waypoints, joint_names)
        knot_times = [0.0]
        for segment_time in segment_times:
            knot_times.append(knot_times[-1] + segment_time)

        total_time_s = float(knot_times[-1])
        sample_count = max(2, int(self.config.num_waypoints))
        sample_times = np.linspace(0.0, total_time_s, sample_count, dtype=np.float64)

        position_arrays: dict[str, np.ndarray] = {}
        velocity_arrays: dict[str, np.ndarray] = {}
        acceleration_arrays: dict[str, np.ndarray] = {}
        knot_time_array = np.asarray(knot_times, dtype=np.float64)

        for joint_name in joint_names:
            knot_values = np.asarray(
                [waypoint[joint_name] for waypoint in normalized_waypoints],
                dtype=np.float64,
            )
            second_derivatives = _natural_cubic_second_derivatives(knot_time_array, knot_values)
            positions, velocities, accelerations = _evaluate_natural_cubic_spline(
                knot_time_array,
                knot_values,
                second_derivatives,
                sample_times,
            )
            position_arrays[joint_name] = positions
            velocity_arrays[joint_name] = velocities
            acceleration_arrays[joint_name] = accelerations

        if self.config.optimize_for == "smoothness":
            position_arrays, velocity_arrays, acceleration_arrays = self._smooth_velocity_profiles(
                position_arrays,
                velocity_arrays,
                sample_times,
            )

        sampled_waypoints = _dict_samples(position_arrays, joint_names)
        sampled_velocities = _dict_samples(velocity_arrays, joint_names)
        sampled_accelerations = _dict_samples(acceleration_arrays, joint_names)

        provisional_result = TrajOptResult(
            waypoints=sampled_waypoints,
            times=sample_times.tolist(),
            joint_velocities=sampled_velocities,
            joint_accelerations=sampled_accelerations,
            total_time_s=total_time_s,
            energy_estimate=0.0,
            feasible=True,
            constraint_violations=[],
        )
        violations = self._collect_constraint_violations(
            robot,
            normalized_waypoints,
            provisional_result,
            joint_names,
        )
        energy_estimate = self.compute_energy(provisional_result, robot)
        feasible = len(violations) == 0 and self.is_feasible(provisional_result)

        return TrajOptResult(
            waypoints=sampled_waypoints,
            times=sample_times.tolist(),
            joint_velocities=sampled_velocities,
            joint_accelerations=sampled_accelerations,
            total_time_s=total_time_s,
            energy_estimate=energy_estimate,
            feasible=feasible,
            constraint_violations=violations,
        )

    def resample(self, result: TrajOptResult, num_samples: int) -> TrajOptResult:
        """Resample an existing trajectory at a new time resolution."""

        sample_count = max(2, int(num_samples))
        if len(result.times) < 2:
            times = np.linspace(0.0, float(result.total_time_s), sample_count, dtype=np.float64)
            if not result.waypoints:
                return TrajOptResult(
                    waypoints=[],
                    times=times.tolist(),
                    joint_velocities=[],
                    joint_accelerations=[],
                    total_time_s=float(times[-1]) if len(times) else 0.0,
                    energy_estimate=0.0,
                    feasible=self.is_feasible(result),
                    constraint_violations=list(result.constraint_violations),
                )
            waypoints = [dict(result.waypoints[0]) for _ in range(sample_count)]
            zeros = [{joint: 0.0 for joint in result.waypoints[0]} for _ in range(sample_count)]
            return TrajOptResult(
                waypoints=waypoints,
                times=times.tolist(),
                joint_velocities=zeros,
                joint_accelerations=zeros,
                total_time_s=float(times[-1]) if len(times) else 0.0,
                energy_estimate=0.0,
                feasible=self.is_feasible(result),
                constraint_violations=list(result.constraint_violations),
            )

        original_times = np.asarray(result.times, dtype=np.float64)
        target_times = np.linspace(original_times[0], original_times[-1], sample_count, dtype=np.float64)
        joint_names = sorted(result.waypoints[0]) if result.waypoints else []

        position_arrays: dict[str, np.ndarray] = {}
        velocity_arrays: dict[str, np.ndarray] = {}
        acceleration_arrays: dict[str, np.ndarray] = {}
        for joint_name in joint_names:
            knot_values = np.asarray([waypoint[joint_name] for waypoint in result.waypoints], dtype=np.float64)
            second_derivatives = _natural_cubic_second_derivatives(original_times, knot_values)
            positions, velocities, accelerations = _evaluate_natural_cubic_spline(
                original_times,
                knot_values,
                second_derivatives,
                target_times,
            )
            position_arrays[joint_name] = positions
            velocity_arrays[joint_name] = velocities
            acceleration_arrays[joint_name] = accelerations

        resampled = TrajOptResult(
            waypoints=_dict_samples(position_arrays, joint_names),
            times=target_times.tolist(),
            joint_velocities=_dict_samples(velocity_arrays, joint_names),
            joint_accelerations=_dict_samples(acceleration_arrays, joint_names),
            total_time_s=float(target_times[-1] - target_times[0]),
            energy_estimate=_compute_energy_from_arrays(
                velocity_arrays,
                target_times,
                {joint_name: 1.0 for joint_name in joint_names},
            ),
            feasible=self.is_feasible(result),
            constraint_violations=list(result.constraint_violations),
        )
        resampled.feasible = self.is_feasible(resampled) and resampled.feasible
        return resampled

    def compute_energy(self, result: TrajOptResult, robot: RobotModel) -> float:
        """Estimate trajectory energy from velocity-squared torque proxy."""

        if not result.joint_velocities or len(result.times) < 2:
            return 0.0
        joint_names = sorted(result.joint_velocities[0])
        velocity_arrays = {
            joint_name: np.asarray(
                [sample.get(joint_name, 0.0) for sample in result.joint_velocities],
                dtype=np.float64,
            )
            for joint_name in joint_names
        }
        inertia = {joint_name: self._joint_inertia_proxy(robot, joint_name) for joint_name in joint_names}
        return _compute_energy_from_arrays(velocity_arrays, np.asarray(result.times, dtype=np.float64), inertia)

    def is_feasible(self, result: TrajOptResult) -> bool:
        """Return whether the result structure is internally consistent and violation-free."""

        if result.constraint_violations:
            return False
        times = np.asarray(result.times, dtype=np.float64)
        if len(times) != len(result.waypoints):
            return False
        if len(times) != len(result.joint_velocities) or len(times) != len(result.joint_accelerations):
            return False
        if len(times) == 0 or not np.all(np.isfinite(times)):
            return False
        if np.any(np.diff(times) < -1e-12):
            return False
        if not np.isfinite(result.total_time_s) or result.total_time_s < 0.0:
            return False
        return True

    def _normalize_waypoints(
        self,
        robot: RobotModel,
        waypoints: list[dict[str, float]],
    ) -> tuple[list[dict[str, float]], list[str]]:
        if not waypoints:
            raise ValueError("trajectory optimization requires at least one waypoint")

        joint_names = sorted(robot.joints)
        normalized: list[dict[str, float]] = []
        previous = dict(robot.joint_positions)
        for waypoint in waypoints:
            merged = dict(previous)
            for joint_name in joint_names:
                if joint_name in waypoint:
                    merged[joint_name] = float(waypoint[joint_name])
                else:
                    merged.setdefault(joint_name, 0.0)
            normalized.append({joint_name: float(merged[joint_name]) for joint_name in joint_names})
            previous = merged
        return normalized, joint_names

    def _compute_segment_times(
        self,
        robot: RobotModel,
        waypoints: list[dict[str, float]],
        joint_names: list[str],
    ) -> list[float]:
        scale_v = max(float(self.config.max_velocity_scale), 1e-6)
        scale_a = max(float(self.config.max_acceleration_scale), 1e-6)
        times: list[float] = []
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            segment_time = float(self.config.min_segment_time)
            for joint_name in joint_names:
                delta = abs(end[joint_name] - start[joint_name])
                joint = robot.joints[joint_name]
                max_velocity = max(joint.velocity_limit * scale_v, 1e-6)
                max_acceleration = max(joint.velocity_limit * scale_a, 1e-6)
                velocity_time = delta / max_velocity
                acceleration_time = float(np.sqrt((6.0 * delta) / max_acceleration)) if delta > 0.0 else 0.0
                segment_time = max(segment_time, velocity_time, acceleration_time)
            if self.config.optimize_for == "energy":
                segment_time *= 1.2
            times.append(segment_time)
        return times

    def _smooth_velocity_profiles(
        self,
        position_arrays: dict[str, np.ndarray],
        velocity_arrays: dict[str, np.ndarray],
        sample_times: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        smoothed_positions: dict[str, np.ndarray] = {}
        smoothed_velocities: dict[str, np.ndarray] = {}
        smoothed_accelerations: dict[str, np.ndarray] = {}
        for joint_name, velocities in velocity_arrays.items():
            filtered_velocities = _gaussian_smooth(velocities, self.config.smoothing_window)
            integrated_positions = _integrate_velocity_profile(
                filtered_velocities,
                sample_times,
                position_arrays[joint_name][0],
            )
            endpoint_error = position_arrays[joint_name][-1] - integrated_positions[-1]
            integrated_positions = integrated_positions + np.linspace(
                0.0,
                endpoint_error,
                len(integrated_positions),
                dtype=np.float64,
            )
            integrated_positions[0] = position_arrays[joint_name][0]
            integrated_positions[-1] = position_arrays[joint_name][-1]
            accelerations = np.gradient(filtered_velocities, sample_times, edge_order=1)
            smoothed_positions[joint_name] = integrated_positions
            smoothed_velocities[joint_name] = filtered_velocities
            smoothed_accelerations[joint_name] = accelerations
        return smoothed_positions, smoothed_velocities, smoothed_accelerations

    def _collect_constraint_violations(
        self,
        robot: RobotModel,
        input_waypoints: list[dict[str, float]],
        result: TrajOptResult,
        joint_names: list[str],
    ) -> list[str]:
        violations: list[str] = []
        scale_v = max(float(self.config.max_velocity_scale), 0.0)
        scale_a = max(float(self.config.max_acceleration_scale), 0.0)

        for index, waypoint in enumerate(input_waypoints):
            for joint_name in joint_names:
                joint = robot.joints[joint_name]
                value = waypoint[joint_name]
                if value < joint.limit_lower - 1e-9 or value > joint.limit_upper + 1e-9:
                    violations.append(
                        f"joint {joint_name} exceeds position limit at waypoint {index}: {value:.3f}"
                    )

        for sample_index, current_time in enumerate(result.times):
            waypoint = result.waypoints[sample_index]
            velocities = result.joint_velocities[sample_index]
            accelerations = result.joint_accelerations[sample_index]
            for joint_name in joint_names:
                joint = robot.joints[joint_name]
                position = waypoint[joint_name]
                if position < joint.limit_lower - 1e-6 or position > joint.limit_upper + 1e-6:
                    violations.append(
                        f"joint {joint_name} exceeds position limit at t={current_time:.2f}"
                    )
                velocity_limit = joint.velocity_limit * scale_v
                if abs(velocities[joint_name]) > velocity_limit + 1e-6:
                    violations.append(
                        f"joint {joint_name} exceeds velocity limit at t={current_time:.2f}"
                    )
                if scale_a > 0.0:
                    acceleration_limit = joint.velocity_limit * scale_a
                    if abs(accelerations[joint_name]) > acceleration_limit + 1e-6:
                        violations.append(
                            f"joint {joint_name} exceeds acceleration limit at t={current_time:.2f}"
                        )

        return violations

    @staticmethod
    def _joint_inertia_proxy(robot: RobotModel, joint_name: str) -> float:
        joint = robot.joints[joint_name]
        if hasattr(joint, "mass"):
            mass = getattr(joint, "mass")
            if isinstance(mass, (int, float)) and mass > 0.0:
                return float(mass)
        masses = getattr(robot, "joint_masses", None)
        if isinstance(masses, dict):
            mass = masses.get(joint_name)
            if isinstance(mass, (int, float)) and mass > 0.0:
                return float(mass)
        return 1.0


def optimize_path(
    robot: RobotModel,
    waypoints: list[dict[str, float]],
    config: TrajOptConfig | None = None,
) -> TrajOptResult:
    """Optimize a waypoint path with a one-shot helper API."""

    return TrajectoryOptimizer(config=config).optimize(robot, waypoints)


def _natural_cubic_second_derivatives(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    count = len(times)
    if count <= 2:
        return np.zeros(count, dtype=np.float64)

    intervals = np.diff(times)
    matrix = np.zeros((count, count), dtype=np.float64)
    rhs = np.zeros(count, dtype=np.float64)
    matrix[0, 0] = 1.0
    matrix[-1, -1] = 1.0

    for index in range(1, count - 1):
        left_interval = intervals[index - 1]
        right_interval = intervals[index]
        matrix[index, index - 1] = left_interval
        matrix[index, index] = 2.0 * (left_interval + right_interval)
        matrix[index, index + 1] = right_interval
        rhs[index] = 6.0 * (
            (values[index + 1] - values[index]) / right_interval
            - (values[index] - values[index - 1]) / left_interval
        )

    return np.linalg.solve(matrix, rhs)


def _evaluate_natural_cubic_spline(
    times: np.ndarray,
    values: np.ndarray,
    second_derivatives: np.ndarray,
    sample_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.empty_like(sample_times, dtype=np.float64)
    velocities = np.empty_like(sample_times, dtype=np.float64)
    accelerations = np.empty_like(sample_times, dtype=np.float64)
    segment_indices = np.searchsorted(times, sample_times, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(times) - 2)

    for sample_index, segment_index in enumerate(segment_indices):
        left_time = times[segment_index]
        right_time = times[segment_index + 1]
        interval = right_time - left_time
        sample_time = sample_times[sample_index]
        left_weight = (right_time - sample_time) / interval
        right_weight = (sample_time - left_time) / interval

        left_value = values[segment_index]
        right_value = values[segment_index + 1]
        left_second = second_derivatives[segment_index]
        right_second = second_derivatives[segment_index + 1]

        positions[sample_index] = (
            left_weight * left_value
            + right_weight * right_value
            + (((left_weight**3) - left_weight) * left_second + ((right_weight**3) - right_weight) * right_second)
            * (interval**2)
            / 6.0
        )
        velocities[sample_index] = (
            (right_value - left_value) / interval
            - ((3.0 * (left_weight**2) - 1.0) * interval * left_second) / 6.0
            + ((3.0 * (right_weight**2) - 1.0) * interval * right_second) / 6.0
        )
        accelerations[sample_index] = left_weight * left_second + right_weight * right_second

    return positions, velocities, accelerations


def _gaussian_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 2:
        return values.copy()
    radius = max(int(window) // 2, 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    sigma = max(window / 3.0, 1e-6)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _integrate_velocity_profile(velocities: np.ndarray, times: np.ndarray, initial_position: float) -> np.ndarray:
    positions = np.empty_like(velocities, dtype=np.float64)
    positions[0] = float(initial_position)
    if len(velocities) == 1:
        return positions
    delta_t = np.diff(times)
    for index in range(1, len(velocities)):
        positions[index] = positions[index - 1] + 0.5 * (velocities[index - 1] + velocities[index]) * delta_t[index - 1]
    return positions


def _dict_samples(arrays: dict[str, np.ndarray], joint_names: list[str]) -> list[dict[str, float]]:
    if not joint_names:
        return []
    sample_count = len(arrays[joint_names[0]])
    return [
        {joint_name: float(arrays[joint_name][index]) for joint_name in joint_names}
        for index in range(sample_count)
    ]


def _compute_energy_from_arrays(
    velocity_arrays: dict[str, np.ndarray],
    times: np.ndarray,
    inertia: dict[str, float],
) -> float:
    if len(times) < 2:
        return 0.0
    energy = 0.0
    delta_t = np.diff(times)
    for joint_name, velocities in velocity_arrays.items():
        inertia_proxy = inertia.get(joint_name, 1.0)
        power = np.abs(velocities * (velocities * inertia_proxy))
        energy += float(np.sum(0.5 * (power[:-1] + power[1:]) * delta_t))
    return float(energy)


__all__ = [
    "TrajOptConfig",
    "TrajOptResult",
    "TrajectoryOptimizer",
    "optimize_path",
]
