"""Sensor simulation primitives for humanoid robotics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


def _as_float_array(values: float | list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


@dataclass
class SensorNoise:
    """Simple additive sensor noise model."""

    gaussian_std: float = 0.0
    bias: float = 0.0
    dropout_prob: float = 0.0

    def apply(self, values: float | list[float] | np.ndarray) -> float | np.ndarray:
        array = _as_float_array(values).copy()
        if self.gaussian_std:
            array += np.random.normal(0.0, self.gaussian_std, size=array.shape)
        if self.bias:
            array += self.bias
        if self.dropout_prob:
            dropout = np.random.random(size=array.shape) < self.dropout_prob
            array[dropout] = np.nan
        if array.ndim == 0:
            return float(array)
        return array


@dataclass
class ForceTorqueSensor:
    """Six-axis force/torque sensor with saturation and contact detection."""

    name: str
    mount_joint: str
    noise: SensorNoise
    max_force_n: float = 100.0
    max_torque_nm: float = 20.0

    def read(self, wrench: list[float] | np.ndarray) -> np.ndarray:
        values = _as_float_array(wrench)
        if values.shape != (6,):
            raise ValueError("ForceTorqueSensor.read expects a 6D wrench vector.")
        noisy = _as_float_array(self.noise.apply(values))
        force = np.clip(noisy[:3], -self.max_force_n, self.max_force_n)
        torque = np.clip(noisy[3:], -self.max_torque_nm, self.max_torque_nm)
        return np.concatenate((force, torque))

    def is_contact(self, wrench: list[float] | np.ndarray, threshold_n: float) -> bool:
        values = self.read(wrench)
        return bool(np.linalg.norm(np.nan_to_num(values[:3], nan=0.0)) > threshold_n)


@dataclass
class ProximitySensor:
    """Simple scalar range sensor."""

    name: str
    noise: SensorNoise
    max_range_m: float = 1.0
    min_range_m: float = 0.02

    def read(self, distance_m: float) -> float:
        reading = self.noise.apply(float(distance_m))
        if np.isnan(reading):
            return float("nan")
        return float(np.clip(reading, self.min_range_m, self.max_range_m))


@dataclass
class JointEncoderArray:
    """Array of joint encoders with quantization."""

    joint_names: list[str]
    noise: SensorNoise
    resolution_rad: float = 0.001

    def read(self, joint_positions: list[float] | np.ndarray) -> np.ndarray:
        values = _as_float_array(joint_positions)
        if values.shape != (len(self.joint_names),):
            raise ValueError("JointEncoderArray.read expects one value per joint.")
        noisy = _as_float_array(self.noise.apply(values))
        return np.round(noisy / self.resolution_rad) * self.resolution_rad


@dataclass
class IMUSensor:
    """Six-axis IMU that reports acceleration and angular velocity."""

    name: str
    noise_accel: SensorNoise
    noise_gyro: SensorNoise
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=float))

    def read_accel(self, linear_accel_sensor_frame: list[float] | np.ndarray) -> np.ndarray:
        accel = _as_float_array(linear_accel_sensor_frame)
        gravity = _as_float_array(self.gravity)
        if accel.shape != (3,) or gravity.shape != (3,):
            raise ValueError("IMUSensor.read_accel expects 3D acceleration and gravity vectors.")
        return _as_float_array(self.noise_accel.apply(accel + gravity))

    def read_gyro(self, angular_velocity_sensor_frame: list[float] | np.ndarray) -> np.ndarray:
        gyro = _as_float_array(angular_velocity_sensor_frame)
        if gyro.shape != (3,):
            raise ValueError("IMUSensor.read_gyro expects a 3D angular velocity vector.")
        return _as_float_array(self.noise_gyro.apply(gyro))

    def read(
        self,
        linear_accel_sensor_frame: list[float] | np.ndarray,
        angular_velocity_sensor_frame: list[float] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "accel": self.read_accel(linear_accel_sensor_frame),
            "gyro": self.read_gyro(angular_velocity_sensor_frame),
        }


@dataclass
class DepthCameraSensor:
    """Simplified depth camera that ray-casts against axis-aligned boxes."""

    name: str
    noise: SensorNoise
    fov_deg: float = 90.0
    resolution: tuple[int, int] = (64, 48)
    max_depth_m: float = 5.0

    def _pixel_rays(self) -> np.ndarray:
        width, height = self.resolution
        aspect = width / height
        tan_half_fov = np.tan(np.deg2rad(self.fov_deg) / 2.0)
        xs = ((np.arange(width) + 0.5) / width * 2.0 - 1.0) * tan_half_fov
        ys = (1.0 - (np.arange(height) + 0.5) / height * 2.0) * tan_half_fov / aspect
        grid_x, grid_y = np.meshgrid(xs, ys)
        rays = np.stack((np.ones_like(grid_x), grid_x, grid_y), axis=-1)
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    @staticmethod
    def _ray_box_intersection(
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        box_center: np.ndarray,
        box_size: np.ndarray,
    ) -> float | None:
        half_size = box_size / 2.0
        bounds_min = box_center - half_size
        bounds_max = box_center + half_size
        inv_dir = np.where(np.abs(ray_dir) > 1e-12, 1.0 / ray_dir, np.inf)
        t0 = (bounds_min - ray_origin) * inv_dir
        t1 = (bounds_max - ray_origin) * inv_dir
        tmin = np.maximum.reduce(np.minimum(t0, t1))
        tmax = np.minimum.reduce(np.maximum(t0, t1))
        if tmax < 0.0 or tmin > tmax:
            return None
        return float(tmin if tmin >= 0.0 else tmax)

    def generate_point_cloud(self, objects: list[dict[str, np.ndarray]]) -> np.ndarray:
        ray_origin = np.zeros(3, dtype=float)
        points: list[np.ndarray] = []

        for ray in self._pixel_rays().reshape(-1, 3):
            nearest_distance = self.max_depth_m
            nearest_point: np.ndarray | None = None
            for obj in objects:
                center = _as_float_array(obj["center"])
                size = _as_float_array(obj["size"])
                distance = self._ray_box_intersection(ray_origin, ray, center, size)
                if distance is None or distance > nearest_distance:
                    continue
                nearest_distance = distance
                nearest_point = ray_origin + ray * distance
            if nearest_point is None:
                continue
            noisy_point = _as_float_array(self.noise.apply(nearest_point))
            if np.isnan(noisy_point).any():
                continue
            points.append(noisy_point)

        if not points:
            return np.empty((0, 3), dtype=float)
        return np.vstack(points)


class SensorSuite:
    """Container for a named set of sensors."""

    def __init__(self, **sensors: Any) -> None:
        self._sensors: dict[str, Any] = dict(sensors)

    def __len__(self) -> int:
        return len(self._sensors)

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return iter(self._sensors.items())

    def __contains__(self, name: str) -> bool:
        return name in self._sensors

    def __getitem__(self, name: str) -> Any:
        return self._sensors[name]

    def keys(self):
        return self._sensors.keys()

    def values(self):
        return self._sensors.values()

    def items(self):
        return self._sensors.items()

    def add_sensor(self, name: str, sensor: Any) -> None:
        self._sensors[name] = sensor

    def remove_sensor(self, name: str) -> Any:
        return self._sensors.pop(name)

    def get_sensor(self, name: str) -> Any:
        return self._sensors[name]

    def read_all(self, readings: dict[str, Any]) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for name, sensor in self._sensors.items():
            sensor_reading = readings[name]
            if isinstance(sensor, IMUSensor):
                accel, gyro = sensor_reading
                outputs[name] = sensor.read(accel, gyro)
            elif isinstance(sensor, DepthCameraSensor):
                outputs[name] = sensor.generate_point_cloud(sensor_reading)
            else:
                outputs[name] = sensor.read(sensor_reading)
        return outputs

    @classmethod
    def default_humanoid_suite(cls) -> SensorSuite:
        return cls(
            left_wrist_ft=ForceTorqueSensor(
                name="left_wrist_ft",
                mount_joint="left_wrist_pitch",
                noise=SensorNoise(gaussian_std=0.2),
            ),
            right_wrist_ft=ForceTorqueSensor(
                name="right_wrist_ft",
                mount_joint="right_wrist_pitch",
                noise=SensorNoise(gaussian_std=0.2),
            ),
            left_foot_proximity=ProximitySensor(
                name="left_foot_proximity",
                noise=SensorNoise(gaussian_std=0.005),
                max_range_m=0.5,
            ),
            right_foot_proximity=ProximitySensor(
                name="right_foot_proximity",
                noise=SensorNoise(gaussian_std=0.005),
                max_range_m=0.5,
            ),
            torso_imu=IMUSensor(
                name="torso_imu",
                noise_accel=SensorNoise(gaussian_std=0.03),
                noise_gyro=SensorNoise(gaussian_std=0.01),
            ),
            head_depth_camera=DepthCameraSensor(
                name="head_depth_camera",
                noise=SensorNoise(gaussian_std=0.002),
                resolution=(64, 48),
                max_depth_m=5.0,
            ),
        )


__all__ = [
    "SensorNoise",
    "ForceTorqueSensor",
    "ProximitySensor",
    "JointEncoderArray",
    "IMUSensor",
    "DepthCameraSensor",
    "SensorSuite",
]
