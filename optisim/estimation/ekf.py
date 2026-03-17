"""Extended Kalman filter utilities for humanoid state estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_vector(values: Any, size: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return array


def _normalize_quaternion(quaternion: Any) -> FloatArray:
    quat = _as_vector(quaternion, 4, "orientation").copy()
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quaternion_multiply(lhs: FloatArray, rhs: FloatArray) -> FloatArray:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quaternion_from_angular_velocity(angular_velocity: FloatArray, dt: float) -> FloatArray:
    omega_norm = float(np.linalg.norm(angular_velocity))
    angle = omega_norm * max(dt, 0.0)
    if angle < 1e-12:
        half_dt = 0.5 * dt
        return _normalize_quaternion(
            np.array([1.0, half_dt * angular_velocity[0], half_dt * angular_velocity[1], half_dt * angular_velocity[2]])
        )
    axis = angular_velocity / omega_norm
    half_angle = 0.5 * angle
    return np.array([np.cos(half_angle), *(axis * np.sin(half_angle))], dtype=np.float64)


def _rotation_matrix_from_quaternion(quaternion: FloatArray) -> FloatArray:
    w, x, y, z = _normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _small_angle_quaternion(delta_theta: FloatArray) -> FloatArray:
    return _normalize_quaternion(
        np.array(
            [1.0, 0.5 * delta_theta[0], 0.5 * delta_theta[1], 0.5 * delta_theta[2]],
            dtype=np.float64,
        )
    )


@dataclass(slots=True)
class RobotState:
    """Nominal humanoid state with full error-state covariance."""

    position: FloatArray
    velocity: FloatArray
    orientation: FloatArray
    angular_velocity: FloatArray
    joint_positions: FloatArray
    joint_velocities: FloatArray
    covariance: FloatArray
    timestamp: float

    def __post_init__(self) -> None:
        self.position = _as_vector(self.position, 3, "position")
        self.velocity = _as_vector(self.velocity, 3, "velocity")
        self.orientation = _normalize_quaternion(self.orientation)
        self.angular_velocity = _as_vector(self.angular_velocity, 3, "angular_velocity")
        self.joint_positions = np.asarray(self.joint_positions, dtype=np.float64)
        self.joint_velocities = np.asarray(self.joint_velocities, dtype=np.float64)
        if self.joint_positions.ndim != 1:
            raise ValueError("joint_positions must be a 1D vector")
        if self.joint_velocities.ndim != 1:
            raise ValueError("joint_velocities must be a 1D vector")
        if self.joint_positions.shape != self.joint_velocities.shape:
            raise ValueError("joint_positions and joint_velocities must have matching shapes")
        expected_size = 12 + 2 * self.joint_positions.size
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        if self.covariance.shape != (expected_size, expected_size):
            raise ValueError(f"covariance must have shape ({expected_size}, {expected_size})")
        self.timestamp = float(self.timestamp)


@dataclass(slots=True)
class EKFConfig:
    """Configuration values for the humanoid state estimator."""

    process_noise_pos: float = 0.001
    process_noise_vel: float = 0.01
    process_noise_ori: float = 0.001
    process_noise_angvel: float = 0.01
    imu_noise_accel: float = 0.05
    imu_noise_gyro: float = 0.005
    encoder_noise: float = 0.001
    contact_update_weight: float = 0.5


def _clone_state(state: RobotState) -> RobotState:
    return RobotState(
        position=state.position.copy(),
        velocity=state.velocity.copy(),
        orientation=state.orientation.copy(),
        angular_velocity=state.angular_velocity.copy(),
        joint_positions=state.joint_positions.copy(),
        joint_velocities=state.joint_velocities.copy(),
        covariance=state.covariance.copy(),
        timestamp=float(state.timestamp),
    )


class RobotStateEstimator:
    """Error-state EKF for humanoid CoM, base attitude, and joint states."""

    def __init__(self, n_joints: int, config: EKFConfig | None = None) -> None:
        if n_joints < 0:
            raise ValueError("n_joints must be non-negative")
        self.n_joints = int(n_joints)
        self.config = config or EKFConfig()
        self._state_dim = 12 + 2 * self.n_joints
        self._gravity_world = np.array([0.0, 0.0, 9.81], dtype=np.float64)
        self._state = RobotState(
            position=np.zeros(3, dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
            joint_positions=np.zeros(self.n_joints, dtype=np.float64),
            joint_velocities=np.zeros(self.n_joints, dtype=np.float64),
            covariance=np.eye(self._state_dim, dtype=np.float64) * 1e-3,
            timestamp=0.0,
        )

    @property
    def state(self) -> RobotState:
        return _clone_state(self._state)

    def reset(self, initial_state: RobotState) -> None:
        if initial_state.joint_positions.size != self.n_joints:
            raise ValueError("initial_state joint count does not match estimator configuration")
        self._state = _clone_state(initial_state)

    def predict(self, dt: float, imu_accel: FloatArray, imu_gyro: FloatArray) -> RobotState:
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        accel = _as_vector(imu_accel, 3, "imu_accel")
        gyro = _as_vector(imu_gyro, 3, "imu_gyro")

        rotation = _rotation_matrix_from_quaternion(self._state.orientation)
        world_accel = rotation @ accel + self._gravity_world

        self._state.position = self._state.position + self._state.velocity * dt + 0.5 * world_accel * dt * dt
        self._state.velocity = self._state.velocity + world_accel * dt
        self._state.orientation = _normalize_quaternion(
            _quaternion_multiply(self._state.orientation, _quaternion_from_angular_velocity(gyro, dt))
        )
        self._state.angular_velocity = gyro.copy()
        self._state.timestamp += dt

        transition = np.eye(self._state_dim, dtype=np.float64)
        transition[0:3, 3:6] = np.eye(3, dtype=np.float64) * dt
        self._state.covariance = transition @ self._state.covariance @ transition.T + self._process_noise(dt)
        self._state.covariance = 0.5 * (self._state.covariance + self._state.covariance.T)
        return self.state

    def update_joints(self, encoder_readings: FloatArray) -> RobotState:
        readings = np.asarray(encoder_readings, dtype=np.float64)
        if readings.shape != (self.n_joints,):
            raise ValueError(f"encoder_readings must have shape ({self.n_joints},)")
        if self.n_joints == 0:
            return self.state
        h_matrix = np.zeros((self.n_joints, self._state_dim), dtype=np.float64)
        h_matrix[:, 12 : 12 + self.n_joints] = np.eye(self.n_joints, dtype=np.float64)
        innovation = readings - self._state.joint_positions
        noise = np.eye(self.n_joints, dtype=np.float64) * self.config.encoder_noise**2
        self._measurement_update(h_matrix, innovation, noise)
        return self.state

    def update_contact(self, contact_positions: list[FloatArray]) -> RobotState:
        if not contact_positions:
            return self.state
        contacts = [np.asarray(contact, dtype=np.float64) for contact in contact_positions]
        if any(contact.shape != (3,) for contact in contacts):
            raise ValueError("each contact position must have shape (3,)")
        contact_centroid = np.mean(np.vstack(contacts), axis=0)
        h_matrix = np.zeros((3, self._state_dim), dtype=np.float64)
        h_matrix[:, 0:3] = np.eye(3, dtype=np.float64)
        sigma = max(1e-6, 0.05 / max(self.config.contact_update_weight, 1e-6))
        innovation = contact_centroid - self._state.position
        noise = np.eye(3, dtype=np.float64) * sigma**2
        self._measurement_update(h_matrix, innovation, noise)
        return self.state

    def update_vision(self, visual_odometry: FloatArray) -> RobotState:
        measurement = _as_vector(visual_odometry, 3, "visual_odometry")
        h_matrix = np.zeros((3, self._state_dim), dtype=np.float64)
        h_matrix[:, 0:3] = np.eye(3, dtype=np.float64)
        innovation = measurement - self._state.position
        noise = np.eye(3, dtype=np.float64) * max(self.config.imu_noise_accel, 1e-6) ** 2
        self._measurement_update(h_matrix, innovation, noise)
        return self.state

    def _process_noise(self, dt: float) -> FloatArray:
        diagonal = np.full(self._state_dim, 1e-6, dtype=np.float64)
        diagonal[0:3] = self.config.process_noise_pos * dt
        diagonal[3:6] = (self.config.process_noise_vel + self.config.imu_noise_accel**2) * dt
        diagonal[6:9] = (self.config.process_noise_ori + self.config.imu_noise_gyro**2) * dt
        diagonal[9:12] = self.config.process_noise_angvel * dt
        if self.n_joints:
            diagonal[12 : 12 + self.n_joints] = self.config.encoder_noise**2 * dt
            diagonal[12 + self.n_joints :] = max(self.config.process_noise_angvel, 1e-6) * dt
        return np.diag(diagonal)

    def _measurement_update(self, h_matrix: FloatArray, innovation: FloatArray, noise: FloatArray) -> None:
        covariance = self._state.covariance
        innovation_covariance = h_matrix @ covariance @ h_matrix.T + noise
        kalman_gain = covariance @ h_matrix.T @ np.linalg.inv(innovation_covariance)
        delta = kalman_gain @ innovation

        self._state.position = self._state.position + delta[0:3]
        self._state.velocity = self._state.velocity + delta[3:6]
        self._state.orientation = _normalize_quaternion(
            _quaternion_multiply(self._state.orientation, _small_angle_quaternion(delta[6:9]))
        )
        self._state.angular_velocity = self._state.angular_velocity + delta[9:12]
        if self.n_joints:
            self._state.joint_positions = self._state.joint_positions + delta[12 : 12 + self.n_joints]
            self._state.joint_velocities = self._state.joint_velocities + delta[12 + self.n_joints :]

        identity = np.eye(self._state_dim, dtype=np.float64)
        left = identity - kalman_gain @ h_matrix
        self._state.covariance = left @ covariance @ left.T + kalman_gain @ noise @ kalman_gain.T
        self._state.covariance = 0.5 * (self._state.covariance + self._state.covariance.T)


class IMUIntegrator:
    """Dead-reckoning pose integrator using only IMU inputs."""

    def __init__(self) -> None:
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._gravity_world = np.array([0.0, 0.0, 9.81], dtype=np.float64)

    def integrate(self, dt: float, accel: FloatArray, gyro: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        accel_vec = _as_vector(accel, 3, "accel")
        gyro_vec = _as_vector(gyro, 3, "gyro")
        rotation = _rotation_matrix_from_quaternion(self.orientation)
        world_accel = rotation @ accel_vec + self._gravity_world
        self.position = self.position + self.velocity * dt + 0.5 * world_accel * dt * dt
        self.velocity = self.velocity + world_accel * dt
        self.orientation = _normalize_quaternion(
            _quaternion_multiply(self.orientation, _quaternion_from_angular_velocity(gyro_vec, dt))
        )
        return self.position.copy(), self.velocity.copy(), self.orientation.copy()


class StateEstimationPipeline:
    """High-level estimator wrapper with history tracking."""

    def __init__(self, n_joints: int, config: EKFConfig | None = None) -> None:
        self.estimator = RobotStateEstimator(n_joints=n_joints, config=config)
        self.history: list[RobotState] = []

    @property
    def uncertainty_norm(self) -> float:
        covariance = self.estimator.state.covariance
        return float(np.trace(covariance) / covariance.shape[0])

    def process_sensor_bundle(
        self,
        imu: dict[str, FloatArray] | tuple[FloatArray, FloatArray],
        encoders: FloatArray,
        contacts: list[FloatArray] | None = None,
        visual_odometry: FloatArray | None = None,
        dt: float = 0.01,
    ) -> RobotState:
        accel, gyro = self._parse_imu_bundle(imu)
        self.estimator.predict(dt=dt, imu_accel=accel, imu_gyro=gyro)
        self.estimator.update_joints(encoders)
        if contacts:
            self.estimator.update_contact(contacts)
        if visual_odometry is not None:
            self.estimator.update_vision(visual_odometry)
        state = self.estimator.state
        self.history.append(state)
        return _clone_state(state)

    def get_history(self) -> list[RobotState]:
        return [_clone_state(state) for state in self.history]

    @staticmethod
    def _parse_imu_bundle(imu: dict[str, FloatArray] | tuple[FloatArray, FloatArray]) -> tuple[FloatArray, FloatArray]:
        if isinstance(imu, tuple):
            if len(imu) != 2:
                raise ValueError("imu tuple must contain (accel, gyro)")
            accel, gyro = imu
            return _as_vector(accel, 3, "imu accel"), _as_vector(gyro, 3, "imu gyro")
        if "accel" not in imu or "gyro" not in imu:
            raise ValueError("imu reading must contain 'accel' and 'gyro'")
        return _as_vector(imu["accel"], 3, "imu accel"), _as_vector(imu["gyro"], 3, "imu gyro")


def build_estimator(n_joints: int, config: EKFConfig | None = None) -> StateEstimationPipeline:
    """Construct a state-estimation pipeline with sensible defaults."""

    return StateEstimationPipeline(n_joints=n_joints, config=config)
