from __future__ import annotations

import numpy as np
import pytest

import optisim.estimation as estimation
from optisim import EKFConfig as PublicEKFConfig
from optisim import RobotState as PublicRobotState
from optisim import StateEstimationPipeline as PublicStateEstimationPipeline
from optisim import build_estimator as public_build_estimator
from optisim.estimation import (
    EKFConfig,
    IMUIntegrator,
    RobotState,
    RobotStateEstimator,
    StateEstimationPipeline,
    build_estimator,
)
from optisim.sensors import IMUSensor, SensorNoise


def _make_state(n_joints: int = 4, covariance_scale: float = 0.1) -> RobotState:
    state_dim = 12 + 2 * n_joints
    return RobotState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
        joint_positions=np.linspace(0.0, 0.3, n_joints),
        joint_velocities=np.zeros(n_joints),
        covariance=np.eye(state_dim) * covariance_scale,
        timestamp=0.0,
    )


def test_robot_state_construction_with_required_fields() -> None:
    state = _make_state(n_joints=3)

    assert state.position.shape == (3,)
    assert state.velocity.shape == (3,)
    assert state.orientation.shape == (4,)
    assert state.angular_velocity.shape == (3,)
    assert state.joint_positions.shape == (3,)
    assert state.joint_velocities.shape == (3,)
    assert state.covariance.shape == (18, 18)
    assert state.timestamp == 0.0


def test_ekf_config_defaults_match_expected_values() -> None:
    config = EKFConfig()

    assert config.process_noise_pos == pytest.approx(0.001)
    assert config.process_noise_vel == pytest.approx(0.01)
    assert config.process_noise_ori == pytest.approx(0.001)
    assert config.process_noise_angvel == pytest.approx(0.01)
    assert config.imu_noise_accel == pytest.approx(0.05)
    assert config.imu_noise_gyro == pytest.approx(0.005)
    assert config.encoder_noise == pytest.approx(0.001)
    assert config.contact_update_weight == pytest.approx(0.5)


def test_robot_state_estimator_initialization() -> None:
    estimator = RobotStateEstimator(n_joints=6)

    assert estimator.n_joints == 6
    assert estimator.state.joint_positions.shape == (6,)
    assert estimator.state.covariance.shape == (24, 24)


def test_reset_replaces_internal_state() -> None:
    estimator = RobotStateEstimator(n_joints=2)
    initial = _make_state(n_joints=2)
    initial.position = np.array([0.1, -0.2, 0.9])

    estimator.reset(initial)

    np.testing.assert_allclose(estimator.state.position, initial.position)


def test_predict_returns_robot_state_with_valid_shape() -> None:
    estimator = RobotStateEstimator(n_joints=5)

    state = estimator.predict(dt=0.01, imu_accel=np.array([0.0, 0.0, -9.81]), imu_gyro=np.zeros(3))

    assert isinstance(state, RobotState)
    assert state.position.shape == (3,)
    assert state.orientation.shape == (4,)
    assert state.covariance.shape == (22, 22)


def test_predict_with_zero_accel_gyro_keeps_position_roughly_stable() -> None:
    estimator = RobotStateEstimator(n_joints=3)

    state = estimator.predict(dt=0.1, imu_accel=np.array([0.0, 0.0, -9.81]), imu_gyro=np.zeros(3))

    np.testing.assert_allclose(state.position, np.zeros(3), atol=1e-8)
    np.testing.assert_allclose(state.velocity, np.zeros(3), atol=1e-8)


def test_predict_with_accel_increases_velocity() -> None:
    estimator = RobotStateEstimator(n_joints=2)

    state = estimator.predict(dt=0.2, imu_accel=np.array([1.0, 0.0, -9.81]), imu_gyro=np.zeros(3))

    assert state.velocity[0] > 0.19
    assert state.position[0] > 0.01


def test_update_joints_reduces_joint_position_uncertainty() -> None:
    estimator = RobotStateEstimator(n_joints=3)
    estimator.reset(_make_state(n_joints=3, covariance_scale=0.5))
    before = np.diag(estimator.state.covariance)[12:15].copy()

    estimator.update_joints(np.array([0.05, 0.1, 0.2]))
    after = np.diag(estimator.state.covariance)[12:15]

    assert np.all(after < before)


def test_update_vision_updates_position_estimate() -> None:
    estimator = RobotStateEstimator(n_joints=1)

    state = estimator.update_vision(np.array([0.4, -0.1, 0.8]))

    assert state.position[0] > 0.0
    assert state.position[2] > 0.0


def test_update_contact_constrains_com_position() -> None:
    estimator = RobotStateEstimator(n_joints=0)
    displaced = _make_state(n_joints=0)
    displaced.position = np.array([0.4, 0.2, 1.0])
    estimator.reset(displaced)

    state = estimator.update_contact([np.array([0.0, 0.0, 0.9]), np.array([0.1, 0.0, 0.9])])

    assert state.position[0] < 0.4
    assert state.position[2] < 1.0


def test_covariance_grows_during_predict() -> None:
    estimator = RobotStateEstimator(n_joints=2)
    before = np.trace(estimator.state.covariance)

    estimator.predict(dt=0.05, imu_accel=np.array([0.0, 0.0, -9.81]), imu_gyro=np.zeros(3))
    after = np.trace(estimator.state.covariance)

    assert after > before


def test_covariance_shrinks_after_measurement_update() -> None:
    estimator = RobotStateEstimator(n_joints=2)
    estimator.predict(dt=0.05, imu_accel=np.array([0.2, 0.0, -9.81]), imu_gyro=np.zeros(3))
    before = np.trace(estimator.state.covariance)

    estimator.update_vision(np.array([0.0, 0.0, 0.0]))
    after = np.trace(estimator.state.covariance)

    assert after < before


def test_multiple_predict_update_cycles_are_stable() -> None:
    estimator = RobotStateEstimator(n_joints=4)

    for _ in range(10):
        estimator.predict(dt=0.01, imu_accel=np.array([0.1, 0.0, -9.81]), imu_gyro=np.array([0.0, 0.0, 0.01]))
        estimator.update_joints(np.zeros(4))
        estimator.update_vision(np.array([0.0, 0.0, 0.9]))

    state = estimator.state
    assert np.isfinite(state.position).all()
    assert np.isfinite(state.orientation).all()
    assert state.timestamp == pytest.approx(0.1)


def test_state_property_returns_copy() -> None:
    estimator = RobotStateEstimator(n_joints=1)

    state = estimator.state
    state.position[:] = 1.0

    np.testing.assert_allclose(estimator.state.position, np.zeros(3))


def test_state_estimation_pipeline_initialization() -> None:
    pipeline = StateEstimationPipeline(n_joints=7)

    assert isinstance(pipeline.estimator, RobotStateEstimator)
    assert pipeline.get_history() == []


def test_process_sensor_bundle_returns_robot_state() -> None:
    pipeline = StateEstimationPipeline(n_joints=3)

    state = pipeline.process_sensor_bundle(
        imu={"accel": np.array([0.0, 0.0, -9.81]), "gyro": np.zeros(3)},
        encoders=np.array([0.1, -0.1, 0.2]),
        dt=0.01,
    )

    assert isinstance(state, RobotState)
    assert state.joint_positions.shape == (3,)


def test_get_history_grows_with_each_call() -> None:
    pipeline = StateEstimationPipeline(n_joints=2)

    for _ in range(3):
        pipeline.process_sensor_bundle(
            imu={"accel": np.array([0.0, 0.0, -9.81]), "gyro": np.zeros(3)},
            encoders=np.zeros(2),
        )

    assert len(pipeline.get_history()) == 3


def test_uncertainty_norm_is_positive_scalar() -> None:
    pipeline = StateEstimationPipeline(n_joints=3)
    pipeline.process_sensor_bundle(
        imu={"accel": np.array([0.0, 0.0, -9.81]), "gyro": np.zeros(3)},
        encoders=np.zeros(3),
    )

    assert isinstance(pipeline.uncertainty_norm, float)
    assert pipeline.uncertainty_norm > 0.0


def test_imu_integrator_integrate_returns_valid_tuple() -> None:
    integrator = IMUIntegrator()

    position, velocity, orientation = integrator.integrate(
        dt=0.02,
        accel=np.array([0.0, 0.0, -9.81]),
        gyro=np.array([0.0, 0.1, 0.0]),
    )

    assert position.shape == (3,)
    assert velocity.shape == (3,)
    assert orientation.shape == (4,)
    assert np.isclose(np.linalg.norm(orientation), 1.0)


def test_full_pipeline_twenty_step_simulation_with_imu_and_encoder_updates() -> None:
    pipeline = build_estimator(n_joints=6)

    for step in range(20):
        state = pipeline.process_sensor_bundle(
            imu={"accel": np.array([0.05, 0.0, -9.81]), "gyro": np.array([0.0, 0.0, 0.01])},
            encoders=np.linspace(0.0, 0.2, 6) + step * 1e-3,
            visual_odometry=np.array([0.01 * step, 0.0, 0.9]),
            dt=0.01,
        )

    assert isinstance(state, RobotState)
    assert len(pipeline.get_history()) == 20
    assert state.position[0] > 0.0


def test_pipeline_accepts_optisim_sensor_imu_reading() -> None:
    sensor = IMUSensor(
        name="torso",
        noise_accel=SensorNoise(),
        noise_gyro=SensorNoise(),
    )
    pipeline = StateEstimationPipeline(n_joints=2)
    imu_reading = sensor.read(np.zeros(3), np.array([0.0, 0.0, 0.02]))

    state = pipeline.process_sensor_bundle(imu=imu_reading, encoders=np.zeros(2), dt=0.01)

    assert isinstance(state, RobotState)
    assert state.angular_velocity[2] == pytest.approx(0.02)


def test_public_exports_expose_estimation_symbols() -> None:
    assert PublicRobotState is RobotState
    assert PublicEKFConfig is EKFConfig
    assert PublicStateEstimationPipeline is StateEstimationPipeline
    assert public_build_estimator is build_estimator


def test_estimation_package_all_matches_expected_symbols() -> None:
    expected = {
        "RobotState",
        "EKFConfig",
        "RobotStateEstimator",
        "IMUIntegrator",
        "StateEstimationPipeline",
        "build_estimator",
    }

    assert set(estimation.__all__) == expected
    for name in expected:
        assert hasattr(estimation, name)


__all__ = [name for name in globals() if name.startswith("test_")]
