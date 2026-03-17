from __future__ import annotations

import numpy as np
import pytest

from optisim.sensors import (
    DepthCameraSensor,
    ForceTorqueSensor,
    IMUSensor,
    JointEncoderArray,
    ProximitySensor,
    SensorNoise,
    SensorSuite,
)


def test_sensor_noise_zero_configuration_leaves_values_unchanged() -> None:
    noise = SensorNoise()

    values = noise.apply(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_allclose(values, np.array([1.0, 2.0, 3.0]))


def test_sensor_noise_bias_is_applied_to_scalar() -> None:
    noise = SensorNoise(bias=0.25)

    value = noise.apply(1.0)

    assert value == pytest.approx(1.25)


def test_sensor_noise_apply_has_near_zero_mean_with_zero_bias() -> None:
    np.random.seed(0)
    noise = SensorNoise(gaussian_std=0.5, bias=0.0)

    samples = np.array([noise.apply(0.0) for _ in range(20_000)])

    assert abs(np.nanmean(samples)) < 0.02


def test_sensor_noise_dropout_produces_expected_nan_rate() -> None:
    np.random.seed(1)
    noise = SensorNoise(dropout_prob=0.3)

    values = noise.apply(np.ones(10_000))

    assert np.isnan(values).mean() == pytest.approx(0.3, abs=0.03)


def test_force_torque_sensor_read_clips_force_and_torque_to_limits() -> None:
    sensor = ForceTorqueSensor(
        name="wrist",
        mount_joint="joint",
        noise=SensorNoise(),
        max_force_n=50.0,
        max_torque_nm=10.0,
    )

    reading = sensor.read(np.array([100.0, -80.0, 25.0, 20.0, -15.0, 5.0]))

    np.testing.assert_allclose(reading, np.array([50.0, -50.0, 25.0, 10.0, -10.0, 5.0]))


def test_force_torque_sensor_applies_noise_before_clipping() -> None:
    sensor = ForceTorqueSensor(
        name="wrist",
        mount_joint="joint",
        noise=SensorNoise(bias=2.0),
        max_force_n=100.0,
        max_torque_nm=100.0,
    )

    reading = sensor.read(np.zeros(6))

    np.testing.assert_allclose(reading, np.full(6, 2.0))


def test_force_torque_sensor_is_contact_detects_force_above_threshold() -> None:
    sensor = ForceTorqueSensor(name="wrist", mount_joint="joint", noise=SensorNoise())

    assert sensor.is_contact(np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0]), threshold_n=3.0)


def test_force_torque_sensor_is_contact_rejects_force_below_threshold() -> None:
    sensor = ForceTorqueSensor(name="wrist", mount_joint="joint", noise=SensorNoise())

    assert not sensor.is_contact(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), threshold_n=2.0)


def test_force_torque_sensor_rejects_invalid_wrench_shape() -> None:
    sensor = ForceTorqueSensor(name="wrist", mount_joint="joint", noise=SensorNoise())

    with pytest.raises(ValueError, match="6D wrench"):
        sensor.read(np.array([1.0, 2.0]))


def test_proximity_sensor_read_clamps_to_min_range() -> None:
    sensor = ProximitySensor(name="foot", noise=SensorNoise(), min_range_m=0.05, max_range_m=1.0)

    assert sensor.read(0.01) == pytest.approx(0.05)


def test_proximity_sensor_read_clamps_to_max_range() -> None:
    sensor = ProximitySensor(name="foot", noise=SensorNoise(), min_range_m=0.05, max_range_m=1.0)

    assert sensor.read(5.0) == pytest.approx(1.0)


def test_proximity_sensor_dropout_returns_nan() -> None:
    sensor = ProximitySensor(name="foot", noise=SensorNoise(dropout_prob=1.0))

    assert np.isnan(sensor.read(0.3))


def test_joint_encoder_array_quantizes_correctly() -> None:
    encoder = JointEncoderArray(
        joint_names=["joint_a", "joint_b"],
        noise=SensorNoise(),
        resolution_rad=0.01,
    )

    readings = encoder.read(np.array([0.014, -0.016]))

    np.testing.assert_allclose(readings, np.array([0.01, -0.02]))


def test_joint_encoder_array_applies_bias_before_quantization() -> None:
    encoder = JointEncoderArray(
        joint_names=["joint_a"],
        noise=SensorNoise(bias=0.006),
        resolution_rad=0.01,
    )

    readings = encoder.read(np.array([0.014]))

    np.testing.assert_allclose(readings, np.array([0.02]))


def test_joint_encoder_array_rejects_mismatched_joint_count() -> None:
    encoder = JointEncoderArray(joint_names=["a", "b"], noise=SensorNoise())

    with pytest.raises(ValueError, match="one value per joint"):
        encoder.read(np.array([0.1]))


def test_imu_read_accel_includes_gravity_in_zero_accel_case() -> None:
    imu = IMUSensor(
        name="imu",
        noise_accel=SensorNoise(),
        noise_gyro=SensorNoise(),
    )

    accel = imu.read_accel(np.zeros(3))

    np.testing.assert_allclose(accel, np.array([0.0, 0.0, -9.81]))


def test_imu_read_gyro_returns_noisy_angular_velocity() -> None:
    imu = IMUSensor(
        name="imu",
        noise_accel=SensorNoise(),
        noise_gyro=SensorNoise(bias=0.1),
    )

    gyro = imu.read_gyro(np.array([0.0, 1.0, -1.0]))

    np.testing.assert_allclose(gyro, np.array([0.1, 1.1, -0.9]))


def test_imu_read_returns_accel_and_gyro_keys() -> None:
    imu = IMUSensor(name="imu", noise_accel=SensorNoise(), noise_gyro=SensorNoise())

    reading = imu.read(np.zeros(3), np.ones(3))

    assert set(reading) == {"accel", "gyro"}
    assert reading["accel"].shape == (3,)
    assert reading["gyro"].shape == (3,)


def test_depth_camera_generate_point_cloud_returns_n_by_3_array() -> None:
    camera = DepthCameraSensor(name="head", noise=SensorNoise(), resolution=(16, 12), max_depth_m=5.0)

    cloud = camera.generate_point_cloud(
        [{"center": np.array([2.0, 0.0, 0.0]), "size": np.array([0.6, 0.6, 0.6])}]
    )

    assert cloud.ndim == 2
    assert cloud.shape[1] == 3
    assert len(cloud) > 0


def test_depth_camera_returns_empty_array_when_no_objects_visible() -> None:
    camera = DepthCameraSensor(name="head", noise=SensorNoise(), resolution=(8, 6))

    cloud = camera.generate_point_cloud([])

    assert cloud.shape == (0, 3)


def test_depth_camera_respects_max_depth() -> None:
    camera = DepthCameraSensor(name="head", noise=SensorNoise(), resolution=(8, 6), max_depth_m=1.0)

    cloud = camera.generate_point_cloud(
        [{"center": np.array([2.5, 0.0, 0.0]), "size": np.array([0.4, 0.4, 0.4])}]
    )

    assert cloud.shape == (0, 3)


def test_sensor_suite_supports_basic_container_access() -> None:
    suite = SensorSuite(foot=ProximitySensor(name="foot", noise=SensorNoise()))

    assert len(suite) == 1
    assert "foot" in suite
    assert suite["foot"].name == "foot"


def test_sensor_suite_read_all_dispatches_to_sensor_types() -> None:
    suite = SensorSuite(
        foot=ProximitySensor(name="foot", noise=SensorNoise()),
        imu=IMUSensor(name="imu", noise_accel=SensorNoise(), noise_gyro=SensorNoise()),
        camera=DepthCameraSensor(name="cam", noise=SensorNoise(), resolution=(8, 6)),
    )

    outputs = suite.read_all(
        {
            "foot": 0.25,
            "imu": (np.zeros(3), np.zeros(3)),
            "camera": [{"center": np.array([1.5, 0.0, 0.0]), "size": np.array([0.4, 0.4, 0.4])}],
        }
    )

    assert outputs["foot"] == pytest.approx(0.25)
    assert set(outputs["imu"]) == {"accel", "gyro"}
    assert outputs["camera"].shape[1] == 3


def test_sensor_suite_default_humanoid_suite_creates_non_empty_suite() -> None:
    suite = SensorSuite.default_humanoid_suite()

    assert len(suite) == 6
    assert "torso_imu" in suite
    assert "head_depth_camera" in suite


def test_sensor_suite_add_and_remove_sensor_work() -> None:
    suite = SensorSuite()
    sensor = ProximitySensor(name="foot", noise=SensorNoise())

    suite.add_sensor("foot", sensor)
    removed = suite.remove_sensor("foot")

    assert removed is sensor
    assert len(suite) == 0


def test_public_exports_match_expected_symbols() -> None:
    import optisim.sensors as sensors

    expected = {
        "SensorNoise",
        "ForceTorqueSensor",
        "ProximitySensor",
        "JointEncoderArray",
        "IMUSensor",
        "DepthCameraSensor",
        "SensorSuite",
    }

    assert set(sensors.__all__) == expected
    for name in expected:
        assert hasattr(sensors, name)
