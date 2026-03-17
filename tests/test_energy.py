from __future__ import annotations

import numpy as np
import pytest

import optisim.energy as energy
from optisim import EnergyBudget as PublicEnergyBudget
from optisim import EnergyEstimator as PublicEnergyEstimator
from optisim import JointPowerModel as PublicJointPowerModel
from optisim import MotorEfficiencyModel as PublicMotorEfficiencyModel
from optisim import TaskEnergyProfile as PublicTaskEnergyProfile
from optisim import energy as public_energy_module
from optisim.energy import EnergyBudget, EnergyEstimator, JointPowerModel, MotorEfficiencyModel, TaskEnergyProfile


def _lookup_model() -> MotorEfficiencyModel:
    return MotorEfficiencyModel(
        torque_grid=np.array([0.0, 10.0, 20.0], dtype=np.float64),
        speed_grid=np.array([0.0, 100.0, 200.0], dtype=np.float64),
        efficiency_map=np.array(
            [
                [0.50, 0.60, 0.70],
                [0.60, 0.70, 0.80],
                [0.70, 0.80, 0.90],
            ],
            dtype=np.float64,
        ),
    )


def _estimator(eta: float = 0.8, damping: float = 0.0) -> EnergyEstimator:
    return EnergyEstimator(
        power_model=JointPowerModel(damping=damping),
        efficiency_model=MotorEfficiencyModel.from_constant(eta),
    )


def test_energy_module_exported_from_root() -> None:
    assert hasattr(public_energy_module, "EnergyEstimator")


def test_root_public_joint_power_model_export() -> None:
    assert PublicJointPowerModel is JointPowerModel


def test_root_public_motor_efficiency_model_export() -> None:
    assert PublicMotorEfficiencyModel is MotorEfficiencyModel


def test_root_public_energy_estimator_export() -> None:
    assert PublicEnergyEstimator is EnergyEstimator


def test_root_public_energy_budget_export() -> None:
    assert PublicEnergyBudget is EnergyBudget


def test_root_public_task_energy_profile_export() -> None:
    assert PublicTaskEnergyProfile is TaskEnergyProfile


def test_joint_power_model_rejects_negative_damping() -> None:
    with pytest.raises(ValueError, match="damping"):
        JointPowerModel(damping=-0.1)


def test_joint_power_model_compute_power_matches_torque_velocity_product() -> None:
    assert JointPowerModel().compute_power(3.0, 4.0) == pytest.approx(12.0)


def test_joint_power_model_compute_power_adds_damping_loss() -> None:
    assert JointPowerModel(damping=0.5).compute_power(3.0, 4.0) == pytest.approx(20.0)


def test_joint_power_model_compute_power_array_vectorizes() -> None:
    model = JointPowerModel(damping=0.5)
    torques = np.array([[1.0, 2.0], [3.0, 4.0]])
    velocities = np.array([[2.0, 3.0], [4.0, 5.0]])

    result = model.compute_power_array(torques, velocities)

    np.testing.assert_allclose(result, torques * velocities + 0.5 * velocities**2)


def test_joint_power_model_compute_power_array_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        JointPowerModel().compute_power_array(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))


def test_motor_efficiency_model_rejects_nond_1_torque_grid() -> None:
    with pytest.raises(ValueError, match="1D"):
        MotorEfficiencyModel(torque_grid=np.array([[0.0, 1.0]]), speed_grid=np.array([0.0, 1.0]), efficiency_map=np.ones((2, 2)))


def test_motor_efficiency_model_rejects_insufficient_grid_points() -> None:
    with pytest.raises(ValueError, match="at least two points"):
        MotorEfficiencyModel(torque_grid=np.array([0.0]), speed_grid=np.array([0.0, 1.0]), efficiency_map=np.ones((1, 2)))


def test_motor_efficiency_model_rejects_nonmonotonic_torque_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        MotorEfficiencyModel(torque_grid=np.array([0.0, 2.0, 1.0]), speed_grid=np.array([0.0, 1.0]), efficiency_map=np.ones((3, 2)))


def test_motor_efficiency_model_rejects_nonmonotonic_speed_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        MotorEfficiencyModel(torque_grid=np.array([0.0, 1.0]), speed_grid=np.array([0.0, 2.0, 1.0]), efficiency_map=np.ones((2, 3)))


def test_motor_efficiency_model_rejects_bad_efficiency_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        MotorEfficiencyModel(torque_grid=np.array([0.0, 1.0]), speed_grid=np.array([0.0, 1.0]), efficiency_map=np.ones((3, 2)))


def test_motor_efficiency_model_from_constant_returns_uniform_lookup() -> None:
    model = MotorEfficiencyModel.from_constant(0.83)

    assert model.get_efficiency(0.2, 0.7) == pytest.approx(0.83)


def test_motor_efficiency_model_returns_exact_grid_value() -> None:
    model = _lookup_model()

    assert model.get_efficiency(10.0, 100.0) == pytest.approx(0.70)


def test_motor_efficiency_model_bilinearly_interpolates_midpoint() -> None:
    model = _lookup_model()

    assert model.get_efficiency(5.0, 50.0) == pytest.approx(0.60)


def test_motor_efficiency_model_uses_absolute_query_values() -> None:
    model = _lookup_model()

    assert model.get_efficiency(-10.0, -100.0) == pytest.approx(0.70)


def test_motor_efficiency_model_clamps_queries_outside_grid() -> None:
    model = _lookup_model()

    assert model.get_efficiency(1000.0, 1000.0) == pytest.approx(0.90)


def test_energy_estimator_estimate_per_joint_returns_joint_vector() -> None:
    estimator = _estimator(eta=0.5)
    torques = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    velocities = np.array([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]])

    result = estimator.estimate_per_joint(torques, velocities, dt=0.5)

    np.testing.assert_allclose(result, np.array([4.0, 4.0]))


def test_energy_estimator_uses_trapezoidal_integration_over_time() -> None:
    estimator = _estimator(eta=1.0)
    torques = np.array([[1.0], [3.0]])
    velocities = np.array([[2.0], [2.0]])

    assert estimator.estimate_energy(torques, velocities, dt=1.0) == pytest.approx(4.0)


def test_energy_estimator_zeroes_negative_power_demand() -> None:
    estimator = _estimator(eta=0.5)
    torques = np.array([[1.0], [-1.0], [1.0]])
    velocities = np.array([[1.0], [1.0], [1.0]])

    assert estimator.estimate_energy(torques, velocities, dt=1.0) == pytest.approx(2.0)


def test_energy_estimator_includes_damping_losses() -> None:
    estimator = _estimator(eta=1.0, damping=0.5)
    torques = np.array([[0.0], [0.0], [0.0]])
    velocities = np.array([[2.0], [2.0], [2.0]])

    assert estimator.estimate_energy(torques, velocities, dt=0.5) == pytest.approx(2.0)


def test_energy_estimator_rejects_non_2d_torques() -> None:
    estimator = _estimator()

    with pytest.raises(ValueError, match="shape"):
        estimator.estimate_energy(np.array([1.0, 2.0]), np.array([[1.0], [2.0]]), dt=0.1)


def test_energy_estimator_rejects_non_2d_velocities() -> None:
    estimator = _estimator()

    with pytest.raises(ValueError, match="shape"):
        estimator.estimate_energy(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]), dt=0.1)


def test_energy_estimator_rejects_shape_mismatch() -> None:
    estimator = _estimator()

    with pytest.raises(ValueError, match="matching shapes"):
        estimator.estimate_energy(np.ones((3, 2)), np.ones((3, 1)), dt=0.1)


def test_energy_estimator_rejects_non_positive_dt() -> None:
    estimator = _estimator()

    with pytest.raises(ValueError, match="dt"):
        estimator.estimate_energy(np.ones((2, 1)), np.ones((2, 1)), dt=0.0)


def test_energy_estimator_raises_when_positive_power_has_zero_efficiency() -> None:
    estimator = _estimator(eta=0.0)

    with pytest.raises(ValueError, match="efficiency"):
        estimator.estimate_energy(np.array([[1.0], [1.0]]), np.array([[1.0], [1.0]]), dt=0.1)


def test_energy_budget_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        EnergyBudget(0.0)


def test_energy_budget_remaining_decreases_after_consumption() -> None:
    budget = EnergyBudget(10.0)
    budget.consume(3.5)

    assert budget.remaining() == pytest.approx(6.5)


def test_energy_budget_is_exhausted_after_budget_is_reached() -> None:
    budget = EnergyBudget(5.0)
    budget.consume(5.0)

    assert budget.is_exhausted() is True


def test_energy_budget_reset_restores_zero_usage() -> None:
    budget = EnergyBudget(5.0)
    budget.consume(2.0)
    budget.reset()

    assert budget.used == pytest.approx(0.0)


def test_task_energy_profile_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        TaskEnergyProfile(name="empty", timestamps=np.array([]), energy_rates=np.array([]))


def test_task_energy_profile_rejects_mismatched_sample_shapes() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        TaskEnergyProfile(name="bad", timestamps=np.array([0.0, 1.0]), energy_rates=np.array([1.0]))


def test_task_energy_profile_rejects_decreasing_timestamps() -> None:
    with pytest.raises(ValueError, match="nondecreasing"):
        TaskEnergyProfile(name="bad", timestamps=np.array([0.0, 0.5, 0.4]), energy_rates=np.array([1.0, 2.0, 3.0]))


def test_task_energy_profile_total_energy_uses_trapezoidal_integration() -> None:
    profile = TaskEnergyProfile(name="task", timestamps=np.array([0.0, 1.0, 2.0]), energy_rates=np.array([0.0, 2.0, 4.0]))

    assert profile.total_energy() == pytest.approx(4.0)


def test_task_energy_profile_mean_power_divides_energy_by_duration() -> None:
    profile = TaskEnergyProfile(name="task", timestamps=np.array([0.0, 1.0, 2.0]), energy_rates=np.array([0.0, 2.0, 4.0]))

    assert profile.mean_power() == pytest.approx(2.0)


def test_task_energy_profile_from_estimator_builds_consistent_profile() -> None:
    estimator = _estimator(eta=0.5)
    torques = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    velocities = np.array([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]])

    profile = TaskEnergyProfile.from_estimator("move", torques, velocities, 0.5, estimator)

    assert profile.name == "move"
    np.testing.assert_allclose(profile.timestamps, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(profile.energy_rates, np.array([8.0, 8.0, 8.0]))
    assert profile.total_energy() == pytest.approx(estimator.estimate_energy(torques, velocities, 0.5))
