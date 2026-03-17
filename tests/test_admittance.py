from __future__ import annotations

from dataclasses import is_dataclass

import numpy as np
import pytest

import optisim
import optisim.admittance as admittance
from optisim import AdmittanceController1D as PublicAdmittanceController1D
from optisim import AdmittanceController6D as PublicAdmittanceController6D
from optisim import AdmittanceLogger as PublicAdmittanceLogger
from optisim import AdmittanceParams as PublicAdmittanceParams
from optisim import ContactCompliantMotion as PublicContactCompliantMotion
from optisim import admittance as public_admittance_module
from optisim.admittance import (
    AdmittanceController1D,
    AdmittanceController6D,
    AdmittanceLogger,
    AdmittanceParams,
    ContactCompliantMotion,
)


def _params() -> AdmittanceParams:
    return AdmittanceParams(mass=1.0, damping=2.0, stiffness=5.0)


def _controller6d() -> AdmittanceController6D:
    return AdmittanceController6D(
        params_linear=AdmittanceParams(mass=1.0, damping=2.0, stiffness=5.0),
        params_angular=AdmittanceParams(mass=2.0, damping=4.0, stiffness=8.0),
        dt=0.1,
    )


def test_admittance_module_exports_public_surface() -> None:
    assert admittance.__all__ == [
        "AdmittanceParams",
        "AdmittanceController1D",
        "AdmittanceController6D",
        "ContactCompliantMotion",
        "AdmittanceLogger",
    ]
    assert admittance.AdmittanceParams is AdmittanceParams
    assert admittance.AdmittanceController1D is AdmittanceController1D
    assert admittance.AdmittanceController6D is AdmittanceController6D
    assert admittance.ContactCompliantMotion is ContactCompliantMotion
    assert admittance.AdmittanceLogger is AdmittanceLogger


def test_top_level_optisim_exports_admittance_module_and_classes() -> None:
    assert public_admittance_module is admittance
    assert PublicAdmittanceParams is AdmittanceParams
    assert PublicAdmittanceController1D is AdmittanceController1D
    assert PublicAdmittanceController6D is AdmittanceController6D
    assert PublicContactCompliantMotion is ContactCompliantMotion
    assert PublicAdmittanceLogger is AdmittanceLogger
    assert optisim.admittance is admittance
    assert optisim.__version__ == "0.28.0"


def test_admittanceparams_is_dataclass() -> None:
    assert is_dataclass(AdmittanceParams)


def test_admittanceparams_is_valid_for_positive_values() -> None:
    assert AdmittanceParams(mass=1.0, damping=2.0, stiffness=3.0).is_valid() is True


def test_admittanceparams_is_invalid_for_zero_or_negative_values() -> None:
    assert AdmittanceParams(mass=0.0, damping=2.0, stiffness=3.0).is_valid() is False
    assert AdmittanceParams(mass=1.0, damping=-1.0, stiffness=3.0).is_valid() is False
    assert AdmittanceParams(mass=1.0, damping=2.0, stiffness=0.0).is_valid() is False


def test_admittanceparams_from_defaults_returns_expected_values() -> None:
    params = AdmittanceParams.from_defaults()

    assert params.mass == pytest.approx(1.0)
    assert params.damping == pytest.approx(10.0)
    assert params.stiffness == pytest.approx(100.0)


def test_admittancecontroller1d_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="params"):
        AdmittanceController1D(AdmittanceParams(mass=0.0, damping=1.0, stiffness=1.0), dt=0.1)


def test_admittancecontroller1d_rejects_non_positive_dt() -> None:
    with pytest.raises(ValueError, match="dt"):
        AdmittanceController1D(_params(), dt=0.0)


def test_admittancecontroller1d_initial_state_is_zero() -> None:
    controller = AdmittanceController1D(_params(), dt=0.1)

    assert controller.position == pytest.approx(0.0)
    assert controller.velocity == pytest.approx(0.0)
    assert controller.acceleration == pytest.approx(0.0)


def test_admittancecontroller1d_zero_force_keeps_zero_state() -> None:
    controller = AdmittanceController1D(_params(), dt=0.1)

    position = controller.step(0.0)

    assert position == pytest.approx(0.0)
    assert controller.velocity == pytest.approx(0.0)
    assert controller.acceleration == pytest.approx(0.0)


def test_admittancecontroller1d_positive_force_increases_position_and_velocity() -> None:
    controller = AdmittanceController1D(AdmittanceParams(mass=1.0, damping=1.0, stiffness=1.0), dt=0.1)

    position = controller.step(2.0)

    assert controller.acceleration == pytest.approx(2.0)
    assert controller.velocity == pytest.approx(0.2)
    assert position == pytest.approx(0.02)


def test_admittancecontroller1d_restoring_force_can_make_acceleration_negative() -> None:
    controller = AdmittanceController1D(AdmittanceParams(mass=1.0, damping=1.0, stiffness=10.0), dt=0.1)
    controller.position = 1.0
    controller.velocity = 0.0

    controller.step(0.0)

    assert controller.acceleration < 0.0


def test_admittancecontroller1d_damping_reduces_velocity_under_zero_force() -> None:
    controller = AdmittanceController1D(AdmittanceParams(mass=1.0, damping=4.0, stiffness=0.1), dt=0.1)
    controller.velocity = 1.0

    controller.step(0.0)

    assert controller.velocity < 1.0


def test_admittancecontroller1d_reset_clears_state() -> None:
    controller = AdmittanceController1D(_params(), dt=0.1)
    controller.step(3.0)
    controller.reset()

    assert controller.position == pytest.approx(0.0)
    assert controller.velocity == pytest.approx(0.0)
    assert controller.acceleration == pytest.approx(0.0)


def test_admittancecontroller6d_rejects_non_positive_dt() -> None:
    with pytest.raises(ValueError, match="dt"):
        AdmittanceController6D(_params(), _params(), dt=0.0)


def test_admittancecontroller6d_initial_state_is_zero_vector() -> None:
    controller = _controller6d()

    np.testing.assert_allclose(controller.state, np.zeros(6))


def test_admittancecontroller6d_rejects_invalid_wrench_shape() -> None:
    with pytest.raises(ValueError, match="wrench"):
        _controller6d().step(np.zeros(5))


def test_admittancecontroller6d_step_returns_6d_deviation() -> None:
    controller = _controller6d()

    deviation = controller.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    assert deviation.shape == (6,)
    assert deviation[0] > 0.0
    np.testing.assert_allclose(deviation[1:], np.zeros(5))


def test_admittancecontroller6d_uses_separate_linear_and_angular_params() -> None:
    controller = _controller6d()

    deviation = controller.step(np.ones(6))

    assert deviation[0] > deviation[3]


def test_admittancecontroller6d_zero_wrench_preserves_zero_state() -> None:
    controller = _controller6d()

    deviation = controller.step(np.zeros(6))

    np.testing.assert_allclose(deviation, np.zeros(6))
    np.testing.assert_allclose(controller.state, np.zeros(6))


def test_admittancecontroller6d_reset_clears_all_axes() -> None:
    controller = _controller6d()
    controller.step(np.ones(6))
    controller.reset()

    np.testing.assert_allclose(controller.state, np.zeros(6))


def test_contactcompliantmotion_rejects_negative_max_deviation() -> None:
    with pytest.raises(ValueError, match="max_deviation"):
        ContactCompliantMotion(_controller6d(), max_deviation=-0.1)


def test_contactcompliantmotion_rejects_invalid_desired_pose_shape() -> None:
    motion = ContactCompliantMotion(_controller6d())

    with pytest.raises(ValueError, match="desired_pose"):
        motion.step(np.zeros(5), np.zeros(6))


def test_contactcompliantmotion_adds_admittance_deviation_to_desired_pose() -> None:
    motion = ContactCompliantMotion(_controller6d(), max_deviation=1.0)
    desired = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

    result = motion.step(desired, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    assert result[0] > desired[0]
    np.testing.assert_allclose(result[1:], desired[1:])


def test_contactcompliantmotion_clamps_deviation_norm() -> None:
    motion = ContactCompliantMotion(_controller6d(), max_deviation=0.01)

    result = motion.step(np.zeros(6), np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    assert np.linalg.norm(result) == pytest.approx(0.01)


def test_contactcompliantmotion_zero_max_deviation_returns_desired_pose() -> None:
    motion = ContactCompliantMotion(_controller6d(), max_deviation=0.0)
    desired = np.arange(6, dtype=np.float64)

    result = motion.step(desired, np.ones(6))

    np.testing.assert_allclose(result, desired)


def test_contactcompliantmotion_deviation_norm_matches_controller_state() -> None:
    motion = ContactCompliantMotion(_controller6d(), max_deviation=1.0)
    motion.step(np.zeros(6), np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0]))

    assert motion.deviation_norm() == pytest.approx(np.linalg.norm(motion.controller.state))


def test_contactcompliantmotion_is_compliant_false_before_contact() -> None:
    assert ContactCompliantMotion(_controller6d()).is_compliant() is False


def test_contactcompliantmotion_is_compliant_true_after_force_response() -> None:
    motion = ContactCompliantMotion(_controller6d(), max_deviation=1.0)
    motion.step(np.zeros(6), np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    assert motion.is_compliant() is True


def test_admittancelogger_rejects_non_positive_max_steps() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        AdmittanceLogger(max_steps=0)


def test_admittancelogger_starts_empty() -> None:
    logger = AdmittanceLogger()

    assert logger.wrench_history == []
    assert logger.deviation_history == []


def test_admittancelogger_rejects_invalid_wrench_shape() -> None:
    logger = AdmittanceLogger()

    with pytest.raises(ValueError, match="wrench"):
        logger.log(0, np.zeros(5), np.zeros(6))


def test_admittancelogger_rejects_invalid_deviation_shape() -> None:
    logger = AdmittanceLogger()

    with pytest.raises(ValueError, match="deviation"):
        logger.log(0, np.zeros(6), np.zeros(5))


def test_admittancelogger_log_appends_copied_arrays() -> None:
    logger = AdmittanceLogger()
    wrench = np.ones(6)
    deviation = np.full(6, 2.0)

    logger.log(3, wrench, deviation)
    wrench[0] = 99.0
    deviation[0] = 88.0

    assert logger.step_history == [3]
    assert logger.wrench_history[0][0] == pytest.approx(1.0)
    assert logger.deviation_history[0][0] == pytest.approx(2.0)


def test_admittancelogger_enforces_max_steps_bound() -> None:
    logger = AdmittanceLogger(max_steps=2)
    logger.log(1, np.ones(6), np.ones(6))
    logger.log(2, np.full(6, 2.0), np.full(6, 2.0))
    logger.log(3, np.full(6, 3.0), np.full(6, 3.0))

    assert logger.step_history == [2, 3]
    assert len(logger.wrench_history) == 2
    assert len(logger.deviation_history) == 2


def test_admittancelogger_mean_wrench_is_zero_when_empty() -> None:
    np.testing.assert_allclose(AdmittanceLogger().mean_wrench(), np.zeros(6))


def test_admittancelogger_mean_wrench_averages_history() -> None:
    logger = AdmittanceLogger()
    logger.log(0, np.array([1.0, 3.0, 5.0, 0.0, 0.0, 0.0]), np.zeros(6))
    logger.log(1, np.array([3.0, 5.0, 7.0, 0.0, 0.0, 0.0]), np.zeros(6))

    np.testing.assert_allclose(logger.mean_wrench(), [2.0, 4.0, 6.0, 0.0, 0.0, 0.0])


def test_admittancelogger_peak_force_is_zero_when_empty() -> None:
    assert AdmittanceLogger().peak_force() == pytest.approx(0.0)


def test_admittancelogger_peak_force_uses_max_translational_norm() -> None:
    logger = AdmittanceLogger()
    logger.log(0, np.array([3.0, 4.0, 0.0, 9.0, 9.0, 9.0]), np.zeros(6))
    logger.log(1, np.array([1.0, 2.0, 2.0, 0.0, 0.0, 0.0]), np.zeros(6))

    assert logger.peak_force() == pytest.approx(5.0)


def test_admittancelogger_clear_removes_all_history() -> None:
    logger = AdmittanceLogger()
    logger.log(0, np.ones(6), np.ones(6))
    logger.clear()

    assert logger.step_history == []
    assert logger.wrench_history == []
    assert logger.deviation_history == []


__all__ = [name for name in globals() if name.startswith("test_")]
