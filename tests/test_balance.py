from __future__ import annotations

from dataclasses import is_dataclass

import numpy as np
import pytest

import optisim
import optisim.balance as balance
from optisim import BalanceMonitor as PublicBalanceMonitor
from optisim import BalanceReport as PublicBalanceReport
from optisim import COMState as PublicCOMState
from optisim import SupportPolygon as PublicSupportPolygon
from optisim import ZMPCalculator as PublicZMPCalculator
from optisim import balance as public_balance_module
from optisim.balance import BalanceMonitor, BalanceReport, COMState, SupportPolygon, ZMPCalculator


def _square_contacts() -> list[tuple[float, float]]:
    return [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)]


def test_balance_module_exported_from_root() -> None:
    assert hasattr(public_balance_module, "BalanceMonitor")


def test_root_public_comstate_export() -> None:
    assert PublicCOMState is COMState


def test_root_public_balancemonitor_export() -> None:
    assert PublicBalanceMonitor is BalanceMonitor


def test_comstate_is_dataclass() -> None:
    assert is_dataclass(COMState)


def test_comstate_from_zeros_returns_zero_vectors() -> None:
    state = COMState.from_zeros()

    np.testing.assert_allclose(state.position, np.zeros(3))
    np.testing.assert_allclose(state.velocity, np.zeros(3))
    np.testing.assert_allclose(state.acceleration, np.zeros(3))


def test_comstate_rejects_invalid_position_shape() -> None:
    with pytest.raises(ValueError, match="position"):
        COMState(position=np.zeros(2), velocity=np.zeros(3), acceleration=np.zeros(3))


def test_comstate_rejects_invalid_acceleration_shape() -> None:
    with pytest.raises(ValueError, match="acceleration"):
        COMState(position=np.zeros(3), velocity=np.zeros(3), acceleration=np.zeros(4))


def test_balancereport_is_dataclass() -> None:
    assert is_dataclass(BalanceReport)


def test_balancereport_default_timestamp_is_zero() -> None:
    report = BalanceReport(zmp=np.array([0.0, 0.0]), stable=True, margin=0.1, support_area=0.2)

    assert report.timestamp == pytest.approx(0.0)


def test_balancereport_rejects_invalid_zmp_shape() -> None:
    with pytest.raises(ValueError, match="zmp"):
        BalanceReport(zmp=np.zeros(3), stable=True, margin=0.0, support_area=0.0)


def test_supportpolygon_from_contacts_computes_convex_hull() -> None:
    polygon = SupportPolygon.from_contacts([(0.0, 0.0), (1.0, 0.0), (0.5, 0.5), (0.0, 1.0), (1.0, 1.0)])

    assert len(polygon.vertices) == 4
    assert {tuple(vertex) for vertex in polygon.vertices} == {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}


def test_supportpolygon_rejects_empty_contacts() -> None:
    with pytest.raises(ValueError, match="contacts"):
        SupportPolygon.from_contacts([])


def test_supportpolygon_rejects_invalid_vertex_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        SupportPolygon(vertices=np.zeros((3, 3)))


def test_supportpolygon_contains_interior_point() -> None:
    polygon = SupportPolygon.from_contacts(_square_contacts())

    assert polygon.contains_point(np.array([0.0, 0.0])) is True


def test_supportpolygon_contains_boundary_point() -> None:
    polygon = SupportPolygon.from_contacts(_square_contacts())

    assert polygon.contains_point(np.array([0.1, 0.0])) is True


def test_supportpolygon_rejects_outside_point() -> None:
    polygon = SupportPolygon.from_contacts(_square_contacts())

    assert polygon.contains_point(np.array([0.2, 0.0])) is False


def test_supportpolygon_contains_point_for_segment_contact() -> None:
    polygon = SupportPolygon.from_contacts([(0.0, 0.0), (1.0, 0.0)])

    assert polygon.contains_point(np.array([0.5, 0.0])) is True


def test_supportpolygon_area_uses_shoelace_formula() -> None:
    polygon = SupportPolygon.from_contacts([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)])

    assert polygon.area() == pytest.approx(2.0)


def test_supportpolygon_area_is_zero_for_segment() -> None:
    polygon = SupportPolygon.from_contacts([(0.0, 0.0), (2.0, 0.0)])

    assert polygon.area() == pytest.approx(0.0)


def test_supportpolygon_centroid_for_polygon() -> None:
    polygon = SupportPolygon.from_contacts([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])

    np.testing.assert_allclose(polygon.centroid(), [1.0, 1.0])


def test_supportpolygon_centroid_for_segment_is_midpoint() -> None:
    polygon = SupportPolygon.from_contacts([(0.0, 0.0), (2.0, 0.0)])

    np.testing.assert_allclose(polygon.centroid(), [1.0, 0.0])


def test_zmpcalculator_compute_zmp_returns_com_projection_for_zero_acceleration() -> None:
    com = COMState(position=np.array([0.1, -0.2, 0.8]), velocity=np.zeros(3), acceleration=np.zeros(3))

    np.testing.assert_allclose(ZMPCalculator.compute_zmp(com, total_mass=50.0), [0.1, -0.2])


def test_zmpcalculator_compute_zmp_uses_horizontal_acceleration() -> None:
    com = COMState(position=np.array([0.0, 0.0, 1.0]), velocity=np.zeros(3), acceleration=np.array([1.0, -2.0, 0.0]))

    zmp = ZMPCalculator.compute_zmp(com, total_mass=60.0, g=10.0)

    np.testing.assert_allclose(zmp, [-0.1, 0.2])


def test_zmpcalculator_compute_zmp_accounts_for_vertical_acceleration() -> None:
    com = COMState(position=np.array([0.0, 0.0, 1.0]), velocity=np.zeros(3), acceleration=np.array([1.0, 0.0, 10.0]))

    zmp = ZMPCalculator.compute_zmp(com, total_mass=60.0, g=10.0)

    np.testing.assert_allclose(zmp, [-0.05, 0.0])


def test_zmpcalculator_compute_zmp_rejects_non_positive_mass() -> None:
    with pytest.raises(ValueError, match="total_mass"):
        ZMPCalculator.compute_zmp(COMState.from_zeros(), total_mass=0.0)


def test_zmpcalculator_compute_zmp_rejects_non_positive_gravity() -> None:
    with pytest.raises(ValueError, match="g"):
        ZMPCalculator.compute_zmp(COMState.from_zeros(), total_mass=1.0, g=0.0)


def test_zmpcalculator_compute_zmp_rejects_singular_vertical_dynamics() -> None:
    com = COMState(position=np.array([0.0, 0.0, 1.0]), velocity=np.zeros(3), acceleration=np.array([0.0, 0.0, -9.81]))

    with pytest.raises(ValueError, match="vertical acceleration"):
        ZMPCalculator.compute_zmp(com, total_mass=1.0)


def test_zmpcalculator_is_stable_returns_false_for_exterior_point() -> None:
    assert ZMPCalculator.is_stable(np.array([0.5, 0.0]), np.asarray(_square_contacts(), dtype=np.float64)) is False


def test_zmpcalculator_stability_margin_is_positive_inside_polygon() -> None:
    margin = ZMPCalculator.stability_margin(np.array([0.0, 0.0]), np.asarray(_square_contacts(), dtype=np.float64))

    assert margin == pytest.approx(0.1)


def test_zmpcalculator_stability_margin_is_zero_on_boundary() -> None:
    margin = ZMPCalculator.stability_margin(np.array([0.1, 0.0]), np.asarray(_square_contacts(), dtype=np.float64))

    assert margin == pytest.approx(0.0, abs=1e-9)


def test_zmpcalculator_stability_margin_is_negative_outside_polygon() -> None:
    margin = ZMPCalculator.stability_margin(np.array([0.2, 0.0]), np.asarray(_square_contacts(), dtype=np.float64))

    assert margin == pytest.approx(-0.1)


def test_zmpcalculator_stability_margin_for_segment_is_negative_distance_when_off_segment() -> None:
    margin = ZMPCalculator.stability_margin(np.array([0.5, 0.2]), np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64))

    assert margin == pytest.approx(-0.2)


def test_balancemonitor_rejects_non_positive_mass() -> None:
    with pytest.raises(ValueError, match="total_mass"):
        BalanceMonitor(total_mass=0.0)


def test_balancemonitor_rejects_non_positive_gravity() -> None:
    with pytest.raises(ValueError, match="g"):
        BalanceMonitor(total_mass=50.0, g=0.0)


def test_balancemonitor_is_balanced_false_without_history() -> None:
    assert BalanceMonitor(total_mass=50.0).is_balanced() is False


def test_balancemonitor_update_returns_report_and_appends_history() -> None:
    monitor = BalanceMonitor(total_mass=50.0)
    report = monitor.update(COMState.from_zeros(), _square_contacts())

    assert isinstance(report, BalanceReport)
    assert monitor.history[-1] is report
    assert report.stable is True
    assert report.support_area == pytest.approx(0.04)
    assert report.timestamp > 0.0


def test_balancemonitor_is_balanced_respects_threshold_margin() -> None:
    monitor = BalanceMonitor(total_mass=50.0)
    monitor.update(COMState.from_zeros(), _square_contacts())

    assert monitor.is_balanced(threshold_margin=0.05) is True
    assert monitor.is_balanced(threshold_margin=0.11) is False


def test_balancemonitor_tracks_unstable_state() -> None:
    monitor = BalanceMonitor(total_mass=50.0)
    com = COMState(position=np.array([0.0, 0.0, 1.0]), velocity=np.zeros(3), acceleration=np.array([-3.0, 0.0, 0.0]))

    report = monitor.update(com, _square_contacts())

    assert report.stable is False
    assert report.margin < 0.0
    assert monitor.is_balanced() is False


def test_balancemonitor_history_is_capped_at_max_history() -> None:
    monitor = BalanceMonitor(total_mass=50.0)
    monitor.max_history = 3

    for offset in range(5):
        com = COMState(position=np.array([offset * 0.001, 0.0, 0.5]), velocity=np.zeros(3), acceleration=np.zeros(3))
        monitor.update(com, _square_contacts())

    assert len(monitor.history) == 3
    np.testing.assert_allclose(monitor.history[0].zmp, [0.002, 0.0])


def test_balance_module_version_is_current() -> None:
    assert balance.__all__ == ["COMState", "ZMPCalculator", "SupportPolygon", "BalanceMonitor", "BalanceReport"]
    assert optisim.__version__ == "0.28.0"
