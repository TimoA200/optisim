from __future__ import annotations

import numpy as np

from optisim import dexterous
from optisim.contact import ContactPoint
from optisim.dexterous import (
    DexterousController,
    Finger,
    FingerCommand,
    FingerControlMode,
    FingerJoint,
    Hand,
    TactileCell,
    TactileSensor,
)


def _contact(position: tuple[float, float, float] = (0.0, 0.0, 0.012)) -> ContactPoint:
    return ContactPoint(
        position=np.asarray(position, dtype=np.float64),
        normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        depth=0.001,
        body_a="finger",
        body_b="object",
    )


def test_dexterous_module_exported_from_root() -> None:
    assert hasattr(dexterous, "Hand")


def test_finger_joint_defaults() -> None:
    joint = FingerJoint(name="j", angle=0.0)
    assert joint.min_angle == -0.1
    assert joint.max_angle == 1.5
    assert joint.link_length == 0.04


def test_finger_forward_kinematics_returns_correct_number_of_positions() -> None:
    finger = Finger(name="index", n_joints=3)
    assert len(finger.forward_kinematics()) == 4


def test_finger_forward_kinematics_positions_are_3d() -> None:
    finger = Finger(name="index", n_joints=3)
    assert all(position.shape == (3,) for position in finger.forward_kinematics())


def test_fingertip_position_returns_shape_three() -> None:
    finger = Finger(name="index", n_joints=3)
    assert finger.fingertip_position().shape == (3,)


def test_set_angles_clamps_to_joint_limits() -> None:
    finger = Finger(name="index", n_joints=3)
    finger.set_angles([10.0, -10.0, 0.5])
    assert finger.get_angles() == [1.5, -0.1, 0.5]


def test_get_angles_returns_list_of_correct_length() -> None:
    finger = Finger(name="index", n_joints=4)
    assert len(finger.get_angles()) == 4


def test_hand_initialization_with_five_fingers() -> None:
    hand = Hand()
    assert len(hand.fingers) == 5


def test_fingertip_positions_shape() -> None:
    hand = Hand()
    assert hand.fingertip_positions().shape == (5, 3)


def test_hand_open_sets_all_angles_to_zero() -> None:
    hand = Hand.closed_fist()
    hand.open()
    assert all(angle == 0.0 for finger in hand.fingers for angle in finger.get_angles())


def test_hand_close_full_sets_angles_near_max() -> None:
    hand = Hand()
    hand.close(1.0)
    assert all(np.isclose(angle, 1.5) for finger in hand.fingers for angle in finger.get_angles())


def test_hand_close_half_sets_angles_to_half_max() -> None:
    hand = Hand()
    hand.close(0.5)
    assert all(np.isclose(angle, 0.75) for finger in hand.fingers for angle in finger.get_angles())


def test_hand_presets_open_hand_returns_hand() -> None:
    assert isinstance(Hand.PRESETS.open_hand(), Hand)


def test_hand_presets_closed_fist_returns_hand() -> None:
    assert isinstance(Hand.PRESETS.closed_fist(), Hand)


def test_hand_presets_pinch_grip_returns_hand() -> None:
    assert isinstance(Hand.PRESETS.pinch_grip(), Hand)


def test_hand_presets_power_grip_returns_hand() -> None:
    assert isinstance(Hand.PRESETS.power_grip(), Hand)


def test_tactile_cell_defaults() -> None:
    cell = TactileCell(position=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))
    assert cell.pressure == 0.0
    assert cell.shear.shape == (3,)


def test_tactile_sensor_initialization_with_n_cells() -> None:
    sensor = TactileSensor(n_cells=16)
    assert isinstance(sensor, TactileSensor)


def test_tactile_sensor_cells_length_matches_n_cells() -> None:
    sensor = TactileSensor(n_cells=12)
    assert len(sensor.cells) == 12


def test_tactile_sensor_update_with_none_contact_has_no_pressure() -> None:
    sensor = TactileSensor(n_cells=8)
    sensor.update(None)
    assert np.allclose(sensor.pressure_map(), np.zeros(8))


def test_tactile_sensor_update_with_contact_sets_positive_pressure() -> None:
    sensor = TactileSensor(n_cells=16)
    sensor.update(_contact(), contact_force=1.0)
    assert np.any(sensor.pressure_map() > 0.0)


def test_tactile_sensor_in_contact_false_initially() -> None:
    sensor = TactileSensor(n_cells=16)
    assert not sensor.in_contact


def test_tactile_sensor_total_force_returns_float() -> None:
    sensor = TactileSensor()
    sensor.update(_contact(), contact_force=1.0)
    assert isinstance(sensor.total_force(), float)


def test_tactile_sensor_pressure_map_shape() -> None:
    sensor = TactileSensor(n_cells=10)
    assert sensor.pressure_map().shape == (10,)


def test_contact_centroid_returns_none_when_no_contact() -> None:
    sensor = TactileSensor()
    assert sensor.contact_centroid() is None


def test_contact_centroid_returns_shape_three_when_in_contact() -> None:
    sensor = TactileSensor()
    sensor.update(_contact(), contact_force=1.0)
    centroid = sensor.contact_centroid()
    assert centroid is not None
    assert centroid.shape == (3,)


def test_dexterous_controller_initialization() -> None:
    controller = DexterousController(Hand())
    assert controller.mode is FingerControlMode.POSITION


def test_step_returns_dict_with_expected_keys() -> None:
    controller = DexterousController(Hand())
    result = controller.step()
    assert set(result) == {"finger_positions", "fingertip_positions"}


def test_grasp_object_returns_bool() -> None:
    controller = DexterousController(Hand())
    result = controller.grasp_object(
        np.asarray([0.08, 0.0, 0.0], dtype=np.float64),
        object_radius=0.05,
    )
    assert isinstance(result, bool)


def test_release_opens_hand() -> None:
    hand = Hand.closed_fist()
    controller = DexterousController(hand)
    controller.release()
    assert all(angle == 0.0 for finger in hand.fingers for angle in finger.get_angles())


def test_dexterous_controller_has_tactile_sensors_dict() -> None:
    controller = DexterousController(Hand())
    assert isinstance(controller.tactile_sensors, dict)


def test_finger_control_mode_enum_values() -> None:
    assert [mode.value for mode in FingerControlMode] == [
        "position",
        "torque",
        "impedance",
    ]


def test_finger_command_defaults() -> None:
    command = FingerCommand()
    assert command.angles is None
    assert command.torques is None
    assert command.stiffness == 50.0
    assert command.damping == 5.0


def test_multiple_step_calls_do_not_raise() -> None:
    controller = DexterousController(Hand())
    for _ in range(5):
        controller.step(0.01)


def test_hand_finger_names_are_unique() -> None:
    hand = Hand()
    assert len({finger.name for finger in hand.fingers}) == len(hand.fingers)


def test_hand_finger_base_positions_are_different() -> None:
    hand = Hand()
    bases = [tuple(finger.base_position.tolist()) for finger in hand.fingers]
    assert len(set(bases)) == len(hand.fingers)


def test_set_all_angles_updates_each_finger() -> None:
    hand = Hand()
    hand.set_all_angles([[0.1, 0.2, 0.3]] * 5)
    assert all(finger.get_angles() == [0.1, 0.2, 0.3] for finger in hand.fingers)


def test_controller_position_command_updates_finger_angles() -> None:
    hand = Hand()
    controller = DexterousController(hand)
    controller.command({"index": FingerCommand(angles=[0.2, 0.3, 0.4])})
    controller.step()
    assert hand.fingers[1].get_angles() == [0.2, 0.3, 0.4]


def test_controller_torque_mode_changes_joint_angles() -> None:
    hand = Hand()
    controller = DexterousController(hand, mode=FingerControlMode.TORQUE)
    controller.command({"index": FingerCommand(torques=[1.0, 1.0, 1.0])})
    controller.step(dt=0.1)
    assert any(angle > 0.0 for angle in hand.fingers[1].get_angles())


def test_controller_impedance_mode_tracks_target() -> None:
    hand = Hand()
    controller = DexterousController(hand, mode=FingerControlMode.IMPEDANCE)
    controller.command(
        {"index": FingerCommand(angles=[0.5, 0.5, 0.5], stiffness=20.0, damping=0.0)}
    )
    controller.step(dt=0.1)
    assert any(angle > 0.0 for angle in hand.fingers[1].get_angles())


def test_release_clears_tactile_contact_state() -> None:
    controller = DexterousController(Hand())
    controller.tactile_sensors["index"].update(_contact(), contact_force=1.0)
    controller.release()
    assert not controller.tactile_sensors["index"].in_contact
