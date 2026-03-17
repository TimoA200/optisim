from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import pytest

import optisim
import optisim.faultdetect as faultdetect
from optisim import FaultCode as PublicFaultCode
from optisim import FaultEvent as PublicFaultEvent
from optisim import FaultHistory as PublicFaultHistory
from optisim import JointMonitor as PublicJointMonitor
from optisim import RobotFaultMonitor as PublicRobotFaultMonitor
from optisim import faultdetect as public_faultdetect_module
from optisim.faultdetect import FaultCode, FaultEvent, FaultHistory, JointMonitor, RobotFaultMonitor


def _monitor(**overrides: float) -> JointMonitor:
    params = {
        "joint_name": "joint1",
        "max_torque": 10.0,
        "max_velocity": 5.0,
        "min_position": -1.0,
        "max_position": 1.0,
        "stall_timeout": 0.5,
        "thermal_warning_threshold": 60.0,
        "thermal_fault_threshold": 80.0,
    }
    params.update(overrides)
    return JointMonitor(**params)


def _event(code: FaultCode, joint_name: str = "joint1", timestamp: float = 1.0) -> FaultEvent:
    return FaultEvent(joint_name=joint_name, code=code, severity=0.5, timestamp=timestamp, message=code.value)


def test_faultcode_enum_values_are_stable() -> None:
    assert FaultCode.NONE.value == "none"
    assert FaultCode.SENSOR_DROPOUT.value == "sensor_dropout"


def test_faultevent_is_frozen_dataclass() -> None:
    event = FaultEvent("joint1", FaultCode.NONE, 0.0, 1.0)

    assert is_dataclass(FaultEvent)
    with pytest.raises(FrozenInstanceError):
        event.message = "changed"  # type: ignore[misc]


def test_faultevent_defaults_message_to_empty_string() -> None:
    assert FaultEvent("joint1", FaultCode.NONE, 0.0, 1.0).message == ""


def test_faultevent_rejects_severity_below_zero() -> None:
    with pytest.raises(ValueError, match="severity"):
        FaultEvent("joint1", FaultCode.NONE, -0.1, 1.0)


def test_faultevent_rejects_severity_above_one() -> None:
    with pytest.raises(ValueError, match="severity"):
        FaultEvent("joint1", FaultCode.NONE, 1.1, 1.0)


def test_jointmonitor_rejects_non_positive_max_torque() -> None:
    with pytest.raises(ValueError, match="max_torque"):
        _monitor(max_torque=0.0)


def test_jointmonitor_rejects_non_positive_max_velocity() -> None:
    with pytest.raises(ValueError, match="max_velocity"):
        _monitor(max_velocity=0.0)


def test_jointmonitor_rejects_inverted_position_limits() -> None:
    with pytest.raises(ValueError, match="max_position"):
        _monitor(min_position=1.0, max_position=0.0)


def test_jointmonitor_rejects_negative_stall_timeout() -> None:
    with pytest.raises(ValueError, match="stall_timeout"):
        _monitor(stall_timeout=-0.1)


def test_jointmonitor_rejects_thermal_fault_threshold_below_warning() -> None:
    with pytest.raises(ValueError, match="thermal_fault_threshold"):
        _monitor(thermal_warning_threshold=70.0, thermal_fault_threshold=60.0)


def test_jointmonitor_returns_no_faults_for_nominal_state() -> None:
    monitor = _monitor()

    events = monitor.check(torque=2.0, velocity=1.0, position=0.0, temperature=40.0, timestamp=1.0)

    assert events == []
    assert monitor.last_fault is None


def test_jointmonitor_detects_torque_saturation() -> None:
    monitor = _monitor()

    events = monitor.check(torque=11.0, velocity=0.0, position=0.0, temperature=40.0, timestamp=1.0)

    assert [event.code for event in events] == [FaultCode.TORQUE_SATURATION]
    assert monitor.last_fault is events[-1]


def test_jointmonitor_detects_velocity_limit() -> None:
    events = _monitor().check(torque=0.0, velocity=6.0, position=0.0, temperature=40.0, timestamp=1.0)

    assert [event.code for event in events] == [FaultCode.VELOCITY_LIMIT]


def test_jointmonitor_detects_position_limit_below_minimum() -> None:
    events = _monitor().check(torque=0.0, velocity=0.0, position=-1.1, temperature=40.0, timestamp=1.0)

    assert [event.code for event in events] == [FaultCode.POSITION_LIMIT]


def test_jointmonitor_detects_position_limit_above_maximum() -> None:
    events = _monitor().check(torque=0.0, velocity=0.0, position=1.1, temperature=40.0, timestamp=1.0)

    assert [event.code for event in events] == [FaultCode.POSITION_LIMIT]


def test_jointmonitor_detects_thermal_warning_without_fault() -> None:
    events = _monitor().check(torque=0.0, velocity=0.0, position=0.0, temperature=65.0, timestamp=1.0)

    assert [event.code for event in events] == [FaultCode.THERMAL_WARNING]


def test_jointmonitor_detects_thermal_fault() -> None:
    events = _monitor().check(torque=0.0, velocity=0.0, position=0.0, temperature=85.0, timestamp=1.0)

    assert [event.code for event in events] == [FaultCode.THERMAL_FAULT]


def test_jointmonitor_prefers_thermal_fault_over_warning_at_fault_threshold() -> None:
    events = _monitor().check(torque=0.0, velocity=0.0, position=0.0, temperature=80.0, timestamp=1.0)

    assert [event.code for event in events] == [FaultCode.THERMAL_FAULT]


def test_jointmonitor_stall_requires_continuous_duration() -> None:
    monitor = _monitor()

    first = monitor.check(torque=2.0, velocity=0.0, position=0.0, temperature=40.0, timestamp=1.0)
    second = monitor.check(torque=2.0, velocity=0.0, position=0.0, temperature=40.0, timestamp=1.4)

    assert first == []
    assert second == []


def test_jointmonitor_detects_stall_after_timeout() -> None:
    monitor = _monitor()
    monitor.check(torque=2.0, velocity=0.0, position=0.0, temperature=40.0, timestamp=1.0)

    events = monitor.check(torque=2.0, velocity=0.0, position=0.0, temperature=40.0, timestamp=1.5)

    assert [event.code for event in events] == [FaultCode.STALL_DETECTED]


def test_jointmonitor_stall_resets_when_velocity_recovers() -> None:
    monitor = _monitor()
    monitor.check(torque=2.0, velocity=0.0, position=0.0, temperature=40.0, timestamp=1.0)
    monitor.check(torque=2.0, velocity=0.02, position=0.0, temperature=40.0, timestamp=1.3)

    events = monitor.check(torque=2.0, velocity=0.0, position=0.0, temperature=40.0, timestamp=1.7)

    assert events == []


def test_jointmonitor_clear_stall_resets_internal_tracking() -> None:
    monitor = _monitor()
    monitor.check(torque=2.0, velocity=0.0, position=0.0, temperature=40.0, timestamp=1.0)
    monitor.clear_stall()

    events = monitor.check(torque=2.0, velocity=0.0, position=0.0, temperature=40.0, timestamp=1.4)

    assert events == []


def test_jointmonitor_can_emit_multiple_faults_in_one_check() -> None:
    events = _monitor().check(torque=12.0, velocity=6.0, position=1.2, temperature=85.0, timestamp=1.0)

    assert [event.code for event in events] == [
        FaultCode.TORQUE_SATURATION,
        FaultCode.VELOCITY_LIMIT,
        FaultCode.POSITION_LIMIT,
        FaultCode.THERMAL_FAULT,
    ]


def test_jointmonitor_last_fault_tracks_last_emitted_event() -> None:
    monitor = _monitor()

    events = monitor.check(torque=12.0, velocity=6.0, position=0.0, temperature=40.0, timestamp=1.0)

    assert monitor.last_fault is events[-1]
    assert monitor.last_fault.code is FaultCode.VELOCITY_LIMIT


def test_jointmonitor_fault_severities_stay_within_unit_interval() -> None:
    events = _monitor().check(torque=20.0, velocity=8.0, position=2.0, temperature=200.0, timestamp=1.0)

    assert events
    assert all(0.0 <= event.severity <= 1.0 for event in events)


def test_robotfaultmonitor_aggregates_faults_across_joints() -> None:
    robot = RobotFaultMonitor({"joint1": _monitor(), "joint2": _monitor(joint_name="joint2")})

    events = robot.check_all(
        {
            "joint1": {"torque": 11.0, "velocity": 0.0, "position": 0.0, "temperature": 40.0, "timestamp": 1.0},
            "joint2": {"torque": 0.0, "velocity": 6.0, "position": 0.0, "temperature": 40.0, "timestamp": 1.0},
        }
    )

    assert [(event.joint_name, event.code) for event in events] == [
        ("joint1", FaultCode.TORQUE_SATURATION),
        ("joint2", FaultCode.VELOCITY_LIMIT),
    ]
    assert robot.active_faults == events


def test_robotfaultmonitor_records_sensor_dropout_for_missing_joint_state() -> None:
    robot = RobotFaultMonitor({"joint1": _monitor()})

    events = robot.check_all({})

    assert [(event.joint_name, event.code) for event in events] == [("joint1", FaultCode.SENSOR_DROPOUT)]


def test_robotfaultmonitor_records_sensor_dropout_for_missing_state_keys() -> None:
    robot = RobotFaultMonitor({"joint1": _monitor()})

    events = robot.check_all({"joint1": {"torque": 0.0, "velocity": 0.0, "position": 0.0, "timestamp": 1.0}})

    assert [event.code for event in events] == [FaultCode.SENSOR_DROPOUT]


def test_robotfaultmonitor_records_sensor_dropout_for_non_finite_state_values() -> None:
    robot = RobotFaultMonitor({"joint1": _monitor()})

    events = robot.check_all(
        {"joint1": {"torque": 0.0, "velocity": float("nan"), "position": 0.0, "temperature": 40.0, "timestamp": 1.0}}
    )

    assert [event.code for event in events] == [FaultCode.SENSOR_DROPOUT]


def test_robotfaultmonitor_has_fault_without_code_checks_any_active_fault() -> None:
    robot = RobotFaultMonitor({"joint1": _monitor()})
    robot.check_all(
        {"joint1": {"torque": 11.0, "velocity": 0.0, "position": 0.0, "temperature": 40.0, "timestamp": 1.0}}
    )

    assert robot.has_fault() is True


def test_robotfaultmonitor_has_fault_with_code_filters_active_faults() -> None:
    robot = RobotFaultMonitor({"joint1": _monitor()})
    robot.check_all(
        {"joint1": {"torque": 11.0, "velocity": 0.0, "position": 0.0, "temperature": 40.0, "timestamp": 1.0}}
    )

    assert robot.has_fault(FaultCode.TORQUE_SATURATION) is True
    assert robot.has_fault(FaultCode.THERMAL_FAULT) is False


def test_robotfaultmonitor_has_fault_returns_false_when_no_faults_active() -> None:
    robot = RobotFaultMonitor({"joint1": _monitor()})
    robot.check_all(
        {"joint1": {"torque": 1.0, "velocity": 0.1, "position": 0.0, "temperature": 40.0, "timestamp": 1.0}}
    )

    assert robot.has_fault() is False


def test_robotfaultmonitor_active_faults_replace_previous_cycle() -> None:
    robot = RobotFaultMonitor({"joint1": _monitor()})
    robot.check_all(
        {"joint1": {"torque": 11.0, "velocity": 0.0, "position": 0.0, "temperature": 40.0, "timestamp": 1.0}}
    )

    events = robot.check_all(
        {"joint1": {"torque": 1.0, "velocity": 0.1, "position": 0.0, "temperature": 40.0, "timestamp": 2.0}}
    )

    assert events == []
    assert robot.active_faults == []


def test_robotfaultmonitor_clear_all_clears_faults_and_joint_state() -> None:
    joint = _monitor()
    robot = RobotFaultMonitor({"joint1": joint})
    joint.check(torque=2.0, velocity=0.0, position=0.0, temperature=40.0, timestamp=1.0)
    robot.check_all(
        {"joint1": {"torque": 11.0, "velocity": 0.0, "position": 0.0, "temperature": 40.0, "timestamp": 2.0}}
    )

    robot.clear_all()

    assert robot.active_faults == []
    assert joint.last_fault is None
    assert joint.check(torque=2.0, velocity=0.0, position=0.0, temperature=40.0, timestamp=2.4) == []


def test_faulthistory_rejects_negative_capacity() -> None:
    with pytest.raises(ValueError, match="max_events"):
        FaultHistory(max_events=-1)


def test_faulthistory_record_limits_stored_events_to_capacity() -> None:
    history = FaultHistory(max_events=2)
    history.record(_event(FaultCode.TORQUE_SATURATION, timestamp=1.0))
    history.record(_event(FaultCode.VELOCITY_LIMIT, timestamp=2.0))
    history.record(_event(FaultCode.POSITION_LIMIT, timestamp=3.0))

    assert [event.code for event in history.since(0.0)] == [FaultCode.VELOCITY_LIMIT, FaultCode.POSITION_LIMIT]


def test_faulthistory_get_by_joint_filters_events() -> None:
    history = FaultHistory()
    history.record(_event(FaultCode.TORQUE_SATURATION, joint_name="joint1"))
    history.record(_event(FaultCode.VELOCITY_LIMIT, joint_name="joint2"))

    assert [event.code for event in history.get_by_joint("joint2")] == [FaultCode.VELOCITY_LIMIT]


def test_faulthistory_get_by_code_filters_events() -> None:
    history = FaultHistory()
    history.record(_event(FaultCode.TORQUE_SATURATION))
    history.record(_event(FaultCode.VELOCITY_LIMIT))
    history.record(_event(FaultCode.TORQUE_SATURATION, timestamp=2.0))

    assert [event.timestamp for event in history.get_by_code(FaultCode.TORQUE_SATURATION)] == [1.0, 2.0]


def test_faulthistory_since_filters_by_timestamp() -> None:
    history = FaultHistory()
    history.record(_event(FaultCode.TORQUE_SATURATION, timestamp=1.0))
    history.record(_event(FaultCode.VELOCITY_LIMIT, timestamp=2.0))

    assert [event.code for event in history.since(2.0)] == [FaultCode.VELOCITY_LIMIT]


def test_faulthistory_count_by_code_returns_frequency_map() -> None:
    history = FaultHistory()
    history.record(_event(FaultCode.TORQUE_SATURATION))
    history.record(_event(FaultCode.TORQUE_SATURATION, timestamp=2.0))
    history.record(_event(FaultCode.VELOCITY_LIMIT, timestamp=3.0))

    assert history.count_by_code() == {FaultCode.TORQUE_SATURATION: 2, FaultCode.VELOCITY_LIMIT: 1}


def test_faulthistory_clear_removes_all_events() -> None:
    history = FaultHistory()
    history.record(_event(FaultCode.TORQUE_SATURATION))

    history.clear()

    assert history.since(0.0) == []


def test_faultdetect_module_exports_public_surface() -> None:
    assert faultdetect.__all__ == ["FaultCode", "FaultEvent", "JointMonitor", "RobotFaultMonitor", "FaultHistory"]
    assert faultdetect.FaultCode is FaultCode
    assert faultdetect.FaultEvent is FaultEvent
    assert faultdetect.JointMonitor is JointMonitor
    assert faultdetect.RobotFaultMonitor is RobotFaultMonitor
    assert faultdetect.FaultHistory is FaultHistory


def test_top_level_optisim_exports_faultdetect_module_and_classes() -> None:
    assert public_faultdetect_module is faultdetect
    assert PublicFaultCode is FaultCode
    assert PublicFaultEvent is FaultEvent
    assert PublicJointMonitor is JointMonitor
    assert PublicRobotFaultMonitor is RobotFaultMonitor
    assert PublicFaultHistory is FaultHistory
    assert optisim.faultdetect is faultdetect
    assert optisim.__version__ == "0.27.0"
