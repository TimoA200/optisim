import json
from pathlib import Path

import numpy as np
import pytest

from optisim.safety import (
    EmergencyStop,
    JointSafetyLimit,
    SafetyConfig,
    SafetyMonitor,
    SafetyViolation,
    SafetyZone,
    ZoneType,
)


def make_zone(
    zone_type: ZoneType = ZoneType.FORBIDDEN,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    half_extents: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> SafetyZone:
    return SafetyZone(
        name=f"{zone_type.value}_zone",
        center=np.array(center, dtype=float),
        half_extents=np.array(half_extents, dtype=float),
        zone_type=zone_type,
    )


def make_limit(joint_name: str = "elbow") -> JointSafetyLimit:
    return JointSafetyLimit(
        joint_name=joint_name,
        min_pos=-1.5,
        max_pos=1.5,
        max_velocity=2.0,
        max_acceleration=5.0,
    )


def test_zone_type_enum_values_match_expected_strings() -> None:
    assert ZoneType.FORBIDDEN.value == "forbidden"
    assert ZoneType.CAUTION.value == "caution"
    assert ZoneType.WORKSPACE.value == "workspace"


def test_safety_zone_contains_point_inside() -> None:
    zone = make_zone()
    assert zone.contains(np.array([0.2, -0.4, 0.8]))


def test_safety_zone_contains_point_on_boundary() -> None:
    zone = make_zone()
    assert zone.contains(np.array([1.0, 0.0, 0.0]))


def test_safety_zone_rejects_point_outside() -> None:
    zone = make_zone()
    assert not zone.contains(np.array([1.1, 0.0, 0.0]))


def test_safety_zone_distance_is_zero_inside() -> None:
    zone = make_zone()
    assert zone.distance_to(np.array([0.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_safety_zone_distance_is_axis_offset_outside() -> None:
    zone = make_zone()
    assert zone.distance_to(np.array([1.25, 0.0, 0.0])) == pytest.approx(0.25)


def test_safety_zone_distance_is_corner_distance_outside() -> None:
    zone = make_zone()
    assert zone.distance_to(np.array([2.0, 2.0, 1.0])) == pytest.approx(np.sqrt(2.0))


def test_safety_zone_validates_vector_shapes() -> None:
    with pytest.raises(ValueError):
        SafetyZone("bad", np.array([0.0, 0.0]), np.ones(3), ZoneType.FORBIDDEN)


def test_joint_safety_limit_rejects_invalid_range() -> None:
    with pytest.raises(ValueError):
        JointSafetyLimit("joint", 1.0, -1.0, 2.0, 3.0)


def test_safety_monitor_detects_forbidden_zone_violations() -> None:
    monitor = SafetyMonitor(zones=[make_zone(ZoneType.FORBIDDEN)])
    violations = monitor.check_positions("atlas", {"hand": np.array([0.0, 0.0, 0.0])})

    assert len(violations) == 1
    assert violations[0].violation_type == "forbidden_zone"
    assert violations[0].severity == pytest.approx(1.0)


def test_safety_monitor_detects_caution_zone_violations() -> None:
    monitor = SafetyMonitor(zones=[make_zone(ZoneType.CAUTION)])
    violations = monitor.check_positions("atlas", {"head": np.array([0.0, 0.0, 0.0])})

    assert len(violations) == 1
    assert violations[0].violation_type == "caution_zone"
    assert violations[0].severity == pytest.approx(0.5)


def test_safety_monitor_ignores_workspace_zone_membership() -> None:
    monitor = SafetyMonitor(zones=[make_zone(ZoneType.WORKSPACE)])
    violations = monitor.check_positions("atlas", {"hand": np.array([0.0, 0.0, 0.0])})
    assert violations == []


def test_safety_monitor_returns_no_position_violations_when_outside_all_zones() -> None:
    monitor = SafetyMonitor(zones=[make_zone(ZoneType.FORBIDDEN)])
    violations = monitor.check_positions("atlas", {"hand": np.array([2.0, 0.0, 0.0])})
    assert violations == []


def test_safety_monitor_detects_velocity_violations() -> None:
    monitor = SafetyMonitor(joint_limits=[make_limit("elbow")])
    violations = monitor.check_velocities("atlas", {"elbow": 2.5})

    assert len(violations) == 1
    assert violations[0].violation_type == "velocity_limit"
    assert 0.0 < violations[0].severity <= 1.0


def test_safety_monitor_detects_joint_position_violations() -> None:
    monitor = SafetyMonitor(joint_limits=[make_limit("elbow")])
    violations = monitor.check_joint_positions("atlas", {"elbow": 2.0})

    assert len(violations) == 1
    assert violations[0].violation_type == "joint_limit"
    assert violations[0].position[0] == pytest.approx(2.0)


def test_safety_monitor_ignores_velocity_within_limit() -> None:
    monitor = SafetyMonitor(joint_limits=[make_limit("elbow")])
    assert monitor.check_velocities("atlas", {"elbow": 1.5}) == []


def test_safety_monitor_detects_acceleration_violations() -> None:
    monitor = SafetyMonitor(joint_limits=[make_limit("elbow")])
    violations = monitor.check_accelerations("atlas", {"elbow": 7.0})

    assert len(violations) == 1
    assert violations[0].violation_type == "joint_limit"


def test_safety_monitor_is_safe_when_only_non_forbidden_violations_exist() -> None:
    monitor = SafetyMonitor()
    violations = [
        SafetyViolation(
            robot_name="atlas",
            joint_name="elbow",
            position=np.zeros(3),
            zone=make_zone(ZoneType.CAUTION),
            violation_type="velocity_limit",
            severity=0.3,
        )
    ]
    assert monitor.is_safe(violations)


def test_safety_monitor_is_unsafe_when_forbidden_violation_exists() -> None:
    monitor = SafetyMonitor()
    violations = [
        SafetyViolation(
            robot_name="atlas",
            joint_name="hand",
            position=np.zeros(3),
            zone=make_zone(ZoneType.FORBIDDEN),
            violation_type="forbidden_zone",
            severity=1.0,
        )
    ]
    assert not monitor.is_safe(violations)


def test_safety_monitor_summarizes_violations_readably() -> None:
    monitor = SafetyMonitor()
    violations = [
        SafetyViolation(
            robot_name="atlas",
            joint_name="hand",
            position=np.zeros(3),
            zone=make_zone(ZoneType.FORBIDDEN),
            violation_type="forbidden_zone",
            severity=1.0,
        )
    ]
    summary = monitor.summarize_violations(violations)
    assert "atlas:hand forbidden_zone" in summary
    assert "forbidden_zone" in summary


def test_emergency_stop_trigger_and_reset_cycle() -> None:
    stop = EmergencyStop()

    stop.trigger("manual")
    assert stop.triggered is True
    assert stop.reason == "manual"

    stop.reset()
    assert stop.triggered is False
    assert stop.reason == ""


def test_emergency_stop_check_and_raise_raises_for_forbidden_violations() -> None:
    stop = EmergencyStop()
    violations = [
        SafetyViolation(
            robot_name="atlas",
            joint_name="hand",
            position=np.zeros(3),
            zone=make_zone(ZoneType.FORBIDDEN),
            violation_type="forbidden_zone",
            severity=1.0,
        )
    ]

    with pytest.raises(RuntimeError, match="Emergency stop triggered"):
        stop.check_and_raise(violations)

    assert stop.triggered is True
    assert "hand" in stop.reason


def test_emergency_stop_check_and_raise_ignores_non_forbidden_violations() -> None:
    stop = EmergencyStop()
    stop.check_and_raise(
        [
            SafetyViolation(
                robot_name="atlas",
                joint_name="elbow",
                position=np.zeros(3),
                zone=make_zone(ZoneType.CAUTION),
                violation_type="velocity_limit",
                severity=0.2,
            )
        ]
    )
    assert stop.triggered is False


def test_safety_config_default_humanoid_creates_valid_defaults() -> None:
    config = SafetyConfig.default_humanoid()

    assert len(config.zones) >= 3
    assert len(config.joint_limits) >= 4
    assert any(zone.zone_type is ZoneType.FORBIDDEN for zone in config.zones)
    assert any(zone.zone_type is ZoneType.CAUTION for zone in config.zones)
    assert any(zone.zone_type is ZoneType.WORKSPACE for zone in config.zones)


def test_safety_config_from_dict_builds_objects() -> None:
    data = {
        "zones": [
            {
                "name": "ceiling",
                "center": [0.0, 0.0, 2.0],
                "half_extents": [1.0, 1.0, 0.1],
                "zone_type": "CAUTION",
            }
        ],
        "joint_limits": [
            {
                "joint_name": "neck_yaw",
                "min_pos": -1.0,
                "max_pos": 1.0,
                "max_velocity": 2.0,
                "max_acceleration": 8.0,
            }
        ],
    }

    config = SafetyConfig.from_dict(data)

    assert config.zones[0].zone_type is ZoneType.CAUTION
    assert np.allclose(config.zones[0].center, np.array([0.0, 0.0, 2.0]))
    assert config.joint_limits[0].joint_name == "neck_yaw"


def test_safety_config_from_dict_round_trip_via_serialized_mapping() -> None:
    config = SafetyConfig.default_humanoid()
    payload = {
        "zones": [
            {
                "name": zone.name,
                "center": zone.center.tolist(),
                "half_extents": zone.half_extents.tolist(),
                "zone_type": zone.zone_type.name,
            }
            for zone in config.zones
        ],
        "joint_limits": [
            {
                "joint_name": limit.joint_name,
                "min_pos": limit.min_pos,
                "max_pos": limit.max_pos,
                "max_velocity": limit.max_velocity,
                "max_acceleration": limit.max_acceleration,
            }
            for limit in config.joint_limits
        ],
    }

    restored = SafetyConfig.from_dict(json.loads(json.dumps(payload)))

    assert [zone.name for zone in restored.zones] == [zone.name for zone in config.zones]
    assert [limit.joint_name for limit in restored.joint_limits] == [
        limit.joint_name for limit in config.joint_limits
    ]


def test_safety_config_from_path_reads_json_compatible_yaml(tmp_path: Path) -> None:
    payload = {
        "zones": [
            {
                "name": "ground",
                "center": [0.0, 0.0, -0.1],
                "half_extents": [1.0, 1.0, 0.1],
                "zone_type": "forbidden",
            }
        ],
        "joint_limits": [],
    }

    path = tmp_path / "safety_config.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = SafetyConfig.from_dict(path)

    assert len(config.zones) == 1
    assert config.zones[0].zone_type is ZoneType.FORBIDDEN


def test_module_exports_public_symbols() -> None:
    from optisim import safety

    expected = {
        "EmergencyStop",
        "JointSafetyLimit",
        "SafetyConfig",
        "SafetyMonitor",
        "SafetyViolation",
        "SafetyZone",
        "ZoneType",
    }

    assert expected.issubset(set(safety.__all__))
