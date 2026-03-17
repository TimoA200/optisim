"""Safety monitoring and enforcement primitives for humanoid task execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


class ZoneType(Enum):
    """Classification for a safety zone."""

    FORBIDDEN = "forbidden"
    CAUTION = "caution"
    WORKSPACE = "workspace"


@dataclass(slots=True)
class SafetyZone:
    """Axis-aligned 3D safety zone in world space."""

    name: str
    center: np.ndarray
    half_extents: np.ndarray
    zone_type: ZoneType

    def __post_init__(self) -> None:
        self.center = _vector3(self.center, "center")
        self.half_extents = _vector3(self.half_extents, "half_extents")
        if np.any(self.half_extents < 0.0):
            raise ValueError("half_extents must be non-negative")

    def contains(self, point: np.ndarray) -> bool:
        """Return True when the point lies inside or on the zone boundary."""

        point_array = _vector3(point, "point")
        offsets = np.abs(point_array - self.center)
        return bool(np.all(offsets <= self.half_extents))

    def distance_to(self, point: np.ndarray) -> float:
        """Return 0 if inside the zone, else the minimum distance to its boundary."""

        point_array = _vector3(point, "point")
        outside = np.abs(point_array - self.center) - self.half_extents
        clamped = np.maximum(outside, 0.0)
        return float(np.linalg.norm(clamped))


@dataclass(slots=True)
class SafetyViolation:
    """Description of a safety-relevant violation."""

    robot_name: str
    joint_name: str
    position: np.ndarray
    zone: SafetyZone
    violation_type: str
    severity: float

    def __post_init__(self) -> None:
        self.position = _vector3(self.position, "position")
        self.severity = float(np.clip(self.severity, 0.0, 1.0))


@dataclass(slots=True)
class JointSafetyLimit:
    """Position, velocity, and acceleration limits for a joint."""

    joint_name: str
    min_pos: float
    max_pos: float
    max_velocity: float
    max_acceleration: float

    def __post_init__(self) -> None:
        if self.min_pos > self.max_pos:
            raise ValueError("min_pos must be <= max_pos")
        if self.max_velocity < 0.0:
            raise ValueError("max_velocity must be non-negative")
        if self.max_acceleration < 0.0:
            raise ValueError("max_acceleration must be non-negative")


class SafetyMonitor:
    """Monitor zones and joint dynamic limits for a robot."""

    def __init__(
        self,
        zones: Iterable[SafetyZone] | None = None,
        joint_limits: Iterable[JointSafetyLimit] | None = None,
    ) -> None:
        self.zones: list[SafetyZone] = list(zones or [])
        self.joint_limits: dict[str, JointSafetyLimit] = {
            limit.joint_name: limit for limit in (joint_limits or [])
        }

    def add_zone(self, zone: SafetyZone) -> None:
        self.zones.append(zone)

    def add_joint_limit(self, limit: JointSafetyLimit) -> None:
        self.joint_limits[limit.joint_name] = limit

    def check_positions(
        self,
        robot_name: str,
        link_positions: Mapping[str, np.ndarray],
    ) -> list[SafetyViolation]:
        violations: list[SafetyViolation] = []
        for joint_name, position in link_positions.items():
            position_array = _vector3(position, "position")
            for zone in self.zones:
                if not zone.contains(position_array):
                    continue
                violation_type = {
                    ZoneType.FORBIDDEN: "forbidden_zone",
                    ZoneType.CAUTION: "caution_zone",
                }.get(zone.zone_type)
                if violation_type is None:
                    continue
                violations.append(
                    SafetyViolation(
                        robot_name=robot_name,
                        joint_name=joint_name,
                        position=position_array,
                        zone=zone,
                        violation_type=violation_type,
                        severity=self._zone_severity(zone),
                    )
                )
        return violations

    def check_velocities(
        self,
        robot_name: str,
        joint_velocities: Mapping[str, float],
    ) -> list[SafetyViolation]:
        violations: list[SafetyViolation] = []
        for joint_name, velocity in joint_velocities.items():
            limit = self.joint_limits.get(joint_name)
            if limit is None or abs(float(velocity)) <= limit.max_velocity:
                continue
            violations.append(
                SafetyViolation(
                    robot_name=robot_name,
                    joint_name=joint_name,
                    position=np.zeros(3, dtype=float),
                    zone=_limit_zone(f"{joint_name}_velocity_limit"),
                    violation_type="velocity_limit",
                    severity=_ratio_severity(abs(float(velocity)), limit.max_velocity),
                )
            )
        return violations

    def check_joint_positions(
        self,
        robot_name: str,
        joint_positions: Mapping[str, float],
    ) -> list[SafetyViolation]:
        violations: list[SafetyViolation] = []
        for joint_name, position in joint_positions.items():
            limit = self.joint_limits.get(joint_name)
            if limit is None:
                continue
            position_value = float(position)
            if limit.min_pos <= position_value <= limit.max_pos:
                continue
            span = max(limit.max_pos - limit.min_pos, 1e-9)
            overage = max(limit.min_pos - position_value, position_value - limit.max_pos)
            violations.append(
                SafetyViolation(
                    robot_name=robot_name,
                    joint_name=joint_name,
                    position=np.array([position_value, 0.0, 0.0], dtype=float),
                    zone=_limit_zone(f"{joint_name}_position_limit"),
                    violation_type="joint_limit",
                    severity=float(np.clip(overage / span, 0.0, 1.0)),
                )
            )
        return violations

    def check_accelerations(
        self,
        robot_name: str,
        joint_accelerations: Mapping[str, float],
    ) -> list[SafetyViolation]:
        violations: list[SafetyViolation] = []
        for joint_name, acceleration in joint_accelerations.items():
            limit = self.joint_limits.get(joint_name)
            if limit is None or abs(float(acceleration)) <= limit.max_acceleration:
                continue
            violations.append(
                SafetyViolation(
                    robot_name=robot_name,
                    joint_name=joint_name,
                    position=np.zeros(3, dtype=float),
                    zone=_limit_zone(f"{joint_name}_acceleration_limit"),
                    violation_type="joint_limit",
                    severity=_ratio_severity(abs(float(acceleration)), limit.max_acceleration),
                )
            )
        return violations

    def is_safe(self, violations: Iterable[SafetyViolation]) -> bool:
        return not any(violation.violation_type == "forbidden_zone" for violation in violations)

    def summarize_violations(self, violations: Iterable[SafetyViolation]) -> str:
        entries = list(violations)
        if not entries:
            return "No safety violations detected."
        return "; ".join(
            f"{violation.robot_name}:{violation.joint_name} {violation.violation_type} in {violation.zone.name} "
            f"(severity={violation.severity:.2f})"
            for violation in entries
        )

    @staticmethod
    def _zone_severity(zone: SafetyZone) -> float:
        if zone.zone_type is ZoneType.FORBIDDEN:
            return 1.0
        if zone.zone_type is ZoneType.CAUTION:
            return 0.5
        return 0.0


class EmergencyStop:
    """Track and enforce emergency stop state."""

    def __init__(self) -> None:
        self._triggered = False
        self._reason = ""

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def reason(self) -> str:
        return self._reason

    def trigger(self, reason: str) -> None:
        self._triggered = True
        self._reason = reason

    def reset(self) -> None:
        self._triggered = False
        self._reason = ""

    def check_and_raise(self, violations: Iterable[SafetyViolation]) -> None:
        forbidden = [violation for violation in violations if violation.violation_type == "forbidden_zone"]
        if not forbidden:
            return
        summary = "; ".join(f"{violation.joint_name} in {violation.zone.name}" for violation in forbidden)
        self.trigger(summary)
        raise RuntimeError(f"Emergency stop triggered: {summary}")


@dataclass(slots=True)
class SafetyConfig:
    """Top-level safety configuration."""

    zones: list[SafetyZone] = field(default_factory=list)
    joint_limits: list[JointSafetyLimit] = field(default_factory=list)

    @classmethod
    def default_humanoid(cls) -> SafetyConfig:
        zones = [
            SafetyZone(
                name="ground_exclusion",
                center=np.array([0.0, 0.0, -0.15], dtype=float),
                half_extents=np.array([2.0, 2.0, 0.15], dtype=float),
                zone_type=ZoneType.FORBIDDEN,
            ),
            SafetyZone(
                name="head_clearance",
                center=np.array([0.0, 0.0, 1.9], dtype=float),
                half_extents=np.array([0.5, 0.5, 0.2], dtype=float),
                zone_type=ZoneType.CAUTION,
            ),
            SafetyZone(
                name="standing_workspace",
                center=np.array([0.6, 0.0, 1.0], dtype=float),
                half_extents=np.array([1.0, 0.9, 1.1], dtype=float),
                zone_type=ZoneType.WORKSPACE,
            ),
        ]
        joint_limits = [
            JointSafetyLimit("neck_yaw", -1.0, 1.0, 2.0, 10.0),
            JointSafetyLimit("torso_pitch", -0.6, 0.6, 1.5, 8.0),
            JointSafetyLimit("left_knee_pitch", 0.0, 2.4, 3.0, 12.0),
            JointSafetyLimit("right_knee_pitch", 0.0, 2.4, 3.0, 12.0),
        ]
        return cls(zones=zones, joint_limits=joint_limits)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | str | Path) -> SafetyConfig:
        if isinstance(data, (str, Path)):
            content = Path(data).read_text(encoding="utf-8")
            parsed = _parse_simple_yaml(content)
        else:
            parsed = dict(data)

        zones_data = parsed.get("zones", [])
        joint_limits_data = parsed.get("joint_limits", [])
        zones = [
            SafetyZone(
                name=str(item["name"]),
                center=np.asarray(item["center"], dtype=float),
                half_extents=np.asarray(item["half_extents"], dtype=float),
                zone_type=_coerce_zone_type(item["zone_type"]),
            )
            for item in zones_data
        ]
        joint_limits = [
            JointSafetyLimit(
                joint_name=str(item["joint_name"]),
                min_pos=float(item["min_pos"]),
                max_pos=float(item["max_pos"]),
                max_velocity=float(item["max_velocity"]),
                max_acceleration=float(item["max_acceleration"]),
            )
            for item in joint_limits_data
        ]
        return cls(zones=zones, joint_limits=joint_limits)


def _vector3(value: np.ndarray, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{field_name} must be a 3D vector")
    return array


def _coerce_zone_type(value: ZoneType | str) -> ZoneType:
    if isinstance(value, ZoneType):
        return value
    normalized = str(value).strip().lower()
    for member in ZoneType:
        if normalized in {member.name.lower(), member.value.lower()}:
            return member
    raise ValueError(f"Unknown zone_type: {value}")


def _ratio_severity(measured: float, limit: float) -> float:
    if limit <= 0.0:
        return 1.0
    return float(np.clip((measured - limit) / limit, 0.0, 1.0))


def _limit_zone(name: str) -> SafetyZone:
    return SafetyZone(
        name=name,
        center=np.zeros(3, dtype=float),
        half_extents=np.zeros(3, dtype=float),
        zone_type=ZoneType.CAUTION,
    )


def _parse_simple_yaml(content: str) -> Mapping[str, Any]:
    try:
        import json

        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "SafetyConfig.from_dict only supports mappings or JSON-compatible YAML strings without external dependencies"
        ) from exc
