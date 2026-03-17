"""Real-time robot fault detection and health monitoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class FaultCode(Enum):
    """Enumerates supported robot health and fault states."""

    NONE = "none"
    TORQUE_SATURATION = "torque_saturation"
    VELOCITY_LIMIT = "velocity_limit"
    POSITION_LIMIT = "position_limit"
    THERMAL_WARNING = "thermal_warning"
    THERMAL_FAULT = "thermal_fault"
    STALL_DETECTED = "stall_detected"
    SENSOR_DROPOUT = "sensor_dropout"


@dataclass(frozen=True, slots=True)
class FaultEvent:
    """Represents a single detected fault condition."""

    joint_name: str
    code: FaultCode
    severity: float
    timestamp: float
    message: str = ""

    def __post_init__(self) -> None:
        severity = float(self.severity)
        timestamp = float(self.timestamp)
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be within [0.0, 1.0]")
        object.__setattr__(self, "joint_name", str(self.joint_name))
        object.__setattr__(self, "code", FaultCode(self.code))
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "message", str(self.message))


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalized_excess(value: float, limit: float) -> float:
    safe_limit = max(abs(float(limit)), 1e-9)
    return _clamp_unit((abs(float(value)) - safe_limit) / safe_limit)


class JointMonitor:
    """Monitors a single joint for common runtime fault conditions."""

    def __init__(
        self,
        joint_name: str,
        max_torque: float,
        max_velocity: float,
        min_position: float,
        max_position: float,
        stall_timeout: float = 0.5,
        thermal_warning_threshold: float = 60.0,
        thermal_fault_threshold: float = 80.0,
    ) -> None:
        self.joint_name = str(joint_name)
        self.max_torque = float(max_torque)
        self.max_velocity = float(max_velocity)
        self.min_position = float(min_position)
        self.max_position = float(max_position)
        self.stall_timeout = float(stall_timeout)
        self.thermal_warning_threshold = float(thermal_warning_threshold)
        self.thermal_fault_threshold = float(thermal_fault_threshold)
        if self.max_torque <= 0.0:
            raise ValueError("max_torque must be positive")
        if self.max_velocity <= 0.0:
            raise ValueError("max_velocity must be positive")
        if self.max_position < self.min_position:
            raise ValueError("max_position must be greater than or equal to min_position")
        if self.stall_timeout < 0.0:
            raise ValueError("stall_timeout must be non-negative")
        if self.thermal_fault_threshold < self.thermal_warning_threshold:
            raise ValueError("thermal_fault_threshold must be >= thermal_warning_threshold")
        self.last_fault: FaultEvent | None = None
        self._stall_started_at: float | None = None

    def check(
        self,
        torque: float,
        velocity: float,
        position: float,
        temperature: float,
        timestamp: float,
    ) -> list[FaultEvent]:
        """Return the active fault conditions for the current joint state."""

        torque = float(torque)
        velocity = float(velocity)
        position = float(position)
        temperature = float(temperature)
        timestamp = float(timestamp)

        events: list[FaultEvent] = []

        if abs(torque) > self.max_torque:
            events.append(
                FaultEvent(
                    joint_name=self.joint_name,
                    code=FaultCode.TORQUE_SATURATION,
                    severity=max(0.1, _normalized_excess(torque, self.max_torque)),
                    timestamp=timestamp,
                    message=f"torque exceeded limit of {self.max_torque:g}",
                )
            )

        if abs(velocity) > self.max_velocity:
            events.append(
                FaultEvent(
                    joint_name=self.joint_name,
                    code=FaultCode.VELOCITY_LIMIT,
                    severity=max(0.1, _normalized_excess(velocity, self.max_velocity)),
                    timestamp=timestamp,
                    message=f"velocity exceeded limit of {self.max_velocity:g}",
                )
            )

        if position < self.min_position or position > self.max_position:
            position_range = max(self.max_position - self.min_position, 1e-9)
            overflow = self.min_position - position if position < self.min_position else position - self.max_position
            events.append(
                FaultEvent(
                    joint_name=self.joint_name,
                    code=FaultCode.POSITION_LIMIT,
                    severity=max(0.1, _clamp_unit(overflow / position_range)),
                    timestamp=timestamp,
                    message=f"position outside range [{self.min_position:g}, {self.max_position:g}]",
                )
            )

        if temperature >= self.thermal_fault_threshold:
            margin = max(self.thermal_fault_threshold, 1.0)
            events.append(
                FaultEvent(
                    joint_name=self.joint_name,
                    code=FaultCode.THERMAL_FAULT,
                    severity=max(0.5, _clamp_unit((temperature - self.thermal_fault_threshold) / margin)),
                    timestamp=timestamp,
                    message=f"temperature exceeded fault threshold of {self.thermal_fault_threshold:g} C",
                )
            )
        elif temperature >= self.thermal_warning_threshold:
            band = max(self.thermal_fault_threshold - self.thermal_warning_threshold, 1.0)
            events.append(
                FaultEvent(
                    joint_name=self.joint_name,
                    code=FaultCode.THERMAL_WARNING,
                    severity=max(0.1, min(0.99, (temperature - self.thermal_warning_threshold) / band)),
                    timestamp=timestamp,
                    message=f"temperature exceeded warning threshold of {self.thermal_warning_threshold:g} C",
                )
            )

        if abs(velocity) < 0.01 and abs(torque) > 0.1 * self.max_torque:
            if self._stall_started_at is None:
                self._stall_started_at = timestamp
            if timestamp - self._stall_started_at >= self.stall_timeout:
                duration = max(timestamp - self._stall_started_at, self.stall_timeout)
                timeout = max(self.stall_timeout, 1e-9)
                events.append(
                    FaultEvent(
                        joint_name=self.joint_name,
                        code=FaultCode.STALL_DETECTED,
                        severity=_clamp_unit(duration / timeout),
                        timestamp=timestamp,
                        message=f"stall persisted for {duration:.3f}s",
                    )
                )
            else:
                self.last_fault = events[-1] if events else None
                return events
        else:
            self._stall_started_at = None

        self.last_fault = events[-1] if events else None
        return events

    def clear_stall(self) -> None:
        """Reset accumulated stall tracking for the joint."""

        self._stall_started_at = None


class RobotFaultMonitor:
    """Runs joint-level monitoring over a full robot state dictionary."""

    _REQUIRED_STATE_KEYS = ("torque", "velocity", "position", "temperature", "timestamp")

    def __init__(self, joint_monitors: dict[str, JointMonitor]) -> None:
        self.joint_monitors = dict(joint_monitors)
        self.active_faults: list[FaultEvent] = []

    def check_all(self, states: dict[str, dict]) -> list[FaultEvent]:
        """Return all currently active non-NONE faults for the provided joint states."""

        active_faults: list[FaultEvent] = []
        for joint_name, monitor in self.joint_monitors.items():
            state = states.get(joint_name)
            if state is None:
                active_faults.append(
                    FaultEvent(
                        joint_name=joint_name,
                        code=FaultCode.SENSOR_DROPOUT,
                        severity=1.0,
                        timestamp=0.0,
                        message="joint state missing",
                    )
                )
                monitor.last_fault = active_faults[-1]
                continue

            missing_keys = [key for key in self._REQUIRED_STATE_KEYS if key not in state]
            if missing_keys:
                timestamp = float(state.get("timestamp", 0.0))
                active_faults.append(
                    FaultEvent(
                        joint_name=joint_name,
                        code=FaultCode.SENSOR_DROPOUT,
                        severity=1.0,
                        timestamp=timestamp,
                        message=f"missing state keys: {', '.join(missing_keys)}",
                    )
                )
                monitor.last_fault = active_faults[-1]
                continue

            values = {key: state[key] for key in self._REQUIRED_STATE_KEYS}
            if any(not self._is_finite_number(value) for value in values.values()):
                timestamp = float(values["timestamp"]) if self._is_finite_number(values["timestamp"]) else 0.0
                active_faults.append(
                    FaultEvent(
                        joint_name=joint_name,
                        code=FaultCode.SENSOR_DROPOUT,
                        severity=1.0,
                        timestamp=timestamp,
                        message="state contains non-finite or missing numeric values",
                    )
                )
                monitor.last_fault = active_faults[-1]
                continue

            active_faults.extend(
                monitor.check(
                    torque=float(values["torque"]),
                    velocity=float(values["velocity"]),
                    position=float(values["position"]),
                    temperature=float(values["temperature"]),
                    timestamp=float(values["timestamp"]),
                )
            )

        self.active_faults = [event for event in active_faults if event.code is not FaultCode.NONE]
        return list(self.active_faults)

    def has_fault(self, code: FaultCode | None = None) -> bool:
        """Return whether any active fault exists, optionally filtered by code."""

        if code is None:
            return bool(self.active_faults)
        return any(event.code is code for event in self.active_faults)

    def clear_all(self) -> None:
        """Clear current active faults and reset joint stall tracking."""

        for monitor in self.joint_monitors.values():
            monitor.clear_stall()
            monitor.last_fault = None
        self.active_faults = []

    @staticmethod
    def _is_finite_number(value: object) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(float(value))


class FaultHistory:
    """Bounded storage and query helpers for fault event history."""

    def __init__(self, max_events: int = 500) -> None:
        if max_events < 0:
            raise ValueError("max_events must be non-negative")
        self.max_events = int(max_events)
        self._events: list[FaultEvent] = []

    def record(self, event: FaultEvent) -> None:
        """Append a fault event and enforce the configured history cap."""

        self._events.append(event)
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]

    def get_by_joint(self, joint_name: str) -> list[FaultEvent]:
        """Return all recorded events for a given joint."""

        return [event for event in self._events if event.joint_name == str(joint_name)]

    def get_by_code(self, code: FaultCode) -> list[FaultEvent]:
        """Return all recorded events matching the given fault code."""

        return [event for event in self._events if event.code is code]

    def since(self, timestamp: float) -> list[FaultEvent]:
        """Return all recorded events with timestamp >= ``timestamp``."""

        threshold = float(timestamp)
        return [event for event in self._events if event.timestamp >= threshold]

    def count_by_code(self) -> dict[FaultCode, int]:
        """Return per-fault counts for the current history."""

        counts: dict[FaultCode, int] = {}
        for event in self._events:
            counts[event.code] = counts.get(event.code, 0) + 1
        return counts

    def clear(self) -> None:
        """Remove all stored fault events."""

        self._events = []
