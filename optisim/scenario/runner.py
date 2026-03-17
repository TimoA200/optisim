"""High-level scenario execution that composes tasks, sensors, safety, and recording."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from optisim.core import TaskDefinition, ValidationReport
from optisim.safety import EmergencyStop, SafetyConfig, SafetyMonitor, SafetyViolation
from optisim.sensors import ForceTorqueSensor, IMUSensor, JointEncoderArray, ProximitySensor, SensorSuite
from optisim.sim import ExecutionEngine, SimulationRecord, SimulationRecording, WorldState


@dataclass
class ScenarioConfig:
    """Top-level configuration for a scenario run."""

    name: str
    task: TaskDefinition
    sensor_suite: SensorSuite | None = None
    safety_config: SafetyConfig | None = None
    dt: float = 0.05
    rng_seed: int = 42

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScenarioConfig":
        """Construct a scenario config from a plain mapping."""

        sensor_suite = payload.get("sensor_suite")
        if sensor_suite is None:
            sensor_suite = SensorSuite.default_humanoid_suite()
        elif not isinstance(sensor_suite, SensorSuite):
            raise TypeError("sensor_suite must be a SensorSuite instance when provided")

        safety_config = payload.get("safety_config")
        if safety_config is None:
            safety_config = SafetyConfig.default_humanoid()
        elif not isinstance(safety_config, SafetyConfig):
            safety_config = SafetyConfig.from_dict(safety_config)

        return cls(
            name=str(payload["name"]),
            task=TaskDefinition.from_dict(payload["task"]),
            sensor_suite=sensor_suite,
            safety_config=safety_config,
            dt=float(payload.get("dt", 0.05)),
            rng_seed=int(payload.get("rng_seed", 42)),
        )


@dataclass(slots=True)
class SensorReading:
    """Sensor data captured for one scenario step."""

    step: int
    time_s: float
    force_torque: dict[str, np.ndarray]
    proximity: dict[str, float]
    imu: dict[str, dict[str, np.ndarray]]
    joint_encoders: dict[str, float]


@dataclass(slots=True)
class ScenarioStepResult:
    """Result of a single scenario step."""

    step: int
    time_s: float
    action_name: str | None
    sensor_reading: SensorReading
    safety_violations: list[SafetyViolation]
    is_safe: bool
    emergency_stopped: bool


@dataclass
class ScenarioResult:
    """Result of a complete scenario run."""

    name: str
    total_steps: int
    total_time_s: float
    step_results: list[ScenarioStepResult]
    sim_record: SimulationRecord
    emergency_stop_reason: str | None = None

    @property
    def safety_violation_count(self) -> int:
        return sum(len(step.safety_violations) for step in self.step_results)

    @property
    def sensor_timeline(self) -> dict[str, list[np.ndarray]]:
        timeline: dict[str, list[np.ndarray]] = {}
        for step in self.step_results:
            for sensor_name, reading in step.sensor_reading.force_torque.items():
                timeline.setdefault(sensor_name, []).append(reading)
        return timeline

    def summary(self) -> str:
        status = "stopped" if self.emergency_stop_reason else "completed"
        summary = (
            f"Scenario '{self.name}' {status} after {self.total_steps} steps in "
            f"{self.total_time_s:.2f}s with {self.safety_violation_count} safety violation(s)."
        )
        if self.emergency_stop_reason:
            summary += f" Reason: {self.emergency_stop_reason}."
        return summary


class _ScenarioAbort(RuntimeError):
    """Internal exception used to stop scenario execution after a safety stop."""


class _ScenarioExecutionEngine(ExecutionEngine):
    """Execution engine variant that exposes a post-step callback."""

    def __init__(
        self,
        *args: Any,
        step_callback: Callable[[str | None, SimulationRecording | None, list[Any]], None] | None = None,
        **kwargs: Any,
    ) -> None:
        self._step_callback = step_callback
        super().__init__(*args, **kwargs)

    def step(
        self,
        visualize: Any | None = None,
        *,
        recording: SimulationRecording | None = None,
        active_action: str | None = None,
    ) -> list[Any]:
        collisions = super().step(visualize=visualize, recording=recording, active_action=active_action)
        if self._step_callback is not None:
            self._step_callback(active_action, recording, collisions)
        return collisions


@dataclass
class ScenarioRunner:
    """Run a task and collect scenario-level telemetry and safety outcomes."""

    config: ScenarioConfig
    _sensor_suite: SensorSuite = field(init=False, repr=False)
    _safety_monitor: SafetyMonitor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sensor_suite = self.config.sensor_suite or SensorSuite.default_humanoid_suite()
        safety_config = self.config.safety_config or SafetyConfig.default_humanoid()
        self._safety_monitor = SafetyMonitor(zones=safety_config.zones, joint_limits=safety_config.joint_limits)

    def validate(self) -> ValidationReport:
        """Validate the configured task without running it."""

        engine = self._build_engine()
        return engine.validate(self.config.task)

    def run(self) -> ScenarioResult:
        """Run the configured scenario to completion or emergency stop."""

        engine = self._build_engine()
        report = engine.validate(self.config.task)
        if not report.is_valid:
            details = "; ".join(issue.message for issue in report.errors)
            raise ValueError(f"task validation failed: {details}")

        recording = SimulationRecording.from_robot(
            engine.robot,
            task_name=self.config.task.name,
            dt=self.config.dt,
            metadata={"scenario_name": self.config.name, "rng_seed": self.config.rng_seed},
        )
        recording.capture_frame(engine.robot, engine.world, active_action=None, collisions=[])

        step_results: list[ScenarioStepResult] = []
        collisions: list[Any] = []
        executed_actions: list[str] = []
        emergency_stop = EmergencyStop()

        def capture_step(
            action_name: str | None,
            current_recording: SimulationRecording | None,
            current_collisions: list[Any],
        ) -> None:
            del current_recording
            step_number = len(step_results) + 1
            sensor_reading = self._read_sensors(engine, step_number)
            safety_violations = self._check_safety(engine)
            collisions.extend(current_collisions)
            emergency_stopped = False
            try:
                emergency_stop.check_and_raise(safety_violations)
            except RuntimeError:
                emergency_stopped = True
            step_results.append(
                ScenarioStepResult(
                    step=step_number,
                    time_s=float(engine.world.time_s),
                    action_name=action_name,
                    sensor_reading=sensor_reading,
                    safety_violations=safety_violations,
                    is_safe=self._safety_monitor.is_safe(safety_violations),
                    emergency_stopped=emergency_stopped,
                )
            )
            if emergency_stopped:
                raise _ScenarioAbort(emergency_stop.reason)

        engine._step_callback = capture_step

        random_state = np.random.get_state()
        np.random.seed(self.config.rng_seed)
        try:
            for action in self.config.task.actions:
                action_name = f"{action.action_type.value} {action.target}"
                executed_actions.append(action.action_type.value)
                frame_count_before = recording.frame_count()
                try:
                    if action.action_type.value == "grasp":
                        engine.world.objects[action.target].held_by = engine._held_by_name(action.end_effector)
                        recording.capture_frame(engine.robot, engine.world, active_action=action_name, collisions=[])
                        capture_step(action_name, recording, [])
                        continue

                    engine._execute_action(action, None, recording, action_name)
                    current_collisions = engine._check_collisions()
                    if recording.frame_count() == frame_count_before:
                        recording.capture_frame(
                            engine.robot,
                            engine.world,
                            active_action=action_name,
                            collisions=current_collisions,
                        )
                        capture_step(action_name, recording, current_collisions)
                except _ScenarioAbort:
                    break
        finally:
            np.random.set_state(random_state)

        sim_record = SimulationRecord(
            steps=len(step_results),
            duration_s=float(engine.world.time_s),
            executed_actions=executed_actions,
            collisions=collisions,
            recording=recording,
        )
        return ScenarioResult(
            name=self.config.name,
            total_steps=len(step_results),
            total_time_s=float(engine.world.time_s),
            step_results=step_results,
            sim_record=sim_record,
            emergency_stop_reason=emergency_stop.reason or None,
        )

    def _build_engine(self) -> _ScenarioExecutionEngine:
        return _ScenarioExecutionEngine(
            world=WorldState.from_dict(self.config.task.world),
            dt=self.config.dt,
        )

    def _read_sensors(self, engine: ExecutionEngine, step: int) -> SensorReading:
        joint_positions = dict(engine.robot.joint_positions)

        force_torque: dict[str, np.ndarray] = {}
        proximity: dict[str, float] = {}
        imu: dict[str, dict[str, np.ndarray]] = {}
        joint_encoders: dict[str, float] = {}

        encoder_found = False
        for sensor_name, sensor in self._sensor_suite.items():
            if isinstance(sensor, ForceTorqueSensor):
                force_torque[sensor_name] = sensor.read(np.zeros(6, dtype=float))
            elif isinstance(sensor, ProximitySensor):
                proximity[sensor_name] = sensor.read(sensor.max_range_m)
            elif isinstance(sensor, IMUSensor):
                imu[sensor_name] = sensor.read(np.zeros(3, dtype=float), np.zeros(3, dtype=float))
            elif isinstance(sensor, JointEncoderArray):
                encoder_inputs = np.array([joint_positions.get(name, 0.0) for name in sensor.joint_names], dtype=float)
                readings = sensor.read(encoder_inputs)
                joint_encoders.update(
                    {joint_name: float(readings[index]) for index, joint_name in enumerate(sensor.joint_names)}
                )
                encoder_found = True

        if not encoder_found:
            joint_encoders = {joint_name: float(value) for joint_name, value in joint_positions.items()}

        return SensorReading(
            step=step,
            time_s=float(engine.world.time_s),
            force_torque=force_torque,
            proximity=proximity,
            imu=imu,
            joint_encoders=joint_encoders,
        )

    def _check_safety(self, engine: ExecutionEngine) -> list[SafetyViolation]:
        joint_positions = {name: float(value) for name, value in engine.robot.joint_positions.items()}
        link_positions = {
            joint_name: np.array([index * 0.01, 0.0, 1.0 + position * 0.01], dtype=float)
            for index, (joint_name, position) in enumerate(joint_positions.items())
        }
        violations = self._safety_monitor.check_positions(engine.robot.name, link_positions)
        violations.extend(self._safety_monitor.check_joint_positions(engine.robot.name, joint_positions))
        return violations
