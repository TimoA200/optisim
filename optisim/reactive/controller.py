"""Reactive contact-aware control primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Callable

import numpy as np

from optisim.robot.model import RobotModel
from optisim.scenario import SensorReading
from optisim.sensors import SensorSuite
from optisim.sim.world import WorldState
from optisim.wbc.controller import WBCController


@dataclass(slots=True)
class ReactiveConfig:
    """Configuration for reactive manipulation control."""

    control_freq_hz: float = 50.0
    safety_stop_on_ft_threshold: float = 50.0
    proximity_slowdown_distance: float = 0.15
    proximity_stop_distance: float = 0.05
    max_step_count: int = 500
    verbose: bool = False


class ContactPhase(Enum):
    """Manipulation phases for contact-rich tasks."""

    PRE_APPROACH = "pre_approach"
    APPROACH = "approach"
    CONTACT = "contact"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    RETRACT = "retract"
    DONE = "done"


@dataclass
class ReactiveState:
    """Current FSM state and derived reactive control signals."""

    phase: ContactPhase
    step: int
    ft_reading: float
    proximity_reading: float
    velocity_scale: float
    triggered_safety_stop: bool
    phase_history: list[ContactPhase] = field(default_factory=list)


class ManipulationFSM:
    """Finite state machine for contact-rich manipulation."""

    def __init__(self, config: ReactiveConfig) -> None:
        self.config = config

    def reset(self) -> ReactiveState:
        return ReactiveState(
            phase=ContactPhase.PRE_APPROACH,
            step=0,
            ft_reading=0.0,
            proximity_reading=float("inf"),
            velocity_scale=1.0,
            triggered_safety_stop=False,
            phase_history=[ContactPhase.PRE_APPROACH],
        )

    def transition(self, state: ReactiveState, sensor_data: dict[str, Any]) -> ContactPhase:
        ft_magnitude = float(sensor_data.get("ft_magnitude", 0.0))
        proximity_min = float(sensor_data.get("proximity_min", float("inf")))
        gripper_closed = bool(sensor_data.get("gripper_closed", False))
        object_held = bool(sensor_data.get("object_held", False))
        at_target = bool(sensor_data.get("at_target", False))

        if state.phase is ContactPhase.PRE_APPROACH:
            return ContactPhase.APPROACH if not at_target else ContactPhase.PRE_APPROACH
        if state.phase is ContactPhase.APPROACH:
            if ft_magnitude > 1.0:
                return ContactPhase.CONTACT
            if proximity_min > self.config.proximity_stop_distance:
                return ContactPhase.APPROACH
            return ContactPhase.APPROACH
        if state.phase is ContactPhase.CONTACT:
            return ContactPhase.GRASP if gripper_closed else ContactPhase.CONTACT
        if state.phase is ContactPhase.GRASP:
            return ContactPhase.LIFT if object_held else ContactPhase.GRASP
        if state.phase is ContactPhase.LIFT:
            if ft_magnitude < self.config.safety_stop_on_ft_threshold and object_held:
                return ContactPhase.TRANSPORT
            return ContactPhase.LIFT
        if state.phase is ContactPhase.TRANSPORT:
            return ContactPhase.PLACE if at_target else ContactPhase.TRANSPORT
        if state.phase is ContactPhase.PLACE:
            return ContactPhase.RETRACT if not object_held else ContactPhase.PLACE
        if state.phase is ContactPhase.RETRACT:
            retract_steps = sum(phase is ContactPhase.RETRACT for phase in state.phase_history)
            return ContactPhase.DONE if retract_steps >= 10 else ContactPhase.RETRACT
        return ContactPhase.DONE

    def step(self, state: ReactiveState, sensor_data: dict[str, Any]) -> ReactiveState:
        ft_magnitude = float(sensor_data.get("ft_magnitude", 0.0))
        proximity_min = float(sensor_data.get("proximity_min", float("inf")))
        next_phase = self.transition(state, sensor_data)
        triggered_safety_stop = state.triggered_safety_stop or (
            ft_magnitude >= self.config.safety_stop_on_ft_threshold
        )

        velocity_scale = 1.0
        if triggered_safety_stop:
            velocity_scale = 0.0
        elif isfinite(proximity_min) and proximity_min <= self.config.proximity_stop_distance:
            velocity_scale = 0.0
        elif isfinite(proximity_min) and proximity_min <= self.config.proximity_slowdown_distance:
            velocity_scale = 0.3

        phase_history = list(state.phase_history)
        phase_history.append(next_phase)
        return ReactiveState(
            phase=next_phase,
            step=state.step + 1,
            ft_reading=ft_magnitude,
            proximity_reading=proximity_min,
            velocity_scale=velocity_scale,
            triggered_safety_stop=triggered_safety_stop,
            phase_history=phase_history,
        )


@dataclass
class ReactiveExecutionResult:
    """Summary of a reactive manipulation run."""

    states: list[ReactiveState]
    final_phase: ContactPhase
    steps_taken: int
    safety_stops: int
    reached_target: bool


class ReactiveController:
    """Sensor-driven wrapper around the WBC controller for manipulation tasks."""

    def __init__(self, config: ReactiveConfig | None = None) -> None:
        self.config = config or ReactiveConfig()
        self.fsm = ManipulationFSM(self.config)
        self.state = self.fsm.reset()
        self._sensor_suite: SensorSuite | Any | None = None
        self._wbc: WBCController | None = None
        self._phase_callback: Callable[[ContactPhase, ReactiveState], None] | None = None

    def attach_sensor_suite(self, suite: SensorSuite) -> None:
        self._sensor_suite = suite

    def attach_wbc(self, controller: WBCController) -> None:
        self._wbc = controller

    def set_phase_callback(self, callback: Callable[[ContactPhase, ReactiveState], None]) -> None:
        self._phase_callback = callback

    def run_step(self, robot: RobotModel, world: WorldState, dt: float) -> ReactiveState:
        sensor_data = self._read_sensor_data(world)
        previous_phase = self.state.phase
        self.state = self.fsm.step(self.state, sensor_data)

        if self._wbc is not None and dt > 0.0:
            velocities = self._wbc.compute_joint_velocities(robot, dt)
            scaled_positions = {
                joint_name: robot.joints[joint_name].clamp(
                    robot.joint_positions[joint_name] + velocity * self.state.velocity_scale * dt
                )
                for joint_name, velocity in velocities.items()
            }
            robot.set_joint_positions(scaled_positions)

        world.time_s += dt

        if self._phase_callback is not None and self.state.phase is not previous_phase:
            self._phase_callback(self.state.phase, self.state)
        return self.state

    def run_loop(self, robot: RobotModel, world: WorldState, max_steps: int | None = None) -> list[ReactiveState]:
        dt = 1.0 / self.config.control_freq_hz
        step_limit = self.config.max_step_count if max_steps is None else int(max_steps)
        history: list[ReactiveState] = []
        for _ in range(step_limit):
            current = self.run_step(robot, world, dt)
            history.append(
                ReactiveState(
                    phase=current.phase,
                    step=current.step,
                    ft_reading=current.ft_reading,
                    proximity_reading=current.proximity_reading,
                    velocity_scale=current.velocity_scale,
                    triggered_safety_stop=current.triggered_safety_stop,
                    phase_history=list(current.phase_history),
                )
            )
            if current.phase is ContactPhase.DONE:
                break
        return history

    def _read_sensor_data(self, world: WorldState) -> dict[str, Any]:
        sensor_data = {
            "ft_magnitude": 0.0,
            "proximity_min": float("inf"),
            "gripper_closed": False,
            "object_held": any(obj.held_by for obj in world.objects.values()),
            "at_target": False,
        }
        if self._sensor_suite is None:
            return sensor_data

        payload: Any | None = None
        read_method = getattr(self._sensor_suite, "read", None)
        if callable(read_method):
            payload = read_method()

        if payload is None:
            return sensor_data

        extracted = self._extract_sensor_data(payload)
        sensor_data.update(extracted)
        return sensor_data

    def _extract_sensor_data(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, SensorReading):
            return {
                "ft_magnitude": self._ft_magnitude_from_collection(payload.force_torque),
                "proximity_min": self._proximity_min_from_collection(payload.proximity),
                "gripper_closed": bool(getattr(payload, "gripper_closed", False)),
                "object_held": bool(getattr(payload, "object_held", False)),
                "at_target": bool(getattr(payload, "at_target", False)),
            }

        if isinstance(payload, dict):
            ft_source = payload.get("force_torque", payload.get("ft_magnitude", 0.0))
            proximity_source = payload.get("proximity", payload.get("proximity_min", float("inf")))
            return {
                "ft_magnitude": self._coerce_ft_magnitude(ft_source),
                "proximity_min": self._coerce_proximity_min(proximity_source),
                "gripper_closed": bool(payload.get("gripper_closed", False)),
                "object_held": bool(payload.get("object_held", False)),
                "at_target": bool(payload.get("at_target", False)),
            }

        return {
            "ft_magnitude": self._coerce_ft_magnitude(getattr(payload, "force_torque", 0.0)),
            "proximity_min": self._coerce_proximity_min(getattr(payload, "proximity", float("inf"))),
            "gripper_closed": bool(getattr(payload, "gripper_closed", False)),
            "object_held": bool(getattr(payload, "object_held", False)),
            "at_target": bool(getattr(payload, "at_target", False)),
        }

    @staticmethod
    def _ft_magnitude_from_collection(force_torque: dict[str, Any]) -> float:
        if not force_torque:
            return 0.0
        return max(ReactiveController._coerce_ft_magnitude(value) for value in force_torque.values())

    @staticmethod
    def _proximity_min_from_collection(proximity: dict[str, Any]) -> float:
        if not proximity:
            return float("inf")
        return min(ReactiveController._coerce_proximity_min(value) for value in proximity.values())

    @staticmethod
    def _coerce_ft_magnitude(value: Any) -> float:
        if isinstance(value, dict):
            return max((ReactiveController._coerce_ft_magnitude(item) for item in value.values()), default=0.0)
        if isinstance(value, (list, tuple, np.ndarray)):
            array = np.asarray(value, dtype=float)
            if array.ndim == 0:
                return float(array)
            force = array[:3] if array.size >= 3 else array
            return float(np.linalg.norm(np.nan_to_num(force, nan=0.0)))
        return float(value)

    @staticmethod
    def _coerce_proximity_min(value: Any) -> float:
        if isinstance(value, dict):
            values = [ReactiveController._coerce_proximity_min(item) for item in value.values()]
            return min(values, default=float("inf"))
        if isinstance(value, (list, tuple, np.ndarray)):
            array = np.asarray(value, dtype=float).reshape(-1)
            finite_values = array[np.isfinite(array)]
            if finite_values.size == 0:
                return float("inf")
            return float(np.min(finite_values))
        coerced = float(value)
        return coerced if isfinite(coerced) else float("inf")


def run_reactive_manipulation(
    robot: RobotModel,
    world: WorldState,
    config: ReactiveConfig | None = None,
) -> ReactiveExecutionResult:
    """Run a reactive manipulation loop to completion or step limit."""

    effective_config = config or ReactiveConfig()
    controller = ReactiveController(config=effective_config)
    states = controller.run_loop(robot, world, max_steps=effective_config.max_step_count)
    final_phase = states[-1].phase if states else ContactPhase.PRE_APPROACH
    return ReactiveExecutionResult(
        states=states,
        final_phase=final_phase,
        steps_taken=len(states),
        safety_stops=sum(state.triggered_safety_stop for state in states),
        reached_target=final_phase in {ContactPhase.PLACE, ContactPhase.RETRACT, ContactPhase.DONE},
    )


__all__ = [
    "ReactiveConfig",
    "ContactPhase",
    "ReactiveState",
    "ManipulationFSM",
    "ReactiveController",
    "ReactiveExecutionResult",
    "run_reactive_manipulation",
]
