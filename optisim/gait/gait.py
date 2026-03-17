"""Gait timing primitives and simple locomotion rhythm control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class GaitPhase(Enum):
    """Discrete locomotion support phase."""

    STANCE = "stance"
    SWING = "swing"
    DOUBLE_SUPPORT = "double_support"


@dataclass(frozen=True)
class LegCycle:
    """Timing definition for a single leg within a repeating gait cycle."""

    swing_duration: float
    stance_duration: float
    phase_offset: float

    def __post_init__(self) -> None:
        if self.swing_duration <= 0.0:
            raise ValueError("swing_duration must be positive")
        if self.stance_duration <= 0.0:
            raise ValueError("stance_duration must be positive")
        if not 0.0 <= self.phase_offset < 1.0:
            raise ValueError("phase_offset must be in [0, 1)")

    @property
    def cycle_duration(self) -> float:
        return self.swing_duration + self.stance_duration

    @property
    def duty_factor(self) -> float:
        return self.stance_duration / self.cycle_duration

    def phase_at(self, t: float) -> GaitPhase:
        local_t = float(t) % self.cycle_duration
        swing_start = self.phase_offset * self.cycle_duration
        swing_end = swing_start + self.swing_duration
        if swing_end <= self.cycle_duration:
            in_swing = swing_start <= local_t < swing_end
        else:
            wrap_end = swing_end - self.cycle_duration
            in_swing = local_t >= swing_start or local_t < wrap_end
        return GaitPhase.SWING if in_swing else GaitPhase.STANCE

    def swing_progress_at(self, t: float) -> float | None:
        """Return normalized swing progress in [0, 1), or None during stance."""

        local_t = float(t) % self.cycle_duration
        swing_start = self.phase_offset * self.cycle_duration
        swing_end = swing_start + self.swing_duration
        if swing_end <= self.cycle_duration:
            if not swing_start <= local_t < swing_end:
                return None
            elapsed = local_t - swing_start
        else:
            wrap_end = swing_end - self.cycle_duration
            if local_t >= swing_start:
                elapsed = local_t - swing_start
            elif local_t < wrap_end:
                elapsed = self.cycle_duration - swing_start + local_t
            else:
                return None
        return elapsed / self.swing_duration


class GaitPattern:
    """A coordinated gait definition for multiple legs."""

    def __init__(self, leg_cycles: dict[str, LegCycle], name: str = "") -> None:
        if not leg_cycles:
            raise ValueError("leg_cycles must not be empty")
        self.leg_cycles = dict(leg_cycles)
        self.name = str(name)

    @property
    def cycle_duration(self) -> float:
        return max(cycle.cycle_duration for cycle in self.leg_cycles.values())

    @property
    def frequency(self) -> float:
        return 1.0 / self.cycle_duration

    def get_phases(self, t: float) -> dict[str, GaitPhase]:
        phases = {name: cycle.phase_at(t) for name, cycle in self.leg_cycles.items()}
        if len(phases) == 2 and all(phase is GaitPhase.STANCE for phase in phases.values()):
            return {name: GaitPhase.DOUBLE_SUPPORT for name in phases}
        return phases

    @classmethod
    def walk(cls, step_duration: float = 0.4, stance_ratio: float = 0.6) -> GaitPattern:
        swing_duration, stance_duration = cls._durations(step_duration, stance_ratio)
        return cls(
            {
                "left": LegCycle(swing_duration=swing_duration, stance_duration=stance_duration, phase_offset=0.0),
                "right": LegCycle(swing_duration=swing_duration, stance_duration=stance_duration, phase_offset=0.5),
            },
            name="walk",
        )

    @classmethod
    def run(cls, step_duration: float = 0.25, stance_ratio: float = 0.35) -> GaitPattern:
        swing_duration, stance_duration = cls._durations(step_duration, stance_ratio)
        return cls(
            {
                "left": LegCycle(swing_duration=swing_duration, stance_duration=stance_duration, phase_offset=0.0),
                "right": LegCycle(swing_duration=swing_duration, stance_duration=stance_duration, phase_offset=0.5),
            },
            name="run",
        )

    @classmethod
    def trot(cls, step_duration: float = 0.3) -> GaitPattern:
        if step_duration <= 0.0:
            raise ValueError("step_duration must be positive")
        swing_duration = 0.5 * step_duration
        stance_duration = 0.5 * step_duration
        return cls(
            {
                "fl": LegCycle(swing_duration=swing_duration, stance_duration=stance_duration, phase_offset=0.0),
                "rr": LegCycle(swing_duration=swing_duration, stance_duration=stance_duration, phase_offset=0.0),
                "fr": LegCycle(swing_duration=swing_duration, stance_duration=stance_duration, phase_offset=0.5),
                "rl": LegCycle(swing_duration=swing_duration, stance_duration=stance_duration, phase_offset=0.5),
            },
            name="trot",
        )

    @staticmethod
    def _durations(step_duration: float, stance_ratio: float) -> tuple[float, float]:
        if step_duration <= 0.0:
            raise ValueError("step_duration must be positive")
        if not 0.0 < stance_ratio < 1.0:
            raise ValueError("stance_ratio must be in (0, 1)")
        stance_duration = step_duration * stance_ratio
        swing_duration = step_duration - stance_duration
        return swing_duration, stance_duration


class CPGOscillator:
    """Simple sinusoidal phase oscillator with optional external coupling."""

    def __init__(self, frequency: float, amplitude: float = 1.0, coupling: float = 0.0) -> None:
        if frequency <= 0.0:
            raise ValueError("frequency must be positive")
        if amplitude < 0.0:
            raise ValueError("amplitude must be non-negative")
        self.frequency = float(frequency)
        self.amplitude = float(amplitude)
        self.coupling = float(coupling)
        self.phase = 0.0
        self.output = 0.0

    def step(self, dt: float, external_input: float = 0.0) -> float:
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        omega = 2.0 * math.pi * self.frequency + self.coupling * float(external_input)
        self.phase = (self.phase + omega * dt) % (2.0 * math.pi)
        self.output = self.amplitude * math.sin(self.phase)
        return self.output

    def reset(self) -> None:
        self.phase = 0.0
        self.output = 0.0


class GaitController:
    """Generate simple vertical swing velocity references for each leg."""

    def __init__(self, pattern: GaitPattern, step_height: float = 0.1, step_length: float = 0.3) -> None:
        if step_height < 0.0:
            raise ValueError("step_height must be non-negative")
        if step_length < 0.0:
            raise ValueError("step_length must be non-negative")
        self.pattern = pattern
        self.step_height = float(step_height)
        self.step_length = float(step_length)
        self.oscillators = {
            name: CPGOscillator(frequency=pattern.frequency, amplitude=1.0)
            for name in pattern.leg_cycles
        }

    def step(self, t: float, dt: float) -> dict[str, float]:
        references: dict[str, float] = {}
        phases = self.pattern.get_phases(t)
        length_scale = 1.0 + 0.5 * self.step_length
        for name, cycle in self.pattern.leg_cycles.items():
            in_contact = phases[name] is not GaitPhase.SWING
            oscillator = self.oscillators[name]
            osc_output = oscillator.step(dt, external_input=-1.0 if in_contact else 1.0)
            swing_progress = cycle.swing_progress_at(t)
            if swing_progress is None or self.step_height == 0.0:
                references[name] = 0.0
                continue
            base_velocity = self.step_height * math.pi * math.cos(math.pi * swing_progress) / cycle.swing_duration
            modulation = 1.0 + 0.25 * osc_output
            references[name] = base_velocity * length_scale * modulation
        return references

    def get_contact_states(self, t: float) -> dict[str, bool]:
        phases = self.pattern.get_phases(t)
        return {name: phase is not GaitPhase.SWING for name, phase in phases.items()}


__all__ = [
    "GaitPhase",
    "LegCycle",
    "GaitPattern",
    "CPGOscillator",
    "GaitController",
]
