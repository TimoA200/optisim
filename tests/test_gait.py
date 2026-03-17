from __future__ import annotations

from dataclasses import is_dataclass
import math

import pytest

import optisim
import optisim.gait as gait
from optisim import CPGOscillator as PublicCPGOscillator
from optisim import GaitController as PublicGaitController
from optisim import GaitPattern as PublicGaitPattern
from optisim import GaitPhase as PublicGaitPhase
from optisim import LegCycle as PublicLegCycle
from optisim import gait as public_gait_module
from optisim.gait import CPGOscillator, GaitController, GaitPattern, GaitPhase, LegCycle


def test_gaitphase_enum_values_are_stable() -> None:
    assert GaitPhase.STANCE.value == "stance"
    assert GaitPhase.SWING.value == "swing"
    assert GaitPhase.DOUBLE_SUPPORT.value == "double_support"


def test_root_exports_gait_types() -> None:
    assert PublicGaitPhase is GaitPhase
    assert PublicLegCycle is LegCycle
    assert PublicGaitPattern is GaitPattern
    assert PublicCPGOscillator is CPGOscillator
    assert PublicGaitController is GaitController


def test_gait_module_all_matches_public_surface() -> None:
    assert gait.__all__ == ["GaitPhase", "LegCycle", "GaitPattern", "CPGOscillator", "GaitController"]


def test_legcycle_rejects_non_positive_swing_duration() -> None:
    with pytest.raises(ValueError, match="swing_duration"):
        LegCycle(swing_duration=0.0, stance_duration=0.2, phase_offset=0.0)


def test_legcycle_rejects_non_positive_stance_duration() -> None:
    with pytest.raises(ValueError, match="stance_duration"):
        LegCycle(swing_duration=0.2, stance_duration=0.0, phase_offset=0.0)


def test_legcycle_rejects_negative_phase_offset() -> None:
    with pytest.raises(ValueError, match="phase_offset"):
        LegCycle(swing_duration=0.2, stance_duration=0.2, phase_offset=-0.1)


def test_legcycle_rejects_phase_offset_of_one() -> None:
    with pytest.raises(ValueError, match="phase_offset"):
        LegCycle(swing_duration=0.2, stance_duration=0.2, phase_offset=1.0)


def test_legcycle_cycle_duration_is_sum_of_swing_and_stance() -> None:
    cycle = LegCycle(swing_duration=0.12, stance_duration=0.18, phase_offset=0.0)

    assert cycle.cycle_duration == pytest.approx(0.3)


def test_legcycle_duty_factor_is_stance_fraction() -> None:
    cycle = LegCycle(swing_duration=0.2, stance_duration=0.3, phase_offset=0.0)

    assert cycle.duty_factor == pytest.approx(0.6)


def test_legcycle_phase_at_returns_swing_during_swing_window() -> None:
    cycle = LegCycle(swing_duration=0.2, stance_duration=0.3, phase_offset=0.0)

    assert cycle.phase_at(0.1) is GaitPhase.SWING


def test_legcycle_phase_at_returns_stance_outside_swing_window() -> None:
    cycle = LegCycle(swing_duration=0.2, stance_duration=0.3, phase_offset=0.0)

    assert cycle.phase_at(0.3) is GaitPhase.STANCE


def test_legcycle_phase_at_supports_wrapped_swing_interval() -> None:
    cycle = LegCycle(swing_duration=0.2, stance_duration=0.3, phase_offset=0.8)

    assert cycle.phase_at(0.45) is GaitPhase.SWING
    assert cycle.phase_at(0.05) is GaitPhase.SWING


def test_legcycle_phase_at_is_periodic_for_negative_time() -> None:
    cycle = LegCycle(swing_duration=0.2, stance_duration=0.3, phase_offset=0.0)

    assert cycle.phase_at(-0.1) is GaitPhase.STANCE


def test_legcycle_swing_progress_is_none_during_stance() -> None:
    cycle = LegCycle(swing_duration=0.2, stance_duration=0.3, phase_offset=0.0)

    assert cycle.swing_progress_at(0.25) is None


def test_legcycle_swing_progress_starts_at_zero() -> None:
    cycle = LegCycle(swing_duration=0.2, stance_duration=0.3, phase_offset=0.0)

    assert cycle.swing_progress_at(0.0) == pytest.approx(0.0)


def test_legcycle_swing_progress_handles_wrapped_swing() -> None:
    cycle = LegCycle(swing_duration=0.2, stance_duration=0.3, phase_offset=0.8)

    assert cycle.swing_progress_at(0.45) == pytest.approx(0.25)
    assert cycle.swing_progress_at(0.05) == pytest.approx(0.75)


def test_gaitpattern_rejects_empty_leg_map() -> None:
    with pytest.raises(ValueError, match="leg_cycles"):
        GaitPattern({})


def test_gaitpattern_cycle_duration_uses_longest_leg_cycle() -> None:
    pattern = GaitPattern(
        {
            "left": LegCycle(swing_duration=0.2, stance_duration=0.3, phase_offset=0.0),
            "right": LegCycle(swing_duration=0.1, stance_duration=0.2, phase_offset=0.0),
        }
    )

    assert pattern.cycle_duration == pytest.approx(0.5)


def test_gaitpattern_frequency_is_inverse_of_cycle_duration() -> None:
    pattern = GaitPattern.walk(step_duration=0.4, stance_ratio=0.6)

    assert pattern.frequency == pytest.approx(2.5)


def test_walk_factory_builds_named_biped_pattern() -> None:
    pattern = GaitPattern.walk()

    assert pattern.name == "walk"
    assert set(pattern.leg_cycles) == {"left", "right"}


def test_walk_factory_offsets_legs_by_half_cycle() -> None:
    pattern = GaitPattern.walk()
    phases = pattern.get_phases(0.0)

    assert phases["left"] is GaitPhase.SWING
    assert phases["right"] is GaitPhase.STANCE


def test_walk_pattern_reports_double_support_when_both_legs_stance() -> None:
    pattern = GaitPattern.walk()
    phases = pattern.get_phases(0.18)

    assert phases["left"] is GaitPhase.DOUBLE_SUPPORT
    assert phases["right"] is GaitPhase.DOUBLE_SUPPORT


def test_run_factory_builds_named_pattern_with_small_duty_factor() -> None:
    pattern = GaitPattern.run()

    assert pattern.name == "run"
    assert pattern.leg_cycles["left"].duty_factor == pytest.approx(0.35)


def test_run_pattern_has_flight_interval_with_both_legs_swinging() -> None:
    pattern = GaitPattern.run()
    phases = pattern.get_phases(0.14)

    assert phases["left"] is GaitPhase.SWING
    assert phases["right"] is GaitPhase.SWING


def test_trot_factory_builds_named_quadruped_pattern() -> None:
    pattern = GaitPattern.trot()

    assert pattern.name == "trot"
    assert set(pattern.leg_cycles) == {"fl", "fr", "rl", "rr"}


def test_trot_factory_keeps_diagonal_pairs_in_phase() -> None:
    pattern = GaitPattern.trot()
    phases = pattern.get_phases(0.05)

    assert phases["fl"] is phases["rr"]


def test_trot_factory_offsets_opposite_diagonal_pair() -> None:
    pattern = GaitPattern.trot()
    phases = pattern.get_phases(0.05)

    assert phases["fl"] is GaitPhase.SWING
    assert phases["fr"] is GaitPhase.STANCE
    assert phases["rl"] is GaitPhase.STANCE


def test_oscillator_rejects_non_positive_frequency() -> None:
    with pytest.raises(ValueError, match="frequency"):
        CPGOscillator(frequency=0.0)


def test_oscillator_rejects_negative_dt() -> None:
    oscillator = CPGOscillator(frequency=1.0)

    with pytest.raises(ValueError, match="dt"):
        oscillator.step(-0.1)


def test_oscillator_quarter_cycle_step_reaches_positive_peak() -> None:
    oscillator = CPGOscillator(frequency=1.0, amplitude=2.0)

    output = oscillator.step(0.25)

    assert output == pytest.approx(2.0)
    assert oscillator.phase == pytest.approx(math.pi / 2.0)


def test_oscillator_external_coupling_changes_phase_advance() -> None:
    uncoupled = CPGOscillator(frequency=1.0, coupling=0.0)
    coupled = CPGOscillator(frequency=1.0, coupling=2.0)

    uncoupled.step(0.1, external_input=1.0)
    coupled.step(0.1, external_input=1.0)

    assert coupled.phase > uncoupled.phase


def test_oscillator_reset_restores_zero_phase_and_output() -> None:
    oscillator = CPGOscillator(frequency=1.0)
    oscillator.step(0.1)
    oscillator.reset()

    assert oscillator.phase == pytest.approx(0.0)
    assert oscillator.output == pytest.approx(0.0)


def test_gaitcontroller_rejects_negative_step_height() -> None:
    with pytest.raises(ValueError, match="step_height"):
        GaitController(GaitPattern.walk(), step_height=-0.1)


def test_gaitcontroller_contact_states_treat_double_support_as_contact() -> None:
    controller = GaitController(GaitPattern.walk())

    assert controller.get_contact_states(0.18) == {"left": True, "right": True}


def test_gaitcontroller_contact_states_report_swing_leg_as_not_in_contact() -> None:
    controller = GaitController(GaitPattern.walk())

    assert controller.get_contact_states(0.0) == {"left": False, "right": True}


def test_gaitcontroller_step_returns_zero_velocity_for_stance_legs() -> None:
    controller = GaitController(GaitPattern.walk(), step_height=0.1)

    references = controller.step(0.18, 0.01)

    assert references["left"] == pytest.approx(0.0)
    assert references["right"] == pytest.approx(0.0)


def test_gaitcontroller_step_returns_positive_velocity_early_in_swing() -> None:
    controller = GaitController(GaitPattern.walk(), step_height=0.1)

    references = controller.step(0.0, 0.01)

    assert references["left"] > 0.0
    assert references["right"] == pytest.approx(0.0)


def test_gaitcontroller_step_crosses_zero_at_mid_swing() -> None:
    controller = GaitController(GaitPattern.walk(), step_height=0.1)

    references = controller.step(0.08, 0.01)

    assert references["left"] == pytest.approx(0.0, abs=1e-9)


def test_gaitcontroller_step_returns_negative_velocity_late_in_swing() -> None:
    controller = GaitController(GaitPattern.walk(), step_height=0.1)

    references = controller.step(0.15, 0.01)

    assert references["left"] < 0.0


def test_package_version_matches_gait_release() -> None:
    assert optisim.gait is gait
    assert public_gait_module is gait
    assert is_dataclass(LegCycle)
    assert optisim.__version__ == "0.28.0"
