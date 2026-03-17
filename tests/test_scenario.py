from __future__ import annotations

import numpy as np

from optisim.core import TaskDefinition, ValidationReport
from optisim.safety import SafetyConfig, SafetyViolation, SafetyZone, ZoneType
from optisim.scenario import (
    ScenarioConfig,
    ScenarioResult,
    ScenarioRunner,
    ScenarioStepResult,
    SensorReading,
)
from optisim.sim import SimulationRecord


def make_task_dict() -> dict:
    return {
        "name": "grasp_box",
        "actions": [{"type": "grasp", "target": "box"}],
    }


def make_safe_config() -> SafetyConfig:
    return SafetyConfig(zones=[], joint_limits=[])


def make_forbidden_violation() -> SafetyViolation:
    zone = SafetyZone(
        name="forbidden",
        center=np.zeros(3, dtype=float),
        half_extents=np.ones(3, dtype=float),
        zone_type=ZoneType.FORBIDDEN,
    )
    return SafetyViolation(
        robot_name="atlas",
        joint_name="joint",
        position=np.zeros(3, dtype=float),
        zone=zone,
        violation_type="forbidden_zone",
        severity=1.0,
    )


def make_result(step_results: list[ScenarioStepResult]) -> ScenarioResult:
    return ScenarioResult(
        name="demo",
        total_steps=len(step_results),
        total_time_s=0.05 * len(step_results),
        step_results=step_results,
        sim_record=SimulationRecord(steps=len(step_results), duration_s=0.05 * len(step_results)),
    )


def test_scenario_config_from_dict_constructs_defaults() -> None:
    config = ScenarioConfig.from_dict({"name": "demo", "task": make_task_dict()})

    assert config.name == "demo"
    assert isinstance(config.task, TaskDefinition)
    assert config.sensor_suite is not None
    assert config.safety_config is not None
    assert config.dt == 0.05
    assert config.rng_seed == 42


def test_scenario_config_from_dict_respects_overrides() -> None:
    config = ScenarioConfig.from_dict(
        {
            "name": "demo",
            "task": make_task_dict(),
            "safety_config": {"zones": [], "joint_limits": []},
            "dt": 0.1,
            "rng_seed": 7,
        }
    )

    assert config.dt == 0.1
    assert config.rng_seed == 7
    assert isinstance(config.safety_config, SafetyConfig)


def test_sensor_reading_fields_are_populated() -> None:
    reading = SensorReading(
        step=1,
        time_s=0.05,
        force_torque={"ft": np.zeros(6, dtype=float)},
        proximity={"prox": 0.4},
        imu={"imu": {"accel": np.zeros(3, dtype=float), "gyro": np.zeros(3, dtype=float)}},
        joint_encoders={"joint": 0.1},
    )

    assert reading.step == 1
    assert reading.time_s == 0.05
    assert reading.force_torque["ft"].shape == (6,)
    assert reading.proximity["prox"] == 0.4
    assert set(reading.imu["imu"]) == {"accel", "gyro"}
    assert reading.joint_encoders["joint"] == 0.1


def test_scenario_step_result_can_be_marked_safe() -> None:
    step_result = ScenarioStepResult(
        step=1,
        time_s=0.0,
        action_name="grasp box",
        sensor_reading=SensorReading(1, 0.0, {}, {}, {}, {}),
        safety_violations=[],
        is_safe=True,
        emergency_stopped=False,
    )

    assert step_result.is_safe is True
    assert step_result.emergency_stopped is False


def test_scenario_step_result_can_be_marked_unsafe() -> None:
    step_result = ScenarioStepResult(
        step=1,
        time_s=0.0,
        action_name="grasp box",
        sensor_reading=SensorReading(1, 0.0, {}, {}, {}, {}),
        safety_violations=[make_forbidden_violation()],
        is_safe=False,
        emergency_stopped=True,
    )

    assert step_result.is_safe is False
    assert step_result.emergency_stopped is True


def test_scenario_result_safety_violation_count_property() -> None:
    step_results = [
        ScenarioStepResult(1, 0.0, None, SensorReading(1, 0.0, {}, {}, {}, {}), [make_forbidden_violation()], False, True),
        ScenarioStepResult(2, 0.1, None, SensorReading(2, 0.1, {}, {}, {}, {}), [], True, False),
    ]

    result = make_result(step_results)

    assert result.safety_violation_count == 1


def test_scenario_result_sensor_timeline_collects_force_torque_readings() -> None:
    step_results = [
        ScenarioStepResult(
            1,
            0.0,
            None,
            SensorReading(1, 0.0, {"ft": np.ones(6, dtype=float)}, {}, {}, {}),
            [],
            True,
            False,
        ),
        ScenarioStepResult(
            2,
            0.1,
            None,
            SensorReading(2, 0.1, {"ft": np.zeros(6, dtype=float)}, {}, {}, {}),
            [],
            True,
            False,
        ),
    ]

    result = make_result(step_results)

    assert len(result.sensor_timeline["ft"]) == 2
    np.testing.assert_allclose(result.sensor_timeline["ft"][0], np.ones(6, dtype=float))


def test_scenario_result_summary_returns_non_empty_string() -> None:
    result = make_result([])

    summary = result.summary()

    assert isinstance(summary, str)
    assert summary


def test_scenario_result_summary_mentions_stop_reason_when_present() -> None:
    result = make_result([])
    result.emergency_stop_reason = "joint in forbidden"

    summary = result.summary()

    assert "stopped" in summary


def test_scenario_runner_validate_returns_validation_report() -> None:
    runner = ScenarioRunner(
        ScenarioConfig(
            name="demo",
            task=TaskDefinition.from_dict(make_task_dict()),
            safety_config=make_safe_config(),
        )
    )

    report = runner.validate()

    assert isinstance(report, ValidationReport)
    assert report.is_valid


def test_scenario_runner_run_returns_scenario_result_with_correct_step_count() -> None:
    runner = ScenarioRunner(
        ScenarioConfig(
            name="demo",
            task=TaskDefinition.from_dict(make_task_dict()),
            safety_config=make_safe_config(),
        )
    )

    result = runner.run()

    assert isinstance(result, ScenarioResult)
    assert result.total_steps == 1
    assert result.sim_record.steps == 1
    assert len(result.step_results) == 1


def test_scenario_runner_run_populates_step_results_with_sensor_readings() -> None:
    runner = ScenarioRunner(
        ScenarioConfig(
            name="demo",
            task=TaskDefinition.from_dict(make_task_dict()),
            safety_config=make_safe_config(),
        )
    )

    result = runner.run()
    reading = result.step_results[0].sensor_reading

    assert reading.force_torque
    assert reading.proximity
    assert reading.imu
    assert reading.joint_encoders


def test_scenario_runner_run_returns_non_emergency_stopped_result_for_safe_scenarios() -> None:
    runner = ScenarioRunner(
        ScenarioConfig(
            name="safe_demo",
            task=TaskDefinition.from_dict(make_task_dict()),
            safety_config=make_safe_config(),
        )
    )

    result = runner.run()

    assert result.emergency_stop_reason is None
    assert result.step_results[0].emergency_stopped is False
    assert result.step_results[0].is_safe is True


def test_scenario_runner_run_is_deterministic_for_same_seed() -> None:
    config = ScenarioConfig(
        name="seeded",
        task=TaskDefinition.from_dict(make_task_dict()),
        safety_config=make_safe_config(),
        rng_seed=11,
    )

    first = ScenarioRunner(config).run()
    second = ScenarioRunner(config).run()

    np.testing.assert_allclose(
        first.step_results[0].sensor_reading.force_torque["left_wrist_ft"],
        second.step_results[0].sensor_reading.force_torque["left_wrist_ft"],
    )


def test_scenario_runner_run_records_action_name_on_step_result() -> None:
    runner = ScenarioRunner(
        ScenarioConfig(
            name="demo",
            task=TaskDefinition.from_dict(make_task_dict()),
            safety_config=make_safe_config(),
        )
    )

    result = runner.run()

    assert result.step_results[0].action_name == "grasp box"


def test_scenario_runner_run_produces_simulation_recording() -> None:
    runner = ScenarioRunner(
        ScenarioConfig(
            name="demo",
            task=TaskDefinition.from_dict(make_task_dict()),
            safety_config=make_safe_config(),
        )
    )

    result = runner.run()

    assert result.sim_record.recording is not None
    assert result.sim_record.recording.frame_count() >= 2


def test_scenario_runner_run_aborts_early_on_emergency_stop() -> None:
    config = ScenarioConfig.from_dict(
        {
            "name": "emergency_demo",
            "task": make_task_dict(),
            "safety_config": {
                "zones": [
                    {
                        "name": "trip_zone",
                        "center": [0.0, 0.0, 1.0],
                        "half_extents": [5.0, 5.0, 5.0],
                        "zone_type": "forbidden",
                    }
                ],
                "joint_limits": [],
            },
        }
    )
    runner = ScenarioRunner(config)

    result = runner.run()

    assert result.total_steps == 1
    assert result.emergency_stop_reason is not None
    assert result.step_results[0].emergency_stopped is True


def test_full_integration_build_config_from_dict_run_and_check_summary() -> None:
    config = ScenarioConfig.from_dict(
        {
            "name": "integration_demo",
            "task": make_task_dict(),
            "safety_config": {"zones": [], "joint_limits": []},
        }
    )
    runner = ScenarioRunner(config)

    result = runner.run()

    assert "integration_demo" in result.summary()
    assert result.total_steps == 1


def test_public_exports_are_available_from_optisim_scenario() -> None:
    from optisim.scenario import ScenarioConfig as ExportedConfig
    from optisim.scenario import ScenarioResult as ExportedResult
    from optisim.scenario import ScenarioRunner as ExportedRunner
    from optisim.scenario import ScenarioStepResult as ExportedStepResult
    from optisim.scenario import SensorReading as ExportedSensorReading

    assert ExportedConfig is ScenarioConfig
    assert ExportedResult is ScenarioResult
    assert ExportedRunner is ScenarioRunner
    assert ExportedStepResult is ScenarioStepResult
    assert ExportedSensorReading is SensorReading
