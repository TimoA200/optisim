"""Comprehensive ScenarioRunner demo with sensors and safety monitoring."""

from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optisim.core.task_definition import TaskDefinition
from optisim.safety.core import SafetyConfig, SafetyZone, ZoneType
from optisim.scenario.runner import ScenarioConfig, ScenarioRunner
from optisim.sensors import SensorSuite


def build_task() -> TaskDefinition:
    return TaskDefinition.from_dict(
        {
            "name": "full_scenario_demo",
            "metadata": {"author": "optisim", "scenario": "full_demo"},
            "robot": {"model": "optimus_humanoid"},
            "world": {
                "gravity": [0.0, 0.0, -9.81],
                "surfaces": [
                    {
                        "name": "table",
                        "pose": {"position": [0.55, 0.0, 0.74], "rpy": [0.0, 0.0, 0.0]},
                        "size": [0.90, 0.60, 0.05],
                    },
                    {
                        "name": "shelf",
                        "pose": {"position": [0.60, -0.25, 1.02], "rpy": [0.0, 0.0, 0.0]},
                        "size": [0.35, 0.25, 0.04],
                    },
                ],
                "objects": [
                    {
                        "name": "box",
                        "pose": {"position": [0.42, -0.12, 0.81], "rpy": [0.0, 0.0, 0.0]},
                        "size": [0.08, 0.08, 0.12],
                        "mass_kg": 0.75,
                    }
                ],
            },
            "actions": [
                {"type": "reach", "target": "box", "end_effector": "right_palm"},
                {"type": "grasp", "target": "box", "end_effector": "right_gripper"},
                {
                    "type": "move",
                    "target": "box",
                    "end_effector": "right_palm",
                    "destination": [0.58, -0.20, 1.08],
                    "speed": 0.28,
                },
                {"type": "place", "target": "box", "end_effector": "right_palm", "support": "shelf"},
            ],
        }
    )


def build_safety_config() -> SafetyConfig:
    return SafetyConfig(
        zones=[
            SafetyZone(
                name="head_height_exclusion",
                center=np.array([0.0, 0.0, 2.30], dtype=float),
                half_extents=np.array([2.0, 2.0, 0.30], dtype=float),
                zone_type=ZoneType.FORBIDDEN,
            ),
            SafetyZone(
                name="upper_torso_caution",
                center=np.array([0.15, 0.0, 1.00], dtype=float),
                half_extents=np.array([0.30, 0.40, 0.12], dtype=float),
                zone_type=ZoneType.CAUTION,
            ),
        ],
        joint_limits=SafetyConfig.default_humanoid().joint_limits,
    )


def main() -> None:
    config = ScenarioConfig(
        name="full_scenario_demo",
        task=build_task(),
        sensor_suite=SensorSuite.default_humanoid_suite(),
        safety_config=build_safety_config(),
        dt=0.05,
        rng_seed=42,
    )
    result = ScenarioRunner(config).run()

    print(result.summary())
    print()
    print("First 5 steps:")
    for step_result in result.step_results[:5]:
        violation_counts = Counter(
            f"{item.violation_type}:{item.zone.name}" for item in step_result.safety_violations
        )
        violations = [f"{name} x{count}" for name, count in sorted(violation_counts.items())]
        print(
            f"step={step_result.step} "
            f"action_name={step_result.action_name!r} "
            f"is_safe={step_result.is_safe} "
            f"violations={violations or ['none']}"
        )

    ft_timeline = result.sensor_timeline
    total_ft_readings = sum(len(readings) for readings in ft_timeline.values())
    print()
    print(
        f"Sensor timeline stats: {total_ft_readings} FT readings "
        f"across {len(ft_timeline)} force/torque sensor(s)."
    )
    print("Demo complete!")


if __name__ == "__main__":
    main()
