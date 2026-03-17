"""Performance benchmark for key optisim subsystems."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optisim.core.task_definition import TaskDefinition
from optisim.math3d import Pose, Quaternion
from optisim.planning.rrt import RRTConfig, plan_rrt
from optisim.robot.humanoid import build_humanoid_model
from optisim.robot.ik import IKOptions, solve_inverse_kinematics
from optisim.safety.core import SafetyConfig, SafetyMonitor, SafetyZone, ZoneType
from optisim.scenario.runner import ScenarioConfig, ScenarioRunner
from optisim.sensors import SensorSuite
from optisim.sim.engine import ExecutionEngine


def build_simple_task() -> TaskDefinition:
    return TaskDefinition.from_dict(
        {
            "name": "benchmark_pick_and_place",
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


def build_benchmark_safety() -> SafetyConfig:
    return SafetyConfig(
        zones=[
            SafetyZone(
                name="head_height_exclusion",
                center=np.array([0.0, 0.0, 2.30], dtype=float),
                half_extents=np.array([2.0, 2.0, 0.30], dtype=float),
                zone_type=ZoneType.FORBIDDEN,
            ),
            SafetyZone(
                name="torso_caution",
                center=np.array([0.20, 0.0, 1.00], dtype=float),
                half_extents=np.array([0.40, 0.40, 0.15], dtype=float),
                zone_type=ZoneType.CAUTION,
            ),
        ],
        joint_limits=SafetyConfig.default_humanoid().joint_limits,
    )


def run_benchmark(name: str, runs: int, fn) -> None:
    samples: list[float] = []
    started = time.perf_counter()
    for _ in range(runs):
        run_started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - run_started)
    total_s = time.perf_counter() - started
    median_ms = statistics.median(samples) * 1000.0
    print(f"{name}: {runs} runs in {total_s:.1f}s -> {median_ms:.1f}ms/run (median)", flush=True)


def main() -> None:
    rng = np.random.default_rng(42)
    task = build_simple_task()

    robot = build_humanoid_model()
    ik_joint_names = [joint.name for joint in robot.joint_chain_for_effector("right_palm")]
    ik_options = IKOptions(max_iterations=8, convergence_threshold=1e-2, damping=0.12, position_only=True)

    def benchmark_ik() -> None:
        target = Pose(
            position=np.array(
                [
                    rng.uniform(0.35, 0.70),
                    rng.uniform(-0.30, 0.05),
                    rng.uniform(0.85, 1.25),
                ],
                dtype=float,
            ),
            orientation=Quaternion.identity(),
        )
        solve_inverse_kinematics(
            robot,
            "right_palm",
            target,
            joint_names=ik_joint_names,
            options=ik_options,
        )

    engine = ExecutionEngine()

    def benchmark_validation() -> None:
        engine.validate(task)

    arm_joints = [joint.name for joint in robot.joint_chain_for_effector("right_palm")]
    lower_bounds = np.array([robot.joints[name].limit_lower for name in arm_joints], dtype=float)
    upper_bounds = np.array([robot.joints[name].limit_upper for name in arm_joints], dtype=float)
    rrt_config = RRTConfig(max_iterations=300, step_size=0.45, goal_bias=0.20, goal_threshold=0.25)

    def benchmark_rrt() -> None:
        start = rng.uniform(lower_bounds, upper_bounds)
        goal = rng.uniform(lower_bounds, upper_bounds)
        plan_rrt(
            start,
            goal,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            is_state_valid=lambda state: bool(np.all(state >= lower_bounds) and np.all(state <= upper_bounds)),
            is_edge_valid=lambda _a, _b: True,
            config=rrt_config,
            rng=rng,
        )

    safety_config = build_benchmark_safety()
    safety_monitor = SafetyMonitor(zones=safety_config.zones, joint_limits=safety_config.joint_limits)
    link_names = [f"link_{index}" for index in range(12)]

    def benchmark_safety() -> None:
        link_positions = {
            name: np.array(
                [
                    rng.uniform(-0.5, 0.8),
                    rng.uniform(-0.6, 0.6),
                    rng.uniform(0.0, 2.4),
                ],
                dtype=float,
            )
            for name in link_names
        }
        safety_monitor.check_positions("benchmark_bot", link_positions)

    def benchmark_scenario() -> None:
        config = ScenarioConfig(
            name="benchmark_scenario",
            task=build_simple_task(),
            sensor_suite=SensorSuite.default_humanoid_suite(),
            safety_config=safety_config,
            dt=0.05,
            rng_seed=42,
        )
        ScenarioRunner(config).run()

    run_benchmark("IK solving", 100, benchmark_ik)
    run_benchmark("Task validation", 50, benchmark_validation)
    run_benchmark("Motion planning (RRT)", 20, benchmark_rrt)
    run_benchmark("Safety checking", 1000, benchmark_safety)
    run_benchmark("Scenario run", 10, benchmark_scenario)


if __name__ == "__main__":
    main()
