"""Performance benchmark for core optisim kinematics and simulation paths."""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optisim.math3d import Pose, Quaternion, vec3
from optisim.robot import IKOptions, JointSpec, LinkSpec, RobotModel, build_humanoid_model, solve_inverse_kinematics
from optisim.sim import ExecutionEngine, WorldState

__all__ = ["main"]


@dataclass(slots=True)
class BenchmarkStats:
    """Small summary of a benchmark sample set."""

    runs: int
    median_ms: float
    p95_ms: float
    min_ms: float


def build_chain_robot(joint_count: int) -> RobotModel:
    """Build a serial chain robot used for scaling benchmarks."""

    links = {"base": LinkSpec(name="base", visual_extent=(0.12, 0.12, 0.12))}
    joints: dict[str, JointSpec] = {}
    parent = "base"
    for index in range(joint_count):
        child = f"link_{index + 1}"
        links[child] = LinkSpec(name=child, visual_extent=(0.22, 0.06, 0.06))
        joint_name = f"joint_{index + 1}"
        joints[joint_name] = JointSpec(
            name=joint_name,
            parent=parent,
            child=child,
            joint_type="revolute",
            origin=Pose.from_xyz_rpy([0.22, 0.0, 0.0], [0.0, 0.0, 0.0]),
            axis=(0.0, 1.0, 0.0),
            limit_lower=-np.pi,
            limit_upper=np.pi,
            velocity_limit=4.0,
        )
        links[child].parent_joint = joint_name
        parent = child
    return RobotModel(
        name=f"chain_{joint_count}dof",
        links=links,
        joints=joints,
        root_link="base",
        end_effectors={"tool": parent},
        base_pose=Pose(position=vec3([0.0, 0.0, 0.25]), orientation=Quaternion.identity()),
    )


def sample_benchmark(runs: int, fn) -> BenchmarkStats:
    """Measure runtime statistics for a callable."""

    samples_ms: list[float] = []
    for _ in range(runs):
        started = time.perf_counter_ns()
        fn()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        samples_ms.append(elapsed_ms)
    return BenchmarkStats(
        runs=runs,
        median_ms=statistics.median(samples_ms),
        p95_ms=np.percentile(samples_ms, 95),
        min_ms=min(samples_ms),
    )


def benchmark_ik_by_joint_count(console: Console, joint_counts: list[int]) -> None:
    """Benchmark IK solve time as joint count increases."""

    table = Table(title="IK Solve Time by Joint Count", header_style="bold cyan")
    table.add_column("Joints", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("Median (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("Best (ms)", justify="right")

    for joint_count in joint_counts:
        robot = build_chain_robot(joint_count)
        target_positions = {
            joint_name: 0.18 * np.sin((index + 1) * 0.45)
            for index, joint_name in enumerate(robot.joints)
        }
        target_pose = robot.end_effector_pose("tool", target_positions)
        stats = sample_benchmark(
            runs=60,
            fn=lambda: solve_inverse_kinematics(
                robot,
                "tool",
                target_pose,
                options=IKOptions(
                    max_iterations=100,
                    convergence_threshold=1e-3,
                    damping=0.08,
                    position_only=True,
                ),
            ),
        )
        table.add_row(
            str(joint_count),
            str(stats.runs),
            f"{stats.median_ms:.3f}",
            f"{stats.p95_ms:.3f}",
            f"{stats.min_ms:.3f}",
        )

    console.print(table)


def benchmark_fk_by_joint_count(console: Console, joint_counts: list[int]) -> None:
    """Benchmark FK across every link in serial chains of increasing size."""

    table = Table(title="Forward Kinematics Across All Links", header_style="bold green")
    table.add_column("Links", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("Median (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("Best (ms)", justify="right")

    for joint_count in joint_counts:
        robot = build_chain_robot(joint_count)
        robot.set_joint_positions(
            {
                joint_name: 0.25 * np.cos((index + 1) * 0.3)
                for index, joint_name in enumerate(robot.joints)
            }
        )
        stats = sample_benchmark(runs=2_000, fn=robot.forward_kinematics)
        table.add_row(
            str(len(robot.links)),
            str(stats.runs),
            f"{stats.median_ms:.4f}",
            f"{stats.p95_ms:.4f}",
            f"{stats.min_ms:.4f}",
        )

    console.print(table)


def benchmark_simulation_step(console: Console) -> None:
    """Benchmark a full simulation step on the built-in humanoid world."""

    engine = ExecutionEngine(robot=build_humanoid_model(), world=WorldState.with_defaults())
    engine.robot.set_joint_positions(
        {
            "torso_yaw": -0.08,
            "right_shoulder_pitch": -0.9,
            "right_shoulder_yaw": 0.45,
            "right_elbow_pitch": 1.2,
        }
    )
    stats = sample_benchmark(runs=5_000, fn=engine.step)

    table = Table(title="Simulation Step Runtime", header_style="bold magenta")
    table.add_column("Workload")
    table.add_column("Runs", justify="right")
    table.add_column("Median (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("Best (ms)", justify="right")
    table.add_row(
        "ExecutionEngine.step()",
        str(stats.runs),
        f"{stats.median_ms:.4f}",
        f"{stats.p95_ms:.4f}",
        f"{stats.min_ms:.4f}",
    )
    console.print(table)


def main() -> None:
    """Run the benchmark suite."""

    console = Console()
    joint_counts = [4, 8, 12, 16]
    console.print("[bold]optisim performance benchmark[/bold]")
    benchmark_ik_by_joint_count(console, joint_counts)
    benchmark_fk_by_joint_count(console, joint_counts)
    benchmark_simulation_step(console)


if __name__ == "__main__":
    main()
