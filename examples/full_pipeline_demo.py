"""Show the full optisim stack working together in one compact script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optisim.benchmark import BenchmarkEvaluator, BenchmarkReporter, BenchmarkSuite
from optisim.curriculum import CurriculumConfig, CurriculumTrainer, CurriculumTrainerConfig, DifficultyLevel, TaskScheduler
from optisim.export import TrajectoryExporter
from optisim.primitives import PrimitiveExecutor
from optisim.scene import SceneBuilder, SceneQuery
from optisim.tamp import Predicate


def make_pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def summarize_surface(scene, surface_id: str) -> str:
    contents = SceneQuery.find_on_surface(scene, surface_id)
    names = ", ".join(node.id for node in contents) if contents else "nothing"
    return f"{surface_id}: {names}"


def main() -> None:
    """Run an end-to-end walkthrough spanning scenes, TAMP, primitives, benchmarks, and export."""

    print("optisim full pipeline demo")
    print("=========================")
    print()

    # 1. Build a kitchen scene and add a robot in front of the table.
    scene = SceneBuilder.build_kitchen()
    SceneBuilder.add_robot(scene, pose=make_pose(0.2, 0.0, 0.75))
    print("1. Scene built")
    print(f"   nodes: {len(scene.nodes)}")
    print(f"   relations: {len(scene.relations)}")
    print()

    # 2. Query the scene for graspables and supported objects.
    graspable = [node.id for node in SceneQuery.find_graspable(scene)]
    surfaces = [node.id for node in SceneQuery.find_by_category(scene, "surface")]
    print("2. Scene queries")
    print(f"   graspable objects: {', '.join(graspable)}")
    for surface_id in surfaces:
        print(f"   {summarize_surface(scene, surface_id)}")
    print()

    # 3. Convert semantic relations into TAMP-style predicates for symbolic planning.
    tamp_predicates = [
        Predicate(item["name"], [str(arg) for arg in item["args"]], bool(item["value"]))
        for item in SceneQuery.to_tamp_predicates(scene)
    ]
    print("3. TAMP predicates")
    print(f"   total predicates: {len(tamp_predicates)}")
    for predicate in tamp_predicates[:5]:
        print(f"   {predicate}")
    print()

    # 4-5. Execute a full primitive chain and report each result in order.
    sequence = [
        {"primitive": "navigate", "params": {"target_id": "cup"}},
        {"primitive": "reach", "params": {"target_id": "cup", "end_effector": "right"}},
        {"primitive": "grasp", "params": {"target_id": "cup", "end_effector": "right", "grasp_force": 12.0}},
        {"primitive": "place", "params": {"object_id": "cup", "surface_id": "countertop"}},
    ]
    executor = PrimitiveExecutor()
    primitive_results = executor.execute_sequence(
        scene=scene,
        robot_id="humanoid",
        robot_joints=np.zeros(31, dtype=float),
        sequence=sequence,
    )

    print("4. Primitive execution")
    for index, (step, result) in enumerate(zip(sequence, primitive_results), start=1):
        print(
            f"   step {index}: {step['primitive']:<8} status={result.status.value:<7} "
            f"traj={len(result.joint_trajectory or []):>2} duration={result.duration_s:.2f}s"
        )
        print(f"     message: {result.message}")
    print()

    # 6. Run the default benchmark suite and print a compact table.
    suite = BenchmarkSuite.DEFAULT
    benchmark_evaluator = BenchmarkEvaluator()
    benchmark_reporter = BenchmarkReporter()
    benchmark_results = benchmark_evaluator.run_suite(suite)
    benchmark_summary = benchmark_reporter.summary(benchmark_results)

    print("5. Default benchmark suite")
    print(benchmark_reporter.format_table(benchmark_results))
    print(
        f"   summary: {benchmark_summary['succeeded']}/{benchmark_summary['total']} tasks succeeded "
        f"({benchmark_summary['success_rate']:.0%})"
    )
    print()

    # 7. Run a short deterministic curriculum training loop.
    scheduler = TaskScheduler(
        suite,
        CurriculumConfig(
            promote_threshold=0.5,
            window_size=1,
            initial_difficulty=DifficultyLevel.EASY,
        ),
    )
    trainer = CurriculumTrainer(
        suite=suite,
        scheduler=scheduler,
        evaluator=benchmark_evaluator,
        config=CurriculumTrainerConfig(n_episodes=20, eval_every=0, verbose=False, rng_seed=7),
    )
    curriculum_history = trainer.train()
    progress = scheduler.progress_summary()

    print("6. Curriculum training")
    print(f"   episodes: {len(curriculum_history)}")
    print(f"   final difficulty: {progress['current_difficulty']}")
    print(f"   success rate: {progress['overall_success_rate']:.0%}")
    print("   progress chart:")
    print(trainer.plot_progress())
    print()

    # 8. Export the primitive trajectory to JSON and preview the first frames.
    trajectory = TrajectoryExporter.from_primitive_results(primitive_results, name="full_pipeline_demo")
    trajectory_json = TrajectoryExporter.to_json(trajectory)
    output_path = Path("/tmp/optisim_full_pipeline_demo_trajectory.json")
    output_path.write_text(trajectory_json + "\n", encoding="utf-8")
    payload = json.loads(trajectory_json)

    print("7. Trajectory export")
    print(f"   json path: {output_path}")
    print(f"   frame count: {len(payload['frames'])}")
    for frame_index, frame in enumerate(payload["frames"][:3]):
        print(f"   frame {frame_index}: {frame[:6]}")
    print()

    # 9. Print one final summary block for the full walkthrough.
    final_surface = SceneQuery.find_on_surface(scene, "countertop")
    print("8. Summary")
    print(f"   robot added to kitchen scene: {'humanoid' in scene.nodes}")
    print(f"   graspable count: {len(graspable)}")
    print(f"   TAMP predicates generated: {len(tamp_predicates)}")
    print(f"   primitive successes: {sum(result.status.value == 'success' for result in primitive_results)}/{len(sequence)}")
    print(f"   benchmarks run: {len(benchmark_results)}")
    print(f"   curriculum episodes: {len(curriculum_history)}")
    print(f"   exported frames previewed: {min(3, len(payload['frames']))}")
    print(f"   cup now on countertop: {any(node.id == 'cup' for node in final_surface)}")


if __name__ == "__main__":
    main()
