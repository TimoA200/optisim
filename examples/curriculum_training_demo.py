"""Showcase curriculum-driven benchmark scheduling and progress tracking."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optisim.benchmark import BenchmarkEvaluator, BenchmarkSuite
from optisim.curriculum import (
    CurriculumConfig,
    CurriculumTrainer,
    CurriculumTrainerConfig,
    DifficultyLevel,
    EarlyStopCallback,
    HistoryCallback,
    TaskScheduler,
)


def print_schedule(results) -> None:
    """Print the tasks sampled during training."""

    print("Scheduled tasks:")
    for result in results:
        print(
            f"  episode {result.episode:02d}: task={result.task_name} "
            f"difficulty={result.difficulty} success={result.success}"
        )


def print_promotions(history_callback: HistoryCallback) -> None:
    """Print promotion events captured by the history callback."""

    print("Promotions:")
    if not history_callback.promotions:
        print("  none")
        return
    for episode, from_diff, to_diff in history_callback.promotions:
        print(f"  episode {episode:02d}: {from_diff.name} -> {to_diff.name}")


def main() -> None:
    """Train across the built-in benchmark curriculum and summarize progress."""

    suite = BenchmarkSuite.DEFAULT
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
        evaluator=BenchmarkEvaluator(),
        config=CurriculumTrainerConfig(
            n_episodes=30,
            eval_every=0,
            rng_seed=42,
            verbose=False,
        ),
    )

    history_callback = HistoryCallback()
    early_stop = EarlyStopCallback(DifficultyLevel.HARD, target_success_rate=0.8)
    trainer.callbacks.extend([history_callback, early_stop])

    results = trainer.train()
    summary = scheduler.progress_summary()

    print("optisim.curriculum demo")
    print("=======================")
    print()
    print_schedule(results)
    print()
    print_promotions(history_callback)
    print()
    print("ASCII progress chart:")
    print(trainer.plot_progress())
    print()
    print("Final progress summary:")
    print(f"  episodes_run: {len(results)}")
    print(f"  current_difficulty: {summary['current_difficulty']}")
    print(f"  overall_success_rate: {summary['overall_success_rate']:.0%}")
    print(f"  early_stop_triggered: {early_stop.stop_requested}")


if __name__ == "__main__":
    main()
