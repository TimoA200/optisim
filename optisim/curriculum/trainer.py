"""Curriculum-driven benchmark training helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optisim.benchmark import BenchmarkEvaluator, BenchmarkSuite
from optisim.curriculum.callbacks import CurriculumCallback
from optisim.curriculum.scheduler import DifficultyLevel, TaskScheduler


@dataclass(slots=True)
class CurriculumTrainerConfig:
    n_episodes: int = 100
    eval_every: int = 10
    eval_episodes: int = 5
    verbose: bool = True
    rng_seed: int = 42


@dataclass(slots=True)
class EpisodeResult:
    episode: int
    task_name: str
    success: bool
    partial_success: float
    difficulty: str


class CurriculumTrainer:
    """Execute benchmark tasks under curriculum scheduling."""

    def __init__(
        self,
        suite: BenchmarkSuite,
        scheduler: TaskScheduler,
        evaluator: BenchmarkEvaluator,
        config: CurriculumTrainerConfig | None = None,
    ) -> None:
        self.suite = suite
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.config = config or CurriculumTrainerConfig()
        self.callbacks: list[CurriculumCallback] = []
        self._history: list[EpisodeResult] = []
        self._rng = np.random.default_rng(self.config.rng_seed)
        self._eval_history: list[dict[str, float]] = []

    @property
    def history(self) -> list[EpisodeResult]:
        return list(self._history)

    def run_episode(self, task_name: str | None = None) -> EpisodeResult:
        selected_task = task_name or self.scheduler.sample_task(self._rng)
        benchmark_result = self.evaluator.run_task(self.suite.get(selected_task))
        previous_difficulty = self.scheduler.current_difficulty
        self.scheduler.record_result(selected_task, benchmark_result.success)
        episode_result = EpisodeResult(
            episode=len(self._history) + 1,
            task_name=benchmark_result.task_name,
            success=benchmark_result.success,
            partial_success=float(benchmark_result.partial_success),
            difficulty=str(benchmark_result.metadata.get("difficulty", previous_difficulty.name.lower())),
        )
        self._history.append(episode_result)
        for callback in self.callbacks:
            callback.on_episode_end(episode_result, self.scheduler)
        self._notify_transition(previous_difficulty, self.scheduler.current_difficulty)
        return episode_result

    def train(self, n_episodes: int | None = None) -> list[EpisodeResult]:
        total_episodes = self.config.n_episodes if n_episodes is None else int(n_episodes)
        for _ in range(total_episodes):
            self.run_episode()
            if self.config.eval_every > 0 and len(self._history) % self.config.eval_every == 0:
                self._eval_history.append(self.eval_current(self.config.eval_episodes))
            if any(getattr(callback, "stop_requested", False) for callback in self.callbacks):
                break
        return self.history

    def eval_current(self, n_episodes: int = 5) -> dict:
        episode_count = max(1, int(n_episodes))
        tasks = self.scheduler.available_tasks
        if not tasks:
            return {
                "difficulty": self.scheduler.current_difficulty.name,
                "episodes": 0,
                "success_rate": 0.0,
                "partial_success_rate": 0.0,
            }
        successes = 0
        partials: list[float] = []
        for _ in range(episode_count):
            task_name = self.scheduler.sample_task(self._rng)
            result = self.evaluator.run_task(self.suite.get(task_name))
            successes += int(result.success)
            partials.append(float(result.partial_success))
        return {
            "difficulty": self.scheduler.current_difficulty.name,
            "episodes": episode_count,
            "success_rate": float(successes / episode_count),
            "partial_success_rate": float(np.mean(partials)) if partials else 0.0,
        }

    def plot_progress(self) -> str:
        if not self._history:
            return "No curriculum history."
        window = min(10, len(self._history))
        rates = [
            float(np.mean([1.0 if entry.success else 0.0 for entry in self._history[max(0, index - window + 1) : index + 1]]))
            for index in range(len(self._history))
        ]
        rows = 8
        blocks = " .:-=+*#%@"
        lines: list[str] = []
        for row in range(rows, 0, -1):
            threshold = row / rows
            chars = []
            for rate in rates:
                level = min(len(blocks) - 1, max(0, int(round(rate * (len(blocks) - 1)))))
                chars.append(blocks[level] if rate >= threshold else " ")
            lines.append(f"{threshold:>4.2f} |{''.join(chars)}")
        lines.append("0.00 +" + "-" * len(rates))
        return "\n".join(lines)

    def _notify_transition(self, previous: DifficultyLevel, current: DifficultyLevel) -> None:
        if previous is current:
            return
        for callback in self.callbacks:
            if current.value > previous.value:
                callback.on_promote(previous, current)
            else:
                callback.on_demote(previous, current)


__all__ = [
    "CurriculumTrainer",
    "CurriculumTrainerConfig",
    "EpisodeResult",
]
