"""Callbacks for curriculum training workflows."""

from __future__ import annotations

from abc import ABC

from optisim.curriculum.scheduler import DifficultyLevel, TaskScheduler


class CurriculumCallback(ABC):
    def on_episode_end(self, result: object, scheduler: TaskScheduler) -> None:
        del result
        del scheduler

    def on_promote(self, from_diff: DifficultyLevel, to_diff: DifficultyLevel) -> None:
        del from_diff
        del to_diff

    def on_demote(self, from_diff: DifficultyLevel, to_diff: DifficultyLevel) -> None:
        del from_diff
        del to_diff


class LoggingCallback(CurriculumCallback):
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = bool(verbose)

    def on_episode_end(self, result: object, scheduler: TaskScheduler) -> None:
        del scheduler
        if not self.verbose:
            return
        print(
            "episode="
            f"{getattr(result, 'episode', '?')} task={getattr(result, 'task_name', '?')} "
            f"success={getattr(result, 'success', False)} partial={getattr(result, 'partial_success', 0.0):.3f} "
            f"difficulty={getattr(result, 'difficulty', '?')}"
        )


class HistoryCallback(CurriculumCallback):
    def __init__(self) -> None:
        self.episodes: list[object] = []
        self.promotions: list[tuple[int, DifficultyLevel, DifficultyLevel]] = []

    def on_episode_end(self, result: object, scheduler: TaskScheduler) -> None:
        del scheduler
        self.episodes.append(result)

    def on_promote(self, from_diff: DifficultyLevel, to_diff: DifficultyLevel) -> None:
        self.promotions.append((len(self.episodes), from_diff, to_diff))


class EarlyStopCallback(CurriculumCallback):
    def __init__(self, target_difficulty: DifficultyLevel, target_success_rate: float = 0.8) -> None:
        self.target_difficulty = target_difficulty
        self.target_success_rate = float(target_success_rate)
        self._stop_requested = False

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def on_episode_end(self, result: object, scheduler: TaskScheduler) -> None:
        del result
        if scheduler.current_difficulty.value < self.target_difficulty.value:
            return
        summary = scheduler.progress_summary()
        current_tasks = [
            details["recent_success_rate"]
            for details in summary["tasks"].values()
            if details["difficulty"] == self.target_difficulty.name and details["unlocked"]
        ]
        if current_tasks and float(sum(current_tasks) / len(current_tasks)) >= self.target_success_rate:
            self._stop_requested = True


__all__ = [
    "CurriculumCallback",
    "EarlyStopCallback",
    "HistoryCallback",
    "LoggingCallback",
]
