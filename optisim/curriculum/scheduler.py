"""Curriculum learning task scheduling utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from optisim.benchmark import BenchmarkSuite


class DifficultyLevel(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3

    @classmethod
    def from_task_difficulty(cls, difficulty: str) -> "DifficultyLevel":
        try:
            return cls[difficulty.upper()]
        except KeyError as exc:
            raise ValueError(f"unsupported task difficulty {difficulty!r}") from exc

    def to_task_difficulty(self) -> str:
        return self.name.lower()


@dataclass(slots=True)
class TaskRecord:
    task_name: str
    difficulty: DifficultyLevel
    attempts: int = 0
    successes: int = 0
    recent_success_rate: float = 0.0
    unlocked: bool = True
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CurriculumConfig:
    promote_threshold: float = 0.7
    demote_threshold: float = 0.3
    window_size: int = 10
    initial_difficulty: DifficultyLevel = DifficultyLevel.EASY
    unlock_on_promote: bool = True


class TaskScheduler:
    """Schedule benchmark tasks by progressive difficulty."""

    def __init__(self, suite: BenchmarkSuite, config: CurriculumConfig | None = None) -> None:
        self.suite = suite
        self.config = config or CurriculumConfig()
        self._current_difficulty = self.config.initial_difficulty
        self._records: dict[str, TaskRecord] = {}
        self._recent_outcomes: dict[str, list[bool]] = {}

        for task_name, task in sorted(self.suite.tasks.items()):
            difficulty = DifficultyLevel.from_task_difficulty(task.difficulty)
            unlocked = difficulty.value <= self.config.initial_difficulty.value
            self._records[task_name] = TaskRecord(
                task_name=task_name,
                difficulty=difficulty,
                unlocked=unlocked,
                tags=list(task.tags),
            )
            self._recent_outcomes[task_name] = []

    @property
    def current_difficulty(self) -> DifficultyLevel:
        return self._current_difficulty

    @property
    def available_tasks(self) -> list[str]:
        return [
            name
            for name, record in sorted(self._records.items())
            if record.difficulty is self._current_difficulty and record.unlocked
        ]

    def sample_task(self, rng: np.random.Generator | None = None) -> str:
        tasks = self.available_tasks
        if not tasks:
            raise RuntimeError("no curriculum tasks are available to sample")
        generator = np.random.default_rng() if rng is None else rng
        index = int(generator.integers(0, len(tasks)))
        return tasks[index]

    def record_result(self, task_name: str, success: bool) -> None:
        record = self.get_record(task_name)
        record.attempts += 1
        if success:
            record.successes += 1

        outcomes = self._recent_outcomes[task_name]
        outcomes.append(bool(success))
        if len(outcomes) > self.config.window_size:
            del outcomes[0]
        record.recent_success_rate = float(np.mean(outcomes)) if outcomes else 0.0

        if self.should_promote():
            self.promote()
        elif self.should_demote():
            self.demote()

    def promote(self) -> bool:
        if self._current_difficulty is DifficultyLevel.HARD:
            return False
        to_diff = DifficultyLevel(self._current_difficulty.value + 1)
        self._current_difficulty = to_diff
        if self.config.unlock_on_promote:
            for record in self._records.values():
                if record.difficulty is to_diff:
                    record.unlocked = True
        return True

    def demote(self) -> bool:
        if self._current_difficulty is DifficultyLevel.EASY:
            return False
        self._current_difficulty = DifficultyLevel(self._current_difficulty.value - 1)
        return True

    def should_promote(self) -> bool:
        records = self._active_records(self._current_difficulty)
        if not records or any(record.attempts == 0 for record in records):
            return False
        return self._mean_success_rate(records) >= self.config.promote_threshold

    def should_demote(self) -> bool:
        records = self._active_records(self._current_difficulty)
        if not records or any(record.attempts == 0 for record in records):
            return False
        return self._mean_success_rate(records) <= self.config.demote_threshold

    def get_record(self, task_name: str) -> TaskRecord:
        try:
            return self._records[task_name]
        except KeyError as exc:
            raise KeyError(f"unknown curriculum task {task_name!r}") from exc

    def all_records(self) -> dict[str, TaskRecord]:
        return dict(self._records)

    def progress_summary(self) -> dict:
        overall_attempts = sum(record.attempts for record in self._records.values())
        overall_successes = sum(record.successes for record in self._records.values())
        overall_rate = float(overall_successes / overall_attempts) if overall_attempts else 0.0
        tasks = {
            name: {
                "difficulty": record.difficulty.name,
                "attempts": record.attempts,
                "successes": record.successes,
                "recent_success_rate": record.recent_success_rate,
                "unlocked": record.unlocked,
                "tags": list(record.tags),
            }
            for name, record in sorted(self._records.items())
        }
        return {
            "current_difficulty": self._current_difficulty.name,
            "tasks": tasks,
            "overall_success_rate": overall_rate,
        }

    def _active_records(self, difficulty: DifficultyLevel) -> list[TaskRecord]:
        return [record for record in self._records.values() if record.difficulty is difficulty and record.unlocked]

    @staticmethod
    def _mean_success_rate(records: list[TaskRecord]) -> float:
        return float(np.mean([record.recent_success_rate for record in records]))


__all__ = [
    "CurriculumConfig",
    "DifficultyLevel",
    "TaskRecord",
    "TaskScheduler",
]
