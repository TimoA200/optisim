from __future__ import annotations

import numpy as np
import pytest

from optisim import (
    BenchmarkResult,
    BenchmarkSuite,
    CurriculumConfig,
    CurriculumEarlyStopCallback,
    CurriculumHistoryCallback,
    CurriculumLoggingCallback,
    CurriculumTrainer,
    CurriculumTrainerConfig,
    DifficultyLevel,
    EpisodeResult,
    PrimitiveResult,
    PrimitiveStatus,
    TaskRecord,
    TaskScheduler,
)


def make_result(
    task_name: str,
    *,
    success: bool = True,
    partial_success: float = 1.0,
    difficulty: str = "easy",
) -> BenchmarkResult:
    primitive = PrimitiveResult(status=PrimitiveStatus.SUCCESS, message="ok", joint_trajectory=[np.zeros(3)], duration_s=0.1)
    return BenchmarkResult(
        task_name=task_name,
        success=success,
        partial_success=partial_success,
        steps_completed=1,
        steps_total=1,
        primitive_results=[primitive],
        elapsed_steps=1,
        error_message=None,
        metadata={"difficulty": difficulty, "tags": ["curriculum"]},
    )


class FakeEvaluator:
    def __init__(self, task_results: dict[str, list[tuple[bool, float]]] | None = None) -> None:
        self.task_results = {name: list(values) for name, values in (task_results or {}).items()}
        self.calls: list[str] = []

    def run_task(self, task: object, robot_joints: np.ndarray | None = None) -> BenchmarkResult:
        del robot_joints
        task_name = getattr(task, "name")
        difficulty = getattr(task, "difficulty")
        self.calls.append(task_name)
        outcomes = self.task_results.get(task_name, [(True, 1.0)])
        success, partial_success = outcomes.pop(0) if outcomes else (True, 1.0)
        return make_result(task_name, success=success, partial_success=partial_success, difficulty=difficulty)


def test_task_record_defaults() -> None:
    record = TaskRecord(task_name="pick_cup", difficulty=DifficultyLevel.EASY)
    assert record.attempts == 0
    assert record.successes == 0
    assert record.recent_success_rate == 0.0
    assert record.unlocked is True
    assert record.tags == []


def test_curriculum_config_defaults() -> None:
    config = CurriculumConfig()
    assert config.promote_threshold == 0.7
    assert config.demote_threshold == 0.3
    assert config.window_size == 10
    assert config.initial_difficulty is DifficultyLevel.EASY
    assert config.unlock_on_promote is True


def test_task_scheduler_initialization_with_default_suite() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    assert scheduler.current_difficulty is DifficultyLevel.EASY


def test_available_tasks_returns_only_easy_tasks_initially() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    expected = BenchmarkSuite.DEFAULT.list_tasks(difficulty="easy")
    assert scheduler.available_tasks == expected


def test_sample_task_returns_valid_task_name() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    assert scheduler.sample_task(np.random.default_rng(0)) in scheduler.available_tasks


def test_record_result_increments_attempts() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    task_name = scheduler.available_tasks[0]
    scheduler.record_result(task_name, success=False)
    assert scheduler.get_record(task_name).attempts == 1


def test_success_rate_updates_after_records() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    task_name = scheduler.available_tasks[0]
    scheduler.record_result(task_name, success=True)
    scheduler.record_result(task_name, success=False)
    assert scheduler.get_record(task_name).recent_success_rate == 0.5


def test_should_promote_returns_true_when_threshold_met() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT, CurriculumConfig(promote_threshold=0.6))
    for task_name in scheduler.available_tasks:
        scheduler.record_result(task_name, success=True)
    assert scheduler.current_difficulty is DifficultyLevel.MEDIUM


def test_promote_moves_to_next_difficulty() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    assert scheduler.promote() is True
    assert scheduler.current_difficulty is DifficultyLevel.MEDIUM


def test_demote_moves_to_previous_difficulty() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    scheduler.promote()
    assert scheduler.demote() is True
    assert scheduler.current_difficulty is DifficultyLevel.EASY


def test_demote_at_easy_returns_false() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    assert scheduler.demote() is False


def test_promote_at_hard_returns_false() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    scheduler.promote()
    scheduler.promote()
    assert scheduler.promote() is False


def test_unlocked_tasks_at_higher_difficulty_after_promotion() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    medium_task = BenchmarkSuite.DEFAULT.list_tasks(difficulty="medium")[0]
    assert scheduler.get_record(medium_task).unlocked is False
    scheduler.promote()
    assert scheduler.get_record(medium_task).unlocked is True


def test_progress_summary_keys_present() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    summary = scheduler.progress_summary()
    assert set(summary) == {"current_difficulty", "tasks", "overall_success_rate"}


def test_overall_success_rate_between_zero_and_one() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    scheduler.record_result(scheduler.available_tasks[0], success=True)
    rate = scheduler.progress_summary()["overall_success_rate"]
    assert 0.0 <= rate <= 1.0


def test_curriculum_trainer_run_episode_returns_episode_result() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    trainer = CurriculumTrainer(BenchmarkSuite.DEFAULT, scheduler, FakeEvaluator())
    result = trainer.run_episode(scheduler.available_tasks[0])
    assert isinstance(result, EpisodeResult)


def test_episode_result_difficulty_is_a_string() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    trainer = CurriculumTrainer(BenchmarkSuite.DEFAULT, scheduler, FakeEvaluator())
    result = trainer.run_episode(scheduler.available_tasks[0])
    assert isinstance(result.difficulty, str)


def test_curriculum_trainer_train_returns_list_of_results() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    trainer = CurriculumTrainer(BenchmarkSuite.DEFAULT, scheduler, FakeEvaluator())
    results = trainer.train(3)
    assert isinstance(results, list)


def test_train_result_length_matches_n_episodes() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    trainer = CurriculumTrainer(BenchmarkSuite.DEFAULT, scheduler, FakeEvaluator())
    assert len(trainer.train(4)) == 4


def test_eval_current_returns_dict_with_success_rate() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    trainer = CurriculumTrainer(BenchmarkSuite.DEFAULT, scheduler, FakeEvaluator())
    result = trainer.eval_current(3)
    assert "success_rate" in result


def test_plot_progress_returns_non_empty_string() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    trainer = CurriculumTrainer(BenchmarkSuite.DEFAULT, scheduler, FakeEvaluator())
    trainer.train(3)
    assert trainer.plot_progress()


def test_history_callback_records_episodes() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    trainer = CurriculumTrainer(BenchmarkSuite.DEFAULT, scheduler, FakeEvaluator())
    callback = CurriculumHistoryCallback()
    trainer.callbacks.append(callback)
    trainer.train(2)
    assert len(callback.episodes) == 2


def test_history_callback_records_promotions() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT, CurriculumConfig(promote_threshold=0.5, window_size=1))
    trainer = CurriculumTrainer(BenchmarkSuite.DEFAULT, scheduler, FakeEvaluator())
    callback = CurriculumHistoryCallback()
    trainer.callbacks.append(callback)
    for task_name in BenchmarkSuite.DEFAULT.list_tasks(difficulty="easy"):
        trainer.run_episode(task_name)
    assert callback.promotions


def test_early_stop_callback_stop_requested_false_initially() -> None:
    callback = CurriculumEarlyStopCallback(DifficultyLevel.MEDIUM)
    assert callback.stop_requested is False


def test_early_stop_callback_triggers_after_target_reached() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT, CurriculumConfig(promote_threshold=0.5, window_size=1))
    trainer = CurriculumTrainer(BenchmarkSuite.DEFAULT, scheduler, FakeEvaluator())
    callback = CurriculumEarlyStopCallback(DifficultyLevel.MEDIUM, target_success_rate=1.0)
    trainer.callbacks.append(callback)
    for task_name in BenchmarkSuite.DEFAULT.list_tasks(difficulty="easy"):
        trainer.run_episode(task_name)
    for task_name in BenchmarkSuite.DEFAULT.list_tasks(difficulty="medium"):
        trainer.run_episode(task_name)
    assert callback.stop_requested is True


def test_logging_callback_does_not_raise() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    trainer = CurriculumTrainer(BenchmarkSuite.DEFAULT, scheduler, FakeEvaluator())
    trainer.callbacks.append(CurriculumLoggingCallback(verbose=True))
    trainer.run_episode(scheduler.available_tasks[0])


def test_multiple_callbacks_can_be_attached() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    trainer = CurriculumTrainer(BenchmarkSuite.DEFAULT, scheduler, FakeEvaluator())
    trainer.callbacks.extend([CurriculumLoggingCallback(verbose=False), CurriculumHistoryCallback()])
    trainer.train(2)
    assert len(trainer.callbacks) == 2


def test_curriculum_trainer_config_defaults() -> None:
    config = CurriculumTrainerConfig()
    assert config.n_episodes == 100
    assert config.eval_every == 10
    assert config.eval_episodes == 5
    assert config.verbose is True
    assert config.rng_seed == 42


def test_task_scheduler_all_records_returns_dict() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    assert isinstance(scheduler.all_records(), dict)


def test_get_record_returns_task_record() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    assert isinstance(scheduler.get_record(scheduler.available_tasks[0]), TaskRecord)


def test_rng_seed_produces_deterministic_sampling() -> None:
    scheduler_a = TaskScheduler(BenchmarkSuite.DEFAULT)
    scheduler_b = TaskScheduler(BenchmarkSuite.DEFAULT)
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    seq_a = [scheduler_a.sample_task(rng_a) for _ in range(5)]
    seq_b = [scheduler_b.sample_task(rng_b) for _ in range(5)]
    assert seq_a == seq_b


def test_window_size_one_rapid_response() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT, CurriculumConfig(window_size=1))
    task_name = scheduler.available_tasks[0]
    scheduler.record_result(task_name, success=True)
    scheduler.record_result(task_name, success=False)
    assert scheduler.get_record(task_name).recent_success_rate == 0.0


def test_should_demote_returns_true_when_threshold_met() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT, CurriculumConfig(demote_threshold=0.4, window_size=1))
    scheduler.promote()
    for task_name in BenchmarkSuite.DEFAULT.list_tasks(difficulty="medium"):
        scheduler.record_result(task_name, success=False)
    assert scheduler.current_difficulty is DifficultyLevel.EASY


def test_scheduler_initially_locks_hard_tasks() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    hard_task = BenchmarkSuite.DEFAULT.list_tasks(difficulty="hard")[0]
    assert scheduler.get_record(hard_task).unlocked is False


def test_progress_summary_includes_task_entries() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    summary = scheduler.progress_summary()
    assert scheduler.available_tasks[0] in summary["tasks"]


def test_sample_task_without_rng_returns_valid_name() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    assert scheduler.sample_task() in scheduler.available_tasks


def test_record_result_updates_successes() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    task_name = scheduler.available_tasks[0]
    scheduler.record_result(task_name, success=True)
    assert scheduler.get_record(task_name).successes == 1


def test_trainer_history_property_returns_episode_results() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    trainer = CurriculumTrainer(BenchmarkSuite.DEFAULT, scheduler, FakeEvaluator())
    trainer.train(2)
    assert all(isinstance(result, EpisodeResult) for result in trainer.history)


def test_eval_current_reports_episode_count() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    trainer = CurriculumTrainer(BenchmarkSuite.DEFAULT, scheduler, FakeEvaluator())
    result = trainer.eval_current(4)
    assert result["episodes"] == 4


def test_trainer_respects_rng_seed_for_sampling_order() -> None:
    trainer_a = CurriculumTrainer(
        BenchmarkSuite.DEFAULT,
        TaskScheduler(BenchmarkSuite.DEFAULT),
        FakeEvaluator(),
        CurriculumTrainerConfig(rng_seed=7),
    )
    trainer_b = CurriculumTrainer(
        BenchmarkSuite.DEFAULT,
        TaskScheduler(BenchmarkSuite.DEFAULT),
        FakeEvaluator(),
        CurriculumTrainerConfig(rng_seed=7),
    )
    seq_a = [trainer_a.run_episode().task_name for _ in range(3)]
    seq_b = [trainer_b.run_episode().task_name for _ in range(3)]
    assert seq_a == seq_b


def test_train_can_stop_early_from_callback() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT, CurriculumConfig(promote_threshold=0.5, window_size=1))
    trainer = CurriculumTrainer(BenchmarkSuite.DEFAULT, scheduler, FakeEvaluator())
    trainer.callbacks.append(CurriculumEarlyStopCallback(DifficultyLevel.MEDIUM, target_success_rate=1.0))
    results = trainer.train(10)
    assert len(results) < 10


def test_available_tasks_after_promotion_match_medium_tasks() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    scheduler.promote()
    assert scheduler.available_tasks == BenchmarkSuite.DEFAULT.list_tasks(difficulty="medium")


def test_all_records_contains_default_suite_tasks() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    assert set(BenchmarkSuite.DEFAULT.tasks).issubset(set(scheduler.all_records()))


def test_get_record_unknown_task_raises_key_error() -> None:
    scheduler = TaskScheduler(BenchmarkSuite.DEFAULT)
    with pytest.raises(KeyError):
        scheduler.get_record("missing_task")
