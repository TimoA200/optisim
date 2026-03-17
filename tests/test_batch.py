from __future__ import annotations

import builtins
import csv
import json
from concurrent.futures import TimeoutError
from pathlib import Path

import pytest

from optisim.batch import BatchConfig, BatchResult, BatchTaskResult, run_batch
from optisim.cli import build_parser, main
from optisim.core import TaskDefinition
from optisim.library import TaskCatalog


def _task(name: str = "batch_demo") -> TaskDefinition:
    return TaskDefinition.from_dict(
        {
            "name": name,
            "world": {
                "surfaces": [
                    {"name": "table", "pose": {"position": [0.55, 0.0, 0.74]}, "size": [0.90, 0.60, 0.05]}
                ],
                "objects": [
                    {"name": "box", "pose": {"position": [0.42, -0.12, 0.81]}, "size": [0.08, 0.08, 0.12]}
                ],
            },
            "actions": [
                {"type": "reach", "target": "box"},
                {"type": "grasp", "target": "box"},
                {"type": "move", "target": "box", "destination": [0.55, 0.0, 0.82]},
                {"type": "place", "target": "box", "support": "table"},
            ],
        }
    )


def test_run_batch_with_one_task_one_worker() -> None:
    result = run_batch(BatchConfig(tasks=[_task()], n_workers=1, include_dynamics=False))

    assert result.n_success == 1
    assert result.n_failed == 0
    assert len(result.results) == 1
    assert result.results[0].success is True
    assert result.results[0].step_count > 0


def test_run_batch_with_multiple_tasks_multiple_workers() -> None:
    result = run_batch(
        BatchConfig(
            tasks=[_task("alpha_batch"), _task("beta_batch")],
            n_workers=2,
            include_dynamics=False,
        )
    )

    assert len(result.results) == 2
    assert [item.task_name for item in result.results] == ["alpha_batch", "beta_batch"]
    assert all(item.success for item in result.results)


def test_batch_result_to_csv_writes_valid_csv(tmp_path: Path) -> None:
    result = BatchResult(
        results=[
            BatchTaskResult(
                task_name="csv_task",
                run_index=0,
                success=True,
                elapsed_seconds=1.25,
                step_count=4,
                collision_count=0,
                total_energy_j=0.5,
                validation_ok=True,
            )
        ],
        total_elapsed=1.25,
    )
    output = tmp_path / "results.csv"

    result.to_csv(output)

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["task_name"] == "csv_task"
    assert rows[0]["success"] == "True"


def test_batch_result_to_json_writes_valid_json(tmp_path: Path) -> None:
    result = BatchResult(
        results=[
            BatchTaskResult(
                task_name="json_task",
                run_index=0,
                success=False,
                elapsed_seconds=0.1,
                step_count=0,
                collision_count=0,
                total_energy_j=None,
                validation_ok=False,
                error="boom",
            )
        ],
        total_elapsed=0.1,
    )
    output = tmp_path / "results.json"

    result.to_json(output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["n_failed"] == 1
    assert payload["results"][0]["error"] == "boom"


def test_failed_task_is_captured_in_result_not_raised(tmp_path: Path) -> None:
    bad_task = tmp_path / "bad_task.yaml"
    bad_task.write_text("actions: []\n", encoding="utf-8")

    result = run_batch(BatchConfig(tasks=[bad_task], n_workers=1, include_dynamics=False))

    assert result.n_failed == 1
    assert result.results[0].success is False
    assert result.results[0].error


def test_timeout_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    from optisim.batch import runner as batch_runner

    class FakeFuture:
        def result(self, timeout: float | None = None) -> BatchTaskResult:
            raise TimeoutError()

        def cancel(self) -> bool:
            return True

    class FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def submit(self, fn, args):  # noqa: ANN001
            del fn, args
            return FakeFuture()

        def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
            del wait, cancel_futures

    monkeypatch.setattr(batch_runner, "ProcessPoolExecutor", FakeExecutor)

    result = run_batch(BatchConfig(tasks=[_task("slow_task")], n_workers=1, timeout_per_task=0.01))

    assert result.n_failed == 1
    assert result.results[0].error == "task timed out"


def test_repeat_produces_multiple_results() -> None:
    result = run_batch(BatchConfig(tasks=[_task("repeat_task")], n_workers=1, repeat=3, include_dynamics=False))

    assert len(result.results) == 3
    assert [item.run_index for item in result.results] == [0, 1, 2]


def test_summary_table_returns_non_empty_string() -> None:
    summary = BatchResult(
        results=[
            BatchTaskResult(
                task_name="summary_task",
                run_index=0,
                success=True,
                elapsed_seconds=0.2,
                step_count=3,
                collision_count=0,
                total_energy_j=None,
                validation_ok=True,
            )
        ],
        total_elapsed=0.2,
    ).summary_table()

    assert isinstance(summary, str)
    assert "summary_task" in summary
    assert "completed 1 runs" in summary


def test_batch_config_defaults_are_sensible() -> None:
    config = BatchConfig(tasks=[_task()])

    assert config.n_workers >= 1
    assert config.n_workers <= 4
    assert config.timeout_per_task == 60.0
    assert config.repeat == 1
    assert config.seed is None
    assert config.robot_spec is None
    assert config.output_dir is None
    assert config.include_dynamics is True
    assert config.include_grasp is False


def test_cli_batch_command_with_builtin_library_tasks(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["batch", "--workers", "1", "--repeat", "1"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "running batch with" in output
    assert "completed" in output


def test_to_dataframe_raises_import_error_when_pandas_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    result = BatchResult(results=[], total_elapsed=0.0)
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, A002
        if name == "pandas":
            raise ImportError("pandas missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pandas is required"):
        result.to_dataframe()


def test_output_dir_creates_recording_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "recordings"
    result = run_batch(
        BatchConfig(
            tasks=[_task("recorded_task")],
            n_workers=1,
            include_dynamics=False,
            output_dir=output_dir,
        )
    )

    assert result.results[0].recording_path is not None
    assert result.results[0].recording_path.exists()
    payload = json.loads(result.results[0].recording_path.read_text(encoding="utf-8"))
    assert payload["task_name"] == "recorded_task"


def test_batch_parser_accepts_subcommand() -> None:
    args = build_parser().parse_args(["batch", "examples/pick_and_place.yaml", "--workers", "2"])

    assert args.command == "batch"
    assert args.workers == 2
    assert args.task_files == [Path("examples/pick_and_place.yaml")]


def test_batch_no_task_files_uses_builtin_templates() -> None:
    catalog = TaskCatalog()
    result = run_batch(
        BatchConfig(
            tasks=[catalog.get(template.name) for template in catalog.list()[:2]],
            n_workers=1,
            include_dynamics=False,
        )
    )

    assert len(result.results) == 2
    assert all(item.success for item in result.results)

__all__ = ["test_run_batch_with_one_task_one_worker", "test_run_batch_with_multiple_tasks_multiple_workers", "test_batch_result_to_csv_writes_valid_csv", "test_batch_result_to_json_writes_valid_json", "test_failed_task_is_captured_in_result_not_raised", "test_timeout_handling", "test_repeat_produces_multiple_results", "test_summary_table_returns_non_empty_string", "test_batch_config_defaults_are_sensible", "test_cli_batch_command_with_builtin_library_tasks", "test_to_dataframe_raises_import_error_when_pandas_absent", "test_output_dir_creates_recording_files", "test_batch_parser_accepts_subcommand", "test_batch_no_task_files_uses_builtin_templates"]
