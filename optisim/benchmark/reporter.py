"""Reporting helpers for benchmark execution results."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import numpy as np

from optisim.benchmark.evaluator import BenchmarkResult


class BenchmarkReporter:
    """Aggregate and serialize benchmark results."""

    def summary(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Return an aggregate benchmark summary."""

        total = len(results)
        succeeded = sum(1 for result in results if result.success)
        success_rate = float(succeeded / total) if total else 0.0
        avg_partial_success = float(sum(result.partial_success for result in results) / total) if total else 0.0

        by_difficulty: dict[str, dict[str, Any]] = {}
        for result in results:
            difficulty = str(result.metadata.get("difficulty", "unknown"))
            group = by_difficulty.setdefault(
                difficulty,
                {"total": 0, "succeeded": 0, "success_rate": 0.0, "avg_partial_success": 0.0},
            )
            group["total"] += 1
            group["succeeded"] += int(result.success)
            group["avg_partial_success"] += float(result.partial_success)
        for group in by_difficulty.values():
            if group["total"]:
                group["success_rate"] = float(group["succeeded"] / group["total"])
                group["avg_partial_success"] = float(group["avg_partial_success"] / group["total"])

        by_task = [
            {
                "task_name": result.task_name,
                "difficulty": result.metadata.get("difficulty", "unknown"),
                "success": result.success,
                "partial_success": result.partial_success,
                "steps_completed": result.steps_completed,
                "steps_total": result.steps_total,
                "elapsed_steps": result.elapsed_steps,
            }
            for result in results
        ]

        return {
            "total": total,
            "succeeded": succeeded,
            "success_rate": success_rate,
            "avg_partial_success": avg_partial_success,
            "by_difficulty": by_difficulty,
            "by_task": by_task,
        }

    def format_table(self, results: list[BenchmarkResult]) -> str:
        """Format results as a plain-text ASCII table."""

        headers = ["Task", "Diff", "Success", "Steps", "Partial"]
        rows = [
            [
                result.task_name,
                str(result.metadata.get("difficulty", "unknown")),
                "yes" if result.success else "no",
                str(result.elapsed_steps),
                f"{result.partial_success:.2f}",
            ]
            for result in results
        ]
        widths = [
            max([len(header), *[len(row[index]) for row in rows]] if rows else [len(header)])
            for index, header in enumerate(headers)
        ]
        header_row = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
        separator_row = "|-" + "-|-".join("-" * widths[index] for index in range(len(headers))) + "-|"
        body_rows = [
            "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |" for row in rows
        ]
        return "\n".join([header_row, separator_row, *body_rows])

    def to_json(self, results: list[BenchmarkResult]) -> str:
        """Serialize results to a JSON string."""

        payload = {
            "summary": self.summary(results),
            "results": [self._result_to_dict(result) for result in results],
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def to_csv(self, results: list[BenchmarkResult]) -> str:
        """Serialize results to a CSV string."""

        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                "task_name",
                "difficulty",
                "success",
                "partial_success",
                "steps_completed",
                "steps_total",
                "elapsed_steps",
                "error_message",
                "tags",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "task_name": result.task_name,
                    "difficulty": result.metadata.get("difficulty", "unknown"),
                    "success": result.success,
                    "partial_success": f"{result.partial_success:.6f}",
                    "steps_completed": result.steps_completed,
                    "steps_total": result.steps_total,
                    "elapsed_steps": result.elapsed_steps,
                    "error_message": result.error_message or "",
                    "tags": "|".join(str(tag) for tag in result.metadata.get("tags", [])),
                }
            )
        return buffer.getvalue()

    def _result_to_dict(self, result: BenchmarkResult) -> dict[str, Any]:
        return {
            "task_name": result.task_name,
            "success": result.success,
            "partial_success": result.partial_success,
            "steps_completed": result.steps_completed,
            "steps_total": result.steps_total,
            "elapsed_steps": result.elapsed_steps,
            "error_message": result.error_message,
            "primitive_results": [self._primitive_result_to_dict(item) for item in result.primitive_results],
            "metadata": _jsonify(result.metadata),
        }

    def _primitive_result_to_dict(self, result: Any) -> dict[str, Any]:
        return {
            "status": result.status.value,
            "message": result.message,
            "duration_s": result.duration_s,
            "trajectory_steps": len(result.joint_trajectory or []),
            "metadata": _jsonify(result.metadata),
        }


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = ["BenchmarkReporter"]
