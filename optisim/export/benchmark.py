"""Benchmark export helpers."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import numpy as np

from optisim.benchmark import BenchmarkResult


class BenchmarkExporter:
    """Serialize benchmark result sets for external reporting."""

    @staticmethod
    def to_json(results: list[BenchmarkResult]) -> str:
        """Serialize benchmark results to JSON."""

        payload = [BenchmarkExporter._result_to_dict(result) for result in results]
        return json.dumps(payload, indent=2, sort_keys=True)

    @staticmethod
    def to_csv(results: list[BenchmarkResult]) -> str:
        """Serialize benchmark results to a CSV summary table."""

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["task_name", "difficulty", "success", "steps", "partial"])
        for result in results:
            writer.writerow(
                [
                    result.task_name,
                    str(result.metadata.get("difficulty", "unknown")),
                    "true" if result.success else "false",
                    result.elapsed_steps,
                    f"{result.partial_success:.6f}",
                ]
            )
        return buffer.getvalue()

    @staticmethod
    def to_markdown(results: list[BenchmarkResult]) -> str:
        """Serialize benchmark results to a GitHub-flavored Markdown table."""

        lines = [
            "| Task | Difficulty | Success | Steps | Partial |",
            "|------|-----------|---------|-------|---------|",
        ]
        successes = 0
        for result in results:
            successes += int(result.success)
            lines.append(
                "| "
                f"{result.task_name} | "
                f"{result.metadata.get('difficulty', 'unknown')} | "
                f"{'✓' if result.success else '✗'} | "
                f"{result.elapsed_steps} | "
                f"{result.partial_success:.2f} |"
            )
        total = len(results)
        rate = 0.0 if total == 0 else (successes / total) * 100.0
        lines.append(f"Success rate: {successes}/{total} ({rate:.1f}%)")
        return "\n".join(lines)

    @staticmethod
    def _result_to_dict(result: BenchmarkResult) -> dict[str, Any]:
        return {
            "task_name": result.task_name,
            "success": result.success,
            "partial_success": result.partial_success,
            "steps_completed": result.steps_completed,
            "steps_total": result.steps_total,
            "elapsed_steps": result.elapsed_steps,
            "error_message": result.error_message,
            "primitive_results": [BenchmarkExporter._primitive_to_dict(item) for item in result.primitive_results],
            "metadata": _jsonify(result.metadata),
        }

    @staticmethod
    def _primitive_to_dict(result: Any) -> dict[str, Any]:
        return {
            "status": result.status.value,
            "message": result.message,
            "duration_s": result.duration_s,
            "joint_trajectory": [np.asarray(frame, dtype=float).tolist() for frame in result.joint_trajectory or []],
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
