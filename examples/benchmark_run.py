"""Run the default benchmark suite and export a Markdown report."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optisim.benchmark import BenchmarkEvaluator, BenchmarkReporter, BenchmarkSuite
from optisim.export import BenchmarkExporter


def main() -> None:
    """Execute all built-in benchmark tasks and write a Markdown summary."""

    suite = BenchmarkSuite.DEFAULT
    evaluator = BenchmarkEvaluator()
    reporter = BenchmarkReporter()

    results = evaluator.run_suite(suite)
    summary = reporter.summary(results)
    markdown = BenchmarkExporter.to_markdown(results)

    output_path = Path(__file__).with_name("benchmark_results.md")
    output_path.write_text(markdown + "\n", encoding="utf-8")

    print("optisim.benchmark + optisim.export demo")
    print("======================================")
    print()
    print(reporter.format_table(results))
    print()
    print("Summary:")
    print(
        f"  success_rate: {summary['succeeded']}/{summary['total']} "
        f"({summary['success_rate']:.0%})"
    )
    for difficulty, details in sorted(summary["by_difficulty"].items()):
        print(
            f"  {difficulty}: {details['succeeded']}/{details['total']} "
            f"({details['success_rate']:.0%})"
        )
    print()
    print(f"Markdown report written to {output_path}")


if __name__ == "__main__":
    main()
