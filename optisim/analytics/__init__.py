"""Analytics helpers for simulation recordings."""

from optisim.analytics.optimizer import ParameterRange, SweepResult, composite_score, find_best, sweep_task
from optisim.analytics.trajectory import TrajectoryMetrics, analyze_trajectory, compare_trajectories

__all__ = [
    "ParameterRange",
    "SweepResult",
    "TrajectoryMetrics",
    "analyze_trajectory",
    "compare_trajectories",
    "composite_score",
    "find_best",
    "sweep_task",
]
