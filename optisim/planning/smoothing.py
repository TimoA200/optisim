"""Path post-processing helpers for sampled motion plans."""

from __future__ import annotations

from typing import Callable

import numpy as np

Config = dict[str, float]
SegmentValidator = Callable[[Config, Config], bool]


def shortcut_path(
    path: list[Config],
    *,
    is_segment_valid: SegmentValidator,
    max_iterations: int = 100,
    rng: np.random.Generator | None = None,
) -> list[Config]:
    """Shortcut a waypoint path by replacing valid subpaths with direct edges."""

    if len(path) <= 2 or max_iterations <= 0:
        return [dict(waypoint) for waypoint in path]

    rng = rng or np.random.default_rng(0)
    smoothed = [dict(waypoint) for waypoint in path]

    for _ in range(max_iterations):
        if len(smoothed) <= 2:
            break
        first, second = sorted(int(index) for index in rng.choice(len(smoothed), size=2, replace=False))
        if second - first <= 1:
            continue
        if not is_segment_valid(smoothed[first], smoothed[second]):
            continue
        smoothed = smoothed[: first + 1] + smoothed[second:]

    return smoothed

__all__ = ["Config", "SegmentValidator", "shortcut_path"]
