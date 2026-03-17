"""Inter-robot collision checks for multi-robot coordination."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optisim.math3d import vec3
from optisim.robot import RobotModel


@dataclass(frozen=True, slots=True)
class InterRobotCollision:
    """Proximity report between links on two different robots."""

    robot_a: str
    link_a: str
    robot_b: str
    link_b: str
    distance: float


def inter_robot_collisions(
    robots: dict[str, RobotModel],
    *,
    threshold: float = 0.05,
) -> list[InterRobotCollision]:
    """Check pairwise link proximity between different robots."""

    names = sorted(robots)
    collisions: list[InterRobotCollision] = []
    for index, robot_a_name in enumerate(names):
        aabbs_a = robots[robot_a_name].link_aabbs()
        for robot_b_name in names[index + 1 :]:
            aabbs_b = robots[robot_b_name].link_aabbs()
            for link_a, (a_min, a_max) in aabbs_a.items():
                for link_b, (b_min, b_max) in aabbs_b.items():
                    distance = _aabb_distance(a_min, a_max, b_min, b_max)
                    if distance <= threshold:
                        collisions.append(
                            InterRobotCollision(
                                robot_a=robot_a_name,
                                link_a=link_a,
                                robot_b=robot_b_name,
                                link_b=link_b,
                                distance=distance,
                            )
                        )
    return collisions


def _aabb_distance(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
) -> float:
    gap = np.maximum(vec3([0.0, 0.0, 0.0]), np.maximum(a_min - b_max, b_min - a_max))
    return float(np.linalg.norm(gap))
