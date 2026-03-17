"""Export format definitions and containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from optisim.scene import SceneGraph


class ExportFormat(Enum):
    """Supported interchange formats for trajectories, scenes, and benchmarks."""

    JSON_TRAJ = "json_traj"
    CSV_TRAJ = "csv_traj"
    MOCAP_CSV = "mocap_csv"
    ROS2_BAG_JSON = "ros2_bag_json"
    SCENE_JSON = "scene_json"
    URDF_ANNOTATION = "urdf_annotation"


@dataclass(slots=True)
class TrajectoryExport:
    """Normalized trajectory export payload."""

    name: str
    joint_names: list[str]
    frames: list[np.ndarray]
    timestamps: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.joint_names = [str(name) for name in self.joint_names]
        self.frames = [np.asarray(frame, dtype=float).copy() for frame in self.frames]
        self.timestamps = [float(timestamp) for timestamp in self.timestamps]
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class SceneExport:
    """Composite export payload for a scene graph and optional robot trajectory."""

    scene_graph: SceneGraph
    robot_trajectory: TrajectoryExport | None
    task_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task_name = str(self.task_name)
        self.metadata = dict(self.metadata)
