"""Trajectory export helpers."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import numpy as np

from optisim.export.formats import TrajectoryExport
from optisim.primitives import PrimitiveResult


class TrajectoryExporter:
    """Serialize and deserialize robot trajectories."""

    @staticmethod
    def from_primitive_results(
        results: list[PrimitiveResult],
        joint_names: list[str] | None = None,
        dt: float = 0.05,
        name: str = "trajectory",
    ) -> TrajectoryExport:
        """Concatenate primitive trajectories into a single export payload."""

        frames: list[np.ndarray] = []
        for result in results:
            for frame in result.joint_trajectory or []:
                frames.append(np.asarray(frame, dtype=float).copy())

        if joint_names is None:
            dof = len(frames[0]) if frames else 31
            joint_names = [f"j{index}" for index in range(dof)]
        else:
            joint_names = [str(joint_name) for joint_name in joint_names]

        timestamps = [float(index * dt) for index in range(len(frames))]
        metadata = {"source": "primitive_results", "dt": float(dt), "result_count": len(results)}
        return TrajectoryExport(
            name=str(name),
            joint_names=joint_names,
            frames=frames,
            timestamps=timestamps,
            metadata=metadata,
        )

    @staticmethod
    def to_json(export: TrajectoryExport) -> str:
        """Serialize a trajectory to JSON."""

        return json.dumps(TrajectoryExporter._to_dict(export), indent=2, sort_keys=True)

    @staticmethod
    def to_csv(export: TrajectoryExport) -> str:
        """Serialize a trajectory to CSV."""

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["timestamp", *export.joint_names])
        for timestamp, frame in zip(export.timestamps, export.frames):
            writer.writerow([f"{timestamp:.12g}", *[f"{float(value):.12g}" for value in frame]])
        return buffer.getvalue()

    @staticmethod
    def to_mocap_csv(export: TrajectoryExport) -> str:
        """Serialize a trajectory to a mocap-style CSV."""

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        header = ["frame", "time"]
        for joint_name in export.joint_names:
            header.extend([f"{joint_name}_x", f"{joint_name}_y", f"{joint_name}_z"])
        writer.writerow(header)

        for frame_number, (timestamp, frame) in enumerate(zip(export.timestamps, export.frames)):
            row: list[str | int] = [frame_number, f"{timestamp:.12g}"]
            for value in frame:
                pseudo_position = float(value)
                row.extend(
                    [
                        f"{pseudo_position:.12g}",
                        f"{pseudo_position:.12g}",
                        f"{pseudo_position:.12g}",
                    ]
                )
            writer.writerow(row)
        return buffer.getvalue()

    @staticmethod
    def to_ros2_bag_json(export: TrajectoryExport, topic: str = "/joint_states") -> str:
        """Serialize a trajectory to a ROS2-bag-friendly JSON list."""

        messages: list[dict[str, Any]] = []
        for timestamp, frame in zip(export.timestamps, export.frames):
            sec = int(timestamp)
            nanosec = int(round((timestamp - sec) * 1_000_000_000))
            if nanosec >= 1_000_000_000:
                sec += 1
                nanosec -= 1_000_000_000
            messages.append(
                {
                    "topic": topic,
                    "sec": sec,
                    "nanosec": nanosec,
                    "data": {
                        "name": list(export.joint_names),
                        "position": [float(value) for value in frame],
                        "velocity": [],
                        "effort": [],
                    },
                }
            )
        return json.dumps(messages, indent=2, sort_keys=True)

    @staticmethod
    def from_json(json_str: str) -> TrajectoryExport:
        """Deserialize a JSON trajectory export."""

        payload = json.loads(json_str)
        return TrajectoryExporter._from_dict(payload)

    @staticmethod
    def from_csv(csv_str: str) -> TrajectoryExport:
        """Deserialize a CSV trajectory export."""

        buffer = io.StringIO(csv_str)
        reader = csv.reader(buffer)
        rows = list(reader)
        if not rows:
            return TrajectoryExport(name="trajectory", joint_names=[], frames=[], timestamps=[], metadata={})

        header = rows[0]
        if not header:
            return TrajectoryExport(name="trajectory", joint_names=[], frames=[], timestamps=[], metadata={})
        joint_names = [str(name) for name in header[1:]]
        timestamps: list[float] = []
        frames: list[np.ndarray] = []
        for row in rows[1:]:
            if not row:
                continue
            timestamps.append(float(row[0]))
            frames.append(np.asarray([float(value) for value in row[1:]], dtype=float))
        return TrajectoryExport(
            name="trajectory",
            joint_names=joint_names,
            frames=frames,
            timestamps=timestamps,
            metadata={"source": "csv"},
        )

    @staticmethod
    def _to_dict(export: TrajectoryExport) -> dict[str, Any]:
        return {
            "name": export.name,
            "joint_names": list(export.joint_names),
            "frames": [frame.tolist() for frame in export.frames],
            "timestamps": [float(timestamp) for timestamp in export.timestamps],
            "metadata": _jsonify(export.metadata),
        }

    @staticmethod
    def _from_dict(payload: dict[str, Any]) -> TrajectoryExport:
        return TrajectoryExport(
            name=str(payload.get("name", "trajectory")),
            joint_names=[str(name) for name in payload.get("joint_names", [])],
            frames=[np.asarray(frame, dtype=float) for frame in payload.get("frames", [])],
            timestamps=[float(timestamp) for timestamp in payload.get("timestamps", [])],
            metadata=dict(payload.get("metadata", {})),
        )


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
