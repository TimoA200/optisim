"""Simulation telemetry recording, replay, and analytics helpers."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _as_float_vector(name: str, value: Any, expected_size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if expected_size is not None and array.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},)")
    return array.copy()


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("base_pose quaternion must have non-zero norm")
    return quaternion / norm


@dataclass(slots=True)
class TelemetryFrame:
    """State snapshot for a single simulation timestep."""

    timestamp: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    base_pose: np.ndarray
    contact_states: dict[str, bool]
    extras: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp)
        self.joint_positions = _as_float_vector("joint_positions", self.joint_positions)
        self.joint_velocities = _as_float_vector("joint_velocities", self.joint_velocities)
        self.joint_torques = _as_float_vector("joint_torques", self.joint_torques)
        self.base_pose = _as_float_vector("base_pose", self.base_pose, expected_size=7)
        if self.joint_positions.shape != self.joint_velocities.shape:
            raise ValueError("joint_positions and joint_velocities must have matching shapes")
        if self.joint_positions.shape != self.joint_torques.shape:
            raise ValueError("joint_positions and joint_torques must have matching shapes")
        self.base_pose[3:] = _normalize_quaternion(self.base_pose[3:])
        self.contact_states = {str(name): bool(value) for name, value in self.contact_states.items()}
        self.extras = {str(name): float(value) for name, value in self.extras.items()}

    def to_dict(self) -> dict[str, Any]:
        """Convert the frame into JSON-serializable data."""

        return {
            "timestamp": self.timestamp,
            "joint_positions": self.joint_positions.tolist(),
            "joint_velocities": self.joint_velocities.tolist(),
            "joint_torques": self.joint_torques.tolist(),
            "base_pose": self.base_pose.tolist(),
            "contact_states": dict(self.contact_states),
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TelemetryFrame":
        """Build a frame from its serialized representation."""

        return cls(
            timestamp=float(d["timestamp"]),
            joint_positions=np.asarray(d["joint_positions"], dtype=float),
            joint_velocities=np.asarray(d["joint_velocities"], dtype=float),
            joint_torques=np.asarray(d["joint_torques"], dtype=float),
            base_pose=np.asarray(d["base_pose"], dtype=float),
            contact_states={str(name): bool(value) for name, value in d["contact_states"].items()},
            extras={str(name): float(value) for name, value in d.get("extras", {}).items()},
        )


class EpisodeRecorder:
    """Records a bounded sequence of telemetry frames."""

    def __init__(self, max_frames: int = 10000) -> None:
        if max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        self.max_frames = int(max_frames)
        self.frames: list[TelemetryFrame] = []

    def record(self, frame: TelemetryFrame) -> None:
        """Append a frame unless the recorder is already full."""

        if self.is_full():
            return
        self.frames.append(frame)

    def clear(self) -> None:
        """Remove all recorded frames."""

        self.frames = []

    def is_full(self) -> bool:
        """Return whether the recorder has reached capacity."""

        return len(self.frames) >= self.max_frames

    def duration(self) -> float:
        """Return elapsed time between the first and last frame."""

        if len(self.frames) < 2:
            return 0.0
        return float(self.frames[-1].timestamp - self.frames[0].timestamp)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Serialize the full episode as a list of frame mappings."""

        return [frame.to_dict() for frame in self.frames]

    @classmethod
    def from_dict_list(cls, records: list[dict[str, Any]]) -> "EpisodeRecorder":
        """Restore a recorder from serialized frame data."""

        recorder = cls(max_frames=max(10000, len(records)))
        recorder.frames = [TelemetryFrame.from_dict(record) for record in records]
        return recorder


class EpisodeReplay:
    """Random-access and interpolation helpers for recorded telemetry."""

    def __init__(self, frames: list[TelemetryFrame]) -> None:
        self.frames = list(frames)
        self._timestamps = [frame.timestamp for frame in self.frames]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> TelemetryFrame:
        return self.frames[idx]

    def get_at_time(self, t: float) -> TelemetryFrame:
        """Return the frame at time ``t``, interpolating numeric state when needed."""

        if not self.frames:
            raise ValueError("cannot replay an empty episode")

        target = float(t)
        if target <= self.frames[0].timestamp:
            return self.frames[0]
        if target >= self.frames[-1].timestamp:
            return self.frames[-1]

        upper_idx = bisect_left(self._timestamps, target)
        upper = self.frames[upper_idx]
        if upper.timestamp == target:
            return upper

        lower = self.frames[upper_idx - 1]
        span = upper.timestamp - lower.timestamp
        nearest = self._nearest_frame(target, lower, upper)
        if span <= 0.0:
            return nearest

        alpha = (target - lower.timestamp) / span
        return TelemetryFrame(
            timestamp=target,
            joint_positions=(1.0 - alpha) * lower.joint_positions + alpha * upper.joint_positions,
            joint_velocities=(1.0 - alpha) * lower.joint_velocities + alpha * upper.joint_velocities,
            joint_torques=(1.0 - alpha) * lower.joint_torques + alpha * upper.joint_torques,
            base_pose=self._interpolate_base_pose(lower.base_pose, upper.base_pose, alpha),
            contact_states=dict(nearest.contact_states),
            extras=dict(nearest.extras),
        )

    def time_range(self) -> tuple[float, float]:
        """Return the minimum and maximum timestamps in the replay."""

        if not self.frames:
            return (0.0, 0.0)
        return (float(self.frames[0].timestamp), float(self.frames[-1].timestamp))

    @staticmethod
    def _interpolate_base_pose(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        position = (1.0 - alpha) * a[:3] + alpha * b[:3]
        quat_a = a[3:]
        quat_b = b[3:]
        if float(np.dot(quat_a, quat_b)) < 0.0:
            quat_b = -quat_b
        quaternion = _normalize_quaternion((1.0 - alpha) * quat_a + alpha * quat_b)
        return np.concatenate((position, quaternion))

    @staticmethod
    def _nearest_frame(target: float, lower: TelemetryFrame, upper: TelemetryFrame) -> TelemetryFrame:
        if abs(target - lower.timestamp) <= abs(upper.timestamp - target):
            return lower
        return upper


class TelemetryStats:
    """Compute aggregate statistics over a replayed episode."""

    def __init__(self, replay: EpisodeReplay) -> None:
        self.replay = replay

    def mean_joint_positions(self) -> np.ndarray:
        """Return the per-joint mean position across all frames."""

        self._require_frames()
        return np.mean([frame.joint_positions for frame in self.replay.frames], axis=0)

    def max_joint_torques(self) -> np.ndarray:
        """Return the per-joint maximum absolute torque across all frames."""

        self._require_frames()
        return np.max(np.abs([frame.joint_torques for frame in self.replay.frames]), axis=0)

    def contact_ratio(self, leg: str) -> float:
        """Return the fraction of frames reporting contact for the given leg."""

        self._require_frames()
        name = str(leg)
        contacts = [frame.contact_states.get(name, False) for frame in self.replay.frames]
        return float(np.mean(contacts))

    def total_duration(self) -> float:
        """Return the replay duration."""

        start, end = self.replay.time_range()
        return float(end - start)

    def _require_frames(self) -> None:
        if len(self.replay) == 0:
            raise ValueError("telemetry statistics require at least one frame")
