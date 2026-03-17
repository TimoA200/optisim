"""Simulation recording, JSON export, and replay utilities."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from optisim.core.task_definition import TaskDefinition
from optisim.math3d import Pose, Quaternion, vec3
from optisim.robot.model import RobotModel
from optisim.sim.collision import Collision
from optisim.sim.world import ObjectState, Surface, WorldState


def _pose_payload(pose: Pose) -> dict[str, list[float]]:
    return {
        "position": [float(value) for value in pose.position.tolist()],
        "orientation": [float(value) for value in pose.orientation.as_np().tolist()],
    }


@dataclass(slots=True)
class RecordingFrame:
    """State snapshot for a single simulation timestep."""

    index: int
    time_s: float
    active_action: str | None
    joint_positions: dict[str, float]
    link_positions: dict[str, list[float]]
    objects: dict[str, dict[str, Any]]
    surfaces: dict[str, dict[str, Any]]
    moving_joints: list[str] = field(default_factory=list)
    collisions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the frame into JSON-friendly data."""

        return {
            "index": self.index,
            "time_s": self.time_s,
            "active_action": self.active_action,
            "joint_positions": self.joint_positions,
            "link_positions": self.link_positions,
            "objects": self.objects,
            "surfaces": self.surfaces,
            "moving_joints": self.moving_joints,
            "collisions": self.collisions,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RecordingFrame":
        """Deserialize a frame from disk."""

        return cls(
            index=int(payload["index"]),
            time_s=float(payload["time_s"]),
            active_action=payload.get("active_action"),
            joint_positions={name: float(value) for name, value in payload["joint_positions"].items()},
            link_positions={
                name: [float(component) for component in position]
                for name, position in payload["link_positions"].items()
            },
            objects={name: dict(item) for name, item in payload["objects"].items()},
            surfaces={name: dict(item) for name, item in payload["surfaces"].items()},
            moving_joints=[str(name) for name in payload.get("moving_joints", [])],
            collisions=[dict(item) for item in payload.get("collisions", [])],
        )


@dataclass
class SimulationRecording:
    """Portable recording of robot and world state across a simulation."""

    robot_name: str
    task_name: str | None = None
    dt: float = 0.05
    joint_names: list[str] = field(default_factory=list)
    skeleton_connections: list[tuple[str, str]] = field(default_factory=list)
    end_effectors: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    frames: list[RecordingFrame] = field(default_factory=list)
    _previous_joint_positions: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_robot(
        cls,
        robot: RobotModel,
        *,
        task_name: str | None = None,
        dt: float = 0.05,
        metadata: dict[str, Any] | None = None,
    ) -> "SimulationRecording":
        """Create an empty recording seeded with robot metadata."""

        return cls(
            robot_name=robot.name,
            task_name=task_name,
            dt=dt,
            joint_names=list(robot.joints),
            skeleton_connections=[(joint.parent, joint.child) for joint in robot.joints.values()],
            end_effectors=dict(robot.end_effectors),
            metadata=dict(metadata or {}),
        )

    def capture_frame(
        self,
        robot: RobotModel,
        world: WorldState,
        *,
        active_action: str | None = None,
        collisions: list[Collision] | None = None,
    ) -> RecordingFrame:
        """Append the current runtime state as a recording frame."""

        poses = robot.forward_kinematics()
        joint_positions = {name: float(value) for name, value in robot.joint_positions.items()}
        moving_joints = [
            name
            for name, value in joint_positions.items()
            if abs(value - self._previous_joint_positions.get(name, value)) > 1e-3
        ]
        frame = RecordingFrame(
            index=len(self.frames),
            time_s=float(world.time_s),
            active_action=active_action,
            joint_positions=joint_positions,
            link_positions={
                name: [float(component) for component in pose.position.tolist()] for name, pose in poses.items()
            },
            objects={
                name: {
                    "pose": _pose_payload(obj.pose),
                    "size": [float(value) for value in obj.size],
                    "mass_kg": float(obj.mass_kg),
                    "held_by": obj.held_by,
                }
                for name, obj in world.objects.items()
            },
            surfaces={
                name: {
                    "pose": _pose_payload(surface.pose),
                    "size": [float(value) for value in surface.size],
                }
                for name, surface in world.surfaces.items()
            },
            moving_joints=moving_joints,
            collisions=[
                {
                    "entity_a": collision.entity_a,
                    "entity_b": collision.entity_b,
                    "penetration_depth": float(collision.penetration_depth),
                }
                for collision in collisions or []
            ],
        )
        self.frames.append(frame)
        self._previous_joint_positions = joint_positions
        return frame

    def to_dict(self) -> dict[str, Any]:
        """Serialize the recording into a plain mapping."""

        return {
            "robot_name": self.robot_name,
            "task_name": self.task_name,
            "dt": self.dt,
            "joint_names": self.joint_names,
            "skeleton_connections": [list(item) for item in self.skeleton_connections],
            "end_effectors": self.end_effectors,
            "metadata": self.metadata,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    def dump(self, path: str | Path) -> None:
        """Write the recording to disk as JSON."""

        destination = Path(path)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SimulationRecording":
        """Load a recording from an in-memory mapping."""

        recording = cls(
            robot_name=str(payload["robot_name"]),
            task_name=payload.get("task_name"),
            dt=float(payload.get("dt", 0.05)),
            joint_names=[str(name) for name in payload.get("joint_names", [])],
            skeleton_connections=[
                (str(parent), str(child)) for parent, child in payload.get("skeleton_connections", [])
            ],
            end_effectors=dict(payload.get("end_effectors", {})),
            metadata=dict(payload.get("metadata", {})),
            frames=[RecordingFrame.from_dict(item) for item in payload.get("frames", [])],
        )
        if recording.frames:
            recording._previous_joint_positions = dict(recording.frames[-1].joint_positions)
        return recording

    @classmethod
    def from_file(cls, path: str | Path) -> "SimulationRecording":
        """Read a recording from a JSON file."""

        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"recording file {source} does not contain a mapping")
        return cls.from_dict(payload)

    def frame_count(self) -> int:
        """Return the number of recorded frames."""

        return len(self.frames)


def apply_frame_to_runtime(frame: RecordingFrame, robot: RobotModel, world: WorldState) -> None:
    """Mutate a robot and world instance to match a recorded frame."""

    robot.set_joint_positions(frame.joint_positions)
    world.time_s = frame.time_s
    world.objects = {
        name: ObjectState(
            name=name,
            pose=Pose(
                position=vec3(item["pose"]["position"]),
                orientation=Quaternion(*item["pose"]["orientation"]).normalized(),
            ),
            size=tuple(float(value) for value in item["size"]),
            mass_kg=float(item.get("mass_kg", 1.0)),
            held_by=item.get("held_by"),
        )
        for name, item in frame.objects.items()
    }
    world.surfaces = {
        name: Surface(
            name=name,
            pose=Pose(
                position=vec3(item["pose"]["position"]),
                orientation=Quaternion(*item["pose"]["orientation"]).normalized(),
            ),
            size=tuple(float(value) for value in item["size"]),
        )
        for name, item in frame.surfaces.items()
    }


def replay_recording(
    recording: SimulationRecording,
    *,
    robot: RobotModel,
    world: WorldState,
    visualizer: Any | None = None,
    realtime: bool = True,
) -> None:
    """Replay a recording through an existing visualization backend."""

    if not recording.frames:
        return

    task = TaskDefinition(name=recording.task_name or "replay", actions=[], metadata={"mode": "replay"})
    if visualizer is not None:
        visualizer.start_task(task, world, robot)

    previous_time = recording.frames[0].time_s
    for frame in recording.frames:
        if realtime:
            delay_s = max(frame.time_s - previous_time, 0.0)
            if delay_s > 0.0:
                time.sleep(delay_s)
        previous_time = frame.time_s
        apply_frame_to_runtime(frame, robot, world)
        if visualizer is not None:
            visualizer.update_collisions(
                [
                    Collision(
                        entity_a=item["entity_a"],
                        entity_b=item["entity_b"],
                        penetration_depth=float(item["penetration_depth"]),
                    )
                    for item in frame.collisions
                ]
            )
            visualizer.render(world, robot)

    if visualizer is not None:
        visualizer.finish(task, world, robot, [])

__all__ = ["RecordingFrame", "SimulationRecording", "apply_frame_to_runtime", "replay_recording"]
