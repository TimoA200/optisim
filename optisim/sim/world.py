"""World state used by the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optisim.math3d import Pose, Quaternion, vec3


@dataclass(slots=True)
class Surface:
    name: str
    pose: Pose
    size: tuple[float, float, float]


@dataclass(slots=True)
class ObjectState:
    name: str
    pose: Pose
    size: tuple[float, float, float]
    mass_kg: float = 1.0
    held_by: str | None = None

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        half = np.asarray(self.size, dtype=np.float64) / 2.0
        return self.pose.position - half, self.pose.position + half


@dataclass
class WorldState:
    gravity: np.ndarray = field(default_factory=lambda: vec3([0.0, 0.0, -9.81]))
    objects: dict[str, ObjectState] = field(default_factory=dict)
    surfaces: dict[str, Surface] = field(default_factory=dict)
    time_s: float = 0.0

    @classmethod
    def with_defaults(cls) -> "WorldState":
        table_pose = Pose(position=vec3([0.55, 0.0, 0.74]), orientation=Quaternion.identity())
        box_pose = Pose(position=vec3([0.42, -0.12, 0.81]), orientation=Quaternion.identity())
        return cls(
            objects={
                "box": ObjectState(name="box", pose=box_pose, size=(0.08, 0.08, 0.12), mass_kg=0.75)
            },
            surfaces={
                "table": Surface(name="table", pose=table_pose, size=(0.90, 0.60, 0.05)),
                "shelf": Surface(
                    name="shelf",
                    pose=Pose(position=vec3([0.60, -0.25, 1.02]), orientation=Quaternion.identity()),
                    size=(0.35, 0.25, 0.04),
                ),
            },
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "WorldState":
        world = cls.with_defaults()
        if not payload:
            return world
        if "gravity" in payload:
            world.gravity = vec3(payload["gravity"])
        for surface in payload.get("surfaces", []):
            world.surfaces[surface["name"]] = Surface(
                name=surface["name"],
                pose=Pose.from_xyz_rpy(surface["pose"]["position"], surface["pose"].get("rpy", [0.0, 0.0, 0.0])),
                size=tuple(float(v) for v in surface["size"]),
            )
        for obj in payload.get("objects", []):
            world.objects[obj["name"]] = ObjectState(
                name=obj["name"],
                pose=Pose.from_xyz_rpy(obj["pose"]["position"], obj["pose"].get("rpy", [0.0, 0.0, 0.0])),
                size=tuple(float(v) for v in obj["size"]),
                mass_kg=float(obj.get("mass_kg", 1.0)),
            )
        return world
