"""Robot fleet management for shared-world multi-robot simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from optisim.math3d import Pose, Quaternion, vec3
from optisim.robot import RobotModel, build_humanoid_model
from optisim.sim import WorldState


@dataclass
class RobotFleet:
    """Manage a set of named humanoids operating in one shared world."""

    world: WorldState = field(default_factory=WorldState.with_defaults)
    robots: dict[str, RobotModel] = field(default_factory=dict)

    def add_robot(self, name: str, base_offset: Pose | Iterable[float]) -> RobotModel:
        """Add a humanoid robot with a unique name and base offset transform."""

        if name in self.robots:
            raise ValueError(f"robot '{name}' already exists in fleet")
        pose = (
            base_offset
            if isinstance(base_offset, Pose)
            else Pose(position=vec3(base_offset), orientation=Quaternion.identity())
        )
        robot = build_humanoid_model()
        robot.name = name
        robot.base_pose = pose
        self.robots[name] = robot
        return robot

    def get_robot(self, name: str) -> RobotModel:
        """Return a robot by its fleet name."""

        return self.robots[name]

    def to_dict(self) -> dict[str, object]:
        """Serialize the fleet definition."""

        return {
            "world": _world_to_dict(self.world),
            "robots": [
                {
                    "name": name,
                    "base_offset": [float(value) for value in robot.base_pose.position.tolist()],
                }
                for name, robot in self.robots.items()
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RobotFleet":
        """Deserialize a fleet from a plain mapping."""

        fleet = cls(world=WorldState.from_dict(payload.get("world", {})))  # type: ignore[arg-type]
        robots = payload.get("robots", [])
        if not isinstance(robots, list):
            return fleet
        for robot_payload in robots:
            if not isinstance(robot_payload, dict):
                continue
            fleet.add_robot(str(robot_payload["name"]), robot_payload.get("base_offset", [0.0, 0.0, 0.0]))
        return fleet


def _world_to_dict(world: WorldState) -> dict[str, object]:
    return {
        "gravity": [float(value) for value in world.gravity.tolist()],
        "objects": [
            {
                "name": obj.name,
                "pose": {"position": [float(value) for value in obj.pose.position.tolist()]},
                "size": [float(value) for value in obj.size],
                "mass_kg": float(obj.mass_kg),
            }
            for obj in world.objects.values()
        ],
        "surfaces": [
            {
                "name": surface.name,
                "pose": {"position": [float(value) for value in surface.pose.position.tolist()]},
                "size": [float(value) for value in surface.size],
            }
            for surface in world.surfaces.values()
        ],
    }
