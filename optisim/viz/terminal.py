"""Terminal visualization for quick iteration over task plans."""

from __future__ import annotations

from dataclasses import dataclass

from optisim.robot.model import RobotModel
from optisim.sim.world import WorldState


@dataclass
class TerminalVisualizer:
    width: int = 50
    height: int = 16
    x_range: tuple[float, float] = (-0.3, 1.0)
    y_range: tuple[float, float] = (-0.6, 0.6)

    def render(self, world: WorldState, robot: RobotModel) -> None:
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]

        def plot(x: float, y: float, char: str) -> None:
            u = int((x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * (self.width - 1))
            v = int((y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * (self.height - 1))
            v = self.height - 1 - v
            if 0 <= u < self.width and 0 <= v < self.height:
                grid[v][u] = char

        for surface in world.surfaces.values():
            plot(surface.pose.position[0], surface.pose.position[1], "T")
        for obj in world.objects.values():
            plot(obj.pose.position[0], obj.pose.position[1], "B" if obj.name == "box" else "O")
        for effector in robot.end_effectors:
            pose = robot.end_effector_pose(effector)
            plot(pose.position[0], pose.position[1], "R")

        frame = "\n".join("".join(row) for row in grid)
        print(f"\n[t={world.time_s:05.2f}s]\n{frame}")
