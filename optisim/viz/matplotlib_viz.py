"""Optional matplotlib visualization."""

from __future__ import annotations

from dataclasses import dataclass, field

from optisim.robot.model import RobotModel
from optisim.sim.world import WorldState


@dataclass
class MatplotlibVisualizer:
    """Minimal matplotlib renderer for 2D top-down state inspection."""

    figure: object | None = field(default=None, init=False)
    axes: object | None = field(default=None, init=False)

    def render(self, world: WorldState, robot: RobotModel) -> None:
        """Render the world, objects, and end effectors with matplotlib."""

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib visualization requires `pip install optisim[viz]`") from exc

        if self.figure is None or self.axes is None:
            self.figure = plt.figure(figsize=(6, 5))
            self.axes = self.figure.add_subplot(111)
            plt.ion()

        ax = self.axes
        assert ax is not None
        ax.clear()
        ax.set_title(f"optisim @ {world.time_s:.2f}s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        for surface in world.surfaces.values():
            ax.scatter(surface.pose.position[0], surface.pose.position[1], marker="s", s=120, label=surface.name)
        for obj in world.objects.values():
            ax.scatter(obj.pose.position[0], obj.pose.position[1], marker="o", s=90, label=obj.name)
        for effector in robot.end_effectors:
            pose = robot.end_effector_pose(effector)
            ax.scatter(pose.position[0], pose.position[1], marker="x", s=80, label=effector)
        ax.legend(loc="upper right")
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
