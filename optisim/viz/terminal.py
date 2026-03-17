"""Rich-powered terminal visualization for task execution."""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from rich.align import Align
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.text import Text
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without Rich installed.
    raise ModuleNotFoundError(
        "Terminal visualization requires the 'rich' dependency. "
        "Install with `pip install optisim` or `pip install rich`."
    ) from exc

from optisim.core.action_primitives import ActionPrimitive
from optisim.core.task_definition import TaskDefinition
from optisim.robot.model import RobotModel
from optisim.sim.collision import Collision
from optisim.sim.world import WorldState


@dataclass
class TerminalVisualizer:
    """Rich-based terminal renderer for task execution progress."""

    width: int = 38
    height: int = 22
    console: Console = field(default_factory=Console)
    _live: Live | None = field(default=None, init=False, repr=False)
    _progress: Progress | None = field(default=None, init=False, repr=False)
    _progress_task_id: int | None = field(default=None, init=False, repr=False)
    _current_action: str = field(default="idle", init=False, repr=False)
    _action_index: int = field(default=0, init=False, repr=False)
    _action_total: int = field(default=0, init=False, repr=False)
    _collision_names: set[str] = field(default_factory=set, init=False, repr=False)
    _previous_positions: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def start_task(self, task: TaskDefinition, world: WorldState, robot: RobotModel) -> None:
        """Prepare live terminal rendering for a new task."""

        self._previous_positions = dict(robot.joint_positions)
        self._collision_names.clear()
        self._current_action = "initializing"
        self._action_index = 0
        self._action_total = len(task.actions)
        self._progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            expand=True,
        )
        self._progress_task_id = self._progress.add_task(task.name, total=max(len(task.actions), 1))
        self._live = Live(self._render_layout(world, robot), console=self.console, refresh_per_second=10)
        self._live.start()

    def start_action(self, action: ActionPrimitive, *, index: int, total_actions: int) -> None:
        """Update the active action label and progress bar state."""

        self._current_action = f"{action.action_type.value} {action.target}"
        self._action_index = index
        self._action_total = total_actions
        if self._progress is not None and self._progress_task_id is not None:
            self._progress.update(self._progress_task_id, completed=index - 1)

    def update_collisions(self, collisions: list[Collision]) -> None:
        """Track colliding entity names for highlighting."""

        self._collision_names = {collision.entity_a for collision in collisions} | {
            collision.entity_b for collision in collisions
        }

    def render(self, world: WorldState, robot: RobotModel) -> None:
        """Render or refresh the current terminal visualization frame."""

        if self._live is None:
            self.console.print(self._render_layout(world, robot))
            return
        self._live.update(self._render_layout(world, robot), refresh=True)
        self._previous_positions = dict(robot.joint_positions)

    def finish(
        self,
        task: TaskDefinition,
        world: WorldState,
        robot: RobotModel,
        collisions: list[Collision],
    ) -> None:
        """Finalize the live session and render the terminal end state."""

        if self._progress is not None and self._progress_task_id is not None:
            self._progress.update(self._progress_task_id, completed=max(len(task.actions), 1))
        if self._live is not None:
            self._live.update(self._render_layout(world, robot), refresh=True)
            self._live.stop()
            self._live = None

    def _render_layout(self, world: WorldState, robot: RobotModel) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="progress", size=3),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(name="figure", ratio=2),
            Layout(name="status", ratio=1),
        )
        layout["progress"].update(self._progress or Text("idle"))
        layout["figure"].update(self._build_figure_panel(world, robot))
        layout["status"].update(self._build_status_panel(world, robot))
        return layout

    def _build_figure_panel(self, world: WorldState, robot: RobotModel) -> Panel:
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        poses = robot.forward_kinematics()
        moving_joints = {
            name
            for name, value in robot.joint_positions.items()
            if abs(value - self._previous_positions.get(name, value)) > 1e-3
        }
        connections = [
            ("pelvis", "chest"),
            ("chest", "neck"),
            ("neck", "head"),
            ("chest", "right_clavicle"),
            ("right_clavicle", "right_upper_arm"),
            ("right_upper_arm", "right_forearm"),
            ("right_forearm", "right_hand"),
            ("right_hand", "right_palm"),
            ("chest", "left_clavicle"),
            ("left_clavicle", "left_upper_arm"),
            ("left_upper_arm", "left_forearm"),
            ("left_forearm", "left_hand"),
            ("left_hand", "left_palm"),
            ("pelvis", "right_thigh"),
            ("right_thigh", "right_shin"),
            ("right_shin", "right_foot"),
            ("pelvis", "left_thigh"),
            ("left_thigh", "left_shin"),
            ("left_shin", "left_foot"),
        ]
        x_min, x_max = -0.9, 1.0
        z_min, z_max = -0.1, 1.8

        def project(link_name: str) -> tuple[int, int]:
            pose = poses.get(link_name, robot.base_pose)
            x, _, z = pose.position
            u = int((x - x_min) / (x_max - x_min) * (self.width - 1))
            v = self.height - 1 - int((z - z_min) / (z_max - z_min) * (self.height - 1))
            return max(0, min(self.width - 1, u)), max(0, min(self.height - 1, v))

        def plot(x: int, y: int, char: str) -> None:
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = char

        def draw_line(start: tuple[int, int], end: tuple[int, int], char: str) -> None:
            x0, y0 = start
            x1, y1 = end
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for step in range(steps + 1):
                alpha = step / steps
                plot(int(round(x0 + (x1 - x0) * alpha)), int(round(y0 + (y1 - y0) * alpha)), char)

        for parent, child in connections:
            if parent in poses and child in poses:
                draw_line(project(parent), project(child), "•")

        for joint in robot.joints.values():
            x, y = project(joint.child)
            char = "●" if joint.name in moving_joints else "○"
            plot(x, y, char)

        held_objects = {name for name, obj in world.objects.items() if obj.held_by is not None}
        for obj in world.objects.values():
            x, _, z = obj.pose.position
            u = int((x - x_min) / (x_max - x_min) * (self.width - 1))
            v = self.height - 1 - int((z - z_min) / (z_max - z_min) * (self.height - 1))
            if obj.name in self._collision_names:
                plot(u, v, "X")
            elif obj.name in held_objects:
                plot(u, v, "G")
            else:
                plot(u, v, "O")

        lines = []
        for row in grid:
            text = Text()
            for char in row:
                if char == "●":
                    text.append(char, style="bold green")
                elif char == "X":
                    text.append(char, style="bold red")
                elif char == "G":
                    text.append(char, style="bold yellow")
                elif char == "O":
                    text.append(char, style="cyan")
                else:
                    text.append(char, style="white")
            lines.append(text)

        legend = Text.assemble(
            ("● moving ", "bold green"),
            ("X collision ", "bold red"),
            ("G grasped", "bold yellow"),
        )
        return Panel(Group(Align.center(Text("Side Elevation", style="bold")), *lines, Align.center(legend)), title="Robot")

    def _build_status_panel(self, world: WorldState, robot: RobotModel) -> Panel:
        status = Table.grid(expand=True)
        status.add_column(justify="left")
        status.add_column(justify="right")
        status.add_row("Action", self._current_action)
        status.add_row("Step", f"{self._action_index}/{max(self._action_total, 1)}")
        status.add_row("Sim Time", f"{world.time_s:0.2f}s")

        ee_pose = robot.end_effector_pose("right_palm")
        status.add_row(
            "Right EE",
            f"x={ee_pose.position[0]:+.2f} y={ee_pose.position[1]:+.2f} z={ee_pose.position[2]:+.2f}",
        )

        joints = Table(title="Joint Positions", show_header=True, header_style="bold cyan", expand=True)
        joints.add_column("Joint")
        joints.add_column("rad", justify="right")
        moving = {
            name
            for name, value in robot.joint_positions.items()
            if abs(value - self._previous_positions.get(name, value)) > 1e-3
        }
        for name in sorted(robot.joint_positions)[:12]:
            style = "green" if name in moving else None
            joints.add_row(name, f"{robot.joint_positions[name]:+.2f}", style=style)

        objects = Table(title="World Objects", show_header=True, header_style="bold magenta", expand=True)
        objects.add_column("Object")
        objects.add_column("State")
        for obj in world.objects.values():
            state = "collision" if obj.name in self._collision_names else "grasped" if obj.held_by else "free"
            style = "red" if state == "collision" else "yellow" if state == "grasped" else None
            objects.add_row(obj.name, state, style=style)

        return Panel(Group(status, joints, objects), title="Status", border_style="cyan")
