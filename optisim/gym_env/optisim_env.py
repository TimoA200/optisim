"""Gymnasium environment wrapper for optisim."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from optisim.core.action_primitives import ActionType
from optisim.core.task_definition import TaskDefinition
from optisim.robot import RobotModel, build_humanoid_model, load_urdf
from optisim.sim import ExecutionEngine, SimulationRecording, WorldState
from optisim.viz import TerminalVisualizer, WebVisualizer

from .reward import CollisionPenalty, CompositeReward, ReachReward, RewardFunction, TaskCompletionReward

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    gym = None
    spaces = None

DEFAULT_ENV_ID = "Optisim-v0"


@dataclass(slots=True)
class ObservationConfig:
    """Observation layout for the RL wrapper."""

    joint_positions: bool = True
    link_positions: bool = True
    object_states: bool = True
    joint_names: list[str] | None = None
    link_names: list[str] | None = None
    object_names: list[str] | None = None


@dataclass(slots=True)
class TaskTargets:
    """Resolved target hints inferred from a task definition."""

    effector: str = "right_palm"
    target_position: np.ndarray | None = None
    controlled_object: str | None = None
    support_name: str | None = None
    use_object_completion: bool = False


def _require_gymnasium() -> None:
    if gym is None or spaces is None:
        raise ModuleNotFoundError(
            "gymnasium is required for optisim.gym_env. Install with `pip install optisim[rl]`."
        )


if gym is not None:

    class OptisimEnv(gym.Env[np.ndarray, np.ndarray]):
        """Continuous-control environment exposing optisim through Gymnasium."""

        metadata = {"render_modes": ["human", "web"], "render_fps": 20}

        def __init__(
            self,
            *,
            task_definition: TaskDefinition | str | Path | None = None,
            observation_config: ObservationConfig | dict[str, Any] | None = None,
            reward_fn: RewardFunction | None = None,
            max_steps: int = 200,
            dt: float = 0.05,
            render_mode: str | None = None,
            render_backend: str = "terminal",
            grasp_distance: float = 0.12,
        ) -> None:
            super().__init__()
            self.task_definition = self._load_task_definition(task_definition)
            self.observation_config = self._normalize_observation_config(observation_config)
            self.reward_fn = reward_fn or CompositeReward(
                [
                    ReachReward(success_bonus=2.0),
                    TaskCompletionReward(bonus=10.0),
                    CollisionPenalty(scale=0.5),
                ]
            )
            self.max_steps = int(max_steps)
            self.dt = float(dt)
            self.render_mode = render_mode
            self.render_backend = render_backend
            self.grasp_distance = float(grasp_distance)
            self._visualizer: TerminalVisualizer | WebVisualizer | None = None
            self._recording: SimulationRecording | None = None
            self._steps = 0
            self.task_complete = False

            self.robot = self._build_robot()
            self.world = self._build_world()
            self.engine = ExecutionEngine(robot=self.robot, world=self.world, dt=self.dt)
            self.targets = self._infer_targets()
            self.target_position = self.targets.target_position
            self._configure_spaces()

        @property
        def recording(self) -> SimulationRecording | None:
            return self._recording

        @property
        def step_count(self) -> int:
            return self._steps

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            super().reset(seed=seed)
            del options
            self.robot = self._build_robot()
            self.world = self._build_world()
            self.engine = ExecutionEngine(robot=self.robot, world=self.world, dt=self.dt)
            self.targets = self._infer_targets()
            self.target_position = self.targets.target_position
            self._steps = 0
            self.task_complete = False
            self._recording = SimulationRecording.from_robot(
                self.robot,
                task_name=self.task_definition.name if self.task_definition is not None else "gym_episode",
                dt=self.dt,
                metadata={"mode": "gym"},
            )
            self._recording.capture_frame(self.robot, self.world, active_action="reset", collisions=[])
            self.reward_fn.reset(self)
            self._start_visualizer()
            observation = self._get_observation()
            info = self._build_info(collisions=[])
            return observation, info

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            clipped_action = np.asarray(action, dtype=np.float64).reshape(self.action_space.shape)
            clipped_action = np.clip(clipped_action, self.action_space.low, self.action_space.high)
            self._apply_action(clipped_action)
            self._update_controlled_object()
            collisions = self.engine.step(
                visualize=self._visualizer if self.render_mode is not None else None,
                recording=self._recording,
                active_action="rl_step",
            )
            self._steps += 1
            self.task_complete = self.is_task_complete()
            terminated = self.task_complete
            truncated = self._steps >= self.max_steps
            observation = self._get_observation()
            info = self._build_info(collisions=collisions)
            reward = self.reward_fn.compute(
                self,
                collisions=collisions,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            return observation, reward, terminated, truncated, info

        def render(self) -> str | None:
            if self.render_mode is None:
                return None
            if self._visualizer is None:
                self._start_visualizer()
            if self._visualizer is None:
                return None
            self._visualizer.update_collisions(self.engine._check_collisions())
            self._visualizer.render(self.world, self.robot)
            if isinstance(self._visualizer, WebVisualizer):
                return self._visualizer.url
            return None

        def close(self) -> None:
            if isinstance(self._visualizer, WebVisualizer):
                self._visualizer.close()
            self._visualizer = None

        def end_effector_position(self, effector: str) -> np.ndarray:
            return self.robot.end_effector_pose(effector).position.copy()

        def is_task_complete(self) -> bool:
            if (
                self.targets.use_object_completion
                and self.targets.controlled_object is not None
                and self.targets.target_position is not None
            ):
                obj = self.world.objects.get(self.targets.controlled_object)
                if obj is None:
                    return False
                if np.linalg.norm(obj.pose.position - self.targets.target_position) <= 0.08:
                    return True
            if self.targets.support_name is not None and self.targets.controlled_object is not None:
                obj = self.world.objects.get(self.targets.controlled_object)
                surface = self.world.surfaces.get(self.targets.support_name)
                if obj is None or surface is None:
                    return False
                half = np.asarray(surface.size, dtype=np.float64) / 2.0
                top_z = surface.pose.position[2] + half[2] + obj.size[2] / 2.0
                within_xy = np.all(np.abs(obj.pose.position[:2] - surface.pose.position[:2]) <= half[:2] + 0.05)
                return bool(within_xy and abs(obj.pose.position[2] - top_z) <= 0.06 and obj.held_by is None)
            if self.target_position is not None:
                distance = np.linalg.norm(self.end_effector_position(self.targets.effector) - self.target_position)
                return bool(distance <= 0.05)
            return False

        def observation_components(self) -> dict[str, np.ndarray]:
            poses = self.robot.forward_kinematics()
            components: dict[str, np.ndarray] = {}
            if self.observation_config.joint_positions:
                joint_names = self.observation_config.joint_names or list(self.robot.joints)
                components["joint_positions"] = np.asarray(
                    [self.robot.joint_positions[name] for name in joint_names],
                    dtype=np.float32,
                )
            if self.observation_config.link_positions:
                link_names = self.observation_config.link_names or list(self.robot.links)
                link_values: list[float] = []
                for name in link_names:
                    link_values.extend(float(value) for value in poses[name].position.tolist())
                components["link_positions"] = np.asarray(link_values, dtype=np.float32)
            if self.observation_config.object_states:
                object_names = self.observation_config.object_names or sorted(self.world.objects)
                object_values: list[float] = []
                for name in object_names:
                    obj = self.world.objects[name]
                    object_values.extend(float(value) for value in obj.pose.position.tolist())
                    object_values.extend(float(value) for value in obj.pose.orientation.as_np().tolist())
                    object_values.extend(float(value) for value in obj.size)
                    object_values.append(1.0 if obj.held_by is not None else 0.0)
                components["object_states"] = np.asarray(object_values, dtype=np.float32)
            return components

        def _get_observation(self) -> np.ndarray:
            components = self.observation_components()
            if not components:
                return np.zeros((0,), dtype=np.float32)
            return np.concatenate(list(components.values())).astype(np.float32)

        def _configure_spaces(self) -> None:
            action_joints = list(self.robot.joints)
            self._action_joint_names = action_joints
            action_bounds = np.asarray([self.robot.joints[name].velocity_limit for name in action_joints], dtype=np.float32)
            self.action_space = spaces.Box(low=-action_bounds, high=action_bounds, dtype=np.float32)

            lows: list[float] = []
            highs: list[float] = []
            if self.observation_config.joint_positions:
                for name in self.observation_config.joint_names or list(self.robot.joints):
                    joint = self.robot.joints[name]
                    lows.append(joint.limit_lower)
                    highs.append(joint.limit_upper)
            if self.observation_config.link_positions:
                link_count = len(self.observation_config.link_names or list(self.robot.links))
                lows.extend([-np.inf] * (link_count * 3))
                highs.extend([np.inf] * (link_count * 3))
            if self.observation_config.object_states:
                object_count = len(self.observation_config.object_names or sorted(self.world.objects))
                per_object_low = [-np.inf] * 3 + [-1.0] * 4 + [0.0] * 3 + [0.0]
                per_object_high = [np.inf] * 3 + [1.0] * 4 + [np.inf] * 3 + [1.0]
                lows.extend(per_object_low * object_count)
                highs.extend(per_object_high * object_count)
            self.observation_space = spaces.Box(
                low=np.asarray(lows, dtype=np.float32),
                high=np.asarray(highs, dtype=np.float32),
                dtype=np.float32,
            )

        def _apply_action(self, action: np.ndarray) -> None:
            updated = dict(self.robot.joint_positions)
            for joint_name, velocity in zip(self._action_joint_names, action, strict=True):
                updated[joint_name] = updated[joint_name] + float(velocity) * self.dt
            self.robot.set_joint_positions(updated)

        def _update_controlled_object(self) -> None:
            object_name = self.targets.controlled_object
            if object_name is None or object_name not in self.world.objects:
                return
            obj = self.world.objects[object_name]
            effector = self.targets.effector
            effector_position = self.end_effector_position(effector)
            distance = float(np.linalg.norm(obj.pose.position - effector_position))
            if obj.held_by is None and distance <= self.grasp_distance:
                obj.held_by = effector
            if obj.held_by is not None:
                obj.pose.position = effector_position.copy()
                if self.targets.support_name is not None and self.is_task_complete():
                    obj.held_by = None

        def _build_info(self, *, collisions: list[Any]) -> dict[str, Any]:
            distance_to_target = None
            if self.target_position is not None:
                distance_to_target = float(
                    np.linalg.norm(self.end_effector_position(self.targets.effector) - self.target_position)
                )
            return {
                "step_count": self._steps,
                "time_s": float(self.world.time_s),
                "task_complete": self.task_complete,
                "distance_to_target": distance_to_target,
                "collisions": [
                    {
                        "entity_a": item.entity_a,
                        "entity_b": item.entity_b,
                        "penetration_depth": float(item.penetration_depth),
                    }
                    for item in collisions
                ],
                "observation_components": {
                    name: value.copy() for name, value in self.observation_components().items()
                },
            }

        def _start_visualizer(self) -> None:
            if self.render_mode is None:
                return
            if self._visualizer is None:
                self._visualizer = WebVisualizer(open_browser=False) if self.render_backend == "web" else TerminalVisualizer()
            task = self.task_definition or TaskDefinition(name="gym_episode", actions=[])
            self._visualizer.start_task(task, self.world, self.robot)
            self._visualizer.render(self.world, self.robot)

        def _load_task_definition(self, task_definition: TaskDefinition | str | Path | None) -> TaskDefinition | None:
            if task_definition is None:
                return None
            if isinstance(task_definition, TaskDefinition):
                return task_definition
            return TaskDefinition.from_file(task_definition)

        def _normalize_observation_config(
            self,
            config: ObservationConfig | dict[str, Any] | None,
        ) -> ObservationConfig:
            if config is None:
                return ObservationConfig()
            if isinstance(config, ObservationConfig):
                return config
            return ObservationConfig(**config)

        def _build_robot(self) -> RobotModel:
            robot_payload = self.task_definition.robot if self.task_definition is not None else {}
            if "urdf" in robot_payload:
                return load_urdf(robot_payload["urdf"])
            return build_humanoid_model()

        def _build_world(self) -> WorldState:
            world_payload = self.task_definition.world if self.task_definition is not None else {}
            return WorldState.from_dict(world_payload)

        def _infer_targets(self) -> TaskTargets:
            if self.task_definition is None:
                default_target = None
                if "box" in self.world.objects:
                    default_target = self.world.objects["box"].pose.position.copy()
                return TaskTargets(
                    target_position=default_target,
                )

            effector = "right_palm"
            target_position: np.ndarray | None = None
            controlled_object: str | None = None
            support_name: str | None = None
            use_object_completion = False
            for action in self.task_definition.actions:
                effector = action.end_effector or effector
                if action.action_type is ActionType.REACH and target_position is None:
                    if action.pose is not None:
                        target_position = action.pose.position.copy()
                    elif action.target in self.world.objects:
                        target_position = self.world.objects[action.target].pose.position.copy()
                elif action.action_type is ActionType.GRASP:
                    controlled_object = action.target
                elif action.action_type is ActionType.MOVE and action.destination is not None:
                    controlled_object = action.target
                    target_position = np.asarray(action.destination, dtype=np.float64)
                    use_object_completion = True
                elif action.action_type is ActionType.PLACE and action.support in self.world.surfaces:
                    controlled_object = action.target
                    support_name = action.support
                    use_object_completion = True
                    surface = self.world.surfaces[action.support]
                    height = surface.pose.position[2] + surface.size[2] / 2.0
                    if action.target in self.world.objects:
                        height += self.world.objects[action.target].size[2] / 2.0
                    target_position = np.asarray(
                        [surface.pose.position[0], surface.pose.position[1], height],
                        dtype=np.float64,
                    )
            return TaskTargets(
                effector=effector,
                target_position=target_position,
                controlled_object=controlled_object,
                support_name=support_name,
                use_object_completion=use_object_completion,
            )


else:

    class OptisimEnv:
        """Placeholder raised when Gymnasium is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            _require_gymnasium()


def register_optisim_env(env_id: str = DEFAULT_ENV_ID, **kwargs: Any) -> str:
    """Register the optisim Gymnasium environment and return the environment id."""

    _require_gymnasium()
    assert gym is not None
    if env_id not in gym.registry:
        gym.register(
            id=env_id,
            entry_point="optisim.gym_env.optisim_env:OptisimEnv",
            kwargs=kwargs,
        )
    return env_id

__all__ = ["DEFAULT_ENV_ID", "ObservationConfig", "TaskTargets", "register_optisim_env"]
