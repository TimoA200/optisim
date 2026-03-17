"""High-level motion planner that combines IK, collision checking, and RRT."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from optisim.math3d import Pose
from optisim.planning.rrt import RRTConfig, plan_rrt, plan_rrt_connect
from optisim.planning.smoothing import shortcut_path
from optisim.robot import IKOptions, RobotModel, solve_inverse_kinematics
from optisim.sim.collision import robot_world_collisions
from optisim.sim.world import WorldState

Vector = NDArray[np.float64]


@dataclass(slots=True)
class PlanningResult:
    """Result payload returned by the motion planner."""

    path: list[dict[str, float]]
    success: bool
    iterations: int
    planning_time: float


@dataclass(slots=True)
class MotionPlanner:
    """Collision-aware joint-space motion planner."""

    robot: RobotModel
    world: WorldState
    smoothing_iterations: int = 100
    interpolation_step_size: float = 0.08
    use_rrt_connect: bool = True
    random_seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)
    _ignored_collision_links: tuple[str, ...] = field(
        default=(
            "pelvis",
            "torso",
            "chest",
            "neck",
            "head",
            "left_clavicle",
            "right_clavicle",
        ),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Create a deterministic RNG for repeatable planning."""

        self.rng = np.random.default_rng(self.random_seed)

    def plan(
        self,
        start_config: dict[str, float],
        goal_config: dict[str, float],
        rrt_config: RRTConfig | None = None,
    ) -> PlanningResult:
        """Plan a collision-free joint-space path between two configurations."""

        started_at = perf_counter()
        config = rrt_config or RRTConfig()
        start_merged = self._merged_config(start_config)
        goal_merged = self._merged_config(goal_config)
        active_joints = self._active_joints(start_config, goal_config)
        monitored_links = self._monitored_links(active_joints)
        start_state = self._state_vector(start_merged, active_joints)
        goal_state = self._state_vector(goal_merged, active_joints)

        if not self.is_collision_free(start_merged, link_names=monitored_links) or not self.is_collision_free(
            goal_merged, link_names=monitored_links
        ):
            return PlanningResult(path=[], success=False, iterations=0, planning_time=perf_counter() - started_at)
        if self.is_path_segment_collision_free(start_merged, goal_merged, link_names=monitored_links):
            path = [start_merged, goal_merged]
            return PlanningResult(
                path=self._smooth_path(path, monitored_links),
                success=True,
                iterations=0,
                planning_time=perf_counter() - started_at,
            )

        lower_bounds = np.asarray([self.robot.joints[name].limit_lower for name in active_joints], dtype=np.float64)
        upper_bounds = np.asarray([self.robot.joints[name].limit_upper for name in active_joints], dtype=np.float64)

        def is_state_valid(state: Vector) -> bool:
            return self.is_collision_free(self._vector_to_config(state, active_joints, start_merged), link_names=monitored_links)

        def is_edge_valid(source: Vector, target: Vector) -> bool:
            return self.is_path_segment_collision_free(
                self._vector_to_config(source, active_joints, start_merged),
                self._vector_to_config(target, active_joints, start_merged),
                link_names=monitored_links,
            )

        planner_fn = plan_rrt_connect if self.use_rrt_connect else plan_rrt
        waypoints, iterations = planner_fn(
            start_state,
            goal_state,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            is_state_valid=is_state_valid,
            is_edge_valid=is_edge_valid,
            config=config,
            rng=self.rng,
        )
        if not waypoints:
            return PlanningResult(path=[], success=False, iterations=iterations, planning_time=perf_counter() - started_at)

        path = [self._vector_to_config(vertex, active_joints, start_merged) for vertex in waypoints]
        return PlanningResult(
            path=self._smooth_path(path, monitored_links),
            success=True,
            iterations=iterations,
            planning_time=perf_counter() - started_at,
        )

    def plan_to_pose(
        self,
        start_config: dict[str, float],
        target_pose: Pose,
        ik_options: IKOptions | dict[str, Any] | None = None,
        rrt_config: RRTConfig | None = None,
    ) -> PlanningResult:
        """Solve IK for a target pose and then plan a collision-free path to it."""

        started_at = perf_counter()
        effector, options = self._normalize_ik_options(ik_options)
        ik_result = solve_inverse_kinematics(
            self.robot,
            effector,
            target_pose,
            initial_positions=self._merged_config(start_config),
            options=options,
        )
        if not ik_result.success:
            return PlanningResult(path=[], success=False, iterations=ik_result.iterations, planning_time=perf_counter() - started_at)

        goal_config = self._merged_config(start_config)
        goal_config.update(ik_result.joint_positions)
        result = self.plan(start_config, goal_config, rrt_config=rrt_config)
        result.iterations += ik_result.iterations
        result.planning_time = perf_counter() - started_at
        return result

    def is_collision_free(
        self,
        config: dict[str, float],
        *,
        link_names: set[str] | None = None,
    ) -> bool:
        """Return whether the supplied robot configuration is collision-free."""

        return not robot_world_collisions(self.robot, self.world, self._merged_config(config), link_names=link_names)

    def is_path_segment_collision_free(
        self,
        start_config: dict[str, float],
        goal_config: dict[str, float],
        *,
        link_names: set[str] | None = None,
    ) -> bool:
        """Return whether linear interpolation between two configurations is collision-free."""

        start = self._merged_config(start_config)
        goal = self._merged_config(goal_config)
        joint_names = self._active_joints(start, goal)
        start_state = self._state_vector(start, joint_names)
        goal_state = self._state_vector(goal, joint_names)
        distance = float(np.linalg.norm(goal_state - start_state))
        steps = max(int(np.ceil(distance / max(self.interpolation_step_size, 1e-6))), 1)
        for step_index in range(steps + 1):
            alpha = step_index / steps
            config = dict(start)
            for joint_name in joint_names:
                config[joint_name] = float(start[joint_name] * (1.0 - alpha) + goal[joint_name] * alpha)
            if not self.is_collision_free(config, link_names=link_names):
                return False
        return True

    def _smooth_path(self, path: list[dict[str, float]], link_names: set[str]) -> list[dict[str, float]]:
        return shortcut_path(
            path,
            is_segment_valid=lambda start, goal: self.is_path_segment_collision_free(
                start,
                goal,
                link_names=link_names,
            ),
            max_iterations=self.smoothing_iterations,
            rng=self.rng,
        )

    def _normalize_ik_options(self, payload: IKOptions | dict[str, Any] | None) -> tuple[str, IKOptions]:
        if payload is None:
            return "right_palm", IKOptions(max_iterations=120, convergence_threshold=2e-3, damping=0.12)
        if isinstance(payload, IKOptions):
            return "right_palm", payload
        options_payload = dict(payload)
        effector = str(options_payload.pop("end_effector", "right_palm"))
        return effector, IKOptions(**options_payload)

    def _merged_config(self, config: dict[str, float] | None = None) -> dict[str, float]:
        merged = dict(self.robot.joint_positions)
        if config:
            for joint_name, value in config.items():
                merged[joint_name] = self.robot.joints[joint_name].clamp(value)
        return merged

    def _active_joints(self, start_config: dict[str, float], goal_config: dict[str, float]) -> list[str]:
        return sorted(set(start_config) | set(goal_config))

    def _state_vector(self, config: dict[str, float], joint_names: list[str]) -> Vector:
        return np.asarray([config[name] for name in joint_names], dtype=np.float64)

    def _vector_to_config(
        self,
        values: Vector,
        joint_names: list[str],
        reference_config: dict[str, float],
    ) -> dict[str, float]:
        config = dict(reference_config)
        for joint_name, value in zip(joint_names, values, strict=True):
            config[joint_name] = self.robot.joints[joint_name].clamp(float(value))
        return config

    def _monitored_links(self, active_joints: list[str]) -> set[str]:
        links = {self.robot.joints[name].child for name in active_joints}
        return {name for name in links if name not in self._ignored_collision_links}
