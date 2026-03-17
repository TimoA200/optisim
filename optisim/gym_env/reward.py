"""Pluggable reward functions for the Gymnasium environment wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from optisim.sim.collision import Collision


class SupportsRewardEnv(Protocol):
    """Subset of environment methods used by reward components."""

    target_position: np.ndarray | None
    task_complete: bool

    def end_effector_position(self, effector: str) -> np.ndarray: ...

    def is_task_complete(self) -> bool: ...


class RewardFunction:
    """Base reward component interface."""

    def reset(self, env: SupportsRewardEnv) -> None:
        """Reset any internal state before a new episode."""

    def compute(
        self,
        env: SupportsRewardEnv,
        *,
        collisions: list[Collision],
        terminated: bool,
        truncated: bool,
        info: dict[str, object],
    ) -> float:
        """Compute a scalar reward contribution."""

        raise NotImplementedError


@dataclass
class ReachReward(RewardFunction):
    """Reward progress toward a target position."""

    effector: str = "right_palm"
    target_position: np.ndarray | None = None
    scale: float = 1.0
    distance_threshold: float = 0.05
    success_bonus: float = 1.0
    _previous_distance: float | None = field(default=None, init=False, repr=False)
    _bonus_paid: bool = field(default=False, init=False, repr=False)

    def reset(self, env: SupportsRewardEnv) -> None:
        target = self._resolve_target(env)
        self._previous_distance = self._distance(env, target) if target is not None else None
        self._bonus_paid = False

    def compute(
        self,
        env: SupportsRewardEnv,
        *,
        collisions: list[Collision],
        terminated: bool,
        truncated: bool,
        info: dict[str, object],
    ) -> float:
        del collisions, terminated, truncated, info
        target = self._resolve_target(env)
        if target is None:
            return 0.0
        distance = self._distance(env, target)
        previous = self._previous_distance if self._previous_distance is not None else distance
        reward = self.scale * (previous - distance)
        if distance <= self.distance_threshold and not self._bonus_paid:
            reward += self.success_bonus
            self._bonus_paid = True
        self._previous_distance = distance
        return float(reward)

    def _resolve_target(self, env: SupportsRewardEnv) -> np.ndarray | None:
        if self.target_position is not None:
            return np.asarray(self.target_position, dtype=np.float64)
        if env.target_position is None:
            return None
        return np.asarray(env.target_position, dtype=np.float64)

    def _distance(self, env: SupportsRewardEnv, target: np.ndarray | None) -> float:
        if target is None:
            return 0.0
        return float(np.linalg.norm(env.end_effector_position(self.effector) - target))


@dataclass
class TaskCompletionReward(RewardFunction):
    """Bonus when the environment marks the current task complete."""

    bonus: float = 10.0
    _awarded: bool = field(default=False, init=False, repr=False)

    def reset(self, env: SupportsRewardEnv) -> None:
        del env
        self._awarded = False

    def compute(
        self,
        env: SupportsRewardEnv,
        *,
        collisions: list[Collision],
        terminated: bool,
        truncated: bool,
        info: dict[str, object],
    ) -> float:
        del collisions, terminated, truncated, info
        if env.is_task_complete() and not self._awarded:
            self._awarded = True
            return float(self.bonus)
        return 0.0


@dataclass
class CollisionPenalty(RewardFunction):
    """Penalty for collisions reported by the simulator."""

    scale: float = 1.0
    use_penetration_depth: bool = False

    def compute(
        self,
        env: SupportsRewardEnv,
        *,
        collisions: list[Collision],
        terminated: bool,
        truncated: bool,
        info: dict[str, object],
    ) -> float:
        del env, terminated, truncated, info
        if self.use_penetration_depth:
            penalty = sum(item.penetration_depth for item in collisions)
        else:
            penalty = float(len(collisions))
        return -self.scale * float(penalty)


@dataclass
class CompositeReward(RewardFunction):
    """Combine multiple reward components additively."""

    rewards: list[RewardFunction]

    def reset(self, env: SupportsRewardEnv) -> None:
        for reward in self.rewards:
            reward.reset(env)

    def compute(
        self,
        env: SupportsRewardEnv,
        *,
        collisions: list[Collision],
        terminated: bool,
        truncated: bool,
        info: dict[str, object],
    ) -> float:
        return float(
            sum(
                reward.compute(
                    env,
                    collisions=collisions,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
                for reward in self.rewards
            )
        )

__all__ = ["SupportsRewardEnv", "RewardFunction", "ReachReward", "TaskCompletionReward", "CollisionPenalty", "CompositeReward"]
