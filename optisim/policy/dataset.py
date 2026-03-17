from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from optisim.lfd import Demonstration


def _as_float_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    return array.reshape(-1)


def _flatten_extras(extras: dict[str, Any]) -> np.ndarray:
    if not extras:
        return np.zeros(0, dtype=float)
    flattened: list[float] = []
    for key in sorted(extras):
        value = extras[key]
        if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            flattened.append(float(value))
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            array = np.asarray(value, dtype=float)
            if array.size:
                flattened.extend(array.reshape(-1).tolist())
    return np.asarray(flattened, dtype=float)


@dataclass(slots=True)
class PolicyObservation:
    joint_positions: np.ndarray
    ee_pose: np.ndarray
    gripper_open: float
    extras_flat: np.ndarray

    def as_array(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.joint_positions, dtype=float).reshape(-1),
                np.asarray(self.ee_pose, dtype=float).reshape(-1),
                np.array([float(self.gripper_open)], dtype=float),
            ]
        )


@dataclass(slots=True)
class PolicyAction:
    joint_delta: np.ndarray
    gripper_cmd: float

    def as_array(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.joint_delta, dtype=float).reshape(-1),
                np.array([float(self.gripper_cmd)], dtype=float),
            ]
        )


@dataclass(slots=True)
class NormStats:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        values = np.asarray(action, dtype=float)
        return values * self.act_std + self.act_mean

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        values = np.asarray(obs, dtype=float)
        return (values - self.obs_mean) / self.obs_std


class PolicyDataset:
    def __init__(self, observations: np.ndarray, actions: np.ndarray) -> None:
        obs_array = np.asarray(observations, dtype=float)
        act_array = np.asarray(actions, dtype=float)
        if obs_array.ndim != 2 or act_array.ndim != 2:
            raise ValueError("observations and actions must be 2D arrays.")
        if obs_array.shape[0] != act_array.shape[0]:
            raise ValueError("observations and actions must have the same number of rows.")
        self.observations = obs_array.copy()
        self.actions = act_array.copy()
        self.obs_dim = int(self.observations.shape[1]) if self.observations.size else 0
        self.act_dim = int(self.actions.shape[1]) if self.actions.size else 0

    @classmethod
    def from_demonstrations(
        cls,
        demos: list[Demonstration],
        delta_actions: bool = True,
    ) -> PolicyDataset:
        if not demos:
            raise ValueError("At least one demonstration is required.")

        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        joint_dim: int | None = None

        for demo in demos:
            if not demo.steps:
                continue
            if joint_dim is None:
                joint_dim = len(demo.steps[0].joint_positions)
            for index, step in enumerate(demo.steps):
                current_joint = _as_float_array(step.joint_positions)
                current_ee = _as_float_array(step.ee_pose)
                if joint_dim is None:
                    joint_dim = current_joint.size
                if current_joint.size != joint_dim:
                    raise ValueError("All demonstration steps must have the same joint dimension.")
                if current_ee.size != 7:
                    raise ValueError("Each ee_pose must have exactly 7 elements.")

                observation = PolicyObservation(
                    joint_positions=current_joint,
                    ee_pose=current_ee,
                    gripper_open=float(step.gripper_open),
                    extras_flat=_flatten_extras(step.extras),
                )
                observations.append(observation.as_array())

                if index + 1 < len(demo.steps):
                    next_step = demo.steps[index + 1]
                    next_joint = _as_float_array(next_step.joint_positions)
                    gripper_cmd = float(next_step.gripper_open)
                else:
                    next_joint = current_joint.copy()
                    gripper_cmd = float(step.gripper_open)

                joint_delta = next_joint - current_joint if delta_actions else next_joint.copy()
                action = PolicyAction(joint_delta=joint_delta, gripper_cmd=gripper_cmd)
                actions.append(action.as_array())

        if joint_dim is None:
            raise ValueError("Demonstrations must contain at least one step.")

        return cls(np.vstack(observations), np.vstack(actions))

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def normalize(self) -> NormStats:
        if len(self) == 0:
            raise ValueError("Cannot normalize an empty dataset.")
        obs_mean = np.mean(self.observations, axis=0)
        obs_std = np.std(self.observations, axis=0)
        obs_std = np.where(obs_std < 1e-8, 1.0, obs_std)
        act_mean = np.mean(self.actions, axis=0)
        act_std = np.std(self.actions, axis=0)
        act_std = np.where(act_std < 1e-8, 1.0, act_std)
        if act_mean.size:
            act_mean[-1] = 0.0
            act_std[-1] = 1.0
        self.observations = (self.observations - obs_mean) / obs_std
        self.actions = (self.actions - act_mean) / act_std
        return NormStats(obs_mean=obs_mean, obs_std=obs_std, act_mean=act_mean, act_std=act_std)

    def apply_norm(self, stats: NormStats) -> None:
        self.observations = (self.observations - stats.obs_mean) / stats.obs_std
        self.actions = (self.actions - stats.act_mean) / stats.act_std

    def split(self, train_fraction: float = 0.8, seed: int = 42) -> tuple[PolicyDataset, PolicyDataset]:
        if len(self) == 0:
            raise ValueError("Cannot split an empty dataset.")
        if len(self) == 1:
            clone = PolicyDataset(self.observations.copy(), self.actions.copy())
            return clone, PolicyDataset(self.observations.copy(), self.actions.copy())

        rng = np.random.default_rng(seed)
        indices = np.arange(len(self))
        rng.shuffle(indices)
        split_index = int(round(len(self) * float(train_fraction)))
        split_index = min(max(split_index, 1), len(self) - 1)
        train_indices = indices[:split_index]
        val_indices = indices[split_index:]
        return (
            PolicyDataset(self.observations[train_indices], self.actions[train_indices]),
            PolicyDataset(self.observations[val_indices], self.actions[val_indices]),
        )

    def get_batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch_indices = np.asarray(indices, dtype=int)
        return self.observations[batch_indices], self.actions[batch_indices]
