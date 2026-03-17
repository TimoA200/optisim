from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from optisim.policy.dataset import NormStats
from optisim.policy.network import PolicyNetwork


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(slots=True)
class PolicyStep:
    joint_positions_next: list[float]
    gripper_open: bool
    raw_action: np.ndarray


class NeuralPolicy:
    def __init__(self, network: PolicyNetwork, norm_stats: NormStats, joint_dim: int) -> None:
        self.network = network
        self.norm_stats = norm_stats
        self.joint_dim = int(joint_dim)

    def _normalized_obs(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape[-1] == self.norm_stats.obs_mean.shape[0]:
            return self.norm_stats.normalize_obs(obs)
        raise ValueError("Observation dimension does not match normalization statistics.")

    def _build_observation(
        self,
        joint_positions: list[float],
        ee_pose: tuple[float, ...],
        gripper_open: bool,
    ) -> np.ndarray:
        joint_array = np.asarray(joint_positions, dtype=float).reshape(-1)
        ee_array = np.asarray(ee_pose, dtype=float).reshape(-1)
        if joint_array.size != self.joint_dim:
            raise ValueError("joint_positions dimension does not match policy joint_dim.")
        if ee_array.size != 7:
            raise ValueError("ee_pose must contain exactly 7 elements.")
        return np.concatenate([joint_array, ee_array, np.array([float(gripper_open)], dtype=float)])

    def act(
        self,
        joint_positions: list[float],
        ee_pose: tuple[float, ...],
        gripper_open: bool,
    ) -> PolicyStep:
        current_joint = np.asarray(joint_positions, dtype=float).reshape(-1)
        obs = self._build_observation(joint_positions, ee_pose, gripper_open)
        normalized_obs = self._normalized_obs(obs)
        normalized_action = self.network.predict(normalized_obs)
        action = self.norm_stats.denormalize_action(normalized_action)
        joint_delta = action[: self.joint_dim]
        gripper_logit = float(action[-1])
        next_joint = current_joint + joint_delta
        return PolicyStep(
            joint_positions_next=next_joint.tolist(),
            gripper_open=bool(gripper_logit >= 0.0),
            raw_action=action,
        )

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "input_dim": np.array(self.network.input_dim, dtype=int),
            "output_dim": np.array(self.network.output_dim, dtype=int),
            "hidden_dims": np.array(self.network.hidden_dims, dtype=int),
            "activation": np.array(self.network.activation),
            "joint_dim": np.array(self.joint_dim, dtype=int),
            "obs_mean": self.norm_stats.obs_mean,
            "obs_std": self.norm_stats.obs_std,
            "act_mean": self.norm_stats.act_mean,
            "act_std": self.norm_stats.act_std,
            "num_layers": np.array(len(self.network.layers), dtype=int),
        }
        for index, layer in enumerate(self.network.layers):
            payload[f"W_{index}"] = layer["W"]
            payload[f"b_{index}"] = layer["b"]
        np.savez(destination, **payload)

    @classmethod
    def load(cls, path: str) -> NeuralPolicy:
        data = np.load(Path(path), allow_pickle=False)
        network = PolicyNetwork(
            input_dim=int(np.asarray(data["input_dim"]).item()),
            hidden_dims=tuple(int(value) for value in np.asarray(data["hidden_dims"], dtype=int).tolist()),
            output_dim=int(np.asarray(data["output_dim"]).item()),
            activation=str(np.asarray(data["activation"]).item()),
        )
        params = []
        for index in range(int(np.asarray(data["num_layers"]).item())):
            params.append(
                {
                    "W": np.asarray(data[f"W_{index}"], dtype=float),
                    "b": np.asarray(data[f"b_{index}"], dtype=float),
                }
            )
        network.set_parameters(params)
        stats = NormStats(
            obs_mean=np.asarray(data["obs_mean"], dtype=float),
            obs_std=np.asarray(data["obs_std"], dtype=float),
            act_mean=np.asarray(data["act_mean"], dtype=float),
            act_std=np.asarray(data["act_std"], dtype=float),
        )
        return cls(network=network, norm_stats=stats, joint_dim=int(np.asarray(data["joint_dim"]).item()))

    def reset(self) -> None:
        return None


class RecurrentNeuralPolicy(NeuralPolicy):
    def __init__(
        self,
        network: PolicyNetwork,
        norm_stats: NormStats,
        joint_dim: int,
        history_len: int = 5,
    ) -> None:
        super().__init__(network=network, norm_stats=norm_stats, joint_dim=joint_dim)
        self.history_len = max(1, int(history_len))
        self._history: list[np.ndarray] = []

    def _normalized_obs(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape[-1] == self.norm_stats.obs_mean.shape[0]:
            return self.norm_stats.normalize_obs(obs)
        base_dim = self.norm_stats.obs_mean.shape[0]
        if obs.shape[-1] == base_dim * self.history_len:
            tiled_mean = np.tile(self.norm_stats.obs_mean, self.history_len)
            tiled_std = np.tile(self.norm_stats.obs_std, self.history_len)
            return (obs - tiled_mean) / tiled_std
        raise ValueError("Observation history dimension does not match normalization statistics.")

    def _history_observation(self, obs: np.ndarray) -> np.ndarray:
        self._history.append(obs.copy())
        if len(self._history) > self.history_len:
            self._history = self._history[-self.history_len :]
        if len(self._history) < self.history_len:
            padding = [np.zeros_like(obs) for _ in range(self.history_len - len(self._history))]
            window = padding + self._history
        else:
            window = self._history
        return np.concatenate(window)

    def act(
        self,
        joint_positions: list[float],
        ee_pose: tuple[float, ...],
        gripper_open: bool,
    ) -> PolicyStep:
        current_joint = np.asarray(joint_positions, dtype=float).reshape(-1)
        obs = self._build_observation(joint_positions, ee_pose, gripper_open)
        history_obs = self._history_observation(obs)
        normalized_obs = self._normalized_obs(history_obs)
        normalized_action = self.network.predict(normalized_obs)
        action = self.norm_stats.denormalize_action(normalized_action)
        next_joint = current_joint + action[: self.joint_dim]
        return PolicyStep(
            joint_positions_next=next_joint.tolist(),
            gripper_open=bool(float(action[-1]) >= 0.0),
            raw_action=action,
        )

    def reset(self) -> None:
        self._history.clear()

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "input_dim": np.array(self.network.input_dim, dtype=int),
            "output_dim": np.array(self.network.output_dim, dtype=int),
            "hidden_dims": np.array(self.network.hidden_dims, dtype=int),
            "activation": np.array(self.network.activation),
            "joint_dim": np.array(self.joint_dim, dtype=int),
            "history_len": np.array(self.history_len, dtype=int),
            "obs_mean": self.norm_stats.obs_mean,
            "obs_std": self.norm_stats.obs_std,
            "act_mean": self.norm_stats.act_mean,
            "act_std": self.norm_stats.act_std,
            "num_layers": np.array(len(self.network.layers), dtype=int),
        }
        for index, layer in enumerate(self.network.layers):
            payload[f"W_{index}"] = layer["W"]
            payload[f"b_{index}"] = layer["b"]
        np.savez(destination, **payload)
