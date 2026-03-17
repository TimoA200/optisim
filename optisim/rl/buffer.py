from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(slots=True)
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
    dones: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class RolloutBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.observations = np.zeros((self.capacity, self.obs_dim), dtype=float)
        self.actions = np.zeros((self.capacity, self.act_dim), dtype=float)
        self.rewards = np.zeros(self.capacity, dtype=float)
        self.values = np.zeros(self.capacity, dtype=float)
        self.log_probs = np.zeros(self.capacity, dtype=float)
        self.dones = np.zeros(self.capacity, dtype=bool)
        self.advantages = np.zeros(self.capacity, dtype=float)
        self.returns = np.zeros(self.capacity, dtype=float)
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        if self.size >= self.capacity:
            raise ValueError("RolloutBuffer is full.")
        self.observations[self.size] = np.asarray(obs, dtype=float)
        self.actions[self.size] = np.asarray(action, dtype=float)
        self.rewards[self.size] = float(reward)
        self.values[self.size] = float(value)
        self.log_probs[self.size] = float(log_prob)
        self.dones[self.size] = bool(done)
        self.size += 1

    def compute_returns_and_advantages(self, last_value: float, done: bool) -> None:
        if self.size == 0:
            return
        advantages = np.zeros(self.size, dtype=float)
        last_advantage = 0.0
        next_value = 0.0 if done else float(last_value)
        next_non_terminal = 0.0 if done else 1.0
        for index in range(self.size - 1, -1, -1):
            if index < self.size - 1:
                next_value = self.values[index + 1]
                next_non_terminal = 0.0 if self.dones[index] else 1.0
            delta = self.rewards[index] + self.gamma * next_value * next_non_terminal - self.values[index]
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            advantages[index] = last_advantage

        self.advantages[: self.size] = advantages
        self.returns[: self.size] = advantages + self.values[: self.size]

        if self.size > 1:
            mean = float(np.mean(self.advantages[: self.size]))
            std = float(np.std(self.advantages[: self.size]))
            if std > 1e-8:
                self.advantages[: self.size] = (self.advantages[: self.size] - mean) / std
            else:
                self.advantages[: self.size] = self.advantages[: self.size] - mean

    def get_minibatches(self, minibatch_size: int) -> Iterator[RolloutBatch]:
        if self.size == 0:
            return iter(())
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        batch_size = max(1, int(minibatch_size))
        for start in range(0, self.size, batch_size):
            batch_indices = indices[start : start + batch_size]
            yield RolloutBatch(
                observations=self.observations[batch_indices].copy(),
                actions=self.actions[batch_indices].copy(),
                rewards=self.rewards[batch_indices].copy(),
                values=self.values[batch_indices].copy(),
                log_probs=self.log_probs[batch_indices].copy(),
                dones=self.dones[batch_indices].copy(),
                advantages=self.advantages[batch_indices].copy(),
                returns=self.returns[batch_indices].copy(),
            )

    def clear(self) -> None:
        self.size = 0

