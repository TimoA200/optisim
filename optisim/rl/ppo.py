from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Iterable

import numpy as np

from optisim.rl.buffer import RolloutBatch, RolloutBuffer
from optisim.rl.callbacks import BaseCallback
from optisim.rl.config import PPOConfig
from optisim.rl.network import ActorCritic
from optisim.rl.optimizer import PPOOptimizer


@dataclass(slots=True)
class TrainingResult:
    episode_rewards: list[float]
    episode_lengths: list[int]
    losses: list[float]
    mean_reward: float
    updates: int
    stopped_early: bool


class PPOTrainer:
    def __init__(
        self,
        config: PPOConfig | None = None,
        network: ActorCritic | None = None,
        callbacks: Iterable[BaseCallback] | None = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.network = network
        self.optimizer = PPOOptimizer()
        self.callbacks = list(callbacks or [])
        self.update_count = 0
        self.latest_metrics: dict[str, float] = {}
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self._obs: np.ndarray | None = None
        self._episode_reward = 0.0
        self._episode_length = 0
        self._last_info: dict[str, Any] = {}

    def _init_network(self, env: Any) -> None:
        if self.network is not None:
            return
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])
        self.network = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=self.config.hidden_sizes,
            tanh_squash=self.config.tanh_squash,
            seed=self.config.seed,
        )

    def collect_rollout(self, env: Any, network: ActorCritic, buffer: RolloutBuffer) -> None:
        if self._obs is None:
            self._obs, self._last_info = env.reset(seed=self.config.seed)
        for callback in self.callbacks:
            callback.on_rollout_start(self)
        buffer.clear()
        for _ in range(self.config.n_steps):
            action, log_prob, value = network.get_action(self._obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            buffer.add(self._obs, action, reward, value, log_prob, done)
            self._episode_reward += float(reward)
            self._episode_length += 1
            self._obs = next_obs
            self._last_info = info
            if done:
                self.episode_rewards.append(self._episode_reward)
                self.episode_lengths.append(self._episode_length)
                self._episode_reward = 0.0
                self._episode_length = 0
                self._obs, self._last_info = env.reset()
        if buffer.size == 0:
            last_value = 0.0
            done = True
        else:
            _, _, last_value = network.forward(self._obs)
            done = bool(buffer.dones[buffer.size - 1])
        buffer.compute_returns_and_advantages(last_value=float(last_value), done=done)
        for callback in self.callbacks:
            callback.on_rollout_end(self)

    def compute_ppo_loss(
        self,
        network: ActorCritic,
        batch: RolloutBatch,
        return_details: bool = False,
    ) -> float | dict[str, float]:
        log_probs, values, entropy = network.evaluate_actions(batch.observations, batch.actions)
        ratio = np.exp(log_probs - batch.log_probs)
        unclipped = ratio * batch.advantages
        clipped = np.clip(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * batch.advantages
        policy_loss = float(np.mean(np.minimum(unclipped, clipped)))
        value_loss = float(np.mean((values - batch.returns) ** 2))
        entropy_bonus = float(np.mean(entropy))
        total_loss = float(-policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy_bonus)
        if return_details:
            return {
                "loss": total_loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy_bonus,
            }
        return total_loss

    def compute_gradients(self, network: ActorCritic, batch: RolloutBatch) -> np.ndarray:
        params = network.get_params()
        grads = np.zeros_like(params)
        h = float(self.config.fd_epsilon)
        for index in range(params.size):
            plus = params.copy()
            minus = params.copy()
            plus[index] += h
            minus[index] -= h
            network.set_params(plus)
            loss_plus = float(self.compute_ppo_loss(network, batch))
            network.set_params(minus)
            loss_minus = float(self.compute_ppo_loss(network, batch))
            grads[index] = (loss_plus - loss_minus) / (2.0 * h)
        network.set_params(params)
        return grads

    def update_network(self, network: ActorCritic, optimizer: PPOOptimizer, grads: np.ndarray) -> None:
        grad_vector = np.asarray(grads, dtype=float)
        grad_norm = float(np.linalg.norm(grad_vector))
        if self.config.max_grad_norm > 0.0 and grad_norm > self.config.max_grad_norm:
            grad_vector = grad_vector * (self.config.max_grad_norm / (grad_norm + 1e-8))
        updated = optimizer.step(network.get_params(), grad_vector, self.config.lr)
        network.set_params(updated)

    def train(self, env: Any) -> TrainingResult:
        self._init_network(env)
        assert self.network is not None
        buffer = RolloutBuffer(
            capacity=self.config.n_steps,
            obs_dim=self.network.obs_dim,
            act_dim=self.network.act_dim,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        losses: list[float] = []
        stopped_early = False
        num_updates = max(1, ceil(self.config.total_timesteps / self.config.n_steps))

        for _ in range(num_updates):
            self.collect_rollout(env, self.network, buffer)
            update_losses: list[float] = []
            for _ in range(self.config.n_epochs):
                for batch in buffer.get_minibatches(self.config.minibatch_size):
                    grads = self.compute_gradients(self.network, batch)
                    self.update_network(self.network, self.optimizer, grads)
                    metrics = self.compute_ppo_loss(self.network, batch, return_details=True)
                    assert isinstance(metrics, dict)
                    update_losses.append(float(metrics["loss"]))
                    self.latest_metrics = {
                        "loss": float(metrics["loss"]),
                        "mean_reward": float(np.mean(self.episode_rewards[-10:])) if self.episode_rewards else 0.0,
                    }
            self.update_count += 1
            losses.append(float(np.mean(update_losses) if update_losses else 0.0))
            for callback in self.callbacks:
                callback.on_update(self)
            if any(callback.stop_training for callback in self.callbacks):
                stopped_early = True
                break

        for callback in self.callbacks:
            callback.on_training_end(self)

        return TrainingResult(
            episode_rewards=list(self.episode_rewards),
            episode_lengths=list(self.episode_lengths),
            losses=losses,
            mean_reward=float(np.mean(self.episode_rewards) if self.episode_rewards else 0.0),
            updates=self.update_count,
            stopped_early=stopped_early,
        )
