from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from optisim.rl.network import ActorCritic


@dataclass(slots=True)
class EvalResult:
    mean_reward: float
    std_reward: float
    mean_length: float
    episode_rewards: list[float]


def evaluate_policy(env: Any, network: ActorCritic, n_episodes: int) -> EvalResult:
    rewards: list[float] = []
    lengths: list[int] = []
    for episode in range(int(n_episodes)):
        obs, _ = env.reset(seed=episode)
        done = False
        total_reward = 0.0
        length = 0
        while not done:
            action, _, _ = network.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            length += 1
            done = bool(terminated or truncated)
        rewards.append(total_reward)
        lengths.append(length)
    return EvalResult(
        mean_reward=float(np.mean(rewards) if rewards else 0.0),
        std_reward=float(np.std(rewards) if rewards else 0.0),
        mean_length=float(np.mean(lengths) if lengths else 0.0),
        episode_rewards=rewards,
    )


def record_episode(env: Any, network: ActorCritic) -> list[tuple[np.ndarray, np.ndarray, float]]:
    obs, _ = env.reset()
    done = False
    frames: list[tuple[np.ndarray, np.ndarray, float]] = []
    while not done:
        action, _, _ = network.get_action(obs, deterministic=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        frames.append((np.asarray(obs, dtype=float).copy(), np.asarray(action, dtype=float).copy(), float(reward)))
        obs = next_obs
        done = bool(terminated or truncated)
    return frames
