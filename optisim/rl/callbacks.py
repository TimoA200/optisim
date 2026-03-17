from __future__ import annotations

from pathlib import Path

import numpy as np

from optisim.rl.network import ActorCritic


class BaseCallback:
    def __init__(self) -> None:
        self.stop_training = False

    def on_rollout_start(self, trainer: object) -> None:
        del trainer

    def on_rollout_end(self, trainer: object) -> None:
        del trainer

    def on_update(self, trainer: object) -> None:
        del trainer

    def on_training_end(self, trainer: object) -> None:
        del trainer


class LoggingCallback(BaseCallback):
    def __init__(self, log_interval: int = 10) -> None:
        super().__init__()
        self.log_interval = max(1, int(log_interval))

    def on_update(self, trainer: object) -> None:
        update_count = int(getattr(trainer, "update_count", 0))
        if update_count % self.log_interval != 0:
            return
        metrics = getattr(trainer, "latest_metrics", {})
        mean_reward = float(metrics.get("mean_reward", 0.0))
        loss = float(metrics.get("loss", 0.0))
        print(f"update={update_count} mean_reward={mean_reward:.3f} loss={loss:.6f}")


class CheckpointCallback(BaseCallback):
    def __init__(self, checkpoint_freq: int, output_dir: str | Path) -> None:
        super().__init__()
        self.checkpoint_freq = max(1, int(checkpoint_freq))
        self.output_dir = Path(output_dir)
        self.saved_paths: list[Path] = []

    def on_update(self, trainer: object) -> None:
        update_count = int(getattr(trainer, "update_count", 0))
        if update_count % self.checkpoint_freq != 0:
            return
        network = getattr(trainer, "network", None)
        if not isinstance(network, ActorCritic):
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"ppo_update_{update_count}.npz"
        network.save(str(path))
        self.saved_paths.append(path)


class EarlyStopCallback(BaseCallback):
    def __init__(self, target_reward: float, window_size: int = 5) -> None:
        super().__init__()
        self.target_reward = float(target_reward)
        self.window_size = max(1, int(window_size))

    def on_update(self, trainer: object) -> None:
        rewards = list(getattr(trainer, "episode_rewards", []))
        if not rewards:
            return
        if float(np.mean(rewards[-self.window_size :])) >= self.target_reward:
            self.stop_training = True
