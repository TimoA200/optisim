"""Useful wrappers for optisim Gymnasium environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from optisim.sim import SimulationRecording

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    gym = None
    spaces = None


def _require_gymnasium() -> None:
    if gym is None or spaces is None:
        raise ModuleNotFoundError(
            "gymnasium is required for optisim.gym_env wrappers. Install with `pip install optisim[rl]`."
        )


if gym is not None:

    class NormalizeObservation(gym.ObservationWrapper[np.ndarray, np.ndarray, np.ndarray]):
        """Online normalization for flat continuous observations."""

        def __init__(self, env: gym.Env[np.ndarray, np.ndarray], epsilon: float = 1e-8) -> None:
            super().__init__(env)
            self.epsilon = float(epsilon)
            self.count = 0
            self.mean = np.zeros(self.observation_space.shape, dtype=np.float64)
            self.m2 = np.ones(self.observation_space.shape, dtype=np.float64)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.env.observation_space.shape,
                dtype=np.float32,
            )

        def observation(self, observation: np.ndarray) -> np.ndarray:
            obs = np.asarray(observation, dtype=np.float64)
            self.count += 1
            delta = obs - self.mean
            self.mean += delta / self.count
            delta2 = obs - self.mean
            self.m2 += delta * delta2
            variance = np.maximum(self.m2 / max(self.count, 1), self.epsilon)
            normalized = (obs - self.mean) / np.sqrt(variance)
            return normalized.astype(np.float32)


    class FlattenJoints(gym.ObservationWrapper[np.ndarray, np.ndarray, np.ndarray]):
        """Expose only the joint-position slice from an optisim observation."""

        def __init__(self, env: gym.Env[np.ndarray, np.ndarray]) -> None:
            super().__init__(env)
            joint_names = self.unwrapped.observation_config.joint_names or list(self.unwrapped.robot.joints)
            self._joint_dim = len(joint_names)
            self.observation_space = spaces.Box(
                low=self.unwrapped.observation_space.low[: self._joint_dim],
                high=self.unwrapped.observation_space.high[: self._joint_dim],
                dtype=np.float32,
            )

        def observation(self, observation: np.ndarray) -> np.ndarray:
            del observation
            components = self.unwrapped.observation_components()
            return components.get("joint_positions", np.zeros((0,), dtype=np.float32)).astype(np.float32)


    class RecordEpisode(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """Persist completed episodes using the existing optisim recording format."""

        def __init__(self, env: gym.Env[np.ndarray, np.ndarray], output_dir: str | Path) -> None:
            super().__init__(env)
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.episode_index = 0
            self.last_recording_path: Path | None = None

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            return self.env.reset(seed=seed, options=options)

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            observation, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                self._dump_recording()
            return observation, reward, terminated, truncated, info

        def _dump_recording(self) -> None:
            recording = getattr(self.unwrapped, "recording", None)
            if not isinstance(recording, SimulationRecording):
                return
            path = self.output_dir / f"episode_{self.episode_index:04d}.json"
            recording.dump(path)
            self.last_recording_path = path
            self.episode_index += 1


else:

    class NormalizeObservation:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            _require_gymnasium()


    class FlattenJoints:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            _require_gymnasium()


    class RecordEpisode:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            _require_gymnasium()
