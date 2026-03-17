"""Basic random-agent demo for the optisim Gymnasium environment."""

from __future__ import annotations

import importlib.util


def main() -> int:
    if importlib.util.find_spec("gymnasium") is None:
        print("Install gym support first: pip install optisim[gym]")
        return 1

    import gymnasium as gym

    from optisim.gym_env import register_optisim_env

    env_id = register_optisim_env(task_definition="examples/pick_and_place.yaml", max_steps=100)
    env = gym.make(env_id)
    try:
        observation, info = env.reset(seed=0)
        total_reward = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
        print(f"episode complete reward={total_reward:.3f} task_complete={info['task_complete']}")
        del observation
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["main"]
