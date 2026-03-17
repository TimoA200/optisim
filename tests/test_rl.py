from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from optisim import ActorCritic, PPOConfig, PPOTrainer, RLTrainingResult, RolloutBuffer, evaluate_policy
from optisim.rl import (
    BaseCallback,
    CheckpointCallback,
    EarlyStopCallback,
    PPOOptimizer,
    RolloutBatch,
    record_episode,
)


class _Space:
    def __init__(self, shape: tuple[int, ...], low: float = -1.0, high: float = 1.0) -> None:
        self.shape = shape
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)

    def sample(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.float32)


class _MockEnv:
    def __init__(self, obs_dim: int = 3, act_dim: int = 2, max_steps: int = 8, reward_bias: float = 0.0) -> None:
        self.observation_space = _Space((obs_dim,), low=-5.0, high=5.0)
        self.action_space = _Space((act_dim,), low=-1.0, high=1.0)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_steps = max_steps
        self.reward_bias = float(reward_bias)
        self.steps = 0
        self.state = np.zeros(self.obs_dim, dtype=float)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, float]]:
        rng = np.random.default_rng(seed)
        self.steps = 0
        self.state = rng.normal(scale=0.1, size=self.obs_dim).astype(float)
        return self.state.copy(), {"seed": float(seed or 0)}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        clipped = np.asarray(action, dtype=float).reshape(self.act_dim)
        self.steps += 1
        self.state[: self.act_dim] = self.state[: self.act_dim] + 0.1 * clipped
        reward = float(self.reward_bias - np.sum(clipped * clipped))
        terminated = False
        truncated = self.steps >= self.max_steps
        return self.state.copy(), reward, terminated, truncated, {"steps": float(self.steps)}


class _CountingCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.rollout_start = 0
        self.rollout_end = 0
        self.update = 0
        self.training_end = 0

    def on_rollout_start(self, trainer: object) -> None:
        del trainer
        self.rollout_start += 1

    def on_rollout_end(self, trainer: object) -> None:
        del trainer
        self.rollout_end += 1

    def on_update(self, trainer: object) -> None:
        del trainer
        self.update += 1

    def on_training_end(self, trainer: object) -> None:
        del trainer
        self.training_end += 1


def _build_network(obs_dim: int = 3, act_dim: int = 2, hidden_sizes: tuple[int, ...] = (16, 16)) -> ActorCritic:
    return ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes, seed=0)


def _make_batch(obs_dim: int = 3, act_dim: int = 2, batch_size: int = 4) -> RolloutBatch:
    observations = np.linspace(-0.2, 0.3, batch_size * obs_dim).reshape(batch_size, obs_dim)
    actions = np.linspace(-0.1, 0.2, batch_size * act_dim).reshape(batch_size, act_dim)
    rewards = np.linspace(0.0, 1.0, batch_size)
    values = np.linspace(0.1, 0.4, batch_size)
    log_probs = np.linspace(-0.5, -0.2, batch_size)
    dones = np.array([False] * batch_size, dtype=bool)
    advantages = np.linspace(-1.0, 1.0, batch_size)
    returns = values + advantages
    return RolloutBatch(observations, actions, rewards, values, log_probs, dones, advantages, returns)


def test_rollout_buffer_add_and_store_values() -> None:
    buffer = RolloutBuffer(capacity=4, obs_dim=2, act_dim=1)
    buffer.add([1.0, 2.0], [0.5], 1.5, 0.2, -0.1, False)

    assert buffer.size == 1
    assert np.allclose(buffer.observations[0], [1.0, 2.0])
    assert np.allclose(buffer.actions[0], [0.5])
    assert buffer.rewards[0] == pytest.approx(1.5)


def test_rollout_buffer_gae_matches_hand_verified_two_step_case() -> None:
    buffer = RolloutBuffer(capacity=2, obs_dim=1, act_dim=1, gamma=0.99, gae_lambda=0.95)
    buffer.add([0.0], [0.0], 1.0, 0.5, -0.1, False)
    buffer.add([0.0], [0.0], 2.0, 0.4, -0.2, True)
    buffer.compute_returns_and_advantages(last_value=0.0, done=True)

    raw_advantages = np.array([2.4008, 1.6])
    normalized = (raw_advantages - raw_advantages.mean()) / raw_advantages.std()
    returns = raw_advantages + np.array([0.5, 0.4])

    assert np.allclose(buffer.advantages[:2], normalized, atol=1e-4)
    assert np.allclose(buffer.returns[:2], returns, atol=1e-4)


def test_rollout_buffer_normalizes_advantages() -> None:
    buffer = RolloutBuffer(capacity=3, obs_dim=1, act_dim=1)
    for reward, value in [(1.0, 0.1), (0.5, 0.2), (0.2, 0.3)]:
        buffer.add([0.0], [0.0], reward, value, 0.0, False)
    buffer.compute_returns_and_advantages(last_value=0.0, done=True)

    assert np.mean(buffer.advantages[:3]) == pytest.approx(0.0, abs=1e-7)
    assert np.std(buffer.advantages[:3]) == pytest.approx(1.0, rel=1e-6)


def test_rollout_buffer_minibatches_cover_all_data_once() -> None:
    np.random.seed(0)
    buffer = RolloutBuffer(capacity=5, obs_dim=1, act_dim=1)
    for index in range(5):
        buffer.add([float(index)], [float(index)], float(index), float(index), -float(index), False)
    buffer.compute_returns_and_advantages(last_value=0.0, done=True)

    seen = []
    for batch in buffer.get_minibatches(minibatch_size=2):
        seen.extend(int(value[0]) for value in batch.observations)

    assert sorted(seen) == [0, 1, 2, 3, 4]
    assert len(seen) == len(set(seen))


def test_rollout_buffer_clear_resets_size() -> None:
    buffer = RolloutBuffer(capacity=2, obs_dim=1, act_dim=1)
    buffer.add([0.0], [0.0], 0.0, 0.0, 0.0, False)
    buffer.clear()

    assert buffer.size == 0


def test_actor_critic_forward_returns_expected_shapes_for_batch() -> None:
    network = _build_network()
    mean, log_std, value = network.forward(np.ones((5, 3), dtype=float))

    assert mean.shape == (5, 2)
    assert log_std.shape == (5, 2)
    assert value.shape == (5,)


def test_actor_critic_forward_returns_expected_shapes_for_single_observation() -> None:
    network = _build_network()
    mean, log_std, value = network.forward(np.ones(3, dtype=float))

    assert mean.shape == (2,)
    assert log_std.shape == (2,)
    assert isinstance(float(value), float)


def test_actor_critic_get_action_returns_finite_values() -> None:
    network = _build_network()
    action, log_prob, value = network.get_action(np.zeros(3, dtype=float))

    assert action.shape == (2,)
    assert np.isfinite(action).all()
    assert np.isfinite(log_prob)
    assert np.isfinite(value)


def test_actor_critic_evaluate_actions_matches_batch_size() -> None:
    network = _build_network()
    batch_obs = np.ones((4, 3), dtype=float)
    batch_actions = np.zeros((4, 2), dtype=float)
    log_probs, values, entropy = network.evaluate_actions(batch_obs, batch_actions)

    assert log_probs.shape == (4,)
    assert values.shape == (4,)
    assert entropy.shape == (4,)


def test_actor_critic_entropy_is_positive() -> None:
    network = _build_network()
    _, _, entropy = network.evaluate_actions(np.ones((3, 3), dtype=float), np.zeros((3, 2), dtype=float))

    assert np.all(entropy > 0.0)


def test_actor_critic_get_set_params_roundtrip() -> None:
    network = _build_network()
    params = network.get_params()
    updated = params + 0.5
    network.set_params(updated)

    assert np.allclose(network.get_params(), updated)


def test_actor_critic_deterministic_action_equals_mean() -> None:
    network = _build_network()
    observation = np.array([0.1, -0.2, 0.3], dtype=float)
    mean, _, _ = network.forward(observation)
    action, _, _ = network.get_action(observation, deterministic=True)

    assert np.allclose(action, mean)


def test_actor_critic_backward_matches_parameter_vector_shape() -> None:
    network = _build_network()
    grads = network.backward(
        np.ones((2, 3), dtype=float),
        np.ones((2, 2), dtype=float),
        np.ones(2, dtype=float),
    )

    assert grads.shape == network.get_params().shape


def test_actor_critic_save_load_roundtrip(tmp_path: Path) -> None:
    network = _build_network()
    path = tmp_path / "actor_critic.npz"
    output_before = network.forward(np.ones((2, 3), dtype=float))
    network.save(str(path))
    restored = ActorCritic.load(str(path))
    output_after = restored.forward(np.ones((2, 3), dtype=float))

    assert np.allclose(output_before[0], output_after[0])
    assert np.allclose(output_before[2], output_after[2])


def test_ppo_optimizer_step_moves_params_in_negative_gradient_direction() -> None:
    optimizer = PPOOptimizer()
    params = np.array([1.0, -1.0], dtype=float)
    grads = np.array([0.2, -0.4], dtype=float)
    updated = optimizer.step(params, grads, lr=0.1)

    assert updated[0] < params[0]
    assert updated[1] > params[1]


def test_ppo_optimizer_accumulates_moments_across_steps() -> None:
    optimizer = PPOOptimizer()
    params = np.array([1.0, 2.0], dtype=float)
    grads = np.array([0.5, -0.25], dtype=float)
    optimizer.step(params, grads, lr=0.1)
    first_m = optimizer.m.copy()
    optimizer.step(params, grads, lr=0.1)

    assert optimizer.t == 2
    assert np.linalg.norm(optimizer.m) > np.linalg.norm(first_m)


def test_ppo_optimizer_reset_clears_state() -> None:
    optimizer = PPOOptimizer()
    optimizer.step(np.array([1.0], dtype=float), np.array([0.1], dtype=float), lr=0.1)
    optimizer.reset()

    assert optimizer.m is None
    assert optimizer.v is None
    assert optimizer.t == 0


def test_collect_rollout_fills_buffer_to_capacity() -> None:
    env = _MockEnv()
    trainer = PPOTrainer(PPOConfig(n_steps=5, total_timesteps=5, hidden_sizes=(16, 16)))
    network = _build_network(obs_dim=3, act_dim=2)
    buffer = RolloutBuffer(capacity=5, obs_dim=3, act_dim=2)
    trainer.collect_rollout(env, network, buffer)

    assert buffer.size == 5
    assert np.isfinite(buffer.advantages[:5]).all()


def test_compute_ppo_loss_returns_scalar_float() -> None:
    trainer = PPOTrainer(PPOConfig(hidden_sizes=(16, 16), total_timesteps=4))
    network = _build_network()
    batch = _make_batch()
    loss = trainer.compute_ppo_loss(network, batch)

    assert isinstance(loss, float)


def test_compute_gradients_returns_flat_vector_matching_params() -> None:
    trainer = PPOTrainer(PPOConfig(hidden_sizes=(4,), fd_epsilon=1e-4, total_timesteps=4))
    network = _build_network(hidden_sizes=(4,))
    batch = _make_batch()
    grads = trainer.compute_gradients(network, batch)

    assert grads.shape == network.get_params().shape
    assert np.isfinite(grads).all()


def test_update_network_changes_parameters() -> None:
    trainer = PPOTrainer(PPOConfig(hidden_sizes=(4,), total_timesteps=4))
    network = _build_network(hidden_sizes=(4,))
    params_before = network.get_params().copy()
    grads = np.ones_like(params_before) * 0.1
    trainer.update_network(network, trainer.optimizer, grads)

    assert not np.allclose(params_before, network.get_params())


def test_training_loop_runs_without_crashing_for_small_mock_env() -> None:
    env = _MockEnv(obs_dim=3, act_dim=2, max_steps=5)
    trainer = PPOTrainer(
        PPOConfig(total_timesteps=200, n_steps=20, n_epochs=1, minibatch_size=20, hidden_sizes=(4,))
    )
    result = trainer.train(env)

    assert isinstance(result, RLTrainingResult)
    assert result.updates >= 1


def test_training_result_contains_episode_rewards_list() -> None:
    env = _MockEnv(obs_dim=3, act_dim=2, max_steps=4)
    trainer = PPOTrainer(
        PPOConfig(total_timesteps=40, n_steps=10, n_epochs=1, minibatch_size=10, hidden_sizes=(4,))
    )
    result = trainer.train(env)

    assert isinstance(result.episode_rewards, list)


def test_callbacks_fire_at_expected_training_points() -> None:
    callback = _CountingCallback()
    env = _MockEnv(obs_dim=3, act_dim=2, max_steps=4)
    trainer = PPOTrainer(
        PPOConfig(total_timesteps=20, n_steps=10, n_epochs=1, minibatch_size=10, hidden_sizes=(4,)),
        callbacks=[callback],
    )
    trainer.train(env)

    assert callback.rollout_start == 2
    assert callback.rollout_end == 2
    assert callback.update == 2
    assert callback.training_end == 1


def test_evaluate_policy_returns_expected_types_and_shapes() -> None:
    env = _MockEnv(obs_dim=3, act_dim=2, max_steps=3)
    network = _build_network()
    result = evaluate_policy(env, network, n_episodes=3)

    assert isinstance(result.mean_reward, float)
    assert isinstance(result.std_reward, float)
    assert isinstance(result.mean_length, float)
    assert len(result.episode_rewards) == 3


def test_record_episode_returns_frame_triplets() -> None:
    env = _MockEnv(obs_dim=3, act_dim=2, max_steps=3)
    network = _build_network()
    frames = record_episode(env, network)

    assert len(frames) == 3
    assert frames[0][0].shape == (3,)
    assert frames[0][1].shape == (2,)
    assert isinstance(frames[0][2], float)


def test_default_ppoconfig_is_valid() -> None:
    config = PPOConfig()

    assert config.n_steps > 0
    assert config.total_timesteps > 0


def test_custom_ppoconfig_overrides_work() -> None:
    config = PPOConfig(total_timesteps=256, n_steps=32, hidden_sizes=(16, 16), tanh_squash=True)

    assert config.total_timesteps == 256
    assert config.n_steps == 32
    assert config.hidden_sizes == (16, 16)
    assert config.tanh_squash is True


def test_config_can_be_derived_with_replace() -> None:
    config = PPOConfig(total_timesteps=100)
    updated = replace(config, total_timesteps=50, n_steps=10)

    assert updated.total_timesteps == 50
    assert updated.n_steps == 10


def test_checkpoint_callback_save_load_roundtrip(tmp_path: Path) -> None:
    env = _MockEnv(obs_dim=3, act_dim=2, max_steps=4)
    callback = CheckpointCallback(checkpoint_freq=1, output_dir=tmp_path)
    trainer = PPOTrainer(
        PPOConfig(total_timesteps=10, n_steps=10, n_epochs=1, minibatch_size=10, hidden_sizes=(4,)),
        callbacks=[callback],
    )
    trainer.train(env)
    restored = ActorCritic.load(str(callback.saved_paths[0]))

    assert callback.saved_paths
    assert np.allclose(restored.get_params(), trainer.network.get_params())


def test_early_stop_callback_stops_training_when_threshold_met() -> None:
    env = _MockEnv(obs_dim=3, act_dim=2, max_steps=3, reward_bias=2.0)
    callback = EarlyStopCallback(target_reward=0.0, window_size=1)
    trainer = PPOTrainer(
        PPOConfig(total_timesteps=100, n_steps=10, n_epochs=1, minibatch_size=10, hidden_sizes=(4,)),
        callbacks=[callback],
    )
    result = trainer.train(env)

    assert result.stopped_early is True
    assert result.updates < 10


def test_logging_callback_does_not_break_training(capsys: pytest.CaptureFixture[str]) -> None:
    from optisim.rl import LoggingCallback

    env = _MockEnv(obs_dim=3, act_dim=2, max_steps=3)
    trainer = PPOTrainer(
        PPOConfig(total_timesteps=10, n_steps=10, n_epochs=1, minibatch_size=10, hidden_sizes=(4,)),
        callbacks=[LoggingCallback(log_interval=1)],
    )
    trainer.train(env)
    output = capsys.readouterr().out

    assert "update=" in output


def test_rl_public_imports_are_available_from_optisim() -> None:
    config = PPOConfig(total_timesteps=10)
    trainer = PPOTrainer(config)
    buffer = RolloutBuffer(capacity=2, obs_dim=1, act_dim=1)

    assert isinstance(config, PPOConfig)
    assert isinstance(trainer, PPOTrainer)
    assert isinstance(buffer, RolloutBuffer)


def test_compute_ppo_loss_can_return_details() -> None:
    trainer = PPOTrainer(PPOConfig(hidden_sizes=(4,), total_timesteps=4))
    network = _build_network(hidden_sizes=(4,))
    details = trainer.compute_ppo_loss(network, _make_batch(), return_details=True)

    assert set(details) == {"loss", "policy_loss", "value_loss", "entropy"}


def test_collect_rollout_tracks_episode_completion() -> None:
    env = _MockEnv(obs_dim=3, act_dim=2, max_steps=2)
    trainer = PPOTrainer(PPOConfig(total_timesteps=4, n_steps=4, n_epochs=1, minibatch_size=4, hidden_sizes=(4,)))
    network = _build_network(hidden_sizes=(4,))
    buffer = RolloutBuffer(capacity=4, obs_dim=3, act_dim=2)
    trainer.collect_rollout(env, network, buffer)

    assert len(trainer.episode_rewards) >= 1
    assert len(trainer.episode_lengths) >= 1


def test_network_with_tanh_squash_returns_bounded_actions() -> None:
    network = ActorCritic(obs_dim=3, act_dim=2, hidden_sizes=(16,), tanh_squash=True, seed=0)
    action, _, _ = network.get_action(np.ones(3, dtype=float))

    assert np.all(np.abs(action) <= 1.0)


def test_minibatches_preserve_action_dimension() -> None:
    buffer = RolloutBuffer(capacity=4, obs_dim=2, act_dim=3)
    for _ in range(4):
        buffer.add(np.ones(2), np.ones(3), 1.0, 0.1, -0.2, False)
    buffer.compute_returns_and_advantages(last_value=0.0, done=True)
    batch = next(buffer.get_minibatches(2))

    assert batch.actions.shape[1] == 3


def test_evaluate_policy_episode_count_matches_requested_count() -> None:
    env = _MockEnv(obs_dim=3, act_dim=2, max_steps=2)
    result = evaluate_policy(env, _build_network(), n_episodes=5)

    assert len(result.episode_rewards) == 5


def test_short_training_run_on_optisim_env_completes_without_error() -> None:
    pytest.importorskip("gymnasium")
    from optisim.gym_env import ObservationConfig, OptisimEnv

    env = OptisimEnv(
        max_steps=10,
        observation_config=ObservationConfig(
            joint_positions=True,
            link_positions=False,
            object_states=False,
            joint_names=["torso_yaw", "torso_pitch"],
        ),
    )
    trainer = PPOTrainer(
        PPOConfig(total_timesteps=100, n_steps=50, n_epochs=1, minibatch_size=50, hidden_sizes=(4,))
    )
    result = trainer.train(env)
    env.close()

    assert result.updates >= 1


def test_optisim_env_checkpoint_roundtrip_works(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    from optisim.gym_env import ObservationConfig, OptisimEnv

    env = OptisimEnv(
        max_steps=5,
        observation_config=ObservationConfig(
            joint_positions=True,
            link_positions=False,
            object_states=False,
            joint_names=["torso_yaw", "torso_pitch"],
        ),
    )
    callback = CheckpointCallback(checkpoint_freq=1, output_dir=tmp_path)
    trainer = PPOTrainer(
        PPOConfig(total_timesteps=20, n_steps=20, n_epochs=1, minibatch_size=20, hidden_sizes=(2,)),
        callbacks=[callback],
    )
    trainer.train(env)
    env.close()
    restored = ActorCritic.load(str(callback.saved_paths[0]))

    assert np.allclose(restored.get_params(), trainer.network.get_params())


def test_optisim_env_early_stop_callback_works() -> None:
    pytest.importorskip("gymnasium")
    from optisim.gym_env import ObservationConfig, OptisimEnv

    env = OptisimEnv(
        max_steps=5,
        observation_config=ObservationConfig(
            joint_positions=True,
            link_positions=False,
            object_states=False,
            joint_names=["torso_yaw", "torso_pitch"],
        ),
    )
    callback = EarlyStopCallback(target_reward=-1e6, window_size=1)
    trainer = PPOTrainer(
        PPOConfig(total_timesteps=40, n_steps=20, n_epochs=1, minibatch_size=20, hidden_sizes=(2,)),
        callbacks=[callback],
    )
    result = trainer.train(env)
    env.close()

    assert result.stopped_early is True
