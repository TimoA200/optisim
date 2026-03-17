from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optisim import (
    BCConfig,
    BehavioralCloningTrainer,
    Demonstration,
    DemonstrationRecorder,
    NeuralPolicy,
    PolicyDataset,
    PolicyNetwork,
    PolicyStep,
    RecurrentNeuralPolicy,
    TrainingResult,
    build_policy_network,
    train_policy,
)
from optisim.policy.dataset import NormStats, PolicyAction, PolicyObservation


def build_demo(
    *,
    task_name: str = "reach",
    n_steps: int = 10,
    joint_dim: int = 4,
    offset: float = 0.0,
    rng_seed: int = 0,
) -> Demonstration:
    rng = np.random.default_rng(rng_seed)
    recorder = DemonstrationRecorder(task_name=task_name, metadata={"source": "policy-test"})
    for step in range(n_steps):
        phase = 0.0 if n_steps == 1 else step / (n_steps - 1)
        joints = [
            offset + 0.2 * axis + 0.3 * phase + 0.01 * float(rng.normal())
            for axis in range(joint_dim)
        ]
        ee_pose = (
            0.4 + 0.1 * phase,
            -0.2 + 0.05 * phase,
            0.8 - 0.03 * phase,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        extras = {"phase": phase, "bias": [phase, phase + 1.0]} if step % 2 == 0 else {}
        recorder.record(step, joints, ee_pose, step < max(n_steps - 2, 1), extras)
    return recorder.demonstration


def build_demo_set(num_demos: int = 5, n_steps: int = 10, joint_dim: int = 4) -> list[Demonstration]:
    return [
        build_demo(task_name="clone", n_steps=n_steps, joint_dim=joint_dim, offset=0.05 * index, rng_seed=index)
        for index in range(num_demos)
    ]


def test_policy_network_construction() -> None:
    network = PolicyNetwork(input_dim=12, hidden_dims=(16, 8), output_dim=5)

    assert network.input_dim == 12
    assert network.hidden_dims == (16, 8)
    assert network.output_dim == 5
    assert len(network.layers) == 3


def test_policy_network_invalid_activation() -> None:
    with pytest.raises(ValueError):
        PolicyNetwork(input_dim=4, hidden_dims=(8,), output_dim=2, activation="gelu")


def test_policy_network_forward_batch_shape() -> None:
    network = PolicyNetwork(input_dim=6, hidden_dims=(10, 7), output_dim=3)
    output = network.forward(np.ones((4, 6), dtype=float))

    assert output.shape == (4, 3)


def test_policy_network_forward_single_shape() -> None:
    network = PolicyNetwork(input_dim=6, hidden_dims=(10,), output_dim=2)
    output = network.forward(np.ones(6, dtype=float))

    assert output.shape == (2,)


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid"])
def test_policy_network_forward_with_supported_activations(activation: str) -> None:
    network = PolicyNetwork(input_dim=5, hidden_dims=(7, 6), output_dim=2, activation=activation)
    output = network.forward(np.full((3, 5), 0.25, dtype=float))

    assert output.shape == (3, 2)
    assert np.isfinite(output).all()


def test_policy_network_forward_with_cache_structure() -> None:
    network = PolicyNetwork(input_dim=4, hidden_dims=(5, 6), output_dim=2)
    _, cache = network._forward_with_cache(np.ones((2, 4), dtype=float))

    assert len(cache) == 3
    assert set(cache[0]) == {"input", "pre_activation", "post_activation"}


def test_policy_network_backward_returns_gradient_dicts() -> None:
    network = PolicyNetwork(input_dim=4, hidden_dims=(5, 3), output_dim=2)
    output, cache = network._forward_with_cache(np.ones((3, 4), dtype=float))
    grads = network.backward(cache, np.ones_like(output))

    assert len(grads) == 3
    assert set(grads[0]) == {"dW", "db"}


def test_policy_network_backward_gradient_shapes_match_parameters() -> None:
    network = PolicyNetwork(input_dim=3, hidden_dims=(4,), output_dim=2)
    output, cache = network._forward_with_cache(np.ones((2, 3), dtype=float))
    grads = network.backward(cache, np.ones_like(output))

    for grad, layer in zip(grads, network.layers):
        assert grad["dW"].shape == layer["W"].shape
        assert grad["db"].shape == layer["b"].shape


def test_policy_network_parameters_and_set_parameters_roundtrip() -> None:
    network = PolicyNetwork(input_dim=3, hidden_dims=(4,), output_dim=2)
    params = [{"W": layer["W"] + 1.0, "b": layer["b"] + 2.0} for layer in network.parameters()]

    network.set_parameters(params)

    for param, layer in zip(params, network.layers):
        assert np.allclose(param["W"], layer["W"])
        assert np.allclose(param["b"], layer["b"])


def test_policy_network_predict_handles_1d_and_2d_inputs() -> None:
    network = PolicyNetwork(input_dim=3, hidden_dims=(4,), output_dim=2)

    single = network.predict(np.array([1.0, 2.0, 3.0], dtype=float))
    batch = network.predict(np.ones((2, 3), dtype=float))

    assert single.shape == (2,)
    assert batch.shape == (2, 2)


def test_policy_network_num_parameters_counts_all_weights_and_biases() -> None:
    network = PolicyNetwork(input_dim=3, hidden_dims=(4, 5), output_dim=2)

    assert network.num_parameters() == (3 * 4 + 4) + (4 * 5 + 5) + (5 * 2 + 2)


def test_policy_network_save_load_roundtrip(tmp_path: Path) -> None:
    network = PolicyNetwork(input_dim=4, hidden_dims=(5,), output_dim=3)
    path = tmp_path / "policy_network.npz"
    saved_output = network.predict(np.ones((2, 4), dtype=float))

    network.save(str(path))
    restored = PolicyNetwork(input_dim=4, hidden_dims=(5,), output_dim=3)
    restored.load(str(path))

    assert np.allclose(saved_output, restored.predict(np.ones((2, 4), dtype=float)))


def test_policy_network_copy_is_independent() -> None:
    network = PolicyNetwork(input_dim=4, hidden_dims=(6,), output_dim=2)
    cloned = network.copy()
    cloned.layers[0]["W"][0, 0] += 5.0

    assert not np.allclose(network.layers[0]["W"], cloned.layers[0]["W"])


def test_build_policy_network_returns_policy_network() -> None:
    network = build_policy_network(input_dim=8, hidden_dims=(16, 12), output_dim=4, activation="tanh")

    assert isinstance(network, PolicyNetwork)
    assert network.activation == "tanh"


def test_policy_observation_and_action_as_array_shapes() -> None:
    observation = PolicyObservation(
        joint_positions=np.array([0.1, 0.2], dtype=float),
        ee_pose=np.zeros(7, dtype=float),
        gripper_open=1.0,
        extras_flat=np.zeros(0, dtype=float),
    )
    action = PolicyAction(joint_delta=np.array([0.3, -0.2], dtype=float), gripper_cmd=0.0)

    assert observation.as_array().shape == (10,)
    assert action.as_array().shape == (3,)


def test_policy_dataset_from_demonstrations_shapes() -> None:
    dataset = PolicyDataset.from_demonstrations(build_demo_set(num_demos=2, n_steps=6, joint_dim=3))

    assert len(dataset) == 12
    assert dataset.obs_dim == 11
    assert dataset.act_dim == 4


def test_policy_dataset_from_demonstrations_delta_actions_false_uses_absolute_next_joint_positions() -> None:
    demo = build_demo(n_steps=3, joint_dim=2)
    dataset = PolicyDataset.from_demonstrations([demo], delta_actions=False)
    expected = np.array(demo.steps[1].joint_positions + [float(demo.steps[1].gripper_open)], dtype=float)

    assert np.allclose(dataset.actions[0], expected)


def test_policy_dataset_single_step_demo_produces_zero_delta_action() -> None:
    demo = build_demo(n_steps=1, joint_dim=2)
    dataset = PolicyDataset.from_demonstrations([demo])

    assert np.allclose(dataset.actions[0][:2], 0.0)
    assert dataset.actions[0][-1] == float(demo.steps[0].gripper_open)


def test_policy_dataset_handles_zero_dim_extras() -> None:
    observation = PolicyObservation(
        joint_positions=np.array([0.1], dtype=float),
        ee_pose=np.ones(7, dtype=float),
        gripper_open=0.0,
        extras_flat=np.zeros(0, dtype=float),
    )

    assert observation.extras_flat.shape == (0,)


def test_policy_dataset_normalize_returns_stats_and_keeps_gripper_action_unnormalized() -> None:
    dataset = PolicyDataset.from_demonstrations(build_demo_set(num_demos=2, n_steps=4, joint_dim=2))
    original_gripper = dataset.actions[:, -1].copy()
    stats = dataset.normalize()

    assert stats.obs_mean.shape == (dataset.obs_dim,)
    assert stats.act_mean[-1] == 0.0
    assert stats.act_std[-1] == 1.0
    assert np.allclose(dataset.actions[:, -1], original_gripper)


def test_policy_dataset_normalization_round_trip() -> None:
    dataset = PolicyDataset.from_demonstrations(build_demo_set(num_demos=2, n_steps=5, joint_dim=3))
    original_obs = dataset.observations.copy()
    original_actions = dataset.actions.copy()
    stats = dataset.normalize()

    restored_obs = dataset.observations * stats.obs_std + stats.obs_mean
    restored_actions = dataset.actions * stats.act_std + stats.act_mean

    assert np.allclose(restored_obs, original_obs)
    assert np.allclose(restored_actions, original_actions)


def test_policy_dataset_apply_norm_matches_manual_normalization() -> None:
    source = PolicyDataset.from_demonstrations(build_demo_set(num_demos=2, n_steps=4, joint_dim=2))
    target = PolicyDataset.from_demonstrations(build_demo_set(num_demos=2, n_steps=4, joint_dim=2))
    stats = source.normalize()
    target.apply_norm(stats)

    assert np.allclose(source.observations, target.observations)
    assert np.allclose(source.actions, target.actions)


def test_policy_dataset_split_sizes_cover_all_rows() -> None:
    dataset = PolicyDataset.from_demonstrations(build_demo_set(num_demos=3, n_steps=5, joint_dim=2))
    train_data, val_data = dataset.split(train_fraction=0.7, seed=7)

    assert len(train_data) + len(val_data) == len(dataset)
    assert len(train_data) >= 1
    assert len(val_data) >= 1


def test_policy_dataset_split_single_sample_returns_non_empty_splits() -> None:
    dataset = PolicyDataset.from_demonstrations([build_demo(n_steps=1, joint_dim=2)])
    train_data, val_data = dataset.split()

    assert len(train_data) == 1
    assert len(val_data) == 1


def test_policy_dataset_get_batch_returns_selected_rows() -> None:
    dataset = PolicyDataset.from_demonstrations(build_demo_set(num_demos=2, n_steps=5, joint_dim=2))
    observations, actions = dataset.get_batch(np.array([1, 3, 5], dtype=int))

    assert observations.shape == (3, dataset.obs_dim)
    assert actions.shape == (3, dataset.act_dim)
    assert np.allclose(observations[0], dataset.observations[1])


def test_norm_stats_denormalize_action_is_inverse_of_normalization() -> None:
    stats = NormStats(
        obs_mean=np.array([1.0, 2.0], dtype=float),
        obs_std=np.array([2.0, 4.0], dtype=float),
        act_mean=np.array([0.5, -1.0], dtype=float),
        act_std=np.array([3.0, 2.0], dtype=float),
    )
    normalized = np.array([2.0, -0.5], dtype=float)

    assert np.allclose(stats.denormalize_action(normalized), np.array([6.5, -2.0], dtype=float))


def test_norm_stats_normalize_obs_matches_expected_values() -> None:
    stats = NormStats(
        obs_mean=np.array([1.0, 2.0, 3.0], dtype=float),
        obs_std=np.array([2.0, 4.0, 5.0], dtype=float),
        act_mean=np.zeros(2, dtype=float),
        act_std=np.ones(2, dtype=float),
    )

    assert np.allclose(stats.normalize_obs(np.array([3.0, 6.0, 8.0], dtype=float)), np.array([1.0, 1.0, 1.0]))


def test_behavioral_cloning_trainer_train_returns_training_result() -> None:
    dataset = PolicyDataset.from_demonstrations(build_demo_set())
    dataset.normalize()
    trainer = BehavioralCloningTrainer(BCConfig(epochs=6, batch_size=8, hidden_dims=(32, 16), patience=4))

    network, result = trainer.train(dataset)

    assert isinstance(network, PolicyNetwork)
    assert isinstance(result, TrainingResult)
    assert result.epochs_trained >= 1
    assert len(result.train_losses) == result.epochs_trained
    assert len(result.val_losses) == result.epochs_trained


def test_behavioral_cloning_trainer_accepts_custom_network() -> None:
    dataset = PolicyDataset.from_demonstrations(build_demo_set())
    dataset.normalize()
    trainer = BehavioralCloningTrainer(BCConfig(epochs=4, batch_size=16, patience=3))
    network = PolicyNetwork(input_dim=dataset.obs_dim, hidden_dims=(20,), output_dim=dataset.act_dim)

    trained_network, _ = trainer.train(dataset, network=network)

    assert trained_network.output_dim == dataset.act_dim
    assert trained_network.predict(dataset.observations[:2]).shape == (2, dataset.act_dim)


def test_behavioral_cloning_trainer_adam_step_updates_parameters() -> None:
    trainer = BehavioralCloningTrainer()
    network = PolicyNetwork(input_dim=3, hidden_dims=(4,), output_dim=2)
    params = [{"W": layer["W"].copy(), "b": layer["b"].copy()} for layer in network.parameters()]
    grads = [{"dW": np.ones_like(layer["W"]), "db": np.ones_like(layer["b"])} for layer in network.layers]
    m = [{"W": np.zeros_like(layer["W"]), "b": np.zeros_like(layer["b"])} for layer in network.layers]
    v = [{"W": np.zeros_like(layer["W"]), "b": np.zeros_like(layer["b"])} for layer in network.layers]

    updated = trainer._adam_step(params, grads, m, v, t=1, lr=1e-3)

    assert not np.allclose(updated[0]["W"], params[0]["W"])
    assert not np.allclose(updated[0]["b"], params[0]["b"])


def test_train_policy_end_to_end_returns_network_stats_and_result() -> None:
    network, stats, result = train_policy(
        build_demo_set(),
        BCConfig(epochs=6, batch_size=8, hidden_dims=(24, 12), patience=4),
    )

    assert isinstance(network, PolicyNetwork)
    assert isinstance(stats, NormStats)
    assert isinstance(result, TrainingResult)


def test_train_policy_output_can_be_used_by_neural_policy() -> None:
    demos = build_demo_set()
    network, stats, _ = train_policy(demos, BCConfig(epochs=5, batch_size=8, hidden_dims=(16, 16), patience=3))
    policy = NeuralPolicy(network=network, norm_stats=stats, joint_dim=demos[0].joint_dim)
    step = demos[0].steps[0]

    action = policy.act(step.joint_positions, step.ee_pose, step.gripper_open)

    assert isinstance(action, PolicyStep)
    assert len(action.joint_positions_next) == demos[0].joint_dim
    assert action.raw_action.shape == (demos[0].joint_dim + 1,)


def test_neural_policy_act_returns_policy_step_with_correct_shapes() -> None:
    dataset = PolicyDataset.from_demonstrations(build_demo_set(num_demos=2, n_steps=4, joint_dim=3))
    stats = dataset.normalize()
    network = PolicyNetwork(input_dim=dataset.obs_dim, hidden_dims=(12,), output_dim=dataset.act_dim)
    policy = NeuralPolicy(network=network, norm_stats=stats, joint_dim=3)
    step = build_demo(joint_dim=3).steps[0]

    result = policy.act(step.joint_positions, step.ee_pose, step.gripper_open)

    assert len(result.joint_positions_next) == 3
    assert isinstance(result.gripper_open, bool)
    assert result.raw_action.shape == (4,)


def test_neural_policy_save_load_roundtrip_preserves_outputs(tmp_path: Path) -> None:
    dataset = PolicyDataset.from_demonstrations(build_demo_set(num_demos=2, n_steps=4, joint_dim=2))
    stats = dataset.normalize()
    network = PolicyNetwork(input_dim=dataset.obs_dim, hidden_dims=(10,), output_dim=dataset.act_dim)
    policy = NeuralPolicy(network=network, norm_stats=stats, joint_dim=2)
    step = build_demo(joint_dim=2).steps[0]
    path = tmp_path / "neural_policy.npz"
    expected = policy.act(step.joint_positions, step.ee_pose, step.gripper_open)

    policy.save(str(path))
    restored = NeuralPolicy.load(str(path))
    actual = restored.act(step.joint_positions, step.ee_pose, step.gripper_open)

    assert actual.joint_positions_next == pytest.approx(expected.joint_positions_next)
    assert actual.gripper_open == expected.gripper_open
    assert np.allclose(actual.raw_action, expected.raw_action)


def test_neural_policy_reset_is_no_op() -> None:
    dataset = PolicyDataset.from_demonstrations(build_demo_set(num_demos=1, n_steps=3, joint_dim=2))
    stats = dataset.normalize()
    network = PolicyNetwork(input_dim=dataset.obs_dim, hidden_dims=(8,), output_dim=dataset.act_dim)
    policy = NeuralPolicy(network=network, norm_stats=stats, joint_dim=2)

    assert policy.reset() is None


def test_recurrent_neural_policy_act_works_with_history_window() -> None:
    demos = build_demo_set(num_demos=2, n_steps=5, joint_dim=2)
    base_dataset = PolicyDataset.from_demonstrations(demos)
    stats = base_dataset.normalize()
    network = PolicyNetwork(input_dim=base_dataset.obs_dim * 3, hidden_dims=(14,), output_dim=base_dataset.act_dim)
    policy = RecurrentNeuralPolicy(network=network, norm_stats=stats, joint_dim=2, history_len=3)
    step = demos[0].steps[0]

    result = policy.act(step.joint_positions, step.ee_pose, step.gripper_open)

    assert isinstance(result, PolicyStep)
    assert len(policy._history) == 1
    assert result.raw_action.shape == (3,)


def test_recurrent_neural_policy_accumulates_and_resets_history() -> None:
    demos = build_demo_set(num_demos=1, n_steps=4, joint_dim=2)
    base_dataset = PolicyDataset.from_demonstrations(demos)
    stats = base_dataset.normalize()
    network = PolicyNetwork(input_dim=base_dataset.obs_dim * 2, hidden_dims=(10,), output_dim=base_dataset.act_dim)
    policy = RecurrentNeuralPolicy(network=network, norm_stats=stats, joint_dim=2, history_len=2)

    for step in demos[0].steps[:3]:
        policy.act(step.joint_positions, step.ee_pose, step.gripper_open)

    assert len(policy._history) == 2
    policy.reset()
    assert policy._history == []


def test_recurrent_neural_policy_save_writes_file(tmp_path: Path) -> None:
    demos = build_demo_set(num_demos=1, n_steps=4, joint_dim=2)
    base_dataset = PolicyDataset.from_demonstrations(demos)
    stats = base_dataset.normalize()
    network = PolicyNetwork(input_dim=base_dataset.obs_dim * 2, hidden_dims=(10,), output_dim=base_dataset.act_dim)
    policy = RecurrentNeuralPolicy(network=network, norm_stats=stats, joint_dim=2, history_len=2)
    path = tmp_path / "recurrent_policy.npz"

    policy.save(str(path))
    data = np.load(path, allow_pickle=False)

    assert path.exists()
    assert int(np.asarray(data["history_len"]).item()) == 2
