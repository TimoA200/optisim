from __future__ import annotations

import numpy as np

from optisim import BenchmarkSuite, PrimitiveExecutor, SceneGraph, SceneNode, SceneRelation
from optisim.worldmodel import (
    MPPConfig,
    ModelPredictivePlanner,
    StateEncoder,
    TransitionSample,
    WorldModelCollector,
    WorldModelNet,
    WorldModelTrainer,
    WorldState,
)


def make_pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def make_scene() -> SceneGraph:
    scene = SceneGraph()
    scene.add_node(SceneNode("humanoid", "humanoid", "robot", pose=make_pose(0.0, 0.0, 0.0), bbox=(0.3, 0.3, 0.9)))
    scene.add_node(
        SceneNode(
            "cup",
            "cup",
            "container",
            pose=make_pose(0.5, 0.0, 0.2),
            bbox=(0.04, 0.04, 0.08),
            properties={"graspable": True},
        )
    )
    scene.add_node(SceneNode("table", "table", "surface", pose=make_pose(0.6, 0.0, 0.1), bbox=(0.5, 0.4, 0.05)))
    scene.add_node(SceneNode("shelf", "shelf", "surface", pose=make_pose(0.9, 0.3, 0.6), bbox=(0.3, 0.2, 0.3)))
    scene.add_node(SceneNode("box", "box", "object", pose=make_pose(0.4, 0.1, 0.2), bbox=(0.08, 0.08, 0.08)))
    scene.add_relation(SceneRelation("cup", "on", "table"))
    scene.add_relation(SceneRelation("table", "near", "shelf"))
    return scene


def make_world_state(encoder: StateEncoder, scene: SceneGraph | None = None, offset: float = 0.0) -> WorldState:
    graph = make_scene() if scene is None else scene
    scene_features = encoder.encode_scene(graph) + offset
    relation_vector = encoder.encode_relations(graph)
    joints = np.linspace(0.0, 0.3, 31, dtype=float)
    return WorldState(joint_positions=joints, scene_features=scene_features, relation_vector=relation_vector, timestamp=offset)


def make_transition_samples(encoder: StateEncoder, count: int = 12) -> list[TransitionSample]:
    samples: list[TransitionSample] = []
    base_scene = make_scene()
    for index in range(count):
        state = make_world_state(encoder, base_scene, offset=0.01 * index)
        next_state = WorldState(
            joint_positions=state.joint_positions.copy(),
            scene_features=state.scene_features + 0.05,
            relation_vector=state.relation_vector.copy(),
            timestamp=state.timestamp + 1.0,
        )
        action_vec = encoder.encode_action("reach", {"target_id": "cup"})
        samples.append(TransitionSample(state=state, action_vec=action_vec, next_state=next_state, reward=1.0))
    return samples


def make_model(encoder: StateEncoder) -> WorldModelNet:
    return WorldModelNet(state_dim=encoder.state_dim + encoder.relation_dim, action_dim=encoder.action_dim, hidden_dim=32)


def test_state_encoder_initialization() -> None:
    encoder = StateEncoder(max_nodes=10, max_relations=30)
    assert encoder.max_nodes == 10
    assert encoder.max_relations == 30


def test_encode_scene_returns_correct_shape() -> None:
    encoder = StateEncoder()
    assert encoder.encode_scene(make_scene()).shape == (encoder.state_dim,)


def test_encode_relations_returns_correct_shape() -> None:
    encoder = StateEncoder()
    assert encoder.encode_relations(make_scene()).shape == (encoder.relation_dim,)


def test_encode_action_returns_shape_eight() -> None:
    encoder = StateEncoder()
    assert encoder.encode_action("grasp", {"target_id": "cup"}).shape == (8,)


def test_encode_scene_with_empty_graph() -> None:
    encoder = StateEncoder()
    empty = SceneGraph()
    np.testing.assert_allclose(encoder.encode_scene(empty), np.zeros(encoder.state_dim, dtype=float))


def test_encode_scene_with_max_nodes_truncates() -> None:
    encoder = StateEncoder(max_nodes=2)
    encoded = encoder.encode_scene(make_scene())
    assert encoded.shape == (26,)
    assert np.count_nonzero(encoded) > 0


def test_world_model_net_initialization() -> None:
    encoder = StateEncoder()
    model = make_model(encoder)
    assert model.hidden_dim == 32


def test_predict_returns_correct_output_shape() -> None:
    encoder = StateEncoder()
    model = make_model(encoder)
    state = make_world_state(encoder)
    prediction = model.predict(state.as_vector(), encoder.encode_action("reach", {"target_id": "cup"}))
    assert prediction.shape == (encoder.state_dim + encoder.relation_dim,)


def test_n_params_positive() -> None:
    encoder = StateEncoder()
    assert make_model(encoder).n_params > 0


def test_transition_sample_creation() -> None:
    encoder = StateEncoder()
    sample = make_transition_samples(encoder, count=1)[0]
    assert isinstance(sample, TransitionSample)
    assert sample.reward == 1.0


def test_world_model_trainer_train_step_returns_float_loss() -> None:
    encoder = StateEncoder()
    trainer = WorldModelTrainer(make_model(encoder), lr=0.01)
    loss = trainer.train_step(make_transition_samples(encoder))
    assert isinstance(loss, float)


def test_world_model_trainer_fit_returns_loss_list_of_correct_length() -> None:
    encoder = StateEncoder()
    trainer = WorldModelTrainer(make_model(encoder), lr=0.01)
    history = trainer.fit(make_transition_samples(encoder), epochs=5)
    assert len(history) == 5


def test_loss_decreases_over_epochs() -> None:
    encoder = StateEncoder()
    trainer = WorldModelTrainer(make_model(encoder), lr=0.01)
    history = trainer.fit(make_transition_samples(encoder), epochs=20)
    assert history[-1] < history[0]


def test_mpp_config_defaults() -> None:
    config = MPPConfig()
    assert config.horizon == 5
    assert config.n_samples == 50
    assert config.discount == 0.95
    assert config.rng_seed == 42


def test_model_predictive_planner_initialization() -> None:
    encoder = StateEncoder()
    planner = ModelPredictivePlanner(make_model(encoder), encoder, PrimitiveExecutor().available_primitives())
    assert isinstance(planner.config, MPPConfig)


def test_plan_returns_list_of_dicts() -> None:
    encoder = StateEncoder()
    planner = ModelPredictivePlanner(make_model(encoder), encoder, PrimitiveExecutor().available_primitives())
    sequence = planner.plan(make_scene(), np.zeros(31, dtype=float), [{"subject_id": "cup", "predicate": "on", "object_id": "table"}])
    assert isinstance(sequence, list)
    assert all(isinstance(step, dict) for step in sequence)


def test_plan_returns_sequence_of_correct_length() -> None:
    encoder = StateEncoder()
    planner = ModelPredictivePlanner(make_model(encoder), encoder, PrimitiveExecutor().available_primitives())
    sequence = planner.plan(make_scene(), np.zeros(31, dtype=float), [], n_steps=3)
    assert len(sequence) == 3


def test_plan_dicts_have_primitive_key() -> None:
    encoder = StateEncoder()
    planner = ModelPredictivePlanner(make_model(encoder), encoder, PrimitiveExecutor().available_primitives())
    sequence = planner.plan(make_scene(), np.zeros(31, dtype=float), [])
    assert all("primitive" in step for step in sequence)


def test_score_state_returns_float_between_zero_and_one() -> None:
    encoder = StateEncoder()
    planner = ModelPredictivePlanner(make_model(encoder), encoder, PrimitiveExecutor().available_primitives())
    state = make_world_state(encoder)
    score = planner.score_state(state.as_vector(), [{"subject_id": "cup", "predicate": "on", "object_id": "table"}], encoder)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_rollout_returns_correct_shape() -> None:
    encoder = StateEncoder()
    planner = ModelPredictivePlanner(make_model(encoder), encoder, PrimitiveExecutor().available_primitives())
    rollout = planner.rollout(
        make_world_state(encoder).as_vector(),
        [encoder.encode_action("reach", {"target_id": "cup"}), encoder.encode_action("grasp", {"target_id": "cup"})],
    )
    assert rollout.shape == (encoder.state_dim + encoder.relation_dim,)


def test_world_model_collector_collect_returns_list() -> None:
    collector = WorldModelCollector()
    samples = collector.collect(BenchmarkSuite.DEFAULT, n_tasks=2)
    assert isinstance(samples, list)


def test_collected_samples_have_correct_attribute_types() -> None:
    collector = WorldModelCollector()
    sample = collector.collect(BenchmarkSuite.DEFAULT, n_tasks=1)[0]
    assert isinstance(sample.state, WorldState)
    assert isinstance(sample.action_vec, np.ndarray)
    assert isinstance(sample.next_state, WorldState)


def test_full_pipeline_collect_fit_plan() -> None:
    encoder = StateEncoder()
    collector = WorldModelCollector(encoder=encoder)
    samples = collector.collect(BenchmarkSuite.DEFAULT, n_tasks=2)
    trainer = WorldModelTrainer(make_model(encoder), lr=0.005)
    history = trainer.fit(samples, epochs=3)
    planner = ModelPredictivePlanner(trainer.model, encoder, PrimitiveExecutor().available_primitives())
    plan = planner.plan(
        make_scene(),
        np.zeros(31, dtype=float),
        [{"subject_id": "cup", "predicate": "on", "object_id": "table"}],
        n_steps=2,
    )
    assert len(samples) > 0
    assert len(history) == 3
    assert isinstance(plan, list)


def test_state_dim_relation_dim_action_dim_properties() -> None:
    encoder = StateEncoder(max_nodes=4, max_relations=7)
    assert encoder.state_dim == 52
    assert encoder.relation_dim == 7
    assert encoder.action_dim == 8


def test_decode_relation_vector_returns_list_of_tuples() -> None:
    encoder = StateEncoder()
    decoded = encoder.decode_relation_vector(encoder.encode_relations(make_scene()), make_scene())
    assert isinstance(decoded, list)
    assert all(isinstance(item, tuple) and len(item) == 3 for item in decoded)


def test_world_model_net_weights_dict_has_expected_keys() -> None:
    encoder = StateEncoder()
    assert set(make_model(encoder).weights) == {"W1", "b1", "W2", "b2", "W3", "b3"}


def test_planner_with_no_goal_predicates_returns_sequence() -> None:
    encoder = StateEncoder()
    planner = ModelPredictivePlanner(make_model(encoder), encoder, PrimitiveExecutor().available_primitives())
    sequence = planner.plan(make_scene(), np.zeros(31, dtype=float), [])
    assert len(sequence) > 0


def test_plan_with_horizon_one_returns_single_step_sequence() -> None:
    encoder = StateEncoder()
    planner = ModelPredictivePlanner(
        make_model(encoder),
        encoder,
        PrimitiveExecutor().available_primitives(),
        config=MPPConfig(horizon=1, n_samples=10),
    )
    sequence = planner.plan(make_scene(), np.zeros(31, dtype=float), [], n_steps=3)
    assert len(sequence) == 1


def test_multiple_calls_to_plan_return_consistent_types() -> None:
    encoder = StateEncoder()
    planner = ModelPredictivePlanner(make_model(encoder), encoder, PrimitiveExecutor().available_primitives())
    first = planner.plan(make_scene(), np.zeros(31, dtype=float), [])
    second = planner.plan(make_scene(), np.zeros(31, dtype=float), [])
    assert isinstance(first, list)
    assert isinstance(second, list)


def test_world_state_as_vector_has_combined_shape() -> None:
    encoder = StateEncoder()
    state = make_world_state(encoder)
    assert state.as_vector().shape == (encoder.state_dim + encoder.relation_dim,)


def test_world_state_preserves_timestamp() -> None:
    encoder = StateEncoder()
    state = make_world_state(encoder, offset=1.5)
    assert state.timestamp == 1.5


def test_encode_action_normalizes_hashed_fields() -> None:
    encoder = StateEncoder()
    action = encoder.encode_action("place", {"object_id": "cup", "surface_id": "table"})
    assert 0.0 <= action[6] <= 1.0
    assert 0.0 <= action[7] <= 1.0


def test_encode_relations_is_binary() -> None:
    encoder = StateEncoder()
    relation_vec = encoder.encode_relations(make_scene())
    assert set(np.unique(relation_vec)).issubset({0.0, 1.0})


def test_collector_done_flag_marks_last_transition() -> None:
    collector = WorldModelCollector()
    samples = collector.collect(BenchmarkSuite.DEFAULT, n_tasks=1)
    assert any(sample.done for sample in samples)


def test_collector_reward_is_float() -> None:
    collector = WorldModelCollector()
    sample = collector.collect(BenchmarkSuite.DEFAULT, n_tasks=1)[0]
    assert isinstance(sample.reward, float)


def test_score_state_returns_zero_for_empty_goals() -> None:
    encoder = StateEncoder()
    planner = ModelPredictivePlanner(make_model(encoder), encoder, PrimitiveExecutor().available_primitives())
    assert planner.score_state(make_world_state(encoder).as_vector(), [], encoder) == 0.0


def test_planner_uses_available_primitives() -> None:
    encoder = StateEncoder()
    available = ["reach", "grasp"]
    planner = ModelPredictivePlanner(make_model(encoder), encoder, available)
    sequence = planner.plan(make_scene(), np.zeros(31, dtype=float), [], n_steps=4)
    assert all(step["primitive"] in available for step in sequence)


def test_fit_handles_empty_sample_list() -> None:
    encoder = StateEncoder()
    trainer = WorldModelTrainer(make_model(encoder))
    assert trainer.fit([], epochs=3) == [0.0, 0.0, 0.0]


def test_train_step_handles_empty_sample_list() -> None:
    encoder = StateEncoder()
    trainer = WorldModelTrainer(make_model(encoder))
    assert trainer.train_step([]) == 0.0
