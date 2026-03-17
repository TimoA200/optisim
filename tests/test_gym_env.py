from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from optisim.gym_env.reward import CollisionPenalty, CompositeReward, ReachReward, TaskCompletionReward
from optisim.sim.collision import Collision
from optisim.sim.recording import SimulationRecording


class _FakeRewardEnv:
    def __init__(
        self,
        *,
        effector_position: list[float] | np.ndarray,
        target_position: list[float] | np.ndarray | None = None,
        task_complete: bool = False,
    ) -> None:
        self._effector_position = np.asarray(effector_position, dtype=np.float64)
        self.target_position = None if target_position is None else np.asarray(target_position, dtype=np.float64)
        self.task_complete = task_complete

    def end_effector_position(self, effector: str) -> np.ndarray:
        del effector
        return self._effector_position.copy()

    def is_task_complete(self) -> bool:
        return self.task_complete


def _gym_modules():
    gym = pytest.importorskip("gymnasium")
    from optisim.gym_env import FlattenJoints, NormalizeObservation, ObservationConfig, OptisimEnv, RecordEpisode
    from optisim.gym_env import register_optisim_env

    return SimpleNamespace(
        gym=gym,
        FlattenJoints=FlattenJoints,
        NormalizeObservation=NormalizeObservation,
        ObservationConfig=ObservationConfig,
        OptisimEnv=OptisimEnv,
        RecordEpisode=RecordEpisode,
        register_optisim_env=register_optisim_env,
    )


def test_reach_reward_increases_when_effector_moves_toward_target() -> None:
    env = _FakeRewardEnv(effector_position=[0.0, 0.0, 0.0], target_position=[1.0, 0.0, 0.0])
    reward = ReachReward(scale=2.0)
    reward.reset(env)

    env._effector_position = np.array([0.4, 0.0, 0.0], dtype=np.float64)

    assert reward.compute(env, collisions=[], terminated=False, truncated=False, info={}) == pytest.approx(0.8)


def test_reach_reward_pays_success_bonus_only_once() -> None:
    env = _FakeRewardEnv(effector_position=[0.0, 0.0, 0.0], target_position=[0.01, 0.0, 0.0])
    reward = ReachReward(distance_threshold=0.05, success_bonus=3.0)
    reward.reset(env)

    first = reward.compute(env, collisions=[], terminated=False, truncated=False, info={})
    second = reward.compute(env, collisions=[], terminated=False, truncated=False, info={})

    assert first == pytest.approx(3.0)
    assert second == pytest.approx(0.0)


def test_task_completion_reward_only_triggers_once() -> None:
    env = _FakeRewardEnv(effector_position=[0.0, 0.0, 0.0], task_complete=True)
    reward = TaskCompletionReward(bonus=7.5)
    reward.reset(env)

    assert reward.compute(env, collisions=[], terminated=True, truncated=False, info={}) == pytest.approx(7.5)
    assert reward.compute(env, collisions=[], terminated=True, truncated=False, info={}) == pytest.approx(0.0)


def test_collision_penalty_uses_collision_count_by_default() -> None:
    env = _FakeRewardEnv(effector_position=[0.0, 0.0, 0.0])
    penalty = CollisionPenalty(scale=0.75)
    collisions = [
        Collision("a", "b", 0.1),
        Collision("c", "d", 0.2),
    ]

    assert penalty.compute(env, collisions=collisions, terminated=False, truncated=False, info={}) == pytest.approx(-1.5)


def test_composite_reward_sums_subrewards() -> None:
    env = _FakeRewardEnv(effector_position=[0.0, 0.0, 0.0], target_position=[0.01, 0.0, 0.0], task_complete=True)
    reward = CompositeReward(
        [
            ReachReward(distance_threshold=0.05, success_bonus=1.5),
            TaskCompletionReward(bonus=2.5),
            CollisionPenalty(scale=1.0),
        ]
    )
    reward.reset(env)
    collisions = [Collision("a", "b", 0.1)]

    value = reward.compute(env, collisions=collisions, terminated=True, truncated=False, info={})

    assert value == pytest.approx(3.0)


def test_default_env_creation_exposes_flat_box_spaces() -> None:
    modules = _gym_modules()
    env = modules.OptisimEnv()

    assert env.observation_space.shape[0] > 0
    assert env.action_space.shape == (len(env.robot.joints),)


def test_custom_observation_config_changes_observation_shape() -> None:
    modules = _gym_modules()
    env = modules.OptisimEnv(
        observation_config=modules.ObservationConfig(
            joint_positions=True,
            link_positions=False,
            object_states=False,
            joint_names=["torso_yaw", "torso_pitch"],
        )
    )
    observation, info = env.reset()

    assert observation.shape == (2,)
    assert info["observation_components"]["joint_positions"].shape == (2,)


def test_reset_returns_observation_inside_space() -> None:
    modules = _gym_modules()
    env = modules.OptisimEnv()
    observation, info = env.reset(seed=123)

    assert env.observation_space.contains(observation)
    assert info["step_count"] == 0
    assert info["time_s"] == pytest.approx(0.0)


def test_step_with_random_action_returns_gymnasium_tuple() -> None:
    modules = _gym_modules()
    env = modules.OptisimEnv(max_steps=5)
    env.reset(seed=0)

    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "collisions" in info


def test_action_and_observation_shapes_match_component_counts() -> None:
    modules = _gym_modules()
    env = modules.OptisimEnv()
    observation, info = env.reset()
    component_dim = sum(value.shape[0] for value in info["observation_components"].values())

    assert observation.shape[0] == component_dim
    assert env.action_space.shape[0] == len(env.robot.joints)


def test_episode_truncates_at_max_steps() -> None:
    modules = _gym_modules()
    env = modules.OptisimEnv(max_steps=2)
    env.reset(seed=0)

    _, _, terminated_1, truncated_1, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    _, _, terminated_2, truncated_2, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))

    assert not terminated_1
    assert not truncated_1
    assert not terminated_2
    assert truncated_2


def test_record_episode_wrapper_writes_optisim_recording(tmp_path: Path) -> None:
    modules = _gym_modules()
    env = modules.RecordEpisode(modules.OptisimEnv(max_steps=1), output_dir=tmp_path)
    env.reset(seed=0)

    env.step(np.zeros(env.action_space.shape, dtype=np.float32))

    assert env.last_recording_path is not None
    recording = SimulationRecording.from_file(env.last_recording_path)
    assert recording.frame_count() >= 2


def test_registration_helper_supports_gymnasium_make() -> None:
    modules = _gym_modules()
    env_id = modules.register_optisim_env("OptisimTest-v0", max_steps=3)
    env = modules.gym.make(env_id)
    try:
        observation, info = env.reset()
        assert observation.shape == env.observation_space.shape
        assert info["step_count"] == 0
    finally:
        env.close()


def test_flatten_joints_wrapper_exposes_joint_slice_only() -> None:
    modules = _gym_modules()
    env = modules.FlattenJoints(modules.OptisimEnv())
    observation, _ = env.reset()

    assert observation.shape == (len(env.unwrapped.robot.joints),)


def test_normalize_observation_wrapper_preserves_shape() -> None:
    modules = _gym_modules()
    env = modules.NormalizeObservation(modules.OptisimEnv())
    observation, _ = env.reset()

    assert observation.shape == env.observation_space.shape
    assert np.isfinite(observation).all()


def test_task_definition_path_can_be_used_for_env_creation() -> None:
    modules = _gym_modules()
    env = modules.OptisimEnv(task_definition="examples/pick_and_place.yaml")
    observation, info = env.reset()

    assert observation.shape == env.observation_space.shape
    assert info["distance_to_target"] is not None


def test_default_reset_creates_recording_handle() -> None:
    modules = _gym_modules()
    env = modules.OptisimEnv()
    env.reset()

    assert env.recording is not None
    assert env.recording.frame_count() == 1


def test_render_without_mode_returns_none() -> None:
    modules = _gym_modules()
    env = modules.OptisimEnv(render_mode=None)
    env.reset()

    assert env.render() is None

__all__ = ["test_reach_reward_increases_when_effector_moves_toward_target", "test_reach_reward_pays_success_bonus_only_once", "test_task_completion_reward_only_triggers_once", "test_collision_penalty_uses_collision_count_by_default", "test_composite_reward_sums_subrewards", "test_default_env_creation_exposes_flat_box_spaces", "test_custom_observation_config_changes_observation_shape", "test_reset_returns_observation_inside_space", "test_step_with_random_action_returns_gymnasium_tuple", "test_action_and_observation_shapes_match_component_counts", "test_episode_truncates_at_max_steps", "test_record_episode_wrapper_writes_optisim_recording", "test_registration_helper_supports_gymnasium_make", "test_flatten_joints_wrapper_exposes_joint_slice_only", "test_normalize_observation_wrapper_preserves_shape", "test_task_definition_path_can_be_used_for_env_creation", "test_default_reset_creates_recording_handle", "test_render_without_mode_returns_none"]
