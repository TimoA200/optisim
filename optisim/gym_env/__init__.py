"""Gymnasium integration for optisim."""

from optisim.gym_env.optisim_env import DEFAULT_ENV_ID, ObservationConfig, OptisimEnv, register_optisim_env
from optisim.gym_env.reward import CollisionPenalty, CompositeReward, ReachReward, TaskCompletionReward
from optisim.gym_env.wrappers import FlattenJoints, NormalizeObservation, RecordEpisode

__all__ = [
    "DEFAULT_ENV_ID",
    "ObservationConfig",
    "OptisimEnv",
    "register_optisim_env",
    "ReachReward",
    "TaskCompletionReward",
    "CollisionPenalty",
    "CompositeReward",
    "NormalizeObservation",
    "FlattenJoints",
    "RecordEpisode",
]
