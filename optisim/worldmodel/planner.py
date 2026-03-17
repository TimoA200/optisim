"""Model-predictive planning over learned scene transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optisim.scene import SceneGraph
from optisim.worldmodel.model import WorldModelNet
from optisim.worldmodel.state import StateEncoder


def _stable_goal_hash(value: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(str(value)))


@dataclass(slots=True)
class MPPConfig:
    """Configuration for random-shooting model-predictive planning."""

    horizon: int = 5
    n_samples: int = 50
    discount: float = 0.95
    rng_seed: int = 42


class ModelPredictivePlanner:
    """Random-shooting planner over a learned world model."""

    def __init__(
        self,
        model: WorldModelNet,
        encoder: StateEncoder,
        available_primitives: list[str],
        config: MPPConfig | None = None,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.available_primitives = list(available_primitives)
        self.config = MPPConfig() if config is None else config
        self._rng = np.random.default_rng(self.config.rng_seed)

    def plan(
        self,
        scene: SceneGraph,
        robot_joints: np.ndarray,
        goal_predicates: list[dict],
        n_steps: int = 3,
    ) -> list[dict]:
        """Sample action sequences and keep the one with the highest predicted goal score."""

        del robot_joints
        rollout_steps = max(1, min(int(n_steps), int(self.config.horizon)))
        state_vec = np.concatenate([self.encoder.encode_scene(scene), self.encoder.encode_relations(scene)])
        best_score = -np.inf
        best_sequence: list[dict] = []

        for _ in range(self.config.n_samples):
            candidate = [self._sample_primitive_step(scene) for _ in range(rollout_steps)]
            action_sequence = [
                self.encoder.encode_action(step["primitive"], dict(step.get("params", {}))) for step in candidate
            ]
            predicted = self.rollout(state_vec, action_sequence)
            score = self.score_state(predicted, goal_predicates, self.encoder)
            if score > best_score:
                best_score = score
                best_sequence = candidate
        return best_sequence

    def score_state(self, predicted_features: np.ndarray, goal_predicates: list[dict], encoder: StateEncoder) -> float:
        """Score a predicted encoded state against goal predicates using relation activations."""

        if not goal_predicates:
            return 0.0
        relation_vec = np.asarray(predicted_features, dtype=float).reshape(-1)[-encoder.relation_dim :]
        matched = 0
        for predicate in goal_predicates:
            subject_id = str(predicate.get("subject_id", predicate.get("subject", predicate.get("node_id", ""))))
            relation_name = str(predicate.get("predicate", ""))
            object_id = str(predicate.get("object_id", predicate.get("object", subject_id)))
            relation_key = f"{subject_id}:{relation_name}:{object_id}"
            hashed_index = _stable_goal_hash(relation_key) % encoder.relation_dim
            if relation_vec[hashed_index] > 0.5:
                matched += 1
        return float(matched / len(goal_predicates))

    def rollout(self, state_vec: np.ndarray, action_sequence: list[np.ndarray]) -> np.ndarray:
        """Iteratively apply the learned model over a sequence of encoded actions."""

        predicted = np.asarray(state_vec, dtype=float).reshape(-1).copy()
        for action_vec in action_sequence:
            predicted = self.model.predict(predicted, action_vec)
        return predicted

    def _sample_primitive_step(self, scene: SceneGraph) -> dict:
        primitive = str(self._rng.choice(self.available_primitives))
        node_ids = sorted(scene.nodes)
        target_id = str(self._rng.choice(node_ids)) if node_ids else ""
        surface_ids = [node.id for node in sorted(scene.nodes.values(), key=lambda node: node.id) if node.category == "surface"]
        surface_id = str(self._rng.choice(surface_ids)) if surface_ids else target_id
        if primitive in {"reach", "grasp", "navigate"}:
            params = {"target_id": target_id}
        elif primitive == "place":
            params = {"object_id": target_id, "surface_id": surface_id}
        elif primitive == "push":
            params = {"target_id": target_id, "direction": [1.0, 0.0, 0.0]}
        elif primitive == "handover":
            params = {"object_id": target_id, "from_arm": "left", "to_arm": "right"}
        else:
            params = {}
        return {"primitive": primitive, "params": params}


__all__ = ["MPPConfig", "ModelPredictivePlanner"]
