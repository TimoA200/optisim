"""Data collection helpers for learned world models."""

from __future__ import annotations

import numpy as np

from optisim.benchmark import BenchmarkEvaluator, BenchmarkSuite
from optisim.primitives import PrimitiveExecutor, PrimitiveStatus, apply_effects
from optisim.scene import SceneGraph
from optisim.worldmodel.model import TransitionSample
from optisim.worldmodel.state import StateEncoder, WorldState


class WorldModelCollector:
    """Collect transition samples by replaying benchmark primitive sequences."""

    def __init__(
        self,
        encoder: StateEncoder | None = None,
        executor: PrimitiveExecutor | None = None,
        evaluator: BenchmarkEvaluator | None = None,
    ) -> None:
        self.encoder = StateEncoder() if encoder is None else encoder
        self.executor = PrimitiveExecutor() if executor is None else executor
        self.evaluator = BenchmarkEvaluator(self.executor) if evaluator is None else evaluator

    def collect(self, suite: BenchmarkSuite, n_tasks: int = 5, rng_seed: int = 42) -> list[TransitionSample]:
        """Run a subset of benchmark tasks and return encoded transition samples."""

        rng = np.random.default_rng(rng_seed)
        task_names = suite.list_tasks()
        if not task_names:
            return []
        count = min(int(n_tasks), len(task_names))
        chosen_names = list(rng.choice(task_names, size=count, replace=False))
        samples: list[TransitionSample] = []

        for task_name in chosen_names:
            task = suite.get(task_name)
            # Drive task construction through the evaluator-compatible task API.
            scene = task.build_scene()
            robot_id = self._find_robot_id(scene)
            robot_joints = np.zeros(31, dtype=float)
            for step_index, step in enumerate(task.primitive_sequence):
                state = self._encode_state(scene, robot_joints, timestamp=float(step_index))
                primitive_name = str(step["primitive"])
                params = dict(step.get("params", {}))
                action_vec = self.encoder.encode_action(primitive_name, params)
                primitive = self.executor.get(primitive_name, params)
                result = primitive.execute(scene, robot_id, robot_joints)
                if result.status is PrimitiveStatus.SUCCESS:
                    apply_effects(scene, primitive.get_effects(scene, robot_id))
                    if result.joint_trajectory:
                        robot_joints = np.asarray(result.joint_trajectory[-1], dtype=float).copy()
                next_state = self._encode_state(scene, robot_joints, timestamp=float(step_index + 1))
                samples.append(
                    TransitionSample(
                        state=state,
                        action_vec=action_vec,
                        next_state=next_state,
                        reward=1.0 if result.status is PrimitiveStatus.SUCCESS else 0.0,
                        done=result.status is not PrimitiveStatus.SUCCESS or step_index == len(task.primitive_sequence) - 1,
                    )
                )
                if result.status is not PrimitiveStatus.SUCCESS:
                    break
        return samples

    def _encode_state(self, scene: SceneGraph, robot_joints: np.ndarray, timestamp: float) -> WorldState:
        return WorldState(
            joint_positions=np.asarray(robot_joints, dtype=float).copy(),
            scene_features=self.encoder.encode_scene(scene),
            relation_vector=self.encoder.encode_relations(scene),
            timestamp=timestamp,
        )

    @staticmethod
    def _find_robot_id(scene: SceneGraph) -> str:
        for node in scene.nodes.values():
            if node.category == "robot":
                return node.id
        raise ValueError("scene does not contain a robot node")


__all__ = ["WorldModelCollector"]
