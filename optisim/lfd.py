"""Learning from Demonstration utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _as_float_list(values: list[float] | tuple[float, ...] | np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _linspace_times(start: float, end: float, n_steps: int) -> np.ndarray:
    if n_steps <= 1:
        return np.array([float(start)], dtype=float)
    return np.linspace(float(start), float(end), int(n_steps))


@dataclass(slots=True)
class DemoStep:
    """Single recorded demonstration step."""

    timestep: float
    joint_positions: list[float]
    ee_pose: tuple[float, ...]
    gripper_open: bool
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestep": float(self.timestep),
            "joint_positions": _as_float_list(self.joint_positions),
            "ee_pose": _as_float_list(self.ee_pose),
            "gripper_open": bool(self.gripper_open),
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DemoStep:
        return cls(
            timestep=float(payload["timestep"]),
            joint_positions=_as_float_list(payload["joint_positions"]),
            ee_pose=tuple(_as_float_list(payload["ee_pose"])),
            gripper_open=bool(payload["gripper_open"]),
            extras=dict(payload.get("extras") or {}),
        )


@dataclass(slots=True)
class Demonstration:
    """Container for an LfD trajectory demonstration."""

    task_name: str
    steps: list[DemoStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.num_steps <= 1:
            return 0.0
        return float(self.steps[-1].timestep - self.steps[0].timestep)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def joint_dim(self) -> int:
        if not self.steps:
            return 0
        return len(self.steps[0].joint_positions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Demonstration:
        return cls(
            task_name=str(payload["task_name"]),
            steps=[DemoStep.from_dict(item) for item in payload.get("steps", [])],
            metadata=dict(payload.get("metadata") or {}),
        )

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> Demonstration:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


class DemonstrationRecorder:
    """Recorder that accumulates teleoperation or policy demonstrations."""

    def __init__(self, task_name: str, metadata: dict[str, Any] | None = None) -> None:
        self.demonstration = Demonstration(task_name=task_name, metadata=dict(metadata or {}))

    def record(
        self,
        step: int,
        joint_positions: list[float],
        ee_pose: tuple[Any, ...],
        gripper_open: bool,
        extras: dict[str, Any] | None = None,
    ) -> None:
        self.demonstration.steps.append(
            DemoStep(
                timestep=float(step),
                joint_positions=_as_float_list(joint_positions),
                ee_pose=tuple(_as_float_list(ee_pose)),
                gripper_open=bool(gripper_open),
                extras=dict(extras or {}),
            )
        )

    @property
    def duration(self) -> float:
        return self.demonstration.duration

    @property
    def num_steps(self) -> int:
        return self.demonstration.num_steps

    @property
    def joint_dim(self) -> int:
        return self.demonstration.joint_dim

    def save(self, path: str) -> None:
        self.demonstration.save(path)

    @classmethod
    def load(cls, path: str) -> DemonstrationRecorder:
        demo = Demonstration.load(path)
        recorder = cls(task_name=demo.task_name, metadata=demo.metadata)
        recorder.demonstration = demo
        return recorder


class DynamicMovementPrimitive:
    """Classical per-joint DMP with Gaussian basis functions."""

    def __init__(
        self,
        alpha_z: float = 25.0,
        beta_z: float = 6.25,
        alpha_x: float = 1.0,
        n_basis: int = 25,
    ) -> None:
        self.alpha_z = float(alpha_z)
        self.beta_z = float(beta_z)
        self.alpha_x = float(alpha_x)
        self.n_basis = int(n_basis)
        self.centers = np.exp(-self.alpha_x * np.linspace(0.0, 1.0, self.n_basis))
        center_diffs = np.diff(self.centers)
        widths = np.empty(self.n_basis, dtype=float)
        if self.n_basis > 1:
            widths[:-1] = 1.0 / np.maximum(center_diffs**2, 1e-8)
            widths[-1] = widths[-2]
        else:
            widths[0] = 1.0
        self.widths = widths
        self.weights: np.ndarray | None = None
        self.start: np.ndarray | None = None
        self.goal: np.ndarray | None = None
        self.tau: float = 1.0
        self.gripper_profile: np.ndarray | None = None
        self.ee_profile: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.task_name: str = ""
        self._trained = False

    def _phase(self, times: np.ndarray, tau: float) -> np.ndarray:
        return np.exp(-self.alpha_x * times / max(float(tau), 1e-8))

    def _basis(self, x: np.ndarray) -> np.ndarray:
        psi = np.exp(-self.widths[:, None] * (x[None, :] - self.centers[:, None]) ** 2)
        denominator = np.sum(psi, axis=0, keepdims=True)
        denominator = np.where(denominator > 1e-12, denominator, 1.0)
        return (psi * x[None, :]) / denominator

    def train(self, demonstration: Demonstration) -> None:
        if demonstration.num_steps < 2:
            raise ValueError("DMP training requires at least two demonstration steps.")
        if demonstration.joint_dim == 0:
            raise ValueError("DMP training requires non-empty joint positions.")

        times = np.array([step.timestep for step in demonstration.steps], dtype=float)
        times = times - times[0]
        tau = max(float(times[-1]), 1.0)
        q = np.array([step.joint_positions for step in demonstration.steps], dtype=float)

        dq = np.gradient(q, times, axis=0)
        ddq = np.gradient(dq, times, axis=0)
        x = self._phase(times, tau)
        phi = self._basis(x).T

        start = q[0].copy()
        goal = q[-1].copy()
        weights = np.zeros((demonstration.joint_dim, self.n_basis), dtype=float)

        for joint in range(demonstration.joint_dim):
            delta = goal[joint] - start[joint]
            scale = delta if abs(delta) > 1e-8 else 1.0
            target = (
                (tau**2) * ddq[:, joint]
                - self.alpha_z * (self.beta_z * (goal[joint] - q[:, joint]) - tau * dq[:, joint])
            ) / scale
            solution, *_ = np.linalg.lstsq(phi, target, rcond=None)
            weights[joint] = solution

        self.weights = weights
        self.start = start
        self.goal = goal
        self.tau = tau
        self.task_name = demonstration.task_name
        self.metadata = dict(demonstration.metadata)
        self.gripper_profile = np.array(
            [1.0 if step.gripper_open else 0.0 for step in demonstration.steps],
            dtype=float,
        )
        self.ee_profile = np.array([step.ee_pose for step in demonstration.steps], dtype=float)
        self._trained = True

    def generate(
        self,
        goal: list[float],
        start: list[float] | None = None,
        duration: float | None = None,
    ) -> Demonstration:
        if not self._trained or self.weights is None or self.goal is None or self.start is None:
            raise ValueError("DMP must be trained before generation.")

        goal_array = np.array(goal, dtype=float)
        if goal_array.shape != self.goal.shape:
            raise ValueError("Goal dimensionality does not match the trained demonstration.")

        start_array = self.start.copy() if start is None else np.array(start, dtype=float)
        if start_array.shape != self.start.shape:
            raise ValueError("Start dimensionality does not match the trained demonstration.")

        tau = float(duration) if duration is not None else self.tau
        tau = max(tau, 1e-6)

        n_steps = len(self.gripper_profile) if self.gripper_profile is not None else int(round(self.tau)) + 1
        n_steps = max(n_steps, 2)
        times = _linspace_times(0.0, tau, n_steps)
        dt = float(times[1] - times[0]) if n_steps > 1 else tau

        y = start_array.copy()
        z = np.zeros_like(y)
        x = 1.0
        steps: list[DemoStep] = []

        for index, time_s in enumerate(times):
            psi = np.exp(-self.widths * (x - self.centers) ** 2)
            psi_sum = float(np.sum(psi))
            ee_pose: tuple[float, ...]
            gripper_open: bool

            if self.ee_profile is not None and len(self.ee_profile) > 0:
                ee_idx = min(index, len(self.ee_profile) - 1)
                ee_pose = tuple(float(value) for value in self.ee_profile[ee_idx])
            else:
                ee_pose = tuple()

            if self.gripper_profile is not None and len(self.gripper_profile) > 0:
                grip_idx = min(index, len(self.gripper_profile) - 1)
                gripper_open = bool(self.gripper_profile[grip_idx] >= 0.5)
            else:
                gripper_open = True

            steps.append(
                DemoStep(
                    timestep=float(time_s),
                    joint_positions=_as_float_list(y),
                    ee_pose=ee_pose,
                    gripper_open=gripper_open,
                    extras={},
                )
            )

            if index == n_steps - 1:
                break

            for joint in range(len(y)):
                delta = goal_array[joint] - start_array[joint]
                scale = delta if abs(delta) > 1e-8 else 1.0
                forcing = 0.0 if psi_sum <= 1e-12 else float(psi @ self.weights[joint]) * x / psi_sum * scale
                dz = (
                    self.alpha_z * (self.beta_z * (goal_array[joint] - y[joint]) - z[joint]) + forcing
                ) * dt / tau
                z[joint] += dz
                y[joint] += z[joint] * dt / tau

            x += (-self.alpha_x * x / tau) * dt

        steps[-1].joint_positions = _as_float_list(goal_array)
        return Demonstration(
            task_name=self.task_name or "generated_dmp",
            steps=steps,
            metadata={**self.metadata, "generated_by": "DynamicMovementPrimitive", "duration": tau},
        )


class DemonstrationLibrary:
    """Simple on-disk and in-memory library of demonstrations."""

    def __init__(self) -> None:
        self._items: dict[str, list[Demonstration]] = {}

    @property
    def len(self) -> int:
        return sum(len(demos) for demos in self._items.values())

    def add(self, demo: Demonstration) -> None:
        self._items.setdefault(demo.task_name, []).append(demo)

    def get(self, task_name: str) -> list[Demonstration]:
        return list(self._items.get(task_name, []))

    def best(self, task_name: str) -> Demonstration | None:
        demos = self._items.get(task_name, [])
        if not demos:
            return None
        return max(demos, key=lambda demo: (demo.duration, demo.num_steps))

    def all_tasks(self) -> list[str]:
        return sorted(self._items)

    def save_all(self, directory: str) -> None:
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        for task_name, demos in self._items.items():
            for index, demo in enumerate(demos):
                file_name = f"{task_name}_{index:03d}.json"
                demo.save(str(target_dir / file_name))

    @classmethod
    def load_all(cls, directory: str) -> DemonstrationLibrary:
        library = cls()
        source_dir = Path(directory)
        if not source_dir.exists():
            return library
        for path in sorted(source_dir.glob("*.json")):
            library.add(Demonstration.load(str(path)))
        return library


class DemonstrationPlayer:
    """Trajectory replay and simple adaptation helpers."""

    def play(self, demo: Demonstration, time_scale: float = 1.0) -> list[dict[str, Any]]:
        scale = float(time_scale) if time_scale > 0.0 else 1.0
        frames: list[dict[str, Any]] = []
        for step in demo.steps:
            frames.append(
                {
                    "timestep": float(step.timestep) * scale,
                    "joint_positions": list(step.joint_positions),
                    "ee_pose": tuple(step.ee_pose),
                    "gripper_open": bool(step.gripper_open),
                    "extras": dict(step.extras),
                }
            )
        return frames

    def interpolate(self, demo: Demonstration, n_steps: int) -> Demonstration:
        if n_steps <= 0:
            raise ValueError("Interpolation requires at least one step.")
        if demo.num_steps == 0:
            return Demonstration(task_name=demo.task_name, metadata=dict(demo.metadata))
        if demo.num_steps == 1:
            step = demo.steps[0]
            return Demonstration(
                task_name=demo.task_name,
                metadata=dict(demo.metadata),
                steps=[
                    DemoStep(
                        timestep=float(index),
                        joint_positions=list(step.joint_positions),
                        ee_pose=tuple(step.ee_pose),
                        gripper_open=bool(step.gripper_open),
                        extras=dict(step.extras),
                    )
                    for index in range(n_steps)
                ],
            )

        original_times = np.array([step.timestep for step in demo.steps], dtype=float)
        target_times = _linspace_times(original_times[0], original_times[-1], n_steps)
        joints = np.array([step.joint_positions for step in demo.steps], dtype=float)
        ee = np.array([step.ee_pose for step in demo.steps], dtype=float)
        gripper = np.array([1.0 if step.gripper_open else 0.0 for step in demo.steps], dtype=float)

        interpolated_steps: list[DemoStep] = []
        for index, time_s in enumerate(target_times):
            joint_positions = [
                float(np.interp(time_s, original_times, joints[:, joint])) for joint in range(joints.shape[1])
            ]
            ee_pose = tuple(float(np.interp(time_s, original_times, ee[:, axis])) for axis in range(ee.shape[1]))
            gripper_open = bool(np.interp(time_s, original_times, gripper) >= 0.5)
            source_idx = min(index * max(demo.num_steps - 1, 1) // max(n_steps - 1, 1), demo.num_steps - 1)
            interpolated_steps.append(
                DemoStep(
                    timestep=float(time_s),
                    joint_positions=joint_positions,
                    ee_pose=ee_pose,
                    gripper_open=gripper_open,
                    extras=dict(demo.steps[source_idx].extras),
                )
            )
        return Demonstration(task_name=demo.task_name, steps=interpolated_steps, metadata=dict(demo.metadata))

    def adapt_to_goal(self, demo: Demonstration, new_goal: list[float]) -> Demonstration:
        if demo.num_steps == 0:
            return Demonstration(task_name=demo.task_name, metadata=dict(demo.metadata))
        if demo.joint_dim != len(new_goal):
            raise ValueError("New goal dimensionality does not match the demonstration.")

        current_goal = np.array(demo.steps[-1].joint_positions, dtype=float)
        delta = np.array(new_goal, dtype=float) - current_goal
        adapted_steps: list[DemoStep] = []
        denominator = max(demo.num_steps - 1, 1)

        for index, step in enumerate(demo.steps):
            blend = index / denominator
            new_positions = np.array(step.joint_positions, dtype=float) + delta * blend
            adapted_steps.append(
                DemoStep(
                    timestep=float(step.timestep),
                    joint_positions=_as_float_list(new_positions),
                    ee_pose=tuple(step.ee_pose),
                    gripper_open=bool(step.gripper_open),
                    extras=dict(step.extras),
                )
            )

        return Demonstration(task_name=demo.task_name, steps=adapted_steps, metadata=dict(demo.metadata))


__all__ = [
    "DemoStep",
    "Demonstration",
    "DemonstrationLibrary",
    "DemonstrationPlayer",
    "DemonstrationRecorder",
    "DynamicMovementPrimitive",
]
