"""Coordinated dual-arm planning and cooperative manipulation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optisim.math3d import Pose, Quaternion, normalize, pose_from_matrix, vec3

_VALID_HANDS = {"left", "right"}
_VALID_CONSTRAINT_TYPES = {"rigid", "soft", "symmetric", "relative_pose"}


def _quat_array(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (4,):
        raise ValueError(f"expected quaternion shape (4,), received {array.shape}")
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return Quaternion.identity().as_np()
    return array / norm


def _mat4(values: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"expected transform shape (4, 4), received {array.shape}")
    return array


def _frame_transform(frame: "GraspFrame") -> np.ndarray:
    pose = Pose(
        position=vec3(frame.position),
        orientation=Quaternion(*_quat_array(frame.orientation).tolist()),
    )
    return pose.matrix()


def _interpolate_points(points: list[np.ndarray], count: int) -> list[np.ndarray]:
    if not points:
        raise ValueError("waypoint lists must not be empty")
    if count <= 1:
        return [vec3(points[0])]
    if len(points) == 1:
        return [vec3(points[0]) for _ in range(count)]
    control = np.stack([vec3(point) for point in points], axis=0)
    samples = np.linspace(0.0, float(len(control) - 1), count)
    out: list[np.ndarray] = []
    for sample in samples:
        lower = int(np.floor(sample))
        upper = min(lower + 1, len(control) - 1)
        alpha = sample - lower
        out.append(((1.0 - alpha) * control[lower] + alpha * control[upper]).astype(np.float64))
    return out


@dataclass(slots=True)
class GraspFrame:
    """Single hand grasp frame in the world frame."""

    hand: str
    position: np.ndarray
    orientation: np.ndarray
    grasp_width: float = 0.08
    contact_normal: np.ndarray = field(default_factory=lambda: np.asarray([0.0, 0.0, 1.0], dtype=np.float64))

    def __post_init__(self) -> None:
        if self.hand not in _VALID_HANDS:
            raise ValueError(f"hand must be one of {_VALID_HANDS}, received '{self.hand}'")
        self.position = vec3(self.position)
        self.orientation = _quat_array(self.orientation)
        self.contact_normal = normalize(vec3(self.contact_normal))
        self.grasp_width = float(self.grasp_width)


@dataclass(slots=True)
class BimanualConstraint:
    """Desired coordination constraint between the two hands."""

    constraint_type: str
    relative_transform: np.ndarray
    compliance: float = 0.0
    max_force_error: float = 10.0

    def __post_init__(self) -> None:
        if self.constraint_type not in _VALID_CONSTRAINT_TYPES:
            raise ValueError(
                f"constraint_type must be one of {_VALID_CONSTRAINT_TYPES}, received '{self.constraint_type}'"
            )
        self.relative_transform = _mat4(self.relative_transform)
        self.compliance = float(np.clip(self.compliance, 0.0, 1.0))
        self.max_force_error = float(self.max_force_error)


@dataclass(slots=True)
class BimanualTask:
    """Bimanual waypoint task with an inter-hand coordination constraint."""

    left_waypoints: list[np.ndarray]
    right_waypoints: list[np.ndarray]
    constraint: BimanualConstraint
    object_pose: np.ndarray | None = None
    task_name: str = "bimanual_task"

    def __post_init__(self) -> None:
        if not self.left_waypoints or not self.right_waypoints:
            raise ValueError("left_waypoints and right_waypoints must both be non-empty")
        self.left_waypoints = [vec3(point) for point in self.left_waypoints]
        self.right_waypoints = [vec3(point) for point in self.right_waypoints]
        self.object_pose = None if self.object_pose is None else _mat4(self.object_pose)
        self.task_name = str(self.task_name)


@dataclass(slots=True)
class BimanualPlan:
    """Synchronized dual-arm plan generated from a bimanual task."""

    left_trajectory: list[GraspFrame]
    right_trajectory: list[GraspFrame]
    timestamps: list[float]
    constraint: BimanualConstraint
    is_synchronized: bool
    total_duration: float


class BimanualCoordinator:
    """Generate synchronized grasp-frame trajectories for two coordinated arms."""

    def __init__(self, robot_model=None) -> None:
        self.robot_model = robot_model

    def plan(self, task: BimanualTask) -> BimanualPlan:
        """Generate synchronized left/right grasp trajectories respecting the task constraint."""

        count = max(len(task.left_waypoints), len(task.right_waypoints))
        raw_left = _interpolate_points(task.left_waypoints, count)
        raw_right = _interpolate_points(task.right_waypoints, count)
        desired_offset = task.constraint.relative_transform[:3, 3]
        desired_orientation = pose_from_matrix(task.constraint.relative_transform).orientation.as_np()

        left_trajectory: list[GraspFrame] = []
        right_trajectory: list[GraspFrame] = []
        timestamps = np.linspace(0.0, max(float(count - 1), 0.0), count).tolist()

        for left_raw, right_raw in zip(raw_left, raw_right, strict=True):
            midpoint = 0.5 * (left_raw + right_raw)
            enforced_left = midpoint - 0.5 * desired_offset
            enforced_right = midpoint + 0.5 * desired_offset
            compliance = task.constraint.compliance
            left_position = ((1.0 - compliance) * enforced_left + compliance * left_raw).astype(np.float64)
            right_position = ((1.0 - compliance) * enforced_right + compliance * right_raw).astype(np.float64)
            left_trajectory.append(
                GraspFrame(
                    hand="left",
                    position=left_position,
                    orientation=Quaternion.identity().as_np(),
                )
            )
            right_trajectory.append(
                GraspFrame(
                    hand="right",
                    position=right_position,
                    orientation=desired_orientation,
                )
            )

        return BimanualPlan(
            left_trajectory=left_trajectory,
            right_trajectory=right_trajectory,
            timestamps=timestamps,
            constraint=task.constraint,
            is_synchronized=len(left_trajectory) == len(right_trajectory) == len(timestamps),
            total_duration=float(timestamps[-1] if timestamps else 0.0),
        )

    def execute_step(self, plan: BimanualPlan, step_idx: int) -> tuple[GraspFrame, GraspFrame]:
        """Return the pair of grasp frames for a synchronized plan step."""

        if step_idx < 0 or step_idx >= len(plan.left_trajectory):
            raise IndexError(f"step index {step_idx} out of range for plan length {len(plan.left_trajectory)}")
        return plan.left_trajectory[step_idx], plan.right_trajectory[step_idx]

    def check_constraint_violation(
        self,
        left: GraspFrame,
        right: GraspFrame,
        constraint: BimanualConstraint,
    ) -> float:
        """Return the transform error magnitude between the measured and desired hand relation."""

        left_tf = _frame_transform(left)
        right_tf = _frame_transform(right)
        actual_relative = np.linalg.inv(left_tf) @ right_tf
        error_transform = np.linalg.inv(constraint.relative_transform) @ actual_relative
        translation_error = float(np.linalg.norm(error_transform[:3, 3]))
        rotation_error = float(np.linalg.norm(np.eye(3, dtype=np.float64) - error_transform[:3, :3], ord="fro"))
        return translation_error + rotation_error


class CooperativeManipulation:
    """Helpers for force sharing when both hands jointly manipulate one object."""

    def __init__(self, object_mass: float = 1.0, object_inertia: np.ndarray | None = None) -> None:
        self.object_mass = float(object_mass)
        self.object_inertia = (
            np.eye(3, dtype=np.float64) if object_inertia is None else np.asarray(object_inertia, dtype=np.float64)
        )
        if self.object_inertia.shape != (3, 3):
            raise ValueError(f"object_inertia must have shape (3, 3), received {self.object_inertia.shape}")

    def distribute_wrench(
        self,
        total_wrench: np.ndarray,
        grasp_left: GraspFrame,
        grasp_right: GraspFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split the task wrench evenly while preserving total wrench conservation."""

        del grasp_left, grasp_right
        wrench = np.asarray(total_wrench, dtype=np.float64)
        if wrench.shape != (6,):
            raise ValueError(f"total_wrench must have shape (6,), received {wrench.shape}")
        left = 0.5 * wrench
        right = wrench - left
        return left, right

    def compute_internal_force(self, grasp_left: GraspFrame, grasp_right: GraspFrame) -> float:
        """Return the squeezing force proxy from the hand span projected onto the contact normals."""

        span = grasp_right.position - grasp_left.position
        span_norm = float(np.linalg.norm(span))
        if span_norm == 0.0:
            return 0.0
        direction = span / span_norm
        left_alignment = abs(float(np.dot(grasp_left.contact_normal, direction)))
        right_alignment = abs(float(np.dot(grasp_right.contact_normal, -direction)))
        support_force = 0.5 * self.object_mass * 9.81
        return float(max((left_alignment + right_alignment) * support_force, 0.0))

    def is_stable_grasp(
        self,
        grasp_left: GraspFrame,
        grasp_right: GraspFrame,
        total_wrench: np.ndarray,
    ) -> bool:
        """Check whether the shared load and internal squeeze remain within simple nominal limits."""

        left_wrench, right_wrench = self.distribute_wrench(total_wrench, grasp_left, grasp_right)
        force_limit = 100.0
        torque_limit = 25.0
        internal_force = self.compute_internal_force(grasp_left, grasp_right)
        return bool(
            np.linalg.norm(left_wrench[:3]) <= force_limit
            and np.linalg.norm(right_wrench[:3]) <= force_limit
            and np.linalg.norm(left_wrench[3:]) <= torque_limit
            and np.linalg.norm(right_wrench[3:]) <= torque_limit
            and internal_force <= force_limit
        )


class TaskPresets:
    """Factory helpers for common coordinated dual-arm manipulation tasks."""

    @staticmethod
    def pick_large_box(box_width: float = 0.4, box_height: float = 0.3, lift_height: float = 0.2) -> BimanualTask:
        half_width = box_width / 2.0
        center = np.asarray([0.55, 0.0, 0.85], dtype=np.float64)
        left_waypoints = [
            center + np.asarray([0.0, -half_width, 0.06], dtype=np.float64),
            center + np.asarray([0.0, -half_width, 0.0], dtype=np.float64),
            center + np.asarray([0.0, -half_width, lift_height], dtype=np.float64),
        ]
        right_waypoints = [
            center + np.asarray([0.0, half_width, 0.06], dtype=np.float64),
            center + np.asarray([0.0, half_width, 0.0], dtype=np.float64),
            center + np.asarray([0.0, half_width, lift_height], dtype=np.float64),
        ]
        object_pose = np.eye(4, dtype=np.float64)
        object_pose[:3, 3] = center + np.asarray([0.0, 0.0, box_height / 2.0], dtype=np.float64)
        relative_transform = np.eye(4, dtype=np.float64)
        relative_transform[:3, 3] = np.asarray([0.0, box_width, 0.0], dtype=np.float64)
        return BimanualTask(
            left_waypoints=left_waypoints,
            right_waypoints=right_waypoints,
            constraint=BimanualConstraint("rigid", relative_transform),
            object_pose=object_pose,
            task_name="pick_large_box",
        )

    @staticmethod
    def handoff(from_hand: str = "right", to_hand: str = "left") -> BimanualTask:
        if {from_hand, to_hand} != {"left", "right"}:
            raise ValueError("handoff presets support one left hand and one right hand")
        center = np.asarray([0.45, 0.0, 0.95], dtype=np.float64)
        left_waypoints = [
            np.asarray([0.40, -0.24, 0.95], dtype=np.float64),
            center + np.asarray([0.0, -0.08, 0.0], dtype=np.float64),
            center + np.asarray([0.06, -0.20, 0.02], dtype=np.float64),
        ]
        right_waypoints = [
            np.asarray([0.40, 0.24, 0.95], dtype=np.float64),
            center + np.asarray([0.0, 0.08, 0.0], dtype=np.float64),
            center + np.asarray([-0.06, 0.20, 0.02], dtype=np.float64),
        ]
        relative_transform = np.eye(4, dtype=np.float64)
        relative_transform[:3, 3] = np.asarray([0.0, 0.16, 0.0], dtype=np.float64)
        return BimanualTask(
            left_waypoints=left_waypoints,
            right_waypoints=right_waypoints,
            constraint=BimanualConstraint("soft", relative_transform, compliance=0.35),
            task_name=f"handoff_{from_hand}_to_{to_hand}",
        )

    @staticmethod
    def carry_tray(tray_width: float = 0.5, waypoints: list[np.ndarray] | None = None) -> BimanualTask:
        centers = (
            [np.asarray([0.50, 0.0, 0.95], dtype=np.float64), np.asarray([0.80, 0.0, 1.00], dtype=np.float64)]
            if waypoints is None
            else [vec3(point) for point in waypoints]
        )
        half_width = tray_width / 2.0
        left_waypoints = [center + np.asarray([0.0, -half_width, 0.0], dtype=np.float64) for center in centers]
        right_waypoints = [center + np.asarray([0.0, half_width, 0.0], dtype=np.float64) for center in centers]
        relative_transform = np.eye(4, dtype=np.float64)
        relative_transform[:3, 3] = np.asarray([0.0, tray_width, 0.0], dtype=np.float64)
        return BimanualTask(
            left_waypoints=left_waypoints,
            right_waypoints=right_waypoints,
            constraint=BimanualConstraint("symmetric", relative_transform),
            task_name="carry_tray",
        )

    @staticmethod
    def assembly_insert(
        peg_approach: np.ndarray | None = None,
        hole_position: np.ndarray | None = None,
    ) -> BimanualTask:
        approach = (
            np.asarray([0.58, 0.0, 1.02], dtype=np.float64)
            if peg_approach is None
            else vec3(peg_approach)
        )
        hole = (
            np.asarray([0.66, 0.0, 0.98], dtype=np.float64)
            if hole_position is None
            else vec3(hole_position)
        )
        left_waypoints = [
            approach + np.asarray([0.0, -0.10, 0.02], dtype=np.float64),
            hole + np.asarray([0.0, -0.08, 0.0], dtype=np.float64),
        ]
        right_waypoints = [
            approach + np.asarray([0.0, 0.10, 0.02], dtype=np.float64),
            hole + np.asarray([0.0, 0.08, 0.0], dtype=np.float64),
        ]
        relative_transform = np.eye(4, dtype=np.float64)
        relative_transform[:3, 3] = np.asarray([0.0, 0.16, 0.0], dtype=np.float64)
        return BimanualTask(
            left_waypoints=left_waypoints,
            right_waypoints=right_waypoints,
            constraint=BimanualConstraint("relative_pose", relative_transform, compliance=0.1),
            task_name="assembly_insert",
        )


__all__ = [
    "GraspFrame",
    "BimanualConstraint",
    "BimanualTask",
    "BimanualPlan",
    "BimanualCoordinator",
    "CooperativeManipulation",
    "TaskPresets",
]
