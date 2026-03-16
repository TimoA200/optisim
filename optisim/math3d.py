"""Core 3D math utilities used throughout optisim."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
QuaternionArray = NDArray[np.float64]


def vec3(values: Iterable[float]) -> Vector:
    array = np.asarray(list(values), dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"expected 3-vector, received shape {array.shape}")
    return array


def normalize(vector: Vector) -> Vector:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector.copy()
    return vector / norm


@dataclass(frozen=True)
class Quaternion:
    """Scalar-first quaternion for stable pose composition."""

    w: float
    x: float
    y: float
    z: float

    def as_np(self) -> QuaternionArray:
        return np.asarray([self.w, self.x, self.y, self.z], dtype=np.float64)

    def normalized(self) -> "Quaternion":
        q = normalize(self.as_np())
        return Quaternion(*q.tolist())

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        a = self.as_np()
        b = other.as_np()
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ).normalized()

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_euler(roll: float, pitch: float, yaw: float) -> "Quaternion":
        cr = cos(roll * 0.5)
        sr = sin(roll * 0.5)
        cp = cos(pitch * 0.5)
        sp = sin(pitch * 0.5)
        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)
        return Quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ).normalized()

    def to_rotation_matrix(self) -> Matrix:
        w, x, y, z = self.normalized().as_np()
        return np.asarray(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Pose:
    """Rigid body transform represented by translation and quaternion."""

    position: Vector
    orientation: Quaternion = Quaternion.identity()

    def matrix(self) -> Matrix:
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = self.orientation.to_rotation_matrix()
        out[:3, 3] = self.position
        return out

    def transform_point(self, point: Iterable[float]) -> Vector:
        return self.position + self.orientation.to_rotation_matrix() @ vec3(point)

    def compose(self, other: "Pose") -> "Pose":
        rotation = self.orientation.to_rotation_matrix()
        position = self.position + rotation @ other.position
        return Pose(position=position, orientation=self.orientation * other.orientation)

    @staticmethod
    def identity() -> "Pose":
        return Pose(position=np.zeros(3, dtype=np.float64), orientation=Quaternion.identity())

    @staticmethod
    def from_xyz_rpy(xyz: Iterable[float], rpy: Iterable[float]) -> "Pose":
        roll, pitch, yaw = list(rpy)
        return Pose(position=vec3(xyz), orientation=Quaternion.from_euler(roll, pitch, yaw))


def dh_transform(a: float, alpha: float, d: float, theta: float) -> Matrix:
    """Standard Denavit-Hartenberg transform."""

    ct, st = cos(theta), sin(theta)
    ca, sa = cos(alpha), sin(alpha)
    return np.asarray(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def pose_from_matrix(matrix: Matrix) -> Pose:
    position = matrix[:3, 3].astype(np.float64)
    m = matrix[:3, :3]
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return Pose(position=position, orientation=Quaternion(w, x, y, z).normalized())
