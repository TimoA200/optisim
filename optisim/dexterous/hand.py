"""Dexterous hand kinematics primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FingerJoint:
    """Single revolute finger joint."""

    name: str
    angle: float
    min_angle: float = -0.1
    max_angle: float = 1.5
    link_length: float = 0.04

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.angle = float(np.clip(self.angle, self.min_angle, self.max_angle))
        self.min_angle = float(self.min_angle)
        self.max_angle = float(self.max_angle)
        self.link_length = float(max(self.link_length, 0.0))


class Finger:
    """Planar finger chain embedded in 3D space."""

    def __init__(
        self,
        name: str,
        n_joints: int = 3,
        base_position: np.ndarray | None = None,
    ) -> None:
        self.name = str(name)
        self.base_position = np.asarray(
            base_position if base_position is not None else np.zeros(3, dtype=np.float64),
            dtype=np.float64,
        )
        if self.base_position.shape != (3,):
            raise ValueError("base_position must be a 3D vector")
        self.joints = [
            FingerJoint(name=f"{self.name}_joint_{index + 1}", angle=0.0)
            for index in range(n_joints)
        ]

    joints: list[FingerJoint]

    def forward_kinematics(self) -> list[np.ndarray]:
        """Return the base plus each successive joint-tip position."""

        positions = [self.base_position.copy()]
        angle_sum = 0.0
        current = self.base_position.astype(np.float64, copy=True)
        for joint in self.joints:
            angle_sum += joint.angle
            delta = np.asarray(
                [
                    joint.link_length * np.cos(angle_sum),
                    0.0,
                    joint.link_length * np.sin(angle_sum),
                ],
                dtype=np.float64,
            )
            current = current + delta
            positions.append(current.copy())
        return positions

    def fingertip_position(self) -> np.ndarray:
        """Return the final link tip position."""

        return self.forward_kinematics()[-1]

    def set_angles(self, angles: list[float]) -> None:
        """Set joint angles, clamping to each joint's limits."""

        for joint, angle in zip(self.joints, angles, strict=False):
            joint.angle = float(np.clip(angle, joint.min_angle, joint.max_angle))

    def get_angles(self) -> list[float]:
        """Return current joint angles."""

        return [joint.angle for joint in self.joints]


class Hand:
    """Simple multi-finger hand model."""

    def __init__(self, name: str = "right", n_fingers: int = 5) -> None:
        self.name = str(name)
        self.fingers = self._build_fingers(int(n_fingers))

    fingers: list[Finger]

    @staticmethod
    def _build_fingers(n_fingers: int) -> list[Finger]:
        if n_fingers <= 0:
            return []
        base_positions = [
            np.asarray([0.015, -0.035, 0.0], dtype=np.float64),
            np.asarray([0.0, -0.015, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.017, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.034, 0.0], dtype=np.float64),
        ]
        names = ["thumb", "index", "middle", "ring", "little"]
        fingers: list[Finger] = []
        for index in range(n_fingers):
            fingers.append(
                Finger(
                    name=names[index % len(names)],
                    n_joints=3,
                    base_position=base_positions[index % len(base_positions)].copy(),
                )
            )
        return fingers

    def fingertip_positions(self) -> np.ndarray:
        """Return all fingertip positions as an ``(n_fingers, 3)`` array."""

        if not self.fingers:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray(
            [finger.fingertip_position() for finger in self.fingers],
            dtype=np.float64,
        )

    def set_all_angles(self, angles: list[list[float]]) -> None:
        """Set all finger joint angles."""

        for finger, finger_angles in zip(self.fingers, angles, strict=False):
            finger.set_angles(finger_angles)

    def open(self) -> None:
        """Open all fingers."""

        for finger in self.fingers:
            finger.set_angles([0.0] * len(finger.joints))

    def close(self, fraction: float = 1.0) -> None:
        """Close each joint to a fraction of its maximum flexion."""

        close_fraction = float(np.clip(fraction, 0.0, 1.0))
        for finger in self.fingers:
            target = [joint.max_angle * close_fraction for joint in finger.joints]
            finger.set_angles(target)

    @classmethod
    def open_hand(cls) -> Hand:
        hand = cls()
        hand.open()
        return hand

    @classmethod
    def closed_fist(cls) -> Hand:
        hand = cls()
        hand.close(1.0)
        return hand

    @classmethod
    def pinch_grip(cls) -> Hand:
        hand = cls()
        hand.open()
        if len(hand.fingers) >= 2:
            hand.fingers[0].set_angles([0.8, 1.0, 0.7])
            hand.fingers[1].set_angles([0.6, 0.9, 0.6])
        return hand

    @classmethod
    def power_grip(cls) -> Hand:
        hand = cls()
        hand.close(0.85)
        if hand.fingers:
            hand.fingers[0].set_angles([0.9, 1.1, 0.8])
        return hand


Hand.PRESETS = type(
    "PRESETS",
    (),
    {
        "open_hand": staticmethod(Hand.open_hand),
        "closed_fist": staticmethod(Hand.closed_fist),
        "pinch_grip": staticmethod(Hand.pinch_grip),
        "power_grip": staticmethod(Hand.power_grip),
    },
)

__all__ = ["FingerJoint", "Finger", "Hand"]
