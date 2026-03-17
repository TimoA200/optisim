"""Gripper models used by the grasp planning module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GripperType(StrEnum):
    """Supported built-in gripper archetypes."""

    PARALLEL_JAW = "parallel_jaw"
    SUCTION = "suction"
    THREE_FINGER = "three_finger"


@dataclass(frozen=True, slots=True)
class Gripper:
    """Compact gripper description for planning and contact checks."""

    type: GripperType
    max_aperture: float
    max_force: float
    finger_width: float
    contact_area: float


def default_parallel_jaw() -> Gripper:
    """Return a deterministic default parallel-jaw gripper preset."""

    return Gripper(
        type=GripperType.PARALLEL_JAW,
        max_aperture=0.16,
        max_force=80.0,
        finger_width=0.012,
        contact_area=1.5e-4,
    )


def default_suction() -> Gripper:
    """Return a deterministic default suction gripper preset."""

    return Gripper(
        type=GripperType.SUCTION,
        max_aperture=0.08,
        max_force=45.0,
        finger_width=0.0,
        contact_area=2.8e-3,
    )


def default_three_finger() -> Gripper:
    """Return a deterministic default three-finger gripper preset."""

    return Gripper(
        type=GripperType.THREE_FINGER,
        max_aperture=0.16,
        max_force=90.0,
        finger_width=0.01,
        contact_area=2.0e-4,
    )
