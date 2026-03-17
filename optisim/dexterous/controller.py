"""Dexterous hand control helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from optisim.contact import ContactPoint
from optisim.grasp import GraspPose, force_closure
from optisim.math3d import Quaternion

from optisim.dexterous.hand import Hand
from optisim.dexterous.tactile import TactileSensor


class FingerControlMode(Enum):
    POSITION = "position"
    TORQUE = "torque"
    IMPEDANCE = "impedance"


@dataclass(slots=True)
class FingerCommand:
    angles: list[float] | None = None
    torques: list[float] | None = None
    stiffness: float = 50.0
    damping: float = 5.0


class DexterousController:
    """Simple command/step controller for a multi-finger hand."""

    def __init__(self, hand: Hand, mode: FingerControlMode = FingerControlMode.POSITION) -> None:
        self.hand = hand
        self.mode = mode
        self.tactile_sensors = {finger.name: TactileSensor() for finger in hand.fingers}
        self._commands = {
            finger.name: FingerCommand(angles=finger.get_angles()) for finger in hand.fingers
        }

    tactile_sensors: dict[str, TactileSensor]

    def command(self, commands: dict[str, FingerCommand]) -> None:
        """Store per-finger commands."""

        for finger_name, finger_command in commands.items():
            if finger_name in self._commands:
                self._commands[finger_name] = finger_command

    def step(self, dt: float = 0.01) -> dict:
        """Advance the hand state one control step."""

        delta_t = max(float(dt), 1e-6)
        for finger in self.hand.fingers:
            command = self._commands.get(finger.name, FingerCommand())
            current = np.asarray(finger.get_angles(), dtype=np.float64)
            if self.mode is FingerControlMode.POSITION and command.angles is not None:
                finger.set_angles(list(command.angles))
            elif self.mode is FingerControlMode.TORQUE:
                torques = np.asarray(
                    command.torques or np.zeros(len(finger.joints)),
                    dtype=np.float64,
                )
                finger.set_angles(list(current + torques * delta_t))
            else:
                target = np.asarray(command.angles or current, dtype=np.float64)
                error = target - current
                delta = (command.stiffness * error - command.damping * current) * delta_t
                finger.set_angles(list(current + delta))
        return {
            "finger_positions": {
                finger.name: finger.forward_kinematics() for finger in self.hand.fingers
            },
            "fingertip_positions": self.hand.fingertip_positions(),
        }

    def grasp_object(self, object_center: np.ndarray, object_radius: float = 0.03) -> bool:
        """Close fingers until tactile contact or joint limits are reached."""

        center = np.asarray(object_center, dtype=np.float64)
        radius = float(max(object_radius, 1e-6))
        latest_contacts: dict[str, ContactPoint] = {}
        for _ in range(80):
            all_contact = True
            any_progress = False
            for finger in self.hand.fingers:
                tip = finger.fingertip_position()
                offset = center - tip
                distance = float(np.linalg.norm(offset))
                sensor = self.tactile_sensors[finger.name]
                if distance <= radius:
                    normal = offset / max(distance, 1e-9)
                    contact_position = center - normal * radius
                    depth = max(radius - distance, 0.0)
                    contact = ContactPoint(
                        position=contact_position,
                        normal=normal,
                        depth=depth,
                        body_a=finger.name,
                        body_b="object",
                    )
                    latest_contacts[finger.name] = contact
                    local_contact = ContactPoint(
                        position=contact_position - tip,
                        normal=normal,
                        depth=depth,
                        body_a=finger.name,
                        body_b="object",
                    )
                    sensor.update(local_contact, contact_force=max(depth * 200.0, 0.1))
                    continue

                all_contact = False
                sensor.update(None)
                next_angles = [min(joint.max_angle, joint.angle + 0.05) for joint in finger.joints]
                any_progress = any_progress or any(
                    next_angle > joint.angle + 1e-12
                    for joint, next_angle in zip(finger.joints, next_angles, strict=True)
                )
                finger.set_angles(next_angles)
            if all_contact:
                break
            if not any_progress:
                break

        if len(latest_contacts) != len(self.hand.fingers):
            return False

        grasp_contacts = [
            self._to_grasp_contact(contact, center) for contact in latest_contacts.values()
        ]
        _ = GraspPose(
            position=center.copy(),
            orientation=Quaternion.identity(),
            aperture=2.0 * radius,
            contact_points=grasp_contacts,
            quality_score=float(force_closure(grasp_contacts)),
            gripper_type="dexterous_hand",
        )
        return True

    def release(self) -> None:
        """Open the hand and clear tactile state."""

        self.hand.open()
        for sensor in self.tactile_sensors.values():
            sensor.update(None)

    @staticmethod
    def _to_grasp_contact(contact: ContactPoint, object_center: np.ndarray):
        from optisim.grasp.contact import ContactPoint as GraspContactPoint

        direction = np.asarray(contact.position, dtype=np.float64) - np.asarray(
            object_center, dtype=np.float64
        )
        norm = float(np.linalg.norm(direction))
        normal = (
            direction / norm
            if norm > 1e-12
            else np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        )
        return GraspContactPoint(position=contact.position, normal=normal, friction_coeff=0.8)


__all__ = ["FingerControlMode", "FingerCommand", "DexterousController"]
