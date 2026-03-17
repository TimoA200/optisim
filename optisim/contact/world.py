"""Small contact world for narrow-phase contact evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from optisim.contact.forces import ContactForceModel, ContactParams
from optisim.contact.geometry import ContactPair, ContactPoint, aabb_aabb_contact, box_sphere_contact, sphere_sphere_contact


def _vec3(value: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("expected a 3D vector")
    return array


@dataclass(slots=True)
class _Body:
    name: str
    shape: str
    params: dict[str, Any]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


class ContactWorld:
    """Track simple rigid bodies and evaluate contact forces between them."""

    def __init__(self, contact_params: ContactParams | None = None) -> None:
        self._bodies: dict[str, _Body] = {}
        self._contact_params = contact_params or ContactParams()
        self._force_model = ContactForceModel(self._contact_params)

    @property
    def bodies(self) -> dict[str, dict[str, Any]]:
        """Return body state keyed by body name."""

        return {
            name: {
                "name": body.name,
                "shape": body.shape,
                "params": body.params,
                "velocity": body.velocity,
            }
            for name, body in self._bodies.items()
        }

    @property
    def contact_params(self) -> ContactParams:
        """Return active contact parameters."""

        return self._contact_params

    @contact_params.setter
    def contact_params(self, value: ContactParams) -> None:
        self._contact_params = value
        self._force_model.params = value

    def add_body(self, name: str, shape: str, params: dict) -> None:
        """Add a body to the contact world."""

        if shape not in {"sphere", "box"}:
            raise ValueError("shape must be 'sphere' or 'box'")
        payload = {key: (_vec3(value) if key in {"center", "half_extents"} else float(value)) for key, value in params.items()}
        if "center" not in payload:
            raise ValueError("body params must include center")
        if shape == "sphere" and "radius" not in payload:
            raise ValueError("sphere params must include radius")
        if shape == "box" and "half_extents" not in payload:
            raise ValueError("box params must include half_extents")
        self._bodies[name] = _Body(name=name, shape=shape, params=payload)

    def update_body(self, name: str, **kwargs: Any) -> None:
        """Update the stored center, velocity, or shape parameters for a body."""

        body = self._bodies[name]
        for key, value in kwargs.items():
            if key == "velocity":
                body.velocity = _vec3(value)
            elif key in {"center", "half_extents"}:
                body.params[key] = _vec3(value)
            elif key == "radius":
                body.params[key] = float(value)
            else:
                body.params[key] = value

    def detect_contacts(self) -> list[ContactPair]:
        """Run narrow-phase contact checks for all body pairs."""

        contacts: list[ContactPair] = []
        for name_a, name_b in combinations(self._bodies, 2):
            body_a = self._bodies[name_a]
            body_b = self._bodies[name_b]
            pair = self._detect_pair(body_a, body_b)
            if pair is not None:
                contacts.append(pair)
        return contacts

    def apply_forces(self, dt: float = 0.01) -> dict[str, np.ndarray]:
        """Return net contact force on each body for the current configuration."""

        _ = float(dt)
        net_forces = {name: np.zeros(3, dtype=np.float64) for name in self._bodies}
        for pair in self.detect_contacts():
            velocity_a = self._bodies[pair.body_a].velocity
            velocity_b = self._bodies[pair.body_b].velocity
            result = self._force_model.apply(pair, velocity_a=velocity_a, velocity_b=velocity_b)
            net_forces[pair.body_a] += result["force_on_a"]
            net_forces[pair.body_b] += result["force_on_b"]
        return net_forces

    def step(self, dt: float = 0.01) -> dict[str, Any]:
        """Detect contacts and return a summary of forces over one step."""

        contacts = self.detect_contacts()
        net_forces = self.apply_forces(dt=dt)
        return {
            "dt": float(dt),
            "contacts": contacts,
            "net_forces": net_forces,
            "num_pairs": len(contacts),
            "num_contacts": sum(len(pair.contacts) for pair in contacts),
        }

    def _detect_pair(self, body_a: _Body, body_b: _Body) -> ContactPair | None:
        if body_a.shape == "sphere" and body_b.shape == "sphere":
            return sphere_sphere_contact(
                body_a.params["center"],
                body_a.params["radius"],
                body_b.params["center"],
                body_b.params["radius"],
                name_a=body_a.name,
                name_b=body_b.name,
            )
        if body_a.shape == "box" and body_b.shape == "sphere":
            return box_sphere_contact(
                body_a.params["center"],
                body_a.params["half_extents"],
                body_b.params["center"],
                body_b.params["radius"],
                name_box=body_a.name,
                name_sphere=body_b.name,
            )
        if body_a.shape == "sphere" and body_b.shape == "box":
            pair = box_sphere_contact(
                body_b.params["center"],
                body_b.params["half_extents"],
                body_a.params["center"],
                body_a.params["radius"],
                name_box=body_b.name,
                name_sphere=body_a.name,
            )
            if pair is None:
                return None
            return ContactPair(
                body_a=body_a.name,
                body_b=body_b.name,
                contacts=[
                    ContactPoint(
                        position=contact.position,
                        normal=-contact.normal,
                        depth=contact.depth,
                        body_a=body_a.name,
                        body_b=body_b.name,
                    )
                    for contact in pair.contacts
                ],
            )
        return aabb_aabb_contact(
            body_a.params["center"],
            body_a.params["half_extents"],
            body_b.params["center"],
            body_b.params["half_extents"],
            name_a=body_a.name,
            name_b=body_b.name,
        )


__all__ = ["ContactWorld"]
