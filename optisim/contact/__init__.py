"""Contact mechanics interfaces."""

from optisim.contact.forces import ContactForceModel, ContactParams, compute_friction_force, compute_normal_force
from optisim.contact.geometry import ContactPair, ContactPoint, aabb_aabb_contact, box_sphere_contact, sphere_sphere_contact
from optisim.contact.world import ContactWorld

__all__ = [
    "ContactPoint",
    "ContactPair",
    "sphere_sphere_contact",
    "box_sphere_contact",
    "aabb_aabb_contact",
    "ContactParams",
    "compute_normal_force",
    "compute_friction_force",
    "ContactForceModel",
    "ContactWorld",
]
