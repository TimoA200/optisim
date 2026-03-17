from __future__ import annotations

import numpy as np

from optisim import ContactForceModel, ContactParams, ContactWorld, contact
from optisim.contact import (
    ContactPair,
    ContactPoint,
    aabb_aabb_contact,
    box_sphere_contact,
    compute_friction_force,
    compute_normal_force,
    sphere_sphere_contact,
)


def test_contact_module_exported_from_root() -> None:
    assert hasattr(contact, "ContactWorld")


def test_sphere_sphere_contact_overlapping_returns_pair() -> None:
    pair = sphere_sphere_contact([0.0, 0.0, 0.0], 1.0, [1.5, 0.0, 0.0], 1.0)
    assert isinstance(pair, ContactPair)


def test_sphere_sphere_contact_separated_returns_none() -> None:
    assert sphere_sphere_contact([0.0, 0.0, 0.0], 0.5, [2.0, 0.0, 0.0], 0.5) is None


def test_sphere_sphere_contact_has_one_contact_point() -> None:
    pair = sphere_sphere_contact([0.0, 0.0, 0.0], 1.0, [1.8, 0.0, 0.0], 1.0)
    assert pair is not None
    assert len(pair.contacts) == 1


def test_contact_point_normal_is_unit_vector() -> None:
    pair = sphere_sphere_contact([0.0, 0.0, 0.0], 1.0, [1.5, 0.0, 0.0], 1.0)
    assert pair is not None
    assert np.isclose(np.linalg.norm(pair.contacts[0].normal), 1.0)


def test_contact_point_depth_positive_on_overlap() -> None:
    pair = sphere_sphere_contact([0.0, 0.0, 0.0], 1.0, [1.5, 0.0, 0.0], 1.0)
    assert pair is not None
    assert pair.contacts[0].depth > 0.0


def test_contact_point_clamps_negative_depth_to_zero() -> None:
    point = ContactPoint(position=np.zeros(3), normal=np.array([2.0, 0.0, 0.0]), depth=-1.0, body_a="a", body_b="b")
    assert point.depth == 0.0


def test_contact_pair_preserves_body_names() -> None:
    pair = sphere_sphere_contact([0.0, 0.0, 0.0], 1.0, [1.5, 0.0, 0.0], 1.0, name_a="left", name_b="right")
    assert pair is not None
    assert (pair.body_a, pair.body_b) == ("left", "right")


def test_box_sphere_contact_inside_box_region_returns_contact() -> None:
    pair = box_sphere_contact([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.0, 0.0], 0.3)
    assert isinstance(pair, ContactPair)


def test_box_sphere_contact_outside_returns_none() -> None:
    assert box_sphere_contact([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0], 0.2) is None


def test_box_sphere_contact_normal_is_unit_vector() -> None:
    pair = box_sphere_contact([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.1, 0.0, 0.0], 0.2)
    assert pair is not None
    assert np.isclose(np.linalg.norm(pair.contacts[0].normal), 1.0)


def test_box_sphere_contact_depth_nonnegative() -> None:
    pair = box_sphere_contact([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.1, 0.0, 0.0], 0.2)
    assert pair is not None
    assert pair.contacts[0].depth >= 0.0


def test_aabb_aabb_contact_overlapping_boxes_returns_pair() -> None:
    pair = aabb_aabb_contact([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.5, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert isinstance(pair, ContactPair)


def test_aabb_aabb_contact_non_overlapping_returns_none() -> None:
    assert aabb_aabb_contact([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [2.0, 0.0, 0.0], [0.5, 0.5, 0.5]) is None


def test_aabb_aabb_partial_overlap_on_one_axis_returns_pair() -> None:
    pair = aabb_aabb_contact([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.9, 0.1, 0.0], [1.0, 0.5, 0.5])
    assert isinstance(pair, ContactPair)


def test_aabb_aabb_contact_normal_is_axis_aligned() -> None:
    pair = aabb_aabb_contact([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.5, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert pair is not None
    assert np.count_nonzero(pair.contacts[0].normal) == 1


def test_compute_normal_force_positive_for_positive_depth() -> None:
    force = compute_normal_force(0.1, 0.0, ContactParams())
    assert force > 0.0


def test_compute_normal_force_zero_for_zero_depth() -> None:
    assert compute_normal_force(0.0, 1.0, ContactParams()) == 0.0


def test_compute_normal_force_increases_with_positive_depth_dot() -> None:
    params = ContactParams(stiffness=100.0, damping=10.0)
    assert compute_normal_force(0.1, 0.5, params) > compute_normal_force(0.1, 0.0, params)


def test_compute_friction_force_shape_is_three() -> None:
    friction = compute_friction_force(10.0, np.array([1.0, 2.0, 0.0]), ContactParams())
    assert friction.shape == (3,)


def test_compute_friction_force_zero_for_zero_velocity() -> None:
    np.testing.assert_allclose(compute_friction_force(10.0, np.zeros(3), ContactParams()), np.zeros(3))


def test_friction_magnitude_respects_coulomb_limit() -> None:
    params = ContactParams(friction_coeff=0.7)
    friction = compute_friction_force(20.0, np.array([3.0, 4.0, 0.0]), params)
    assert np.linalg.norm(friction) <= params.friction_coeff * 20.0 + 1e-9


def test_contact_force_model_apply_returns_expected_keys() -> None:
    pair = sphere_sphere_contact([0.0, 0.0, 0.0], 1.0, [1.8, 0.0, 0.0], 1.0)
    assert pair is not None
    result = ContactForceModel().apply(pair, np.zeros(3), np.zeros(3))
    assert set(result) == {"force_on_a", "force_on_b", "normal_force", "friction_force"}


def test_contact_force_model_force_vectors_have_shape_three() -> None:
    pair = sphere_sphere_contact([0.0, 0.0, 0.0], 1.0, [1.8, 0.0, 0.0], 1.0)
    assert pair is not None
    result = ContactForceModel().apply(pair, np.zeros(3), np.zeros(3))
    assert result["force_on_a"].shape == (3,)


def test_contact_force_model_forces_are_equal_and_opposite() -> None:
    pair = sphere_sphere_contact([0.0, 0.0, 0.0], 1.0, [1.8, 0.0, 0.0], 1.0)
    assert pair is not None
    result = ContactForceModel().apply(pair, np.array([0.1, 0.0, 0.0]), np.zeros(3))
    np.testing.assert_allclose(result["force_on_a"], -result["force_on_b"])


def test_contact_force_model_friction_force_zero_without_tangential_velocity() -> None:
    pair = sphere_sphere_contact([0.0, 0.0, 0.0], 1.0, [1.8, 0.0, 0.0], 1.0)
    assert pair is not None
    result = ContactForceModel().apply(pair, np.array([0.1, 0.0, 0.0]), np.zeros(3))
    np.testing.assert_allclose(result["friction_force"], np.zeros(3))


def test_contact_params_defaults() -> None:
    params = ContactParams()
    assert params.stiffness == 1000.0
    assert params.damping == 10.0
    assert params.friction_coeff == 0.5
    assert params.restitution == 0.2


def test_contact_world_add_body_then_detect_contacts() -> None:
    world = ContactWorld()
    world.add_body("a", "sphere", {"center": [0.0, 0.0, 0.0], "radius": 1.0})
    world.add_body("b", "sphere", {"center": [1.5, 0.0, 0.0], "radius": 1.0})
    assert len(world.detect_contacts()) == 1


def test_contact_world_two_overlapping_spheres_returns_one_contact_pair() -> None:
    world = ContactWorld()
    world.add_body("a", "sphere", {"center": [0.0, 0.0, 0.0], "radius": 1.0})
    world.add_body("b", "sphere", {"center": [1.9, 0.0, 0.0], "radius": 1.0})
    assert len(world.detect_contacts()) == 1


def test_contact_world_separated_spheres_returns_zero_contact_pairs() -> None:
    world = ContactWorld()
    world.add_body("a", "sphere", {"center": [0.0, 0.0, 0.0], "radius": 0.5})
    world.add_body("b", "sphere", {"center": [2.0, 0.0, 0.0], "radius": 0.5})
    assert len(world.detect_contacts()) == 0


def test_contact_world_step_returns_dict() -> None:
    world = ContactWorld()
    world.add_body("a", "sphere", {"center": [0.0, 0.0, 0.0], "radius": 1.0})
    world.add_body("b", "sphere", {"center": [1.5, 0.0, 0.0], "radius": 1.0})
    assert isinstance(world.step(), dict)


def test_contact_world_apply_forces_returns_dict_keyed_by_body_name() -> None:
    world = ContactWorld()
    world.add_body("a", "sphere", {"center": [0.0, 0.0, 0.0], "radius": 1.0})
    world.add_body("b", "sphere", {"center": [1.5, 0.0, 0.0], "radius": 1.0})
    forces = world.apply_forces()
    assert set(forces) == {"a", "b"}


def test_contact_world_multiple_bodies_supported() -> None:
    world = ContactWorld()
    world.add_body("a", "sphere", {"center": [0.0, 0.0, 0.0], "radius": 1.0})
    world.add_body("b", "sphere", {"center": [1.5, 0.0, 0.0], "radius": 1.0})
    world.add_body("c", "box", {"center": [5.0, 0.0, 0.0], "half_extents": [0.5, 0.5, 0.5]})
    assert len(world.bodies) == 3


def test_update_body_changes_center() -> None:
    world = ContactWorld()
    world.add_body("a", "sphere", {"center": [0.0, 0.0, 0.0], "radius": 1.0})
    world.update_body("a", center=[1.0, 2.0, 3.0])
    np.testing.assert_allclose(world.bodies["a"]["params"]["center"], np.array([1.0, 2.0, 3.0]))


def test_contact_world_contact_params_property_settable() -> None:
    world = ContactWorld()
    params = ContactParams(stiffness=250.0)
    world.contact_params = params
    assert world.contact_params.stiffness == 250.0


def test_contact_world_detects_box_sphere_contacts() -> None:
    world = ContactWorld()
    world.add_body("box", "box", {"center": [0.0, 0.0, 0.0], "half_extents": [1.0, 1.0, 1.0]})
    world.add_body("ball", "sphere", {"center": [1.1, 0.0, 0.0], "radius": 0.2})
    assert len(world.detect_contacts()) == 1


def test_contact_world_detects_box_box_contacts() -> None:
    world = ContactWorld()
    world.add_body("box_a", "box", {"center": [0.0, 0.0, 0.0], "half_extents": [1.0, 1.0, 1.0]})
    world.add_body("box_b", "box", {"center": [1.5, 0.0, 0.0], "half_extents": [1.0, 1.0, 1.0]})
    assert len(world.detect_contacts()) == 1


def test_contact_world_step_includes_contact_counts() -> None:
    world = ContactWorld()
    world.add_body("a", "sphere", {"center": [0.0, 0.0, 0.0], "radius": 1.0})
    world.add_body("b", "sphere", {"center": [1.5, 0.0, 0.0], "radius": 1.0})
    summary = world.step()
    assert {"contacts", "net_forces", "num_pairs", "num_contacts"} <= set(summary)
