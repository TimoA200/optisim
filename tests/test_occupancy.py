from __future__ import annotations

import numpy as np
import pytest

import optisim
import optisim.occupancy as occupancy
from optisim import CollisionChecker as PublicCollisionChecker
from optisim import OccupancyStats as PublicOccupancyStats
from optisim import OccupancyUpdater as PublicOccupancyUpdater
from optisim import VoxelGrid as PublicVoxelGrid
from optisim.occupancy import CollisionChecker, OccupancyStats, OccupancyUpdater, VoxelGrid


def _grid() -> VoxelGrid:
    return VoxelGrid.from_shape(resolution=1.0, origin=(0.0, 0.0, 0.0), shape=(5, 5, 5))


def test_voxelgrid_construction_and_fields() -> None:
    grid = VoxelGrid(resolution=0.5, origin=(1.0, 2.0, 3.0), shape=(2, 3, 4), data=np.zeros((2, 3, 4), dtype=np.uint8))

    assert grid.resolution == pytest.approx(0.5)
    assert grid.origin == (1.0, 2.0, 3.0)
    assert grid.shape == (2, 3, 4)
    assert grid.data.dtype == np.uint8


def test_voxelgrid_rejects_non_positive_resolution() -> None:
    with pytest.raises(ValueError, match="resolution"):
        VoxelGrid(resolution=0.0, origin=(0.0, 0.0, 0.0), shape=(1, 1, 1), data=np.zeros((1, 1, 1), dtype=np.uint8))


def test_voxelgrid_rejects_invalid_shape_tuple() -> None:
    with pytest.raises(ValueError, match="shape"):
        VoxelGrid.from_shape(resolution=1.0, origin=(0.0, 0.0, 0.0), shape=(1, 0, 1))


def test_voxelgrid_rejects_invalid_data_shape() -> None:
    with pytest.raises(ValueError, match="data must have shape"):
        VoxelGrid(resolution=1.0, origin=(0.0, 0.0, 0.0), shape=(2, 2, 2), data=np.zeros((2, 2, 1), dtype=np.uint8))


def test_from_shape_builds_zero_initialized_uint8_grid() -> None:
    grid = _grid()

    assert grid.data.shape == (5, 5, 5)
    assert grid.data.dtype == np.uint8
    assert np.count_nonzero(grid.data) == 0


def test_world_to_voxel_maps_world_coordinate_to_index() -> None:
    grid = VoxelGrid.from_shape(resolution=0.5, origin=(1.0, 2.0, 3.0), shape=(4, 4, 4))

    assert grid.world_to_voxel(1.26, 2.74, 4.0) == (0, 1, 2)


def test_world_to_voxel_uses_floor_for_negative_relative_coordinate() -> None:
    grid = _grid()

    assert grid.world_to_voxel(-0.1, 0.0, 0.0) == (-1, 0, 0)


def test_voxel_to_world_returns_voxel_center() -> None:
    grid = VoxelGrid.from_shape(resolution=0.5, origin=(1.0, 2.0, 3.0), shape=(4, 4, 4))

    assert grid.voxel_to_world(2, 0, 1) == pytest.approx((2.25, 2.25, 3.75))


def test_set_occupied_marks_cell() -> None:
    grid = _grid()
    grid.set_occupied(1.1, 1.1, 1.1)

    assert grid.data[1, 1, 1] == 255


def test_set_free_marks_cell_as_zero() -> None:
    grid = _grid()
    grid.data[1, 1, 1] = np.uint8(255)
    grid.set_free(1.1, 1.1, 1.1)

    assert grid.data[1, 1, 1] == 0


def test_is_occupied_returns_true_only_for_occupied_voxel() -> None:
    grid = _grid()
    grid.data[2, 2, 2] = np.uint8(255)

    assert grid.is_occupied(2.2, 2.2, 2.2) is True


def test_is_free_returns_true_for_zero_voxel() -> None:
    assert _grid().is_free(0.1, 0.1, 0.1) is True


def test_is_free_returns_false_for_occupied_voxel() -> None:
    grid = _grid()
    grid.data[0, 0, 0] = np.uint8(255)

    assert grid.is_free(0.1, 0.1, 0.1) is False


def test_count_occupied_counts_only_255_values() -> None:
    grid = _grid()
    grid.data[0, 0, 0] = np.uint8(255)
    grid.data[1, 1, 1] = np.uint8(128)
    grid.data[2, 2, 2] = np.uint8(255)

    assert grid.count_occupied() == 2


def test_count_free_counts_only_zero_values() -> None:
    grid = _grid()
    grid.data[0, 0, 0] = np.uint8(255)
    grid.data[1, 1, 1] = np.uint8(128)

    assert grid.count_free() == 123


def test_to_point_cloud_returns_empty_array_for_no_occupied_voxels() -> None:
    points = _grid().to_point_cloud()

    assert points.shape == (0, 3)


def test_to_point_cloud_returns_centers_of_occupied_voxels() -> None:
    grid = _grid()
    grid.data[1, 2, 3] = np.uint8(255)
    grid.data[0, 0, 0] = np.uint8(255)

    points = grid.to_point_cloud()

    assert points.shape == (2, 3)
    assert {tuple(point) for point in points} == {(0.5, 0.5, 0.5), (1.5, 2.5, 3.5)}


def test_mark_occupied_marks_each_input_point() -> None:
    grid = _grid()
    updater = OccupancyUpdater(grid)
    updater.mark_occupied(np.array([[0.1, 0.1, 0.1], [2.2, 2.2, 2.2]]))

    assert grid.count_occupied() == 2


def test_mark_occupied_rejects_invalid_point_shape() -> None:
    with pytest.raises(ValueError, match="points"):
        OccupancyUpdater(_grid()).mark_occupied(np.zeros((2, 2)))


def test_mark_free_ray_clears_intermediate_voxels() -> None:
    grid = _grid()
    grid.data[0:4, 0, 0] = np.uint8(255)

    OccupancyUpdater(grid).mark_free_ray((0.1, 0.1, 0.1), (3.9, 0.1, 0.1), steps=4)

    assert np.all(grid.data[0:3, 0, 0] == 0)
    assert grid.data[3, 0, 0] == 255


def test_mark_free_ray_does_not_clear_endpoint_voxel() -> None:
    grid = _grid()
    grid.data[4, 0, 0] = np.uint8(255)

    OccupancyUpdater(grid).mark_free_ray((0.1, 0.1, 0.1), (4.9, 0.1, 0.1), steps=5)

    assert grid.data[4, 0, 0] == 255


def test_mark_free_ray_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        OccupancyUpdater(_grid()).mark_free_ray((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), steps=0)


def test_clear_region_clears_occupied_voxels_inside_sphere() -> None:
    grid = _grid()
    grid.data[2, 2, 2] = np.uint8(255)
    grid.data[4, 4, 4] = np.uint8(255)

    OccupancyUpdater(grid).clear_region((2.5, 2.5, 2.5), radius=1.0)

    assert grid.data[2, 2, 2] == 0
    assert grid.data[4, 4, 4] == 255


def test_check_sphere_returns_true_when_occupied_voxel_is_within_radius() -> None:
    grid = _grid()
    grid.data[2, 2, 2] = np.uint8(255)

    assert CollisionChecker(grid).check_sphere((2.5, 2.5, 2.5), radius=0.2) is True


def test_check_sphere_returns_false_when_region_is_free() -> None:
    assert CollisionChecker(_grid()).check_sphere((2.5, 2.5, 2.5), radius=0.5) is False


def test_check_capsule_returns_true_when_sampled_segment_intersects_occupied_voxel() -> None:
    grid = _grid()
    grid.data[2, 0, 0] = np.uint8(255)

    assert CollisionChecker(grid).check_capsule((0.5, 0.5, 0.5), (4.5, 0.5, 0.5), radius=0.6) is True


def test_check_capsule_returns_false_when_segment_is_clear() -> None:
    assert CollisionChecker(_grid()).check_capsule((0.5, 0.5, 0.5), (4.5, 0.5, 0.5), radius=0.4) is False


def test_check_capsule_rejects_negative_radius() -> None:
    with pytest.raises(ValueError, match="radius"):
        CollisionChecker(_grid()).check_capsule((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), radius=-1.0)


def test_check_path_returns_true_if_any_waypoint_collides() -> None:
    grid = _grid()
    grid.data[1, 1, 1] = np.uint8(255)
    waypoints = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [3.5, 3.5, 3.5]])

    assert CollisionChecker(grid).check_path(waypoints, radius=0.4) is True


def test_check_path_returns_false_if_all_waypoints_are_free() -> None:
    waypoints = np.array([[0.5, 0.5, 0.5], [3.5, 3.5, 3.5]])

    assert CollisionChecker(_grid()).check_path(waypoints, radius=0.4) is False


def test_check_path_rejects_invalid_waypoint_shape() -> None:
    with pytest.raises(ValueError, match="waypoints"):
        CollisionChecker(_grid()).check_path(np.zeros((3, 2)), radius=0.1)


def test_occupancy_ratio_returns_fraction_of_occupied_voxels() -> None:
    grid = _grid()
    grid.data[0, 0, 0] = np.uint8(255)
    grid.data[1, 1, 1] = np.uint8(255)

    assert OccupancyStats(grid).occupancy_ratio() == pytest.approx(2.0 / 125.0)


def test_bounding_box_occupied_returns_none_when_grid_has_no_occupied_voxels() -> None:
    assert OccupancyStats(_grid()).bounding_box_occupied() is None


def test_bounding_box_occupied_returns_min_and_max_world_centers() -> None:
    grid = _grid()
    grid.data[1, 2, 0] = np.uint8(255)
    grid.data[3, 4, 2] = np.uint8(255)

    bbox = OccupancyStats(grid).bounding_box_occupied()

    assert bbox == ((1.5, 2.5, 0.5), (3.5, 4.5, 2.5))


def test_density_by_layer_default_axis_uses_z_slices() -> None:
    grid = _grid()
    grid.data[0, 0, 0] = np.uint8(255)
    grid.data[1, 1, 0] = np.uint8(255)
    grid.data[2, 2, 1] = np.uint8(255)

    density = OccupancyStats(grid).density_by_layer()

    assert density.shape == (5,)
    assert density[0] == pytest.approx(2.0 / 25.0)
    assert density[1] == pytest.approx(1.0 / 25.0)


def test_density_by_layer_supports_other_axis() -> None:
    grid = _grid()
    grid.data[0, 0, 0] = np.uint8(255)
    grid.data[0, 1, 1] = np.uint8(255)
    grid.data[1, 0, 0] = np.uint8(255)

    density = OccupancyStats(grid).density_by_layer(axis=0)

    assert density.shape == (5,)
    assert density[0] == pytest.approx(2.0 / 25.0)
    assert density[1] == pytest.approx(1.0 / 25.0)


def test_density_by_layer_rejects_invalid_axis() -> None:
    with pytest.raises(ValueError, match="axis"):
        OccupancyStats(_grid()).density_by_layer(axis=3)


def test_public_exports_match_module_types() -> None:
    assert PublicVoxelGrid is VoxelGrid
    assert PublicOccupancyUpdater is OccupancyUpdater
    assert PublicCollisionChecker is CollisionChecker
    assert PublicOccupancyStats is OccupancyStats
    assert occupancy.VoxelGrid is VoxelGrid


def test_optisim_exports_occupancy_module() -> None:
    assert optisim.occupancy is occupancy


def test_package_version_matches_new_release() -> None:
    assert optisim.__version__ == "0.24.0"
