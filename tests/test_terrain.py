from __future__ import annotations

import numpy as np
import pytest

import optisim.terrain as terrain
from optisim import HeightMap as PublicHeightMap
from optisim import TerrainAdaptiveFootstep as PublicTerrainAdaptiveFootstep
from optisim import TerrainAnalyzer as PublicTerrainAnalyzer
from optisim import TerrainCostMap as PublicTerrainCostMap
from optisim import TerrainPatch as PublicTerrainPatch
from optisim.terrain import HeightMap, TerrainAdaptiveFootstep, TerrainAnalyzer, TerrainCostMap, TerrainPatch


def _plane_heightmap(ax: float = 0.1, by: float = 0.2, width: int = 6, height: int = 5, resolution: float = 1.0) -> HeightMap:
    return HeightMap.from_heightfn(lambda x, y: ax * x + by * y, width=width, height=height, resolution=resolution)


def _step_heightmap() -> HeightMap:
    heights = np.zeros((5, 5), dtype=np.float64)
    heights[:, 3:] = 1.0
    return HeightMap(resolution=1.0, width=5, height=5, heights=heights)


def test_heightmap_construction_and_fields() -> None:
    heightmap = HeightMap(resolution=0.5, width=3, height=2, heights=np.zeros((2, 3)))

    assert heightmap.resolution == pytest.approx(0.5)
    assert heightmap.width == 3
    assert heightmap.height == 2
    assert heightmap.heights.shape == (2, 3)


def test_heightmap_rejects_non_positive_resolution() -> None:
    with pytest.raises(ValueError, match="resolution"):
        HeightMap(resolution=0.0, width=3, height=3, heights=np.zeros((3, 3)))


def test_heightmap_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="width and height"):
        HeightMap(resolution=1.0, width=0, height=3, heights=np.zeros((3, 0)))


def test_heightmap_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        HeightMap(resolution=1.0, width=3, height=3, heights=np.zeros((2, 3)))


def test_from_flat_builds_constant_heightmap() -> None:
    heightmap = HeightMap.from_flat(width=4, height=3, resolution=0.2, height_value=1.5)

    assert heightmap.heights.shape == (3, 4)
    assert np.allclose(heightmap.heights, 1.5)


def test_from_random_is_deterministic_for_seed() -> None:
    first = HeightMap.from_random(seed=7, width=6, height=6)
    second = HeightMap.from_random(seed=7, width=6, height=6)

    np.testing.assert_allclose(first.heights, second.heights)


def test_from_random_varies_for_different_seed() -> None:
    first = HeightMap.from_random(seed=7, width=6, height=6)
    second = HeightMap.from_random(seed=8, width=6, height=6)

    assert not np.allclose(first.heights, second.heights)


def test_from_heightfn_samples_function_values() -> None:
    heightmap = HeightMap.from_heightfn(lambda x, y: x + 2.0 * y, width=4, height=3, resolution=0.5)

    assert heightmap.heights[2, 3] == pytest.approx(3 * 0.5 + 2.0 * 2 * 0.5)


def test_get_height_returns_grid_value_at_integer_cell() -> None:
    heightmap = HeightMap(resolution=1.0, width=3, height=3, heights=np.arange(9, dtype=np.float64).reshape(3, 3))

    assert heightmap.get_height(2.0, 1.0) == pytest.approx(5.0)


def test_get_height_bilinear_interpolation_at_cell_center() -> None:
    heightmap = HeightMap(resolution=1.0, width=2, height=2, heights=np.array([[0.0, 1.0], [2.0, 3.0]]))

    assert heightmap.get_height(0.5, 0.5) == pytest.approx(1.5)


def test_get_height_clamps_out_of_bounds_queries() -> None:
    heightmap = HeightMap(resolution=1.0, width=3, height=3, heights=np.arange(9, dtype=np.float64).reshape(3, 3))

    assert heightmap.get_height(-10.0, 99.0) == pytest.approx(6.0)


def test_get_normal_on_flat_map_points_up() -> None:
    normal = HeightMap.from_flat(width=5, height=5, resolution=1.0).get_normal(2.0, 2.0)

    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-8)


def test_get_normal_on_plane_matches_gradient_direction() -> None:
    normal = _plane_heightmap(ax=0.2, by=0.0).get_normal(2.0, 2.0)
    expected = np.array([-0.2, 0.0, 1.0], dtype=np.float64)
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(normal, expected, atol=1e-6)


def test_get_slope_on_flat_map_is_zero() -> None:
    slope = HeightMap.from_flat(width=5, height=5).get_slope(2.0, 2.0)

    assert slope == pytest.approx(0.0, abs=1e-8)


def test_get_slope_matches_plane_gradient() -> None:
    slope = _plane_heightmap(ax=0.2, by=0.1).get_slope(2.0, 2.0)
    expected = np.degrees(np.arctan(np.sqrt(0.2**2 + 0.1**2)))

    assert slope == pytest.approx(expected, abs=1e-6)


def test_is_traversable_returns_true_for_low_slope() -> None:
    heightmap = _plane_heightmap(ax=0.1, by=0.0)

    assert heightmap.is_traversable(2.0, 2.0, max_slope_deg=15.0) is True


def test_is_traversable_returns_false_for_high_slope() -> None:
    heightmap = _plane_heightmap(ax=1.0, by=0.0)

    assert heightmap.is_traversable(2.0, 2.0, max_slope_deg=30.0) is False


def test_terrain_patch_construction() -> None:
    patch = TerrainPatch(center=(1.0, 2.0), radius=0.5, label="flat")

    assert patch.center == (1.0, 2.0)
    assert patch.radius == pytest.approx(0.5)
    assert patch.label == "flat"


def test_terrain_patch_rejects_negative_radius() -> None:
    with pytest.raises(ValueError, match="radius"):
        TerrainPatch(center=(0.0, 0.0), radius=-1.0, label="bad")


def test_terrain_analyzer_initialization() -> None:
    analyzer = TerrainAnalyzer(HeightMap.from_flat(width=4, height=4))

    assert analyzer.traversability_map.shape == (4, 4)
    assert analyzer.get_flat_regions() == []


def test_analyzer_marks_flat_map_as_traversable() -> None:
    analyzer = TerrainAnalyzer(HeightMap.from_flat(width=4, height=4)).analyze()

    assert analyzer.traversability_map.all()


def test_analyzer_finds_flat_regions_on_flat_map() -> None:
    analyzer = TerrainAnalyzer(HeightMap.from_flat(width=4, height=4)).analyze()
    regions = analyzer.get_flat_regions()

    assert len(regions) == 1
    assert regions[0].label == "flat"


def test_analyzer_flat_region_center_is_reasonable() -> None:
    analyzer = TerrainAnalyzer(HeightMap.from_flat(width=4, height=4, resolution=1.0)).analyze()
    center = analyzer.get_flat_regions()[0].center

    assert center[0] == pytest.approx(1.5)
    assert center[1] == pytest.approx(1.5)


def test_analyzer_finds_no_step_regions_on_flat_map() -> None:
    analyzer = TerrainAnalyzer(HeightMap.from_flat(width=4, height=4)).analyze()

    assert analyzer.get_step_regions() == []


def test_analyzer_detects_step_regions() -> None:
    analyzer = TerrainAnalyzer(_step_heightmap(), step_height_threshold=0.5).analyze()

    assert len(analyzer.get_step_regions()) >= 1


def test_analyzer_step_map_reduces_traversable_cells() -> None:
    analyzer = TerrainAnalyzer(_step_heightmap(), flat_slope_deg=10.0, step_height_threshold=0.5).analyze()

    assert analyzer.traversability_map.sum() < analyzer.traversability_map.size


def test_analyzer_respects_flat_slope_threshold() -> None:
    analyzer = TerrainAnalyzer(_plane_heightmap(ax=0.2, by=0.0), flat_slope_deg=1.0).analyze()

    assert not analyzer.traversability_map.any()


def test_analyzer_getters_return_copies() -> None:
    analyzer = TerrainAnalyzer(HeightMap.from_flat(width=4, height=4)).analyze()
    regions = analyzer.get_flat_regions()
    regions.clear()

    assert len(analyzer.get_flat_regions()) == 1


def test_adaptive_footstep_adjusts_pair_targets() -> None:
    footstep = TerrainAdaptiveFootstep(_plane_heightmap(ax=0.5, by=0.0))
    adjusted = footstep.adjust_footsteps([(2.0, 1.0)])

    assert adjusted == [(2.0, 1.0, pytest.approx(1.0))]


def test_adaptive_footstep_ignores_existing_z_values() -> None:
    footstep = TerrainAdaptiveFootstep(_plane_heightmap(ax=0.5, by=0.0))
    adjusted = footstep.adjust_footsteps([(2.0, 1.0, 99.0)])

    assert adjusted[0][2] == pytest.approx(1.0)


def test_adaptive_footstep_rejects_invalid_step_tuple() -> None:
    footstep = TerrainAdaptiveFootstep(HeightMap.from_flat(width=4, height=4))

    with pytest.raises(ValueError, match="at least x and y"):
        footstep.adjust_footsteps([(1.0,)])


def test_cost_map_initialization() -> None:
    cost_map = TerrainCostMap(HeightMap.from_flat(width=4, height=3))

    assert cost_map.cost_map.shape == (3, 4)


def test_build_costmap_is_zero_on_flat_map() -> None:
    built = TerrainCostMap(HeightMap.from_flat(width=4, height=4)).build_costmap()

    np.testing.assert_allclose(built, 0.0)


def test_build_costmap_increases_with_slope() -> None:
    low = TerrainCostMap(_plane_heightmap(ax=0.1, by=0.0)).build_costmap().mean()
    high = TerrainCostMap(_plane_heightmap(ax=0.5, by=0.0)).build_costmap().mean()

    assert high > low


def test_build_costmap_adds_step_penalty() -> None:
    flat = TerrainCostMap(HeightMap.from_flat(width=5, height=5), step_height_threshold=0.5).build_costmap().max()
    stepped = TerrainCostMap(_step_heightmap(), step_height_threshold=0.5).build_costmap().max()

    assert stepped > flat


def test_get_cost_interpolates_from_built_costmap() -> None:
    heightmap = _plane_heightmap(ax=0.2, by=0.0, width=2, height=2)
    cost_map = TerrainCostMap(heightmap)
    built = cost_map.build_costmap()

    assert cost_map.get_cost(0.5, 0.5) == pytest.approx(built.mean())


def test_get_cost_builds_map_lazily() -> None:
    cost_map = TerrainCostMap(_plane_heightmap(ax=0.2, by=0.0))

    assert cost_map.get_cost(1.0, 1.0) >= 0.0


def test_public_terrain_module_exports_classes() -> None:
    assert terrain.HeightMap is HeightMap
    assert terrain.TerrainPatch is TerrainPatch
    assert terrain.TerrainAnalyzer is TerrainAnalyzer
    assert terrain.TerrainAdaptiveFootstep is TerrainAdaptiveFootstep
    assert terrain.TerrainCostMap is TerrainCostMap


def test_top_level_optisim_exports_terrain_classes() -> None:
    assert PublicHeightMap is HeightMap
    assert PublicTerrainPatch is TerrainPatch
    assert PublicTerrainAnalyzer is TerrainAnalyzer
    assert PublicTerrainAdaptiveFootstep is TerrainAdaptiveFootstep
    assert PublicTerrainCostMap is TerrainCostMap


def test_top_level_optisim_exports_terrain_module_and_version() -> None:
    import optisim

    assert optisim.terrain is terrain
    assert optisim.__version__ == "0.21.0"
