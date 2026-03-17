from __future__ import annotations

from dataclasses import is_dataclass

import pytest

import optisim
import optisim.pathplan as pathplan
from optisim import AStarPlanner as PublicAStarPlanner
from optisim import GridNode as PublicGridNode
from optisim import RoadmapPlanner as PublicRoadmapPlanner
from optisim import VoxelGrid
from optisim import WaypointSmoother as PublicWaypointSmoother
from optisim.pathplan import AStarPlanner, GridNode, RoadmapPlanner, WaypointSmoother


def _grid() -> VoxelGrid:
    return VoxelGrid.from_shape(resolution=1.0, origin=(0.0, 0.0, 0.0), shape=(5, 5, 5))


def _world(i: int, j: int, k: int) -> tuple[float, float, float]:
    return _grid().voxel_to_world(i, j, k)


def test_gridnode_is_dataclass() -> None:
    assert is_dataclass(GridNode)


def test_gridnode_stores_parent() -> None:
    parent = GridNode(0, 0, 0)
    node = GridNode(1, 2, 3, parent=parent)

    assert node.parent is parent


def test_astar_uses_manhattan_heuristic_without_diagonals() -> None:
    planner = AStarPlanner(_grid(), allow_diagonal=False)

    assert planner.heuristic((0, 0, 0), (2, 3, 4)) == pytest.approx(9.0)


def test_astar_uses_chebyshev_heuristic_with_diagonals() -> None:
    planner = AStarPlanner(_grid(), allow_diagonal=True)

    assert planner.heuristic((0, 0, 0), (2, 3, 4)) == pytest.approx(4.0)


def test_reconstruct_path_returns_ordered_indices() -> None:
    start = GridNode(0, 0, 0)
    mid = GridNode(1, 0, 0, parent=start)
    goal = GridNode(1, 1, 0, parent=mid)

    assert AStarPlanner(_grid()).reconstruct_path(goal) == [(0, 0, 0), (1, 0, 0), (1, 1, 0)]


def test_astar_returns_none_for_occupied_start() -> None:
    grid = _grid()
    grid.data[0, 0, 0] = 255

    assert AStarPlanner(grid).plan((0.5, 0.5, 0.5), (2.5, 2.5, 2.5)) is None


def test_astar_returns_none_for_occupied_goal() -> None:
    grid = _grid()
    grid.data[2, 2, 2] = 255

    assert AStarPlanner(grid).plan((0.5, 0.5, 0.5), (2.5, 2.5, 2.5)) is None


def test_astar_returns_none_for_out_of_bounds_start() -> None:
    assert AStarPlanner(_grid()).plan((-0.1, 0.5, 0.5), (2.5, 2.5, 2.5)) is None


def test_astar_returns_single_waypoint_when_start_equals_goal_voxel() -> None:
    path = AStarPlanner(_grid()).plan((0.6, 0.6, 0.6), (0.9, 0.9, 0.9))

    assert path == [(0.5, 0.5, 0.5)]


def test_astar_finds_straight_axis_aligned_path() -> None:
    path = AStarPlanner(_grid()).plan((0.5, 0.5, 0.5), (3.5, 0.5, 0.5))

    assert path == [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (2.5, 0.5, 0.5), (3.5, 0.5, 0.5)]


def test_astar_avoids_occupied_voxels() -> None:
    grid = _grid()
    grid.data[1, 0, 0] = 255

    path = AStarPlanner(grid).plan((0.5, 0.5, 0.5), (2.5, 0.5, 0.5))

    assert path is not None
    assert (1.5, 0.5, 0.5) not in path


def test_astar_returns_none_when_wall_blocks_all_routes() -> None:
    grid = _grid()
    grid.data[1, :, :] = 255

    assert AStarPlanner(grid).plan((0.5, 0.5, 0.5), (2.5, 2.5, 2.5)) is None


def test_astar_with_diagonals_takes_shorter_route() -> None:
    axis_path = AStarPlanner(_grid(), allow_diagonal=False).plan((0.5, 0.5, 0.5), (2.5, 2.5, 2.5))
    diagonal_path = AStarPlanner(_grid(), allow_diagonal=True).plan((0.5, 0.5, 0.5), (2.5, 2.5, 2.5))

    assert axis_path is not None
    assert diagonal_path is not None
    assert len(diagonal_path) < len(axis_path)


def test_astar_accepts_list_like_world_points() -> None:
    path = AStarPlanner(_grid()).plan([0.5, 0.5, 0.5], [1.5, 0.5, 0.5])  # type: ignore[arg-type]

    assert path == [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5)]


def test_astar_rejects_invalid_point_shape() -> None:
    with pytest.raises(ValueError, match="start_world"):
        AStarPlanner(_grid()).plan((0.5, 0.5), (1.5, 0.5, 0.5))  # type: ignore[arg-type]


def test_smoother_rejects_negative_robot_radius() -> None:
    with pytest.raises(ValueError, match="robot_radius"):
        WaypointSmoother(_grid(), robot_radius=-0.1)


def test_smoother_returns_short_path_unchanged() -> None:
    waypoints = [(0.5, 0.5, 0.5), (1.5, 1.5, 1.5)]

    assert WaypointSmoother(_grid()).smooth(waypoints) == waypoints


def test_line_of_sight_is_true_in_empty_grid() -> None:
    assert WaypointSmoother(_grid()).has_line_of_sight((0.5, 0.5, 0.5), (4.5, 4.5, 4.5)) is True


def test_line_of_sight_is_false_when_segment_crosses_occupied_voxel() -> None:
    grid = _grid()
    grid.data[2, 2, 2] = 255

    assert WaypointSmoother(grid).has_line_of_sight((0.5, 0.5, 0.5), (4.5, 4.5, 4.5)) is False


def test_line_of_sight_is_false_when_endpoint_is_out_of_bounds() -> None:
    assert WaypointSmoother(_grid()).has_line_of_sight((0.5, 0.5, 0.5), (5.1, 0.5, 0.5)) is False


def test_line_of_sight_respects_robot_radius() -> None:
    grid = _grid()
    grid.data[1, 1, 0] = 255

    assert WaypointSmoother(grid, robot_radius=1.1).has_line_of_sight((0.5, 0.5, 0.5), (2.5, 0.5, 0.5)) is False


def test_smooth_removes_unnecessary_intermediate_waypoints() -> None:
    waypoints = [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (2.5, 0.5, 0.5), (3.5, 0.5, 0.5)]

    assert WaypointSmoother(_grid()).smooth(waypoints) == [(0.5, 0.5, 0.5), (3.5, 0.5, 0.5)]


def test_smooth_keeps_detour_when_direct_line_is_blocked() -> None:
    grid = _grid()
    grid.data[1, 0, 0] = 255
    grid.data[1, 1, 0] = 255
    grid.data[1, 2, 0] = 255
    waypoints = [(0.5, 0.5, 0.5), (0.5, 2.5, 0.5), (2.5, 2.5, 0.5), (2.5, 0.5, 0.5)]

    smoothed = WaypointSmoother(grid).smooth(waypoints)

    assert len(smoothed) > 2


def test_smooth_preserves_first_and_last_waypoint() -> None:
    waypoints = [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (2.5, 0.5, 0.5)]

    smoothed = WaypointSmoother(_grid()).smooth(waypoints)

    assert smoothed[0] == waypoints[0]
    assert smoothed[-1] == waypoints[-1]


def test_smooth_accepts_list_like_waypoints() -> None:
    smoothed = WaypointSmoother(_grid()).smooth([(0.5, 0.5, 0.5), [2.5, 0.5, 0.5]])  # type: ignore[list-item]

    assert smoothed == [(0.5, 0.5, 0.5), (2.5, 0.5, 0.5)]


def test_smooth_rejects_invalid_waypoint_shape() -> None:
    with pytest.raises(ValueError, match="waypoint"):
        WaypointSmoother(_grid()).smooth([(0.5, 0.5, 0.5), (1.0, 2.0)])  # type: ignore[list-item]


def test_roadmap_rejects_negative_node_count() -> None:
    with pytest.raises(ValueError, match="n_nodes"):
        RoadmapPlanner(_grid(), n_nodes=-1)


def test_roadmap_rejects_negative_robot_radius() -> None:
    with pytest.raises(ValueError, match="robot_radius"):
        RoadmapPlanner(_grid(), robot_radius=-0.1)


def test_roadmap_build_samples_at_most_n_nodes() -> None:
    planner = RoadmapPlanner(_grid(), n_nodes=7, seed=1).build()

    assert len(planner.nodes) == 7


def test_roadmap_build_on_fully_occupied_grid_has_no_nodes() -> None:
    grid = _grid()
    grid.data[:, :, :] = 255
    planner = RoadmapPlanner(grid, n_nodes=5).build()

    assert planner.nodes == []
    assert planner.edges == []


def test_roadmap_build_is_deterministic_for_seed() -> None:
    first = RoadmapPlanner(_grid(), n_nodes=8, seed=3).build()
    second = RoadmapPlanner(_grid(), n_nodes=8, seed=3).build()

    assert first.nodes == second.nodes
    assert first.edges == second.edges


def test_roadmap_edges_reference_valid_node_indices() -> None:
    planner = RoadmapPlanner(_grid(), n_nodes=12, seed=2).build()

    assert all(0 <= a < len(planner.nodes) and 0 <= b < len(planner.nodes) for a, b in planner.edges)


def test_roadmap_plan_returns_none_for_occupied_start() -> None:
    grid = _grid()
    grid.data[0, 0, 0] = 255

    assert RoadmapPlanner(grid, n_nodes=10).build().plan((0.5, 0.5, 0.5), (4.5, 4.5, 4.5)) is None


def test_roadmap_plan_returns_direct_path_when_visible() -> None:
    path = RoadmapPlanner(_grid(), n_nodes=0).build().plan((0.5, 0.5, 0.5), (4.5, 0.5, 0.5))

    assert path == [(0.5, 0.5, 0.5), (4.5, 0.5, 0.5)]


def test_roadmap_plan_returns_none_when_start_and_goal_cannot_connect() -> None:
    grid = _grid()
    grid.data[2, :, :] = 255

    assert RoadmapPlanner(grid, n_nodes=20, seed=4).build().plan((0.5, 0.5, 0.5), (4.5, 4.5, 4.5)) is None


def test_roadmap_plan_finds_route_around_obstacle_barrier() -> None:
    grid = _grid()
    grid.data[2, 1:4, 2] = 255
    planner = RoadmapPlanner(grid, n_nodes=80, seed=5).build()
    planner.max_connect_dist = 3.5

    path = planner.plan((0.5, 2.5, 2.5), (4.5, 2.5, 2.5))

    assert path is not None
    assert path[0] == (0.5, 2.5, 2.5)
    assert path[-1] == (4.5, 2.5, 2.5)


def test_roadmap_plan_respects_robot_radius_for_start_state() -> None:
    grid = _grid()
    grid.data[1, 0, 0] = 255

    assert RoadmapPlanner(grid, n_nodes=20, robot_radius=1.1).build().plan((0.5, 0.5, 0.5), (4.5, 0.5, 0.5)) is None


def test_pathplan_module_exports_classes() -> None:
    assert pathplan.GridNode is GridNode
    assert pathplan.AStarPlanner is AStarPlanner
    assert pathplan.WaypointSmoother is WaypointSmoother
    assert pathplan.RoadmapPlanner is RoadmapPlanner


def test_top_level_optisim_exports_pathplan_classes() -> None:
    assert PublicGridNode is GridNode
    assert PublicAStarPlanner is AStarPlanner
    assert PublicWaypointSmoother is WaypointSmoother
    assert PublicRoadmapPlanner is RoadmapPlanner


def test_top_level_optisim_exports_pathplan_module_and_version() -> None:
    assert optisim.pathplan is pathplan
    assert optisim.__version__ == "0.28.0"
