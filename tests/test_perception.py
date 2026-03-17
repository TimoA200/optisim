from __future__ import annotations

import numpy as np
import pytest

import optisim.perception as perception
from optisim import PerceptionPipeline as PublicPerceptionPipeline
from optisim import PointCloud as PublicPointCloud
from optisim import PoseEstimate as PublicPoseEstimate
from optisim import build_perception_pipeline as public_build_perception_pipeline
from optisim.grasp import GraspPlanner, default_parallel_jaw
from optisim.math3d import Pose, Quaternion
from optisim.perception import (
    DepthCameraProcessor,
    DetectedObject,
    ICPRegistration,
    ObjectDetector,
    PerceptionPipeline,
    PointCloud,
    PoseEstimate,
    build_perception_pipeline,
)
from optisim.sim.world import ObjectState


def _synthetic_depth_image(width: int = 32, height: int = 24) -> np.ndarray:
    depth = np.ones((height, width), dtype=np.float64)
    depth[8:16, 12:20] = 0.72
    return depth


def _plane_cloud() -> PointCloud:
    xs, ys = np.meshgrid(np.linspace(-0.2, 0.2, 14), np.linspace(-0.2, 0.2, 14))
    plane = np.column_stack((xs.reshape(-1), ys.reshape(-1), np.ones(xs.size)))
    return PointCloud(points=plane)


def _plane_with_object_cloud() -> PointCloud:
    plane = _plane_cloud().points
    xs, ys = np.meshgrid(np.linspace(-0.05, 0.05, 8), np.linspace(-0.05, 0.05, 8))
    box = np.column_stack((xs.reshape(-1), ys.reshape(-1), np.full(xs.size, 0.74)))
    return PointCloud(points=np.vstack((plane, box)))


def test_pointcloud_construction_and_fields() -> None:
    cloud = PointCloud(points=np.zeros((5, 3)), colors=np.ones((5, 3)), frame="cam", timestamp=1.2)

    assert cloud.points.shape == (5, 3)
    assert cloud.colors is not None
    assert cloud.frame == "cam"
    assert cloud.timestamp == pytest.approx(1.2)


def test_pointcloud_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        PointCloud(points=np.zeros((5, 2)))


def test_detected_object_construction() -> None:
    obj = DetectedObject(
        label="box",
        confidence=0.8,
        bounding_box_3d=np.zeros((8, 3)),
        pose=np.eye(4),
        dimensions=np.array([0.1, 0.2, 0.3]),
        point_indices=[0, 1, 2],
    )

    assert obj.label == "box"
    assert obj.bounding_box_3d.shape == (8, 3)
    assert obj.pose.shape == (4, 4)
    assert obj.dimensions.shape == (3,)


def test_pose_estimate_construction() -> None:
    pose = PoseEstimate(
        object_label="box",
        position_world=np.array([0.1, 0.2, 0.3]),
        orientation_world=np.array([1.0, 0.0, 0.0, 0.0]),
        position_uncertainty=0.01,
        orientation_uncertainty=0.05,
    )

    assert pose.position_world.shape == (3,)
    assert pose.orientation_world.shape == (4,)
    assert pose.source == "depth_camera"


def test_depth_camera_processor_initialization() -> None:
    processor = DepthCameraProcessor(width=32, height=24)

    assert processor.width == 32
    assert processor.height == 24
    assert processor.fx == pytest.approx(525.0)


def test_depth_to_pointcloud_with_zeros() -> None:
    processor = DepthCameraProcessor(width=4, height=3, cx=1.5, cy=1.0)
    cloud = processor.depth_to_pointcloud(np.zeros((3, 4), dtype=np.float64))

    assert cloud.points.shape == (12, 3)
    np.testing.assert_allclose(cloud.points, 0.0)


def test_depth_to_pointcloud_with_uniform_depth() -> None:
    processor = DepthCameraProcessor(width=4, height=3, fx=2.0, fy=2.0, cx=1.5, cy=1.0)
    cloud = processor.depth_to_pointcloud(np.full((3, 4), 2.0, dtype=np.float64))

    assert cloud.points.shape == (12, 3)
    assert np.allclose(cloud.points[:, 2], 2.0)


def test_depth_to_pointcloud_result_has_h_times_w_points() -> None:
    processor = DepthCameraProcessor(width=7, height=5)
    cloud = processor.depth_to_pointcloud(np.ones((5, 7), dtype=np.float64))

    assert cloud.points.shape == (35, 3)


def test_filter_range_removes_out_of_range_points() -> None:
    processor = DepthCameraProcessor()
    cloud = PointCloud(points=np.array([[0.0, 0.0, 0.05], [0.0, 0.0, 0.5], [0.0, 0.0, 5.0]]))

    filtered = processor.filter_range(cloud, min_dist=0.1, max_dist=3.0)

    assert filtered.points.shape == (1, 3)
    np.testing.assert_allclose(filtered.points[0], [0.0, 0.0, 0.5])


def test_voxel_downsample_reduces_point_count() -> None:
    processor = DepthCameraProcessor()
    points = np.array([[0.0, 0.0, 1.0], [0.001, 0.001, 1.0], [0.1, 0.1, 1.0]], dtype=np.float64)

    downsampled = processor.voxel_downsample(PointCloud(points=points), voxel_size=0.01)

    assert len(downsampled.points) < len(points)


def test_estimate_normals_returns_n_by_3_array() -> None:
    processor = DepthCameraProcessor()
    normals = processor.estimate_normals(_plane_cloud(), k_neighbors=6)

    assert normals.shape == (196, 3)
    assert np.isfinite(normals).all()


def test_object_detector_initialization() -> None:
    detector = ObjectDetector(confidence_threshold=0.4)

    assert detector.confidence_threshold == pytest.approx(0.4)


def test_detect_planar_surfaces_with_synthetic_plane() -> None:
    detector = ObjectDetector(confidence_threshold=0.1)
    surfaces = detector.detect_planar_surfaces(_plane_cloud())

    assert len(surfaces) == 1
    assert surfaces[0].label in {"table", "plane"}
    assert len(surfaces[0].point_indices) >= 10


def test_detect_planar_surfaces_returns_empty_for_small_cloud() -> None:
    detector = ObjectDetector()

    assert detector.detect_planar_surfaces(PointCloud(points=np.zeros((2, 3)))) == []


def test_detect_objects_on_surface_returns_list() -> None:
    detector = ObjectDetector(confidence_threshold=0.1)
    cloud = _plane_with_object_cloud()
    surface = detector.detect_planar_surfaces(cloud)[0]

    objects = detector.detect_objects_on_surface(cloud, surface)

    assert isinstance(objects, list)
    assert objects


def test_detect_objects_on_surface_assigns_known_label() -> None:
    detector = ObjectDetector(confidence_threshold=0.1)
    cloud = _plane_with_object_cloud()
    surface = detector.detect_planar_surfaces(cloud)[0]

    obj = detector.detect_objects_on_surface(cloud, surface)[0]

    assert obj.label in {"box", "cylinder", "sphere"}
    assert obj.bounding_box_3d.shape == (8, 3)


def test_estimate_object_pose_returns_pose_estimate() -> None:
    detector = ObjectDetector()
    obj = DetectedObject(
        label="box",
        confidence=0.9,
        bounding_box_3d=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1],
                [0.0, 0.1, 0.0],
                [0.0, 0.1, 0.1],
                [0.1, 0.0, 0.0],
                [0.1, 0.0, 0.1],
                [0.1, 0.1, 0.0],
                [0.1, 0.1, 0.1],
            ]
        ),
        pose=np.eye(4),
        dimensions=np.array([0.1, 0.1, 0.1]),
        point_indices=list(range(20)),
    )

    estimate = detector.estimate_object_pose(obj)

    assert isinstance(estimate, PoseEstimate)
    assert estimate.position_world.shape == (3,)
    assert estimate.orientation_world.shape == (4,)


def test_estimate_object_pose_applies_camera_to_world_transform() -> None:
    detector = ObjectDetector()
    obj = DetectedObject(
        label="box",
        confidence=0.9,
        bounding_box_3d=np.zeros((8, 3)),
        pose=np.eye(4),
        dimensions=np.array([0.1, 0.1, 0.1]),
        point_indices=list(range(10)),
    )
    transform = np.eye(4)
    transform[:3, 3] = np.array([1.0, 2.0, 3.0])

    estimate = detector.estimate_object_pose(obj, camera_to_world=transform)

    np.testing.assert_allclose(estimate.position_world, [1.0, 2.0, 3.0])
    assert estimate.source == "icp"


def test_icp_registration_initialization() -> None:
    icp = ICPRegistration(max_iterations=10, tolerance=1e-4)

    assert icp.max_iterations == 10
    assert icp.tolerance == pytest.approx(1e-4)


def test_icp_align_identical_clouds_returns_near_identity() -> None:
    points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
    cloud = PointCloud(points=points)
    transform, residual = ICPRegistration().align(cloud, cloud)

    np.testing.assert_allclose(transform, np.eye(4), atol=1e-6)
    assert residual <= 1e-6


def test_icp_align_translated_cloud_converges() -> None:
    base = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]])
    translation = np.array([0.05, -0.03, 0.04])
    source = PointCloud(points=base + translation)
    target = PointCloud(points=base)

    transform, residual = ICPRegistration(max_iterations=40).align(source, target)

    np.testing.assert_allclose(transform[:3, 3], -translation, atol=1e-2)
    assert residual < 1e-2


def test_perception_pipeline_initialization() -> None:
    pipeline = PerceptionPipeline(camera_intrinsics={"width": 32, "height": 24})

    assert isinstance(pipeline.camera, DepthCameraProcessor)
    assert isinstance(pipeline.detector, ObjectDetector)


def test_perception_pipeline_process_returns_list() -> None:
    pipeline = build_perception_pipeline({"width": 32, "height": 24, "fx": 40.0, "fy": 40.0, "cx": 16.0, "cy": 12.0})

    estimates = pipeline.process(_synthetic_depth_image())

    assert isinstance(estimates, list)


def test_perception_pipeline_process_detects_object_in_synthetic_scene() -> None:
    pipeline = build_perception_pipeline({"width": 32, "height": 24, "fx": 40.0, "fy": 40.0, "cx": 16.0, "cy": 12.0})

    estimates = pipeline.process(_synthetic_depth_image())

    assert estimates
    assert all(isinstance(estimate, PoseEstimate) for estimate in estimates)


def test_to_grasp_targets_returns_required_keys() -> None:
    pipeline = PerceptionPipeline()
    estimates = [
        PoseEstimate(
            object_label="box",
            position_world=np.array([0.1, 0.2, 0.3]),
            orientation_world=np.array([1.0, 0.0, 0.0, 0.0]),
            position_uncertainty=0.01,
            orientation_uncertainty=0.02,
        )
    ]

    targets = pipeline.to_grasp_targets(estimates)

    assert isinstance(targets, list)
    assert set(targets[0]) == {"position", "approach_direction", "object_label"}


def test_full_pipeline_to_grasp_targets_is_compatible_with_grasp_planner() -> None:
    pipeline = build_perception_pipeline({"width": 32, "height": 24, "fx": 40.0, "fy": 40.0, "cx": 16.0, "cy": 12.0})
    estimate = pipeline.process(_synthetic_depth_image())[0]
    target = pipeline.to_grasp_targets([estimate])[0]
    obj = ObjectState(
        name=target["object_label"],
        pose=Pose(position=target["position"], orientation=Quaternion.identity()),
        size=(0.08, 0.08, 0.08),
    )

    grasps = GraspPlanner().plan_grasps(obj, default_parallel_jaw(), n_candidates=2)

    assert grasps


def test_build_perception_pipeline_returns_pipeline() -> None:
    pipeline = build_perception_pipeline({"width": 16, "height": 12})

    assert isinstance(pipeline, PerceptionPipeline)


def test_public_exports_match_expected_symbols() -> None:
    expected = {
        "PointCloud",
        "DetectedObject",
        "PoseEstimate",
        "DepthCameraProcessor",
        "ObjectDetector",
        "ICPRegistration",
        "PerceptionPipeline",
        "build_perception_pipeline",
    }

    assert expected.issubset(set(perception.__all__))
    assert PublicPointCloud is PointCloud
    assert PublicPoseEstimate is PoseEstimate
    assert PublicPerceptionPipeline is PerceptionPipeline
    assert public_build_perception_pipeline is build_perception_pipeline


__all__ = [name for name in globals() if name.startswith("test_")]
