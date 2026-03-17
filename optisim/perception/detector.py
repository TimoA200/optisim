"""Lightweight depth-based object detection and pose estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optisim.math3d import pose_from_matrix


def _as_array(values: np.ndarray | list[float], shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}, received {array.shape}")
    return array


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    return pose_from_matrix(matrix).orientation.as_np()


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.asarray(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


def _pose_from_components(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _bbox_corners(center: np.ndarray, rotation: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    half = np.asarray(dimensions, dtype=np.float64) / 2.0
    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return center + (signs * half) @ rotation.T


def _oriented_bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0:
        rotation = np.eye(3, dtype=np.float64)
        center = np.zeros(3, dtype=np.float64)
        dimensions = np.zeros(3, dtype=np.float64)
        return _bbox_corners(center, rotation, dimensions), center, dimensions, rotation
    centered = points - points.mean(axis=0)
    if len(points) >= 3:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        rotation = vh.T
    else:
        rotation = np.eye(3, dtype=np.float64)
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    local = centered @ rotation
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    dimensions = np.maximum(maxs - mins, 1e-6)
    local_center = 0.5 * (mins + maxs)
    center = points.mean(axis=0) + rotation @ local_center
    return _bbox_corners(center, rotation, dimensions), center, dimensions, rotation


def _cluster_points(points: np.ndarray, radius: float = 0.05, min_points: int = 8) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    visited = np.zeros(len(points), dtype=bool)
    clusters: list[np.ndarray] = []
    for start in range(len(points)):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        cluster = []
        while queue:
            index = queue.pop()
            cluster.append(index)
            distances = np.linalg.norm(points - points[index], axis=1)
            neighbors = np.flatnonzero((distances <= radius) & (~visited))
            visited[neighbors] = True
            queue.extend(neighbors.tolist())
        if len(cluster) >= min_points:
            clusters.append(np.asarray(cluster, dtype=int))
    return clusters


def _best_fit_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid
    covariance = src_centered.T @ tgt_centered
    u, _, vh = np.linalg.svd(covariance, full_matrices=False)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    translation = tgt_centroid - rotation @ src_centroid
    return rotation, translation


@dataclass(slots=True)
class PointCloud:
    """Point cloud in the camera frame."""

    points: np.ndarray
    colors: np.ndarray | None = None
    frame: str = "camera"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.points = _as_array(self.points)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("PointCloud.points must have shape (N, 3)")
        if self.colors is not None:
            self.colors = _as_array(self.colors)
            if self.colors.shape != self.points.shape:
                raise ValueError("PointCloud.colors must have shape (N, 3)")
        self.frame = str(self.frame)
        self.timestamp = float(self.timestamp)


@dataclass(slots=True)
class DetectedObject:
    """Detected support surface or object candidate."""

    label: str
    confidence: float
    bounding_box_3d: np.ndarray
    pose: np.ndarray
    dimensions: np.ndarray
    point_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bounding_box_3d = _as_array(self.bounding_box_3d, (8, 3))
        self.pose = _as_array(self.pose, (4, 4))
        self.dimensions = _as_array(self.dimensions, (3,))
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        self.point_indices = [int(index) for index in self.point_indices]


@dataclass(slots=True)
class PoseEstimate:
    """Pose estimate in the world frame."""

    object_label: str
    position_world: np.ndarray
    orientation_world: np.ndarray
    position_uncertainty: float
    orientation_uncertainty: float
    source: str = "depth_camera"

    def __post_init__(self) -> None:
        self.position_world = _as_array(self.position_world, (3,))
        self.orientation_world = _as_array(self.orientation_world, (4,))
        self.position_uncertainty = float(self.position_uncertainty)
        self.orientation_uncertainty = float(self.orientation_uncertainty)
        self.source = str(self.source)


class DepthCameraProcessor:
    """Project and pre-process depth camera data."""

    def __init__(
        self,
        fx: float = 525.0,
        fy: float = 525.0,
        cx: float = 320.0,
        cy: float = 240.0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(width)
        self.height = int(height)

    def depth_to_pointcloud(self, depth_image: np.ndarray) -> PointCloud:
        depth = _as_array(depth_image)
        if depth.shape != (self.height, self.width):
            raise ValueError(f"depth image must have shape ({self.height}, {self.width})")
        us, vs = np.meshgrid(np.arange(self.width, dtype=np.float64), np.arange(self.height, dtype=np.float64))
        z = depth.reshape(-1)
        x = (us.reshape(-1) - self.cx) * z / self.fx
        y = (vs.reshape(-1) - self.cy) * z / self.fy
        points = np.column_stack((x, y, z))
        return PointCloud(points=points, frame="camera")

    def filter_range(self, cloud: PointCloud, min_dist: float = 0.1, max_dist: float = 3.0) -> PointCloud:
        distances = np.linalg.norm(cloud.points, axis=1)
        mask = np.isfinite(distances) & (distances >= float(min_dist)) & (distances <= float(max_dist))
        colors = cloud.colors[mask] if cloud.colors is not None else None
        return PointCloud(points=cloud.points[mask], colors=colors, frame=cloud.frame, timestamp=cloud.timestamp)

    def voxel_downsample(self, cloud: PointCloud, voxel_size: float = 0.01) -> PointCloud:
        if len(cloud.points) == 0:
            return PointCloud(points=cloud.points.copy(), colors=None if cloud.colors is None else cloud.colors.copy(), frame=cloud.frame, timestamp=cloud.timestamp)
        voxel = max(float(voxel_size), 1e-6)
        keys = np.floor(cloud.points / voxel).astype(np.int64)
        _, first_indices = np.unique(keys, axis=0, return_index=True)
        first_indices = np.sort(first_indices)
        colors = cloud.colors[first_indices] if cloud.colors is not None else None
        return PointCloud(points=cloud.points[first_indices], colors=colors, frame=cloud.frame, timestamp=cloud.timestamp)

    def estimate_normals(self, cloud: PointCloud, k_neighbors: int = 10) -> np.ndarray:
        points = cloud.points
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float64)
        k = max(1, min(int(k_neighbors), len(points)))
        normals = np.zeros_like(points)
        for index, point in enumerate(points):
            distances = np.linalg.norm(points - point, axis=1)
            neighbors = points[np.argsort(distances)[:k]]
            centered = neighbors - neighbors.mean(axis=0)
            covariance = centered.T @ centered / max(len(neighbors), 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            normal = eigenvectors[:, int(np.argmin(eigenvalues))]
            if normal[2] > 0.0:
                normal *= -1.0
            normals[index] = _normalize(normal)
        return normals


class ObjectDetector:
    """Small deterministic detector over point clouds."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = float(confidence_threshold)

    def detect_planar_surfaces(self, cloud: PointCloud) -> list[DetectedObject]:
        points = cloud.points
        if len(points) < 3:
            return []
        best_inliers = np.empty(0, dtype=int)
        best_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        rng = np.random.default_rng(0)
        sample_count = min(60, max(10, len(points) // 4))
        for _ in range(sample_count):
            sample = points[rng.choice(len(points), size=3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = np.linalg.norm(normal)
            if norm <= 1e-9:
                continue
            normal = normal / norm
            if normal[2] < 0.0:
                normal *= -1.0
            distances = np.abs((points - sample[0]) @ normal)
            inliers = np.flatnonzero(distances < 0.015)
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_normal = normal
        if len(best_inliers) < 10:
            return []
        plane_points = points[best_inliers]
        centroid = plane_points.mean(axis=0)
        tangent_x = _normalize(np.cross(np.array([0.0, 0.0, 1.0]), best_normal))
        if np.linalg.norm(tangent_x) <= 1e-9:
            tangent_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        tangent_y = _normalize(np.cross(best_normal, tangent_x))
        rotation = np.column_stack((tangent_x, tangent_y, best_normal))
        local = (plane_points - centroid) @ rotation
        mins = local.min(axis=0)
        maxs = local.max(axis=0)
        dimensions = np.maximum(maxs - mins, 1e-6)
        dimensions[2] = max(dimensions[2], 0.01)
        confidence = min(1.0, len(best_inliers) / max(len(points), 1))
        if confidence < self.confidence_threshold:
            return []
        pose = _pose_from_components(rotation, centroid)
        bbox = _bbox_corners(centroid, rotation, dimensions)
        label = "table" if abs(best_normal[2]) > 0.75 else "plane"
        return [
            DetectedObject(
                label=label,
                confidence=confidence,
                bounding_box_3d=bbox,
                pose=pose,
                dimensions=dimensions,
                point_indices=best_inliers.tolist(),
            )
        ]

    def detect_objects_on_surface(self, cloud: PointCloud, surface: DetectedObject) -> list[DetectedObject]:
        points = cloud.points
        if len(points) == 0:
            return []
        normal = _normalize(surface.pose[:3, 2])
        origin = surface.pose[:3, 3]
        signed = (points - origin) @ normal
        above_surface = np.flatnonzero((np.abs(signed) > 0.02) & (np.abs(signed) < 0.5))
        if len(above_surface) == 0:
            return []
        clusters = _cluster_points(points[above_surface], radius=0.05, min_points=6)
        detected: list[DetectedObject] = []
        for cluster in clusters:
            point_indices = above_surface[cluster]
            cluster_points = points[point_indices]
            bbox, center, dimensions, rotation = _oriented_bbox(cluster_points)
            dims_sorted = np.sort(np.maximum(dimensions, 1e-6))
            ratio = dims_sorted[-1] / dims_sorted[0]
            if ratio < 1.35:
                label = "sphere"
            elif dims_sorted[-1] > 1.6 * dims_sorted[1]:
                label = "cylinder"
            else:
                label = "box"
            pose = _pose_from_components(rotation, center)
            confidence = min(0.99, 0.55 + 0.02 * len(point_indices))
            if confidence < self.confidence_threshold:
                continue
            detected.append(
                DetectedObject(
                    label=label,
                    confidence=confidence,
                    bounding_box_3d=bbox,
                    pose=pose,
                    dimensions=dimensions,
                    point_indices=point_indices.tolist(),
                )
            )
        return detected

    def estimate_object_pose(self, obj: DetectedObject, camera_to_world: np.ndarray | None = None) -> PoseEstimate:
        rotation = obj.pose[:3, :3]
        camera_pose = _pose_from_components(rotation, obj.pose[:3, 3])
        world_pose = camera_pose if camera_to_world is None else _as_array(camera_to_world, (4, 4)) @ camera_pose
        position_uncertainty = max(0.002, 0.02 / np.sqrt(max(len(obj.point_indices), 1)))
        orientation_uncertainty = max(0.01, 0.2 / np.sqrt(max(len(obj.point_indices), 1)))
        return PoseEstimate(
            object_label=obj.label,
            position_world=world_pose[:3, 3],
            orientation_world=_rotation_to_quaternion(world_pose[:3, :3]),
            position_uncertainty=position_uncertainty,
            orientation_uncertainty=orientation_uncertainty,
            source="depth_camera" if camera_to_world is None else "icp",
        )


class ICPRegistration:
    """Pure NumPy point-to-point ICP alignment."""

    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-5) -> None:
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

    def align(self, source: PointCloud, target: PointCloud) -> tuple[np.ndarray, float]:
        source_points = source.points.copy()
        target_points = target.points.copy()
        if len(source_points) == 0 or len(target_points) == 0:
            return np.eye(4, dtype=np.float64), float("inf")
        transform = np.eye(4, dtype=np.float64)
        previous_residual = float("inf")
        for _ in range(self.max_iterations):
            distances = np.linalg.norm(source_points[:, None, :] - target_points[None, :, :], axis=2)
            nearest_distances = np.min(distances, axis=1)
            nearest = target_points[np.argmin(distances, axis=1)]
            residual = float(np.mean(nearest_distances))
            rotation, translation = _best_fit_transform(source_points, nearest)
            update = _pose_from_components(rotation, translation)
            source_points = (rotation @ source_points.T).T + translation
            transform = update @ transform
            if abs(previous_residual - residual) < self.tolerance:
                previous_residual = residual
                break
            previous_residual = residual
        return transform, previous_residual


class PerceptionPipeline:
    """End-to-end perception pipeline from depth to grasp targets."""

    def __init__(self, camera_intrinsics: dict | None = None) -> None:
        intrinsics = dict(camera_intrinsics or {})
        self.camera = DepthCameraProcessor(**intrinsics)
        self.detector = ObjectDetector()

    def process(self, depth_image: np.ndarray, camera_to_world: np.ndarray | None = None) -> list[PoseEstimate]:
        cloud = self.camera.depth_to_pointcloud(depth_image)
        cloud = self.camera.filter_range(cloud)
        cloud = self.camera.voxel_downsample(cloud, voxel_size=0.01)
        surfaces = self.detector.detect_planar_surfaces(cloud)
        estimates: list[PoseEstimate] = []
        for surface in surfaces:
            for obj in self.detector.detect_objects_on_surface(cloud, surface):
                estimates.append(self.detector.estimate_object_pose(obj, camera_to_world=camera_to_world))
        return estimates

    def to_grasp_targets(self, pose_estimates: list[PoseEstimate]) -> list[dict]:
        targets: list[dict] = []
        for estimate in pose_estimates:
            rot = _quaternion_to_rotation(estimate.orientation_world)
            approach_direction = -rot[:, 2]
            if np.linalg.norm(approach_direction) <= 1e-9:
                approach_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            targets.append(
                {
                    "position": estimate.position_world.copy(),
                    "approach_direction": _normalize(approach_direction),
                    "object_label": estimate.object_label,
                }
            )
        return targets


def build_perception_pipeline(camera_config: dict | None = None) -> PerceptionPipeline:
    """Construct a perception pipeline with optional camera overrides."""

    return PerceptionPipeline(camera_intrinsics=camera_config)


__all__ = [
    "PointCloud",
    "DetectedObject",
    "PoseEstimate",
    "DepthCameraProcessor",
    "ObjectDetector",
    "ICPRegistration",
    "PerceptionPipeline",
    "build_perception_pipeline",
]
