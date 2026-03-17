"""Perception primitives for depth-camera point clouds."""

from optisim.perception.detector import (
    DepthCameraProcessor,
    DetectedObject,
    ICPRegistration,
    ObjectDetector,
    PerceptionPipeline,
    PointCloud,
    PoseEstimate,
    build_perception_pipeline,
)

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
