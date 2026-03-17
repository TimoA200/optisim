"""Perception-to-grasp demo using a synthetic depth image."""

from __future__ import annotations

import numpy as np

from optisim.grasp import GraspPlanner, default_parallel_jaw
from optisim.math3d import Pose, Quaternion
from optisim.perception import build_perception_pipeline
from optisim.sim.world import ObjectState


def _synthetic_depth_image(width: int = 64, height: int = 48) -> np.ndarray:
    depth = np.full((height, width), 1.0, dtype=np.float64)
    cy = height // 2
    cx = width // 2
    radius = min(width, height) // 6
    for v in range(height):
        for u in range(width):
            du = u - cx
            dv = v - cy
            r2 = du * du + dv * dv
            if r2 > radius * radius:
                continue
            bump = np.sqrt(1.0 - r2 / float(radius * radius))
            depth[v, u] = 0.78 - 0.08 * bump
    return depth


def main() -> None:
    pipeline = build_perception_pipeline(
        {"width": 64, "height": 48, "fx": 70.0, "fy": 70.0, "cx": 32.0, "cy": 24.0}
    )
    depth_image = _synthetic_depth_image()
    pose_estimates = pipeline.process(depth_image)

    print("detected objects:")
    if not pose_estimates:
        print("  none")
        return
    for estimate in pose_estimates:
        print(
            f"  label={estimate.object_label} position={np.round(estimate.position_world, 3).tolist()} "
            f"orientation={np.round(estimate.orientation_world, 3).tolist()} "
            f"pos_sigma={estimate.position_uncertainty:.4f}"
        )

    grasp_targets = pipeline.to_grasp_targets(pose_estimates)
    print("\ngrasp targets:")
    for target in grasp_targets:
        print(
            f"  label={target['object_label']} position={np.round(target['position'], 3).tolist()} "
            f"approach={np.round(target['approach_direction'], 3).tolist()}"
        )

    planner = GraspPlanner()
    gripper = default_parallel_jaw()
    print("\nfull sense -> plan -> grasp preview:")
    for target in grasp_targets:
        obj = ObjectState(
            name=target["object_label"],
            pose=Pose(position=target["position"], orientation=Quaternion.identity()),
            size=(0.10, 0.10, 0.10),
            mass_kg=0.5,
        )
        grasps = planner.plan_grasps(obj, gripper, n_candidates=3)
        print(f"  object={obj.name} grasp_candidates={len(grasps)}")
        if grasps:
            best = grasps[0]
            print(
                f"    best position={np.round(best.position, 3).tolist()} "
                f"aperture={best.aperture:.3f} score={best.quality_score:.3f}"
            )


if __name__ == "__main__":
    main()
