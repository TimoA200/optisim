from __future__ import annotations

import numpy as np

from optisim import Joint3D, JointMapping, ReferenceSkeleton, RetargetMapping, RetargetResult, RetargetSolver


def make_t_pose_skeleton() -> ReferenceSkeleton:
    return ReferenceSkeleton(
        [
            Joint3D("pelvis", [0.0, 0.0, 1.00]),
            Joint3D("spine", [0.0, 0.0, 1.18], parent="pelvis"),
            Joint3D("chest", [0.0, 0.0, 1.40], parent="spine"),
            Joint3D("neck", [0.0, 0.0, 1.56], parent="chest"),
            Joint3D("head", [0.0, 0.0, 1.74], parent="neck"),
            Joint3D("left_shoulder", [0.0, 0.20, 1.46], parent="chest"),
            Joint3D("left_elbow", [0.0, 0.52, 1.46], parent="left_shoulder"),
            Joint3D("left_wrist", [0.0, 0.82, 1.46], parent="left_elbow"),
            Joint3D("right_shoulder", [0.0, -0.20, 1.46], parent="chest"),
            Joint3D("right_elbow", [0.0, -0.52, 1.46], parent="right_shoulder"),
            Joint3D("right_wrist", [0.0, -0.82, 1.46], parent="right_elbow"),
            Joint3D("left_hip", [0.0, 0.10, 1.00], parent="pelvis"),
            Joint3D("left_knee", [0.0, 0.10, 0.57], parent="left_hip"),
            Joint3D("left_ankle", [0.0, 0.10, 0.12], parent="left_knee"),
            Joint3D("right_hip", [0.0, -0.10, 1.00], parent="pelvis"),
            Joint3D("right_knee", [0.0, -0.10, 0.57], parent="right_hip"),
            Joint3D("right_ankle", [0.0, -0.10, 0.12], parent="right_knee"),
        ]
    )


def make_bent_pose_skeleton() -> ReferenceSkeleton:
    return ReferenceSkeleton(
        [
            Joint3D("pelvis", [0.0, 0.0, 1.00]),
            Joint3D("spine", [0.02, 0.0, 1.16], parent="pelvis"),
            Joint3D("chest", [0.08, 0.03, 1.37], parent="spine"),
            Joint3D("neck", [0.10, 0.03, 1.52], parent="chest"),
            Joint3D("head", [0.14, 0.08, 1.69], parent="neck"),
            Joint3D("left_shoulder", [0.08, 0.22, 1.45], parent="chest"),
            Joint3D("left_elbow", [0.34, 0.38, 1.28], parent="left_shoulder"),
            Joint3D("left_wrist", [0.55, 0.46, 1.08], parent="left_elbow"),
            Joint3D("right_shoulder", [0.08, -0.16, 1.42], parent="chest"),
            Joint3D("right_elbow", [0.30, -0.36, 1.20], parent="right_shoulder"),
            Joint3D("right_wrist", [0.45, -0.30, 0.96], parent="right_elbow"),
            Joint3D("left_hip", [0.01, 0.10, 1.00], parent="pelvis"),
            Joint3D("left_knee", [0.10, 0.12, 0.60], parent="left_hip"),
            Joint3D("left_ankle", [0.12, 0.10, 0.18], parent="left_knee"),
            Joint3D("right_hip", [0.01, -0.10, 1.00], parent="pelvis"),
            Joint3D("right_knee", [0.17, -0.12, 0.66], parent="right_hip"),
            Joint3D("right_ankle", [0.24, -0.10, 0.33], parent="right_knee"),
        ]
    )


def make_pose_dict() -> dict[str, list[float]]:
    skeleton = ReferenceSkeleton.HUMAN_SKELETON()
    return {name: skeleton.get_position(name).tolist() for name in skeleton.joints}


def test_joint3d_creation() -> None:
    joint = Joint3D("pelvis", [0.0, 0.0, 1.0])
    assert joint.name == "pelvis"


def test_joint3d_position_has_shape_three() -> None:
    joint = Joint3D("pelvis", [0.0, 0.0, 1.0])
    assert joint.position.shape == (3,)


def test_joint3d_parent_defaults_to_none() -> None:
    joint = Joint3D("pelvis", [0.0, 0.0, 1.0])
    assert joint.parent is None


def test_reference_skeleton_initialization() -> None:
    skeleton = ReferenceSkeleton([Joint3D("pelvis", [0.0, 0.0, 1.0])])
    assert "pelvis" in skeleton.joints


def test_reference_skeleton_duplicate_joint_raises_value_error() -> None:
    try:
        ReferenceSkeleton([Joint3D("pelvis", [0.0, 0.0, 1.0]), Joint3D("pelvis", [0.0, 0.0, 1.0])])
    except ValueError:
        return
    raise AssertionError("expected duplicate joint failure")


def test_human_skeleton_has_17_joints() -> None:
    assert len(ReferenceSkeleton.HUMAN_SKELETON().joints) == 17


def test_human_skeleton_has_named_keypoints() -> None:
    skeleton = ReferenceSkeleton.HUMAN_SKELETON()
    assert {"left_shoulder", "right_hip", "head", "left_ankle"} <= set(skeleton.joints)


def test_get_position_returns_shape_three() -> None:
    position = ReferenceSkeleton.HUMAN_SKELETON().get_position("pelvis")
    assert position.shape == (3,)


def test_bone_vector_returns_shape_three() -> None:
    vector = ReferenceSkeleton.HUMAN_SKELETON().bone_vector("pelvis", "spine")
    assert vector.shape == (3,)


def test_bone_length_returns_positive_float() -> None:
    length = ReferenceSkeleton.HUMAN_SKELETON().bone_length("pelvis", "spine")
    assert isinstance(length, float)
    assert length > 0.0


def test_from_dict_round_trip_preserves_joint_positions() -> None:
    pose_dict = make_pose_dict()
    skeleton = ReferenceSkeleton.from_dict(pose_dict)
    np.testing.assert_allclose(skeleton.get_position("left_wrist"), pose_dict["left_wrist"])


def test_from_dict_preserves_joint_count() -> None:
    skeleton = ReferenceSkeleton.from_dict(make_pose_dict())
    assert len(skeleton.joints) == 17


def test_joint_mapping_creation() -> None:
    mapping = JointMapping(["left_shoulder", "left_elbow"], "left_shoulder_pitch")
    assert mapping.target_joint == "left_shoulder_pitch"


def test_joint_mapping_defaults_scale_and_offset() -> None:
    mapping = JointMapping(["left_shoulder", "left_elbow"], "left_shoulder_pitch")
    assert mapping.scale == 1.0 and mapping.offset == 0.0


def test_retarget_mapping_default_has_mappings() -> None:
    assert len(RetargetMapping.HUMANOID_DEFAULT().mappings) > 0


def test_retarget_mapping_default_covers_31_joints() -> None:
    assert len(RetargetMapping.HUMANOID_DEFAULT().joint_names) == 31


def test_get_mapping_returns_joint_mapping_for_known_joint() -> None:
    mapping = RetargetMapping.HUMANOID_DEFAULT().get_mapping("torso_pitch")
    assert isinstance(mapping, JointMapping)


def test_get_mapping_returns_none_for_unknown_joint() -> None:
    assert RetargetMapping.HUMANOID_DEFAULT().get_mapping("missing_joint") is None


def test_joint_names_returns_non_empty_list() -> None:
    joint_names = RetargetMapping.HUMANOID_DEFAULT().joint_names
    assert isinstance(joint_names, list)
    assert joint_names


def test_retarget_solver_initialization() -> None:
    solver = RetargetSolver()
    assert isinstance(solver.mapping, RetargetMapping)


def test_retarget_solver_robot_has_31_dof() -> None:
    solver = RetargetSolver()
    assert solver.robot.dof == 31


def test_retarget_returns_retarget_result() -> None:
    result = RetargetSolver().retarget(ReferenceSkeleton.HUMAN_SKELETON())
    assert isinstance(result, RetargetResult)


def test_retarget_joint_angles_shape_is_31() -> None:
    result = RetargetSolver().retarget(ReferenceSkeleton.HUMAN_SKELETON())
    assert result.joint_angles.shape == (31,)


def test_retarget_coverage_between_zero_and_one() -> None:
    result = RetargetSolver().retarget(ReferenceSkeleton.HUMAN_SKELETON())
    assert 0.0 <= result.coverage <= 1.0


def test_retarget_n_mapped_is_positive() -> None:
    result = RetargetSolver().retarget(ReferenceSkeleton.HUMAN_SKELETON())
    assert result.n_mapped > 0


def test_retarget_residuals_shape_matches_n_mapped() -> None:
    result = RetargetSolver().retarget(ReferenceSkeleton.HUMAN_SKELETON())
    assert result.residuals.shape == (result.n_mapped,)


def test_retarget_sequence_returns_list_of_correct_length() -> None:
    solver = RetargetSolver()
    results = solver.retarget_sequence([ReferenceSkeleton.HUMAN_SKELETON(), make_t_pose_skeleton()])
    assert len(results) == 2


def test_retarget_sequence_items_are_results() -> None:
    solver = RetargetSolver()
    results = solver.retarget_sequence([ReferenceSkeleton.HUMAN_SKELETON()])
    assert isinstance(results[0], RetargetResult)


def test_smooth_sequence_returns_same_length() -> None:
    solver = RetargetSolver()
    results = solver.retarget_sequence([ReferenceSkeleton.HUMAN_SKELETON(), make_bent_pose_skeleton(), make_t_pose_skeleton()])
    assert len(solver.smooth_sequence(results)) == len(results)


def test_smooth_sequence_differs_when_joint_angles_change() -> None:
    solver = RetargetSolver()
    results = solver.retarget_sequence([make_t_pose_skeleton(), make_bent_pose_skeleton(), make_t_pose_skeleton()])
    smoothed = solver.smooth_sequence(results, window=3)
    assert not np.allclose(smoothed[1].joint_angles, results[1].joint_angles)


def test_smooth_sequence_window_one_returns_original_values() -> None:
    solver = RetargetSolver()
    results = solver.retarget_sequence([make_t_pose_skeleton(), make_bent_pose_skeleton()])
    smoothed = solver.smooth_sequence(results, window=1)
    assert np.allclose(smoothed[0].joint_angles, results[0].joint_angles)


def test_t_pose_skeleton_retargets_without_error() -> None:
    result = RetargetSolver().retarget(make_t_pose_skeleton())
    assert np.isfinite(result.joint_angles).all()


def test_retarget_produces_angles_within_joint_limits() -> None:
    solver = RetargetSolver()
    result = solver.retarget(make_bent_pose_skeleton())
    for joint_name, value in zip(solver.robot.joint_names, result.joint_angles, strict=True):
        spec = solver.robot.joints[joint_name]
        assert spec.limit_lower <= value <= spec.limit_upper


def test_retarget_result_coverage_positive_for_human_skeleton() -> None:
    result = RetargetSolver().retarget(ReferenceSkeleton.HUMAN_SKELETON())
    assert result.coverage > 0.0


def test_retarget_result_residuals_are_finite() -> None:
    result = RetargetSolver().retarget(make_bent_pose_skeleton())
    assert np.isfinite(result.residuals).all()


def test_retarget_result_joint_angles_are_finite() -> None:
    result = RetargetSolver().retarget(make_bent_pose_skeleton())
    assert np.isfinite(result.joint_angles).all()


def test_t_pose_gives_nonzero_shoulder_rolls() -> None:
    solver = RetargetSolver()
    result = solver.retarget(make_t_pose_skeleton())
    left_index = solver.robot.joint_names.index("left_shoulder_roll")
    right_index = solver.robot.joint_names.index("right_shoulder_roll")
    assert abs(result.joint_angles[left_index]) > 0.1
    assert abs(result.joint_angles[right_index]) > 0.1


def test_bent_pose_increases_right_knee_pitch() -> None:
    solver = RetargetSolver()
    result = solver.retarget(make_bent_pose_skeleton())
    index = solver.robot.joint_names.index("right_knee_pitch")
    assert result.joint_angles[index] > 0.0


def test_bent_pose_increases_left_elbow_pitch() -> None:
    solver = RetargetSolver()
    result = solver.retarget(make_bent_pose_skeleton())
    index = solver.robot.joint_names.index("left_elbow_pitch")
    assert result.joint_angles[index] > 0.0


def test_optisim_root_exports_retarget_symbols() -> None:
    solver = RetargetSolver()
    skeleton = ReferenceSkeleton.HUMAN_SKELETON()
    assert isinstance(solver.retarget(skeleton), RetargetResult)

