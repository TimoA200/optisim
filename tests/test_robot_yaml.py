from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optisim.cli import build_parser
from optisim.robot import RobotModel, load_robot_yaml


def test_load_humanoid_variant_returns_named_robot_model(tmp_path: Path) -> None:
    path = tmp_path / "mini.yaml"
    path.write_text(
        "\n".join(
            [
                "kind: humanoid_variant",
                "name: mini_humanoid",
                "spec:",
                "  upper_arm_m: 0.22",
                "  forearm_m: 0.19",
            ]
        ),
        encoding="utf-8",
    )

    robot = load_robot_yaml(path)

    assert isinstance(robot, RobotModel)
    assert robot.name == "mini_humanoid"


def test_humanoid_variant_height_override_changes_forward_kinematics(tmp_path: Path) -> None:
    default_path = tmp_path / "default.yaml"
    scaled_path = tmp_path / "scaled.yaml"
    default_path.write_text("kind: humanoid_variant\nname: default_humanoid\nspec: {}\n", encoding="utf-8")
    scaled_path.write_text(
        "kind: humanoid_variant\nname: short_humanoid\nspec:\n  height_m: 1.2\n",
        encoding="utf-8",
    )

    default_robot = load_robot_yaml(default_path)
    scaled_robot = load_robot_yaml(scaled_path)

    default_pose = default_robot.forward_kinematics()["head"].position
    scaled_pose = scaled_robot.forward_kinematics()["head"].position

    assert scaled_pose[2] < default_pose[2]


def test_load_custom_robot_with_fixed_and_revolute_joints(tmp_path: Path) -> None:
    robot = load_robot_yaml(_write_simple_arm_yaml(tmp_path))

    assert robot.name == "simple_arm"
    assert robot.root_link == "base_link"
    assert robot.joints["joint1"].joint_type == "revolute"
    assert robot.joints["joint3"].joint_type == "fixed"


def test_custom_robot_forward_kinematics_returns_end_effector_pose(tmp_path: Path) -> None:
    robot = load_robot_yaml(_write_simple_arm_yaml(tmp_path))

    poses = robot.forward_kinematics({"joint1": 0.35, "joint2": -0.4})

    assert "end_effector" in poses
    assert np.isfinite(poses["end_effector"].position).all()
    assert poses["end_effector"].position[2] > 0.5


def test_missing_required_field_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "missing_root.yaml"
    path.write_text("kind: custom\nname: bad_robot\nlinks: []\njoints: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="field 'root'"):
        load_robot_yaml(path)


def test_unsupported_kind_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "bad_kind.yaml"
    path.write_text("kind: octopus\nname: nope\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported robot kind 'octopus'"):
        load_robot_yaml(path)


def test_invalid_origin_format_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "bad_origin.yaml"
    path.write_text(
        "\n".join(
            [
                "kind: custom",
                "name: bad_origin",
                "root: base",
                "links:",
                "  - name: base",
                "    visual_extent: [0.1, 0.1, 0.1]",
                "  - name: tip",
                "    visual_extent: [0.1, 0.1, 0.1]",
                "joints:",
                "  - name: joint1",
                "    parent: base",
                "    child: tip",
                "    type: fixed",
                "    origin: nope",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="joints\\[0\\]\\.origin"):
        load_robot_yaml(path)


def test_humanoid_variant_end_effectors_are_merged(tmp_path: Path) -> None:
    path = tmp_path / "mini.yaml"
    path.write_text(
        "\n".join(
            [
                "kind: humanoid_variant",
                "name: mini_humanoid",
                "spec: {}",
                "end_effectors:",
                "  right: right_palm",
                "  left: left_palm",
            ]
        ),
        encoding="utf-8",
    )

    robot = load_robot_yaml(path)

    assert robot.end_effectors["right"] == "right_palm"
    assert robot.end_effectors["left"] == "left_palm"
    assert robot.end_effectors["right_palm"] == "right_palm"


def test_custom_robot_end_effectors_are_set_correctly(tmp_path: Path) -> None:
    robot = load_robot_yaml(_write_simple_arm_yaml(tmp_path))

    assert robot.end_effectors == {"tool": "end_effector"}


def test_load_robot_yaml_is_exported_from_optisim_robot() -> None:
    from optisim.robot import load_robot_yaml as exported

    assert exported is load_robot_yaml


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        ("run", ["run", "examples/pick_and_place.yaml", "--robot-spec", "builtin"]),
        ("sim", ["sim", "--robot-spec", "builtin"]),
        ("scenario", ["scenario", "examples/pick_and_place.yaml", "--robot-spec", "builtin"]),
        ("batch", ["batch", "--robot-spec", "builtin"]),
    ],
)
def test_cli_robot_spec_flag_is_accepted(command: str, argv: list[str]) -> None:
    args = build_parser().parse_args(argv)

    assert args.command == command
    assert args.robot_spec == "builtin"


def _write_simple_arm_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "simple_arm.yaml"
    path.write_text(
        "\n".join(
            [
                "kind: custom",
                "name: simple_arm",
                "root: base_link",
                "links:",
                "  - name: base_link",
                "    visual_extent: [0.1, 0.1, 0.05]",
                "  - name: link1",
                "    visual_extent: [0.05, 0.05, 0.3]",
                "  - name: link2",
                "    visual_extent: [0.05, 0.05, 0.25]",
                "  - name: end_effector",
                "    visual_extent: [0.04, 0.04, 0.06]",
                "joints:",
                "  - name: joint1",
                "    type: revolute",
                "    parent: base_link",
                "    child: link1",
                "    origin: [0.0, 0.0, 0.025]",
                "    axis: [0, 0, 1]",
                "    limit: [-3.14159, 3.14159]",
                "    velocity_limit: 2.0",
                "  - name: joint2",
                "    type: revolute",
                "    parent: link1",
                "    child: link2",
                "    origin: [0.0, 0.0, 0.3]",
                "    axis: [0, 1, 0]",
                "    limit: [-1.57, 1.57]",
                "  - name: joint3",
                "    type: fixed",
                "    parent: link2",
                "    child: end_effector",
                "    origin:",
                "      xyz: [0.0, 0.0, 0.25]",
                "      rpy: [0.0, 0.0, 0.0]",
                "end_effectors:",
                "  tool: end_effector",
            ]
        ),
        encoding="utf-8",
    )
    return path
