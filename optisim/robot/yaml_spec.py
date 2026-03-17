"""YAML robot specification loading into the internal robot model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import pi
from pathlib import Path
from typing import Any

import yaml
from yaml import YAMLError

from optisim.math3d import Pose
from optisim.robot.humanoid import HumanoidSpec, build_humanoid_model
from optisim.robot.model import JointSpec, LinkSpec, RobotModel

__all__ = ["load_robot_yaml"]


@dataclass(slots=True)
class _CustomLinkDefinition:
    name: str
    visual_extent: tuple[float, float, float]


@dataclass(slots=True)
class _CustomJointDefinition:
    name: str
    parent: str
    child: str
    joint_type: str
    origin: Pose
    axis: tuple[float, float, float]
    limit_lower: float
    limit_upper: float
    velocity_limit: float


def load_robot_yaml(path: str | Path) -> RobotModel:
    """Load a robot YAML spec file into a ``RobotModel``."""

    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"failed to read robot YAML {source}: {exc}") from exc
    try:
        payload = yaml.safe_load(raw)
    except YAMLError as exc:
        raise ValueError(f"invalid robot YAML in {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"robot YAML {source} must contain a mapping")

    kind = _require_non_empty_string(payload, "kind", source)
    name = _require_non_empty_string(payload, "name", source)
    if kind == "humanoid_variant":
        return _load_humanoid_variant(name=name, payload=payload, source=source)
    if kind == "custom":
        return _load_custom_robot(name=name, payload=payload, source=source)
    raise ValueError(f"unsupported robot kind '{kind}' in {source}")


def _load_humanoid_variant(*, name: str, payload: Mapping[str, Any], source: Path) -> RobotModel:
    raw_spec = payload.get("spec", {})
    if not isinstance(raw_spec, Mapping):
        raise ValueError(f"field 'spec' in {source} must be a mapping")

    robot = build_humanoid_model(_build_humanoid_spec(raw_spec, source))
    robot.name = name
    robot.end_effectors = _parse_end_effectors(
        payload.get("end_effectors"),
        links=robot.links,
        source=source,
        base=robot.end_effectors,
    )
    return robot


def _load_custom_robot(*, name: str, payload: Mapping[str, Any], source: Path) -> RobotModel:
    root_link = _require_non_empty_string(payload, "root", source)
    raw_links = payload.get("links")
    raw_joints = payload.get("joints")
    if not isinstance(raw_links, list) or not raw_links:
        raise ValueError(f"field 'links' in {source} must be a non-empty list")
    if not isinstance(raw_joints, list):
        raise ValueError(f"field 'joints' in {source} must be a list")

    parsed_links = [_parse_custom_link(item, index, source) for index, item in enumerate(raw_links)]
    links: dict[str, LinkSpec] = {}
    for link in parsed_links:
        if link.name in links:
            raise ValueError(f"duplicate link '{link.name}' in {source}")
        links[link.name] = LinkSpec(name=link.name, visual_extent=link.visual_extent)
    if root_link not in links:
        raise ValueError(f"root link '{root_link}' in {source} is not defined in 'links'")

    parsed_joints = [_parse_custom_joint(item, index, source) for index, item in enumerate(raw_joints)]
    joints: dict[str, JointSpec] = {}
    child_links: set[str] = set()
    for joint in parsed_joints:
        if joint.name in joints:
            raise ValueError(f"duplicate joint '{joint.name}' in {source}")
        if joint.parent not in links:
            raise ValueError(f"joint '{joint.name}' in {source} references unknown parent link '{joint.parent}'")
        if joint.child not in links:
            raise ValueError(f"joint '{joint.name}' in {source} references unknown child link '{joint.child}'")
        if joint.child in child_links:
            raise ValueError(f"link '{joint.child}' in {source} is the child of more than one joint")
        child_links.add(joint.child)
        joints[joint.name] = JointSpec(
            name=joint.name,
            parent=joint.parent,
            child=joint.child,
            joint_type=joint.joint_type,
            origin=joint.origin,
            axis=joint.axis,
            limit_lower=joint.limit_lower,
            limit_upper=joint.limit_upper,
            velocity_limit=joint.velocity_limit,
        )
        links[joint.child].parent_joint = joint.name

    if root_link in child_links:
        raise ValueError(f"root link '{root_link}' in {source} cannot be the child of a joint")

    raw_end_effectors = payload.get("end_effectors")
    end_effectors = _parse_end_effectors(
        raw_end_effectors,
        links=links,
        source=source,
        base=None if raw_end_effectors is not None else _infer_custom_end_effectors(links, joints, root_link),
    )
    return RobotModel(
        name=name,
        links=links,
        joints=joints,
        root_link=root_link,
        end_effectors=end_effectors,
    )


def _build_humanoid_spec(overrides: Mapping[str, Any], source: Path) -> HumanoidSpec:
    defaults = HumanoidSpec()
    allowed_fields = {field.name for field in fields(HumanoidSpec)}
    unknown_fields = sorted(set(overrides) - allowed_fields)
    if unknown_fields:
        joined = ", ".join(unknown_fields)
        raise ValueError(f"unknown humanoid spec field(s) in {source}: {joined}")

    scale = 1.0
    if "height_m" in overrides:
        scale = _coerce_positive_float(overrides["height_m"], field_name="spec.height_m", source=source) / defaults.height_m

    values: dict[str, float] = {}
    for field in fields(HumanoidSpec):
        field_name = field.name
        if field_name in overrides:
            values[field_name] = _coerce_positive_float(overrides[field_name], field_name=f"spec.{field_name}", source=source)
            continue
        default_value = getattr(defaults, field_name)
        values[field_name] = float(default_value if field_name == "height_m" else default_value * scale)
    return HumanoidSpec(**values)


def _parse_custom_link(item: Any, index: int, source: Path) -> _CustomLinkDefinition:
    if not isinstance(item, Mapping):
        raise ValueError(f"links[{index}] in {source} must be a mapping")
    name = _require_non_empty_string(item, "name", source, prefix=f"links[{index}]")
    visual_extent = _parse_triplet(
        item.get("visual_extent"),
        field_name=f"links[{index}].visual_extent",
        source=source,
    )
    return _CustomLinkDefinition(name=name, visual_extent=visual_extent)


def _parse_custom_joint(item: Any, index: int, source: Path) -> _CustomJointDefinition:
    if not isinstance(item, Mapping):
        raise ValueError(f"joints[{index}] in {source} must be a mapping")

    name = _require_non_empty_string(item, "name", source, prefix=f"joints[{index}]")
    joint_type = item.get("type", "revolute")
    if joint_type not in {"revolute", "prismatic", "fixed"}:
        raise ValueError(f"joints[{index}].type in {source} must be one of revolute, prismatic, or fixed")

    parent = _require_non_empty_string(item, "parent", source, prefix=f"joints[{index}]")
    child = _require_non_empty_string(item, "child", source, prefix=f"joints[{index}]")
    origin = _parse_origin(item.get("origin", [0.0, 0.0, 0.0]), field_name=f"joints[{index}].origin", source=source)
    axis = _parse_triplet(item.get("axis", [0.0, 0.0, 1.0]), field_name=f"joints[{index}].axis", source=source)
    limit_lower, limit_upper = _parse_limit(item.get("limit", [-pi, pi]), field_name=f"joints[{index}].limit", source=source)
    velocity_limit = _coerce_float(item.get("velocity_limit", 2.0), field_name=f"joints[{index}].velocity_limit", source=source)
    return _CustomJointDefinition(
        name=name,
        parent=parent,
        child=child,
        joint_type=joint_type,
        origin=origin,
        axis=axis,
        limit_lower=limit_lower,
        limit_upper=limit_upper,
        velocity_limit=velocity_limit,
    )


def _parse_end_effectors(
    raw: Any,
    *,
    links: Mapping[str, LinkSpec],
    source: Path,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    end_effectors = dict(base or {})
    if raw is None:
        return end_effectors
    if not isinstance(raw, Mapping):
        raise ValueError(f"field 'end_effectors' in {source} must be a mapping")
    for effector_name, link_name in raw.items():
        if not isinstance(effector_name, str) or not effector_name.strip():
            raise ValueError(f"end effector names in {source} must be non-empty strings")
        if not isinstance(link_name, str) or not link_name.strip():
            raise ValueError(f"end_effector '{effector_name}' in {source} must reference a non-empty link name")
        if link_name not in links:
            raise ValueError(f"end_effector '{effector_name}' in {source} references unknown link '{link_name}'")
        end_effectors[effector_name] = link_name
    return end_effectors


def _infer_custom_end_effectors(
    links: Mapping[str, LinkSpec],
    joints: Mapping[str, JointSpec],
    root_link: str,
) -> dict[str, str]:
    parent_links = {joint.parent for joint in joints.values()}
    leaf_links = [name for name in links if name != root_link and name not in parent_links]
    if leaf_links:
        return {name: name for name in sorted(leaf_links)}
    return {root_link: root_link}


def _parse_origin(raw: Any, *, field_name: str, source: Path) -> Pose:
    if isinstance(raw, list):
        xyz = _parse_triplet(raw, field_name=field_name, source=source)
        return Pose.from_xyz_rpy(xyz, [0.0, 0.0, 0.0])
    if isinstance(raw, Mapping):
        xyz = _parse_triplet(raw.get("xyz", [0.0, 0.0, 0.0]), field_name=f"{field_name}.xyz", source=source)
        rpy = _parse_triplet(raw.get("rpy", [0.0, 0.0, 0.0]), field_name=f"{field_name}.rpy", source=source)
        return Pose.from_xyz_rpy(xyz, rpy)
    raise ValueError(f"{field_name} in {source} must be [x, y, z] or a mapping with xyz/rpy")


def _parse_limit(raw: Any, *, field_name: str, source: Path) -> tuple[float, float]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"{field_name} in {source} must be a two-item list [lower, upper]")
    lower = _coerce_float(raw[0], field_name=f"{field_name}[0]", source=source)
    upper = _coerce_float(raw[1], field_name=f"{field_name}[1]", source=source)
    return (lower, upper)


def _parse_triplet(raw: Any, *, field_name: str, source: Path) -> tuple[float, float, float]:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{field_name} in {source} must be a three-item list")
    return tuple(_coerce_float(value, field_name=field_name, source=source) for value in raw)


def _require_non_empty_string(
    payload: Mapping[str, Any],
    key: str,
    source: Path,
    *,
    prefix: str | None = None,
) -> str:
    value = payload.get(key)
    label = f"{prefix}.{key}" if prefix else key
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"field '{label}' in {source} must be a non-empty string")
    return value


def _coerce_float(value: Any, *, field_name: str, source: Path) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} in {source} must be a number") from exc


def _coerce_positive_float(value: Any, *, field_name: str, source: Path) -> float:
    number = _coerce_float(value, field_name=field_name, source=source)
    if number <= 0.0:
        raise ValueError(f"{field_name} in {source} must be greater than 0")
    return number
