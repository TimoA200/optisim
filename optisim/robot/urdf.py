"""URDF loading into the internal robot model."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from optisim.math3d import Pose
from optisim.robot.model import JointSpec, LinkSpec, RobotModel

__all__ = ["load_urdf"]


def load_urdf(path: str | Path) -> RobotModel:
    """Load a URDF file into the internal ``RobotModel`` representation."""

    source = Path(path)
    root = ET.fromstring(source.read_text(encoding="utf-8"))
    if root.tag != "robot":
        raise ValueError(f"{source} is not a URDF robot document")

    links = {
        element.attrib["name"]: LinkSpec(
            name=element.attrib["name"],
            visual_extent=_parse_visual_extent(element),
        )
        for element in root.findall("link")
    }
    joints: dict[str, JointSpec] = {}
    child_links: set[str] = set()

    for element in root.findall("joint"):
        name = element.attrib["name"]
        joint_type = element.attrib.get("type", "revolute")
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint '{name}' is missing parent/child elements")
        origin_node = element.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin_node is not None:
            xyz = _parse_float_triplet(origin_node.attrib.get("xyz", "0 0 0"))
            rpy = _parse_float_triplet(origin_node.attrib.get("rpy", "0 0 0"))
        axis_node = element.find("axis")
        axis = tuple(_parse_float_triplet(axis_node.attrib.get("xyz", "0 0 1"))) if axis_node is not None else (0.0, 0.0, 1.0)
        limit_node = element.find("limit")
        lower, upper, velocity = -3.14159, 3.14159, 2.0
        if limit_node is not None:
            lower = float(limit_node.attrib.get("lower", lower))
            upper = float(limit_node.attrib.get("upper", upper))
            velocity = float(limit_node.attrib.get("velocity", velocity))
        joints[name] = JointSpec(
            name=name,
            parent=parent.attrib["link"],
            child=child.attrib["link"],
            joint_type=joint_type,
            origin=Pose.from_xyz_rpy(xyz, rpy),
            axis=axis,
            limit_lower=lower,
            limit_upper=upper,
            velocity_limit=velocity,
        )
        child_links.add(child.attrib["link"])

    root_candidates = sorted(set(links) - child_links)
    if not root_candidates:
        raise ValueError("failed to infer URDF root link")

    end_effectors = _infer_end_effectors(links, joints, child_links)

    return RobotModel(
        name=root.attrib.get("name", source.stem),
        links=links,
        joints=joints,
        root_link=root_candidates[0],
        end_effectors=end_effectors,
    )


def _parse_float_triplet(raw: str) -> list[float]:
    return [float(value) for value in raw.split()]


def _parse_visual_extent(element: ET.Element) -> tuple[float, float, float]:
    geometry = element.find("./visual/geometry")
    if geometry is None:
        return (0.08, 0.08, 0.08)
    box = geometry.find("box")
    if box is not None:
        size = _parse_float_triplet(box.attrib.get("size", "0.08 0.08 0.08"))
        return (float(size[0]), float(size[1]), float(size[2]))
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        radius = float(cylinder.attrib.get("radius", "0.04"))
        length = float(cylinder.attrib.get("length", "0.08"))
        diameter = radius * 2.0
        return (diameter, diameter, length)
    sphere = geometry.find("sphere")
    if sphere is not None:
        diameter = float(sphere.attrib.get("radius", "0.04")) * 2.0
        return (diameter, diameter, diameter)
    return (0.08, 0.08, 0.08)


def _infer_end_effectors(
    links: dict[str, LinkSpec],
    joints: dict[str, JointSpec],
    child_links: set[str],
) -> dict[str, str]:
    preferred = {
        name: name
        for name in links
        if any(token in name for token in ("hand", "gripper", "palm", "tool", "wrist", "ee"))
    }
    if preferred:
        return preferred
    parent_links = {joint.parent for joint in joints.values()}
    leaf_links = [name for name in links if name in child_links and name not in parent_links]
    if leaf_links:
        return {name: name for name in sorted(leaf_links)}
    leaf_name = sorted(links)[-1]
    return {leaf_name: leaf_name}
