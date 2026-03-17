"""URDF loading into the internal robot model."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from optisim.math3d import Pose
from optisim.robot.model import JointSpec, LinkSpec, RobotModel


def load_urdf(path: str | Path) -> RobotModel:
    """Load a URDF file into the internal ``RobotModel`` representation."""

    source = Path(path)
    root = ET.fromstring(source.read_text(encoding="utf-8"))
    if root.tag != "robot":
        raise ValueError(f"{source} is not a URDF robot document")

    links = {
        element.attrib["name"]: LinkSpec(name=element.attrib["name"])
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
            xyz = [float(v) for v in origin_node.attrib.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin_node.attrib.get("rpy", "0 0 0").split()]
        axis_node = element.find("axis")
        axis = tuple(float(v) for v in axis_node.attrib.get("xyz", "0 0 1").split()) if axis_node is not None else (0.0, 0.0, 1.0)
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

    end_effectors = {
        name: name for name in links if name.endswith("hand") or name.endswith("gripper") or name.endswith("palm")
    }
    if not end_effectors:
        end_effectors = {root_candidates[-1]: root_candidates[-1]}

    return RobotModel(
        name=root.attrib.get("name", source.stem),
        links=links,
        joints=joints,
        root_link=root_candidates[0],
        end_effectors=end_effectors,
    )
