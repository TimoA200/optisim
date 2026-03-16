from __future__ import annotations

from pathlib import Path

from optisim.robot import load_urdf


def test_load_urdf(tmp_path: Path) -> None:
    urdf = tmp_path / "mini.urdf"
    urdf.write_text(
        """
<robot name="mini">
  <link name="base"/>
  <link name="tool"/>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="tool"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" velocity="2.0"/>
  </joint>
</robot>
""".strip(),
        encoding="utf-8",
    )
    robot = load_urdf(urdf)
    assert robot.root_link == "base"
    assert "joint1" in robot.joints
    assert robot.end_effectors
