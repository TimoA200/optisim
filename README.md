#   ____        __  _      _
#  / __ \____  / /_(_)____(_)___ ___
# / / / / __ \/ __/ / ___/ / __ `__ \
#/ /_/ / /_/ / /_/ (__  ) / / / / / /
#\____/ .___/\__/_/____/_/_/ /_/ /_/
#    /_/
#
# Humanoid task planning and lightweight simulation in pure Python.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Tests](https://github.com/TimoA200/optisim/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/TimoA200/optisim/actions/workflows/ci.yml)

`optisim` is a lightweight humanoid robot task planner and simulator for teams that want to move faster than a full robotics stack allows. It sits in the practical middle ground between task specs and heavyweight sim/control infrastructure, giving you a 31-DOF humanoid, inverse kinematics, deterministic execution, motion planning, behavior trees, grasp analysis, safety monitoring, sensor simulation, and terminal-first visualization in a small pure-Python package shaped for the current wave of general-purpose humanoid robotics, including Tesla Optimus-style manipulation workflows. The package currently ships with 12 built-in task templates and 231 tests.

## Features

- ✅ `optisim.core`: task authoring with `ActionPrimitive`, `TaskComposer`, `TaskDefinition`, and validation utilities
- ✅ `optisim.robot`: built-in 31-DOF humanoid model, URDF loading, and Jacobian IK
- ✅ `optisim.sim`: deterministic step-based execution engine, world state management, and recording/replay
- ✅ `optisim.viz`: terminal-first visualization plus optional matplotlib inspection
- ✅ `optisim.planning`: RRT and RRT-Connect motion planning with path smoothing
- ✅ `optisim.behavior`: behavior tree execution with YAML loading for structured task logic
- ✅ `optisim.dynamics`: rigid-body dynamics, energy analysis, and joint/payload/workspace constraints
- ✅ `optisim.grasp`: contact-based grasp planning, force-closure checks, friction-cone checks, and gripper presets
- ✅ `optisim.multi`: shared-world multi-robot fleet coordination with dependency-aware task scheduling
- ✅ `optisim.library`: 12 built-in humanoid task templates for rapid scenario bootstrapping
- ✅ `optisim.analytics`: trajectory metrics, run comparison, and profiling helpers
- ✅ `optisim.gym_env`: OpenAI Gymnasium environment wrapper for RL-style experimentation
- ✅ `optisim.safety`: safety zones, joint limits, emergency stop handling, and `SafetyConfig` humanoid presets
- ✅ `optisim.sensors`: force/torque, proximity, encoder, IMU, and depth-camera simulation with configurable noise models
- ✅ Typed pure-Python API with PEP 561 marker support and a pytest-backed CI suite covering 231 tests

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate && pip install -e .
python -m optisim run examples/stack_blocks.yaml --visualize
python -m optisim multi examples/multi_robot_warehouse.yaml
```

## Architecture

```text
 task.yaml / behavior.yaml / Python API
                 |
                 v
        +----------------------+
        | optisim.core         |
        | authoring / validate |
        +----+-----------+-----+
             |           |
     +-------+           +-------------------+
     |                                       |
     v                                       v
 +----------------------+         +----------------------+
 | optisim.library      |         | optisim.behavior     |
 | built-in templates   |         | BT exec / YAML load  |
 +----------+-----------+         +----------+-----------+
            |                                |
            +----------------+---------------+
                             |
                             v
                    +----------------------+
                    | task graph / intents |
                    +----------+-----------+
                               |
      +------------------------+-------------------------+
      |                        |                         |
      v                        v                         v
 +------------------+  +----------------------+  +----------------------+
 | optisim.robot    |  | optisim.planning     |  | optisim.grasp        |
 | humanoid / URDF  |  | RRT / RRT-Connect    |  | contacts / closure   |
 | Jacobian IK      |  | smoothing            |  | friction / grippers  |
 +---------+--------+  +----------+-----------+  +----------+-----------+
           |                      |                         |
           +-----------+----------+-------------------------+
                       |
                       v
               +----------------------+
               | optisim.sim          |
               | engine / world / log |
               | recording / replay   |
               +---+---------+--------+
                   |         |        |
          +--------+         |        +--------------------+
          |                  |                             |
          v                  v                             v
 +------------------+ +----------------------+   +----------------------+
 | optisim.dynamics | | optisim.safety       |   | optisim.sensors      |
 | rigid body /     | | zones / limits /     |   | FT / proximity /     |
 | energy /         | | estop / presets      |   | encoders / IMU /     |
 | constraints      | |                      |   | depth / noise / suite|
 +--------+---------+ +----------+-----------+   +----------+-----------+
          |                      |                          |
          +----------------------+--------------------------+
                                 |
                                 v
                        +----------------------+
                        | validated execution  |
                        +---+-------------+-------------+------+
                            |             |             |
             +--------------+             |             +------------------+
             |                            |                                |
             v                            v                                v
 +----------------------+                         +----------------------+
 | optisim.multi        |                         | optisim.gym_env      |
 | fleet / shared world |                         | Gymnasium wrapper    |
 | dependency schedule  |                         | RL-style interface   |
 +----------+-----------+                         +----------------------+
            |
            v
 +----------------------+   +----------------------+
 | optisim.analytics    |   | optisim.viz          |
 | metrics / compare /  |   | terminal / matplotlib|
 | profiling            |   |                      |
 +----------------------+   +----------------------+
```

## Why optisim?

Most robotics tools force a bad tradeoff. Low-level kinematics libraries are too bare to express manipulation tasks, while full simulation stacks are expensive to learn, slow to modify, and often overkill when the real question is, "Can this humanoid plausibly do this task sequence?" `optisim` fills that gap with a small, inspectable, typed codebase for validating humanoid task logic before you commit to middleware, planners, perception pipelines, or robot-specific control code.

That matters because humanoid robotics is now moving from research demos toward iterative product engineering. In that phase, teams need something lighter than a full digital twin but more concrete than slides, spreadsheets, or unvalidated YAML. `optisim` is built for that middle layer.

## Examples

See [examples/README.md](/root/.openclaw/workspace/optisim/examples/README.md) for a full list of bundled scenarios.

- `pick_and_place`: reach, grasp, move, and place an object on a support surface
- `pour_water`: manipulate a pitcher and perform a tilt-style pouring sequence
- `open_door`: reach, rotate, and pull a door interaction target
- `stack_blocks`: chain multiple object transfers into a simple assembly task
- `multi_robot_warehouse`: two humanoids sort warehouse parcels with dependency-aware fleet coordination

## Task Library

`optisim` now includes a built-in task library so you can start from realistic humanoid manipulation templates instead of authoring YAML from scratch.

```bash
python -m optisim library list
python -m optisim library run pick_and_place --param object=mug --param surface=shelf --visualize
python -m optisim library export multi_room_delivery --output task.yaml
```

| Template | Difficulty |
| --- | --- |
| `pick_and_place` | beginner |
| `table_clearing` | beginner |
| `dish_loading` | beginner |
| `laundry_folding` | intermediate |
| `shelf_stocking` | intermediate |
| `drawer_open_close` | intermediate |
| `assembly_line_pick` | beginner |
| `pallet_stacking` | intermediate |
| `bin_sorting` | beginner |
| `tool_handover` | intermediate |
| `walk_and_pickup` | intermediate |
| `multi_room_delivery` | advanced |

## Safety

`optisim.safety` adds zone monitoring, joint-limit enforcement, emergency stop handling, and reusable humanoid defaults.

```python
import numpy as np

from optisim.safety import EmergencyStop, SafetyConfig, SafetyMonitor, SafetyZone, ZoneType

config = SafetyConfig.default_humanoid()
monitor = SafetyMonitor(zones=config.zones, joint_limits=config.joint_limits)
estop = EmergencyStop()

monitor.add_zone(
    SafetyZone(
        name="operator_cell",
        center=np.array([0.8, 0.0, 1.0]),
        half_extents=np.array([0.3, 0.4, 0.8]),
        zone_type=ZoneType.FORBIDDEN,
    )
)

link_positions = {
    "left_hand": np.array([0.82, 0.0, 1.05]),
    "torso": np.array([0.0, 0.0, 1.2]),
}

violations = monitor.check_positions("optimus_alpha", link_positions)

if violations:
    print(monitor.summarize_violations(violations))
    estop.check_and_raise(violations)
```

## Sensor Simulation

`optisim.sensors` provides configurable sensor models for contact, proprioception, inertial sensing, and depth perception.

```python
import numpy as np

from optisim.sensors import ForceTorqueSensor, IMUSensor, SensorSuite

suite = SensorSuite.default_humanoid_suite()

ft = suite.get_sensor("left_wrist_ft")
imu = suite.get_sensor("torso_imu")

assert isinstance(ft, ForceTorqueSensor)
assert isinstance(imu, IMUSensor)

wrist_wrench = ft.read([12.0, -3.5, 48.0, 0.2, 0.0, -0.1])
imu_reading = imu.read(
    linear_accel_sensor_frame=np.array([0.15, 0.0, 0.05]),
    angular_velocity_sensor_frame=np.array([0.0, 0.1, -0.02]),
)

print("force/torque:", wrist_wrench)
print("imu accel:", imu_reading["accel"])
print("imu gyro:", imu_reading["gyro"])
```

## Development

```bash
pip install -e .[dev]
pytest -q
```

GitHub Actions runs the same test suite on pushes and pull requests to `main`.

## Contributing

Contributions are expected to keep the project lightweight, deterministic, typed, and easy to inspect. Start with [CONTRIBUTING.md](/root/.openclaw/workspace/optisim/CONTRIBUTING.md), open an issue or draft PR for behavioral changes, and include tests for simulator, IK, task, or robot-model work.

## License

`optisim` is released under the MIT License. See [LICENSE](/root/.openclaw/workspace/optisim/LICENSE).
