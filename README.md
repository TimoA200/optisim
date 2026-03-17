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
[![Tests](https://github.com/example/optisim/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/example/optisim/actions/workflows/ci.yml)

`optisim` is a lightweight humanoid robot task planner and simulator for teams that want to move faster than a full robotics stack allows. It sits in the practical middle ground between task specs and heavyweight sim/control infrastructure, giving you a 31-DOF humanoid, inverse kinematics, deterministic execution, and terminal-first visualization in a small pure-Python package shaped for the current wave of general-purpose humanoid robotics, including Tesla Optimus-style manipulation workflows.

## Features

- ✅ 31-DOF built-in humanoid model inspired by modern general-purpose robots
- ✅ Jacobian-based damped least-squares IK for reach and pose targets
- ✅ Deterministic step-based execution for reproducible task development
- ✅ YAML and Python task authoring paths for fast iteration
- ✅ Lightweight world modeling with objects, surfaces, and coarse collision checks
- ✅ RRT and RRT-Connect collision-aware motion planning with path smoothing
- ✅ Rich terminal visualization that works locally, over SSH, and in CI logs
- ✅ Optional matplotlib backend for quick spatial inspection
- ✅ Runnable end-to-end examples for pick/place, pouring, door opening, and stacking
- ✅ Typed Python API with PEP 561 marker support
- ✅ Pytest-backed CI on Python 3.11 and 3.12

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate && pip install -e .
python -m optisim run examples/stack_blocks.yaml --visualize
```

## Architecture

```text
 task.yaml / Python API
          |
          v
 +---------------------+
 | optisim.core        |
 | actions / composer  |
 | validation          |
 +----------+----------+
            |
            v
 +---------------------+      +---------------------+
 | optisim.robot       |----->| optisim.sim         |
 | humanoid / URDF / IK|      | execution / world   |
 +----------+----------+      +----------+----------+
            |                            |
            +-------------+--------------+
                          |
                          v
                 +------------------+
                 | optisim.viz      |
                 | rich / matplotlib|
                 +------------------+
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
