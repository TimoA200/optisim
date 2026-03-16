# optisim

[![CI Placeholder](https://img.shields.io/badge/ci-pending-lightgrey)](#)
[![PyPI Placeholder](https://img.shields.io/badge/pypi-unreleased-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#)

> `[ logo placeholder ]`

Lightweight humanoid robot task planning and simulation for Python. `optisim` focuses on the first layer robotics teams need before they commit to a heavyweight stack: explicit task models, physically-aware validation, deterministic stepping, and a robot model that can be loaded from URDF or instantiated from a built-in humanoid.

## Why

`optisim` is designed for fast iteration on manipulation tasks:

- Compose tasks from reusable action primitives.
- Validate sequence semantics and coarse physical plausibility before execution.
- Run a deterministic step-based simulation with collision checks.
- Load URDF robots or use the built-in demo humanoid model.
- Visualize state in the terminal or with optional matplotlib.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional visualization extras:

```bash
pip install -e .[viz]
```

## Quick Start

Validate the bundled demo task:

```bash
optisim validate examples/pick_and_place.yaml
```

Run it with terminal visualization:

```bash
optisim run examples/pick_and_place.yaml --visualize
```

Use the Python API:

```python
from optisim.core import ActionPrimitive, TaskComposer, TaskDefinition
from optisim.robot import build_demo_humanoid
from optisim.sim import ExecutionEngine, WorldState

robot = build_demo_humanoid()
world = WorldState.with_defaults()
composer = TaskComposer("pick_and_place")
composer.append(ActionPrimitive.reach(target="box", end_effector="right_palm"))
composer.append(ActionPrimitive.grasp(target="box", gripper="right_gripper"))
composer.append(ActionPrimitive.move(target="box", destination=[0.55, -0.20, 0.90]))
composer.append(ActionPrimitive.place(target="box", support="table"))
task = TaskDefinition.from_composer(composer)

engine = ExecutionEngine(robot=robot, world=world)
report = engine.validate(task)
assert report.is_valid, report.errors
engine.run(task)
```

## Architecture

```text
                   +-----------------------+
                   |  task.yaml / JSON     |
                   +-----------+-----------+
                               |
                     TaskDefinition Loader
                               |
               +---------------+----------------+
               |                                |
        +------+-------+                +-------+------+
        | Task Composer |                | Task Validator|
        +------+-------+                +-------+------+
               |                                |
               +---------------+----------------+
                               |
                      Execution Engine
                               |
        +----------------------+----------------------+
        |                                             |
   Robot Model / FK                           World + Collision
        |                                             |
        +----------------------+----------------------+
                               |
                          Visualization
```

## Package Layout

```text
optisim/
  core/     task primitives, composition, validation, task IO
  robot/    URDF loading, robot model, kinematics, joint control
  sim/      world state, collision detection, execution engine
  viz/      terminal and matplotlib visualization
examples/   runnable demos and task files
tests/      unit and integration coverage
docs/       architecture and schema notes
```

## Development

```bash
pytest
python -m optisim validate examples/pick_and_place.yaml
python -m optisim run examples/pick_and_place.yaml --visualize
```

## Status

This repository is the initial foundation: deterministic, typed, testable, and intentionally compact enough to evolve into richer planning, control, and perception modules.
