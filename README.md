# optisim

[![License](https://img.shields.io/badge/license-MIT-green)](#license)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#quick-start)
[![Humanoid](https://img.shields.io/badge/focus-humanoid_robotics-black)](#why-optisim)

`optisim` is a lightweight humanoid robot task planner and simulator for manipulation workflows, built for the gap between a static task spec and a full robotics stack. It gives you a pure-Python humanoid model, typed task primitives, deterministic task execution, inverse kinematics, and rich terminal visualization in a package that is small enough to modify quickly.

The built-in robot is a 31-DOF humanoid sized around a 1.75 m platform and intentionally aimed at the Tesla Optimus / general-purpose humanoid robotics workflow: rapid task authoring, kinematics validation, and simulation-driven iteration before a team commits to heavier control, perception, and motion-planning infrastructure.

## Why optisim?

Most robotics tooling is either:

- Too low-level: FK/IK and geometry without task semantics.
- Too heavy: large sim stacks when you only need to iterate on manipulation tasks.
- Too hardware-specific: tightly coupled to one robot, controller, or middleware stack.

`optisim` fills that middle layer:

- A concise task model for humanoid manipulation sequences.
- A built-in pure-Python humanoid you can inspect and change directly.
- Deterministic execution for repeatable tests and examples.
- Lightweight physical checks for reachability and coarse collisions.
- Terminal-first visualization that works over SSH and inside CI logs.

## Features

- [x] Jacobian-based damped least-squares inverse kinematics
- [x] Position-only and full pose IK targets
- [x] Joint limit enforcement during control and IK solving
- [x] Built-in 31-DOF humanoid robot model inspired by modern humanoid platforms
- [x] Deterministic step-based task execution engine
- [x] YAML/JSON task definitions and Python composition API
- [x] Rich terminal visualization with live status and progress
- [x] Runnable end-to-end examples for pick/place, pouring, door opening, and stacking
- [x] Pytest coverage for kinematics, IK, URDF loading, and task flow

## Architecture

```text
                          +----------------------+
                          | task.yaml / Python   |
                          +----------+-----------+
                                     |
                    +----------------+----------------+
                    | TaskDefinition / TaskComposer   |
                    +----------------+----------------+
                                     |
                              +------+------+
                              | Validator   |
                              +------+------+
                                     |
                           +---------+----------+
                           | Execution Engine   |
                           +----+----------+----+
                                |          |
                    +-----------+          +------------+
                    |                                   |
             +------+-------+                    +------+------+
             | Robot Model  |                    | World State |
             | FK / IK      |                    | Collisions  |
             +------+-------+                    +------+------+
                    |                                   |
                    +----------------+------------------+
                                     |
                              +------+------+
                              | Visualization|
                              | Rich / MPL   |
                              +-------------+
```

## Quick Start

Install it in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional matplotlib backend:

```bash
pip install -e .[viz]
```

Validate a bundled task:

```bash
python -m optisim validate examples/pour_water.yaml
```

Run a bundled example with the Rich terminal visualizer:

```bash
python -m optisim run examples/stack_blocks.yaml --visualize
```

Run the Python entry point directly:

```bash
python examples/open_door.py
```

Minimal API example:

```python
from optisim.core import ActionPrimitive, TaskComposer, TaskDefinition
from optisim.robot import build_humanoid_model
from optisim.sim import ExecutionEngine, WorldState

robot = build_humanoid_model()
world = WorldState.with_defaults()

composer = TaskComposer("pick_and_place")
composer.append(ActionPrimitive.reach(target="box", end_effector="right_palm"))
composer.append(ActionPrimitive.grasp(target="box", gripper="right_gripper"))
composer.append(ActionPrimitive.move(target="box", destination=[0.58, -0.20, 1.08]))
composer.append(ActionPrimitive.place(target="box", support="shelf"))

task = TaskDefinition.from_composer(composer)
engine = ExecutionEngine(robot=robot, world=world)

report = engine.validate(task)
assert report.is_valid, report.summary()
record = engine.run(task)
print(record)
```

## Built-In Humanoid

The built-in humanoid is defined in pure Python in [`optisim/robot/humanoid.py`](/root/.openclaw/workspace/optisim/optisim/robot/humanoid.py). It ships with:

- Torso: 3 DOF
- Head: 2 DOF
- Right arm: 7 DOF
- Left arm: 7 DOF
- Right leg: 6 DOF
- Left leg: 6 DOF

Total: 31 DOF

The link lengths and joint limits are tuned to feel plausible for a modern general-purpose humanoid rather than a toy manipulator.

## Examples

- [`examples/pick_and_place.yaml`](/root/.openclaw/workspace/optisim/examples/pick_and_place.yaml): reach, grasp, move, place to a shelf
- [`examples/pour_water.yaml`](/root/.openclaw/workspace/optisim/examples/pour_water.yaml): pick up a pitcher, align above a cup, tilt, and return
- [`examples/open_door.yaml`](/root/.openclaw/workspace/optisim/examples/open_door.yaml): reach handle, turn handle, pull door panel
- [`examples/stack_blocks.yaml`](/root/.openclaw/workspace/optisim/examples/stack_blocks.yaml): sequentially stack three blocks

## Development

Run the tests:

```bash
pytest -q
```

Smoke-test the CLI:

```bash
python -m optisim validate examples/pick_and_place.yaml
python -m optisim run examples/pick_and_place.yaml --visualize
```

## Contributing

Contributions should keep the project small, typed, and easy to reason about.

1. Open an issue or draft a PR describing the behavior change.
2. Add or update tests for any simulator, planner, IK, or robot-model change.
3. Keep public APIs documented and examples runnable end-to-end.
4. Prefer deterministic behavior over hidden randomness.

## License

MIT. See `pyproject.toml` for the package license declaration.
