# Quickstart

`optisim` is a lightweight humanoid task planner and simulator for quickly validating manipulation workflows without a heavyweight robotics stack.

## Installation

Install from PyPI:

```bash
python -m venv .venv
source .venv/bin/activate
pip install optisim
```

Install optional extras when you need them:

```bash
pip install "optisim[web,viz,rl]"
```

Install from source:

```bash
git clone https://github.com/TimoA200/optisim.git
cd optisim
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,web,viz,rl]"
```

## Your First Simulation

Five lines of code are enough to run a full pick-and-place scenario:

```python
from optisim.core import TaskDefinition
from optisim.robot import build_humanoid_model
from optisim.sim import ExecutionEngine, WorldState

task = TaskDefinition.from_file("examples/pick_and_place.yaml")
record = ExecutionEngine(build_humanoid_model(), WorldState.from_dict(task.world)).run(task)
```

The returned `record` contains executed actions, elapsed simulation time, detected collisions, and a portable recording for replay or analysis.

## Running Examples

Run the bundled YAML scenarios from the CLI:

```bash
python -m optisim validate examples/pick_and_place.yaml
python -m optisim run examples/pick_and_place.yaml
python -m optisim sim examples/stack_blocks.yaml --recording-out stack_blocks.json
```

Run examples directly from Python:

```bash
python examples/pick_and_place.py
python examples/open_door.py
python examples/pour_water.py
```

## Using the Web Visualizer

Install the `web` extra first:

```bash
pip install "optisim[web]"
```

Launch a task in the browser-backed visualizer:

```bash
python -m optisim sim examples/pick_and_place.yaml --web
```

Or instantiate it directly:

```python
from optisim.viz import WebVisualizer

visualizer = WebVisualizer(open_browser=False)
```

The web view streams robot link positions, object motion, active actions, and collision highlights over WebSockets.

## Writing Custom Tasks

You can author tasks as YAML or through the Python API.

Minimal YAML task:

```yaml
name: mug_transfer
world:
  objects:
    - name: mug
      pose:
        position: [0.42, -0.12, 0.81]
      size: [0.08, 0.08, 0.12]
  surfaces:
    - name: shelf
      pose:
        position: [0.60, -0.25, 1.02]
      size: [0.35, 0.25, 0.04]
actions:
  - type: reach
    target: mug
  - type: grasp
    target: mug
  - type: move
    target: mug
    destination: [0.58, -0.20, 1.08]
  - type: place
    target: mug
    support: shelf
```

Equivalent Python authoring flow:

```python
from optisim.core import TaskComposer, TaskDefinition

task = TaskDefinition.from_composer(
    TaskComposer("mug_transfer").pick_and_place(
        target="mug",
        pickup_effector="right_palm",
        destination=(0.58, -0.20, 1.08),
        support="shelf",
    )
)
```

For the full schema, see `docs/task_schema.md`.
