# Examples

Bundled examples cover the main manipulation flows currently modeled by `optisim`. Each scenario is available both as a YAML task file for CLI execution and as a small Python entry point.

## Included Scenarios

- `pick_and_place.yaml` / `pick_and_place.py`: reach for a box, grasp it, move it to a new location, and place it on a support surface.
- `pour_water.yaml` / `pour_water.py`: pick up a pitcher, move above a cup, rotate to simulate pouring, and return.
- `open_door.yaml` / `open_door.py`: reach a handle target, rotate it, and pull to model a door-opening interaction.
- `stack_blocks.yaml` / `stack_blocks.py`: execute multiple coordinated transfers to stack blocks in sequence.
- `urdf_demo.py`: load a bundled URDF robot, solve a reach target with IK, and animate the result in the terminal.

## Run an Example

```bash
python -m optisim run examples/stack_blocks.yaml --visualize
```

Or use the Python scripts directly:

```bash
python examples/stack_blocks.py
python examples/urdf_demo.py
```
