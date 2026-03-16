# Task Schema

Supported task document fields:

```yaml
name: task_name
metadata: {}
robot:
  model: demo_humanoid
  urdf: path/to/robot.urdf
world:
  gravity: [0.0, 0.0, -9.81]
  surfaces:
    - name: table
      pose:
        position: [0.5, 0.0, 0.75]
        rpy: [0.0, 0.0, 0.0]
      size: [0.8, 0.5, 0.05]
  objects:
    - name: box
      pose:
        position: [0.4, -0.1, 0.8]
        rpy: [0.0, 0.0, 0.0]
      size: [0.08, 0.08, 0.12]
      mass_kg: 0.7
actions:
  - type: reach
    target: box
    end_effector: right_palm
  - type: grasp
    target: box
    end_effector: right_gripper
  - type: move
    target: box
    destination: [0.6, -0.2, 1.0]
  - type: place
    target: box
    support: shelf
```

Supported action types:

- `reach`
- `grasp`
- `move`
- `place`
- `push`
- `pull`
- `rotate`
