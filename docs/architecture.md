# Architecture

`optisim` is split into four subsystems:

1. `optisim.core`: task primitives, composition, serialization, and validation.
2. `optisim.robot`: robot model abstractions, URDF import, forward kinematics, and joint control.
3. `optisim.sim`: world state, collision checks, deterministic execution, and simulation stepping.
4. `optisim.viz`: thin visualization backends for terminal iteration or optional matplotlib inspection.

Design principles:

- Deterministic simulation first. The engine steps with a fixed `dt`.
- Explicit task semantics. Action sequences are human-readable and serializable.
- Kinematics before dynamics. This foundation is optimized for planning and plausibility checks.
- Easy extension. The internal model is compact enough to add IK, trajectory generation, or higher-fidelity physics later.
