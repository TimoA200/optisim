# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [0.27.0] - 2026-03-18

### Added

- Added `optisim.faultdetect`: `FaultCode`, `FaultEvent`, `JointMonitor`, `RobotFaultMonitor`, and `FaultHistory`.
- Added joint-level torque, velocity, position, thermal, stall, and sensor-dropout fault detection for real-time health monitoring.
- Added pytest coverage for fault event validation, joint/robot fault aggregation, bounded history queries, package exports, and version metadata.

## [0.26.0] - 2026-03-17

### Added

- Added `optisim.recorder`: `TelemetryFrame`, `EpisodeRecorder`, `EpisodeReplay`, and `TelemetryStats`.
- Added telemetry frame serialization, bounded episode recording, interpolated replay queries, and aggregate contact/joint statistics.
- Added pytest coverage for recorder serialization, capacity handling, interpolation behavior, replay indexing, package exports, and version metadata.

## [0.25.0] - 2026-03-17

### Added

- Added `optisim.gait`: `GaitPhase`, `LegCycle`, `GaitPattern`, `CPGOscillator`, and `GaitController`.
- Added gait-cycle timing helpers for bipedal walking/running and quadrupedal trotting patterns.
- Added pytest coverage for gait phase timing, controller contact states, oscillator stepping, and top-level package exports.

## [0.24.0] - 2026-03-17

### Added

- Added `optisim.balance`: `COMState`, `ZMPCalculator`, `SupportPolygon`, `BalanceMonitor`, and `BalanceReport`.
- Added whole-body balance helpers for CoM state tracking, convex support polygon construction, ZMP evaluation, and balance-history monitoring.
- Added pytest coverage for ZMP calculation, support polygon geometry, stability margins, monitor history behavior, and package exports.

## [0.23.0] - 2026-03-17

### Added

- Added `optisim.pathplan`: `GridNode`, `AStarPlanner`, `WaypointSmoother`, and `RoadmapPlanner`.
- Added voxel-grid A* planning, greedy line-of-sight waypoint pruning, and 3D probabilistic-roadmap planning against `VoxelGrid` occupancy maps.
- Added pytest coverage for path reconstruction, heuristic behavior, line-of-sight edge cases, roadmap building, package exports, and version metadata.

## [0.22.0] - 2026-03-17

### Added

- Added `optisim.occupancy`: `VoxelGrid`, `OccupancyUpdater`, `CollisionChecker`, and `OccupancyStats`.
- Added voxel/world coordinate transforms, occupancy/free-space updates, point-cloud export, and occupancy statistics helpers.
- Added pytest coverage for occupancy mapping, ray clearing, collision checks, statistics, package exports, and version metadata.

## [0.21.0] - 2026-03-17

### Added

- Added `optisim.energy`: `JointPowerModel`, `MotorEfficiencyModel`, `EnergyEstimator`, `EnergyBudget`, and `TaskEnergyProfile`.
- Added torque-speed efficiency interpolation, cumulative energy budgeting, and task-level power profiling helpers.
- Added pytest coverage for energy estimation, efficiency lookup behavior, budgeting, and profile integration edge cases.

## [0.20.0] - 2026-03-17

### Added

- Added `optisim.terrain`: `HeightMap`, `TerrainPatch`, `TerrainAnalyzer`, `TerrainAdaptiveFootstep`, and `TerrainCostMap`.
- Added bilinear terrain sampling, slope and normal estimation, flat and step region analysis, and terrain-aware footstep height adjustment.
- Added terrain module pytest coverage for terrain representation, analysis, traversability, and planning cost behavior.

## [0.15.0] - 2026-03-17

### Added

- Added `optisim.contact`: pure-Python contact mechanics helpers with sphere-sphere, box-sphere, and AABB contact generation.
- Added `ContactForceModel`, `ContactParams`, and `ContactWorld` for spring-damper normal forces, Coulomb friction, and small contact-world stepping.
- Added pytest coverage for contact geometry, contact-force computation, and contact-world integration behavior.

## [0.14.0] - 2026-03-17

### Added

- Added `optisim.worldmodel`: `StateEncoder`, `WorldModelNet`, `WorldModelTrainer`, `ModelPredictivePlanner`, and `WorldModelCollector`.
- Added lightweight learned transition prediction using a pure-numpy MLP with backpropagation.
- Added random-shooting model-predictive planning over primitive sequences.

## [0.13.0] - 2026-03-17

### Added

- Added `optisim.curriculum`: `TaskScheduler`, `CurriculumTrainer`, and curriculum callbacks for logging, history capture, and early stopping.
- Added promotion and demotion logic for progressive task-difficulty scheduling.

## [0.12.0] - 2026-03-17

### Added

- Added `optisim.export`: JSON, CSV, mocap CSV, ROS2-bag JSON, URDF annotation, and Markdown export workflows.
- Added export support for trajectories, scenes, and benchmark results.

## [0.11.0] - 2026-03-17

### Added

- Added `optisim.benchmark`: benchmark suite with 8 built-in tasks, `BenchmarkEvaluator`, and `BenchmarkReporter`.
- Added benchmark execution support to the CLI.

## [0.10.0] - 2026-03-17

### Added

- Added `optisim.primitives`: `Reach`, `Grasp`, `Place`, `Push`, `Handover`, and `Navigate` motion primitives.
- Added `PrimitiveExecutor` with scene-effect application for semantic action execution.

## [0.9.0] - 2026-03-17

### Added

- Added `optisim.scene`: `SceneNode`, `SceneRelation`, `SceneGraph`, `SceneBuilder`, and `SceneQuery`.
- Added semantic scene-graph integration with TAMP workflows.

## [0.8.0] - 2026-03-17

### Added

- Added `optisim.rl`: PPO reinforcement learning with `ActorCritic`, GAE `RolloutBuffer`, and a Gym wrapper.
- Added finite-difference gradient support for pure-numpy RL optimization.

## [0.7.0] - 2026-03-17

### Added

- Added `optisim.policy`: neural behavioral cloning with a pure-numpy MLP and Adam optimizer.
- Added stateless and recurrent policy executors.

## [0.6.0] - 2026-03-17

### Added

- Added `optisim.tamp`: symbolic task planner using BFS, geometric feasibility checks, and `HouseholdDomain`.

## [0.5.0] - 2026-03-17

### Added

- Added `optisim.lfd`: Learning from Demonstration with DMP support.
- Added `DemonstrationRecorder`, `DemonstrationLibrary`, and `DemonstrationPlayer`.

## [0.4.0] - 2026-03-17

### Added

- Added behavior trees, dynamics, grasp analysis, whole-body control, safety monitoring, and sensor simulation.

## [0.3.0] - 2026-03-17

### Added

- Added motion planning with RRT, RRT-Connect, and path smoothing.
- Added trajectory optimization with cubic splines.

## [0.2.0] - 2026-03-17

### Added

- Added a 31-DOF humanoid robot model, URDF loading, Jacobian IK, simulation engine, and visualization support.

## [0.1.0] - 2026-03-17

### Added

- Initial release with core task authoring via `ActionPrimitive`, `TaskComposer`, and `TaskDefinition`.
