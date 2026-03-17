# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [0.7.0] - 2026-03-17

### Added

- Added `optisim.server`: a FastAPI-based REST API with health, robot preset listing, synthetic simulation, TAMP planning, LfD recording/generation, and an embedded dashboard.
- Added `optisim-server` as a console script entry point for serving the API via Uvicorn on port 8080.
- Added API-focused pytest coverage for server endpoints, validation failures, and dashboard delivery.

### Changed

- Kept `optisim.server` out of `optisim.__init__` so importing the package does not require optional web dependencies.
- Expanded the `web` optional dependency group to include `httpx` for FastAPI `TestClient` support.

## [0.6.0] - 2026-03-17

### Added

- Added `optisim.tamp`: lightweight task and motion planning utilities with symbolic predicates, PDDL-style operators, BFS planning, geometric feasibility checks, and a bundled household manipulation domain.
- Added `TAMPPlanner` and `TAMPPlan` for integrating symbolic search with simple geometric backtracking over candidate plans.
- Added `examples/tamp_demo.py` demonstrating a household cup transfer from table to counter.
- Added pytest coverage for symbolic planning, geometric validation, household-domain operator integrity, and end-to-end TAMP planning.

## [0.5.0] - 2026-03-17

### Added

- Added `optisim.lfd`: demonstration recording, JSON serialization, DMP-based trajectory learning, replay, interpolation, and goal adaptation utilities.
- Added a bundled `examples/lfd_demo.py` workflow showing recording, DMP training, and retargeted generation.
- Added pytest coverage for demonstration recording, persistence, library management, DMP generation, replay, interpolation, and edge cases.

## [0.4.0] - 2026-03-17

### Added

- Added `optisim.mpc`: linear inverted-pendulum MPC for humanoid CoM balance and locomotion.
- Added `FootstepPlanner` and `FootstepPlan` helpers for simple alternating humanoid walking patterns.
- Added `optisim.estimation`: EKF-based humanoid state estimation with IMU, encoder, contact, and vision updates.
- Added `StateEstimationPipeline`, `RobotState`, and an IMU dead-reckoning utility plus a bundled estimation demo.
- Added a bundled `examples/mpc_balance.py` example and MPC-focused pytest coverage.
- Added `optisim.bimanual`: coordinated dual-arm task planning, cooperative wrench sharing, and common bimanual task presets.
- Added `optisim.perception`: depth-camera point-cloud processing, lightweight surface/object detection, pose estimation, ICP refinement, and grasp-target conversion.

## [0.3.0] - 2026-03-17

### Added

- Added `optisim.wbc`: whole-body control with hierarchical null-space task stacking.
- Added `optisim.trajopt`: trajectory optimization with cubic splines and velocity/acceleration profiles.
- Added `optisim.reactive`: reactive manipulation controller with contact-phase FSM and sensor-driven velocity scaling.

### Changed

- Test count updated to 347 passing.

## [0.2.0] - 2026-03-17

### Added

- Built-in 31-DOF humanoid robot model with FK, damped-least-squares IK, and URDF loading support.
- Deterministic step-based simulation engine with task validation, execution recording, and replay.
- Task authoring APIs for action primitives, fluent composition, YAML/JSON task definitions, and a built-in task library.
- Motion-planning utilities including RRT, RRT-Connect, and path smoothing.
- Behavior-tree execution with YAML-defined structured task logic.
- Grasp planning, contact analysis, force-closure checks, and reusable gripper presets.
- Multi-robot coordination, dependency-aware task scheduling, and shared-world execution.
- Safety monitoring, sensor simulation, lightweight dynamics analysis, analytics, and Gymnasium integration.
- Terminal, matplotlib, and web visualization backends.
- Expanded integration coverage for end-to-end execution, CLI flows, IK/FK consistency, and recording replay.

### Changed

- Improved task-file parsing and schema validation so malformed YAML/JSON surfaces actionable errors.
- Improved IK failure reporting to distinguish reachability, joint-limit, and solver-stall cases.
- Standardized optional dependency guidance for terminal and web visualization imports.
- Refreshed packaging metadata, optional dependency groups, and public-release documentation.

## [0.1.0] - 2026-03-17

### Added

- Initial public alpha release of `optisim` with core task planning, humanoid simulation, and CLI support.
