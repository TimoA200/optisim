"""REST API server for optisim."""

from __future__ import annotations

import time
from importlib import metadata
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

try:
    from optisim.lfd import Demonstration, DemonstrationRecorder, DynamicMovementPrimitive
    from optisim.robot import RobotModel, build_humanoid_model
    from optisim.tamp import GeometricChecker, HouseholdDomain, PlanningState, Predicate, SymbolicPlanner, TAMPPlanner

    _IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised only in broken environments
    Demonstration = Any  # type: ignore[assignment]
    DemonstrationRecorder = Any  # type: ignore[assignment]
    DynamicMovementPrimitive = Any  # type: ignore[assignment]
    RobotModel = Any  # type: ignore[assignment]
    build_humanoid_model = None  # type: ignore[assignment]
    GeometricChecker = Any  # type: ignore[assignment]
    HouseholdDomain = Any  # type: ignore[assignment]
    PlanningState = Any  # type: ignore[assignment]
    Predicate = Any  # type: ignore[assignment]
    SymbolicPlanner = Any  # type: ignore[assignment]
    TAMPPlanner = Any  # type: ignore[assignment]
    _IMPORT_ERROR = str(exc)


def _resolve_version() -> str:
    try:
        return metadata.version("optisim")
    except metadata.PackageNotFoundError:
        return "0.7.0"


VERSION = _resolve_version()
DEFAULT_EFFECTOR = "right_palm"
DEMO_STORAGE: dict[str, Demonstration] = {}


class HealthResponse(BaseModel):
    status: str
    version: str


class RobotPreset(BaseModel):
    id: str
    name: str
    dof: int
    description: str


class SimulationRequest(BaseModel):
    task_name: str
    joint_positions: list[float] | None = None
    target_pose: list[float] | None = None
    num_steps: int = Field(default=50, ge=2, le=500)
    include_dynamics: bool = True
    include_safety: bool = True


class SimulationResult(BaseModel):
    success: bool
    num_steps: int
    joint_trajectory: list[list[float]]
    ee_trajectory: list[list[float]]
    energy_profile: list[float]
    safety_violations: list[str]
    execution_time_ms: float


class PredicateModel(BaseModel):
    name: str
    args: list[str]
    value: bool = True


class PlanRequest(BaseModel):
    initial_predicates: list[PredicateModel]
    goal_predicates: list[PredicateModel]
    object_poses: dict[str, list[float]]
    objects: list[str]


class PlanStep(BaseModel):
    operator: str
    parameters: list[str]
    geometry: dict[str, Any]


class PlanResponse(BaseModel):
    feasible: bool
    steps: list[PlanStep]
    num_steps: int


class LFDRecordRequest(BaseModel):
    task_name: str
    num_steps: int = Field(default=30, ge=2, le=500)
    start: list[float]
    end: list[float]


class LFDRecordResponse(BaseModel):
    demo_id: str
    task_name: str
    num_steps: int
    duration: float
    joint_dim: int


class LFDGenerateRequest(BaseModel):
    demo_id: str
    new_goal: list[float]


class LFDGenerateResponse(BaseModel):
    success: bool
    trajectory: list[list[float]]
    num_steps: int


app = FastAPI(title="optisim API", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_runtime() -> None:
    if _IMPORT_ERROR is not None:
        raise HTTPException(status_code=503, detail=f"optisim runtime unavailable: {_IMPORT_ERROR}")


def _build_robot() -> RobotModel:
    _require_runtime()
    assert build_humanoid_model is not None
    return build_humanoid_model()


def _validate_vec3(name: str, values: list[float] | None) -> list[float]:
    if values is None:
        raise HTTPException(status_code=422, detail=f"{name} is required")
    if len(values) != 3:
        raise HTTPException(status_code=422, detail=f"{name} must contain exactly 3 floats")
    return [float(value) for value in values]


def _validate_joint_vector(robot: RobotModel, positions: list[float] | None) -> np.ndarray:
    joint_names = list(robot.joints)
    if positions is None:
        return np.array([robot.joint_positions[name] for name in joint_names], dtype=float)
    if len(positions) != len(joint_names):
        raise HTTPException(status_code=422, detail=f"joint_positions must contain {len(joint_names)} floats")
    clamped = [robot.joints[name].clamp(float(value)) for name, value in zip(joint_names, positions, strict=True)]
    return np.array(clamped, dtype=float)


def _effector_name(robot: RobotModel) -> str:
    if DEFAULT_EFFECTOR in robot.end_effectors:
        return DEFAULT_EFFECTOR
    return next(iter(robot.end_effectors))


def _joint_map(robot: RobotModel, values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(robot.joints, values, strict=True)}


def _current_ee_position(robot: RobotModel, joints: np.ndarray) -> np.ndarray:
    pose = robot.end_effector_pose(_effector_name(robot), _joint_map(robot, joints))
    return pose.position.astype(float)


def _simulation_target(robot: RobotModel, start_joints: np.ndarray, target_pose: list[float] | None) -> np.ndarray:
    start_ee = _current_ee_position(robot, start_joints)
    if target_pose is None:
        return start_ee + np.array([0.18, -0.04, 0.08], dtype=float)
    target = np.array(_validate_vec3("target_pose", target_pose), dtype=float)
    if float(np.linalg.norm(target)) > robot.max_reach() * 1.5:
        raise HTTPException(status_code=422, detail="target_pose is outside the synthetic workspace")
    return target


def _synthetic_joint_goal(robot: RobotModel, start_joints: np.ndarray, target: np.ndarray) -> np.ndarray:
    reach_scale = float(np.clip(np.linalg.norm(target), 0.0, robot.max_reach()))
    target_vector = np.zeros_like(start_joints)
    usable = min(6, len(target_vector))
    if usable >= 1:
        target_vector[0] = 0.35 * target[1]
    if usable >= 2:
        target_vector[1] = -0.25 * target[1]
    if usable >= 3:
        target_vector[2] = 0.15 * target[2]
    if usable >= 4:
        target_vector[3] = 0.45 * target[0]
    if usable >= 5:
        target_vector[4] = 0.55 * target[2]
    if usable >= 6:
        target_vector[5] = 0.30 * reach_scale
    for index, name in enumerate(robot.joints):
        target_vector[index] = robot.joints[name].clamp(float(start_joints[index] + target_vector[index]))
    return target_vector


def _run_synthetic_simulation(payload: SimulationRequest) -> SimulationResult:
    robot = _build_robot()
    start_ts = time.perf_counter()
    start_joints = _validate_joint_vector(robot, payload.joint_positions)
    target = _simulation_target(robot, start_joints, payload.target_pose)
    goal_joints = _synthetic_joint_goal(robot, start_joints, target)

    joint_trajectory: list[list[float]] = []
    ee_trajectory: list[list[float]] = []
    energy_profile: list[float] = []

    steps = max(payload.num_steps, 2)
    for step in range(steps):
        phase = 1.0 if steps == 1 else step / (steps - 1)
        smooth_phase = 3.0 * phase**2 - 2.0 * phase**3
        joints = start_joints + smooth_phase * (goal_joints - start_joints)
        joint_trajectory.append([float(value) for value in joints])
        ee_position = _current_ee_position(robot, joints)
        blended_ee = (1.0 - smooth_phase) * ee_position + smooth_phase * target
        ee_trajectory.append([float(value) for value in blended_ee])
        delta = goal_joints - start_joints
        base_energy = float(np.linalg.norm(delta) ** 2 * (0.3 + smooth_phase))
        energy_profile.append(base_energy if payload.include_dynamics else 0.0)

    safety_violations: list[str] = []
    if payload.include_safety:
        if float(np.linalg.norm(target)) > robot.max_reach():
            safety_violations.append("target_near_workspace_limit")
        for index, name in enumerate(robot.joints):
            joint = robot.joints[name]
            value = goal_joints[index]
            if abs(value - joint.limit_lower) < 0.05 or abs(joint.limit_upper - value) < 0.05:
                safety_violations.append(f"joint_limit_margin:{name}")
                break

    execution_time_ms = (time.perf_counter() - start_ts) * 1000.0
    return SimulationResult(
        success=len(joint_trajectory) == steps,
        num_steps=steps,
        joint_trajectory=joint_trajectory,
        ee_trajectory=ee_trajectory,
        energy_profile=energy_profile,
        safety_violations=safety_violations,
        execution_time_ms=round(execution_time_ms, 3),
    )


def _predicate_from_model(model: PredicateModel) -> Predicate:
    return Predicate(name=model.name, args=list(model.args), value=model.value)


def _validate_object_poses(object_poses: dict[str, list[float]]) -> dict[str, list[float]]:
    normalized: dict[str, list[float]] = {}
    for name, pose in object_poses.items():
        if len(pose) != 3:
            raise HTTPException(status_code=422, detail=f"object pose for '{name}' must contain 3 floats")
        normalized[name] = [float(value) for value in pose]
    return normalized


def _plan(payload: PlanRequest) -> PlanResponse:
    _require_runtime()
    initial_state = PlanningState(
        predicates={_predicate_from_model(item) for item in payload.initial_predicates},
        objects=list(payload.objects),
    )
    goals = [_predicate_from_model(item) for item in payload.goal_predicates]
    object_poses = _validate_object_poses(payload.object_poses)
    operators, _ = HouseholdDomain.build()
    planner = TAMPPlanner(SymbolicPlanner(operators, max_depth=8), GeometricChecker(_build_robot()))
    result = planner.plan(initial_state, goals, object_poses)
    if result is None:
        return PlanResponse(feasible=False, steps=[], num_steps=0)
    return PlanResponse(
        feasible=result.feasible,
        steps=[
            PlanStep(operator=operator.name, parameters=list(operator.parameters), geometry=geometry)
            for operator, geometry in zip(result.operators, result.geometric_params, strict=True)
        ],
        num_steps=result.num_steps,
    )


def _record_demo(payload: LFDRecordRequest) -> LFDRecordResponse:
    _require_runtime()
    if len(payload.start) != len(payload.end):
        raise HTTPException(status_code=422, detail="start and end must have the same dimensionality")
    if len(payload.start) == 0:
        raise HTTPException(status_code=422, detail="start and end must not be empty")

    recorder = DemonstrationRecorder(task_name=payload.task_name)
    start = np.array(payload.start, dtype=float)
    end = np.array(payload.end, dtype=float)
    for step in range(payload.num_steps):
        phase = 1.0 if payload.num_steps == 1 else step / (payload.num_steps - 1)
        joints = start + phase * (end - start)
        ee_pose = tuple(float(value) for value in joints[:3])
        recorder.record(step, joints.tolist(), ee_pose, gripper_open=step < payload.num_steps - 3)

    demo_id = str(uuid4())
    DEMO_STORAGE[demo_id] = recorder.demonstration
    return LFDRecordResponse(
        demo_id=demo_id,
        task_name=payload.task_name,
        num_steps=recorder.num_steps,
        duration=recorder.duration,
        joint_dim=recorder.joint_dim,
    )


def _generate_demo(payload: LFDGenerateRequest) -> LFDGenerateResponse:
    _require_runtime()
    demonstration = DEMO_STORAGE.get(payload.demo_id)
    if demonstration is None:
        raise HTTPException(status_code=404, detail=f"demo '{payload.demo_id}' not found")
    if len(payload.new_goal) != demonstration.joint_dim:
        raise HTTPException(status_code=422, detail="new_goal dimensionality does not match the stored demonstration")

    dmp = DynamicMovementPrimitive()
    dmp.train(demonstration)
    generated = dmp.generate(goal=[float(value) for value in payload.new_goal])
    trajectory = [list(step.joint_positions) for step in generated.steps]
    return LFDGenerateResponse(success=True, trajectory=trajectory, num_steps=generated.num_steps)


def _dashboard_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>optisim — Humanoid Robot Simulator</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0a0f17;
      --bg-soft: #111926;
      --card: rgba(18, 27, 41, 0.92);
      --card-border: rgba(118, 146, 179, 0.18);
      --text: #e7edf5;
      --muted: #96a6ba;
      --accent: #5cc8ff;
      --accent-soft: rgba(92, 200, 255, 0.14);
      --success: #73f5b0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      background:
        radial-gradient(circle at top, rgba(92, 200, 255, 0.14), transparent 32%),
        linear-gradient(180deg, #09111b 0%, #060a11 100%);
      color: var(--text);
      min-height: 100vh;
    }
    .shell {
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 40px;
    }
    .hero {
      padding: 24px 0 28px;
    }
    .eyebrow {
      display: inline-block;
      padding: 6px 10px;
      border: 1px solid var(--card-border);
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    h1 {
      margin: 16px 0 10px;
      font-size: clamp(2rem, 5vw, 3.4rem);
      line-height: 1.05;
      letter-spacing: -0.04em;
    }
    .subtitle {
      margin: 0;
      max-width: 700px;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.6;
    }
    .grid {
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      margin-top: 26px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
      backdrop-filter: blur(14px);
    }
    .card h2 {
      margin: 0 0 6px;
      font-size: 1.1rem;
    }
    .card p {
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.5;
      font-size: 0.94rem;
    }
    label {
      display: block;
      margin: 0 0 12px;
      font-size: 0.85rem;
      color: var(--muted);
    }
    input, textarea, button {
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(150, 166, 186, 0.18);
      background: rgba(8, 13, 21, 0.85);
      color: var(--text);
      padding: 12px 13px;
      font: inherit;
    }
    textarea {
      min-height: 104px;
      resize: vertical;
    }
    button {
      margin-top: 6px;
      border: none;
      background: linear-gradient(135deg, #5cc8ff, #3b82f6);
      color: #04111d;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { filter: brightness(1.05); }
    pre {
      margin: 22px 0 0;
      padding: 18px;
      min-height: 220px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      border-radius: 16px;
      border: 1px solid var(--card-border);
      background: rgba(5, 10, 17, 0.96);
      color: var(--success);
    }
    @media (max-width: 640px) {
      .shell { padding: 20px 14px 28px; }
      .card { padding: 16px; }
      pre { min-height: 180px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <span class="eyebrow">REST Dashboard</span>
      <h1>optisim — Humanoid Robot Simulator</h1>
      <p class="subtitle">Open-source task planner &amp; simulator for humanoid robots</p>
    </section>
    <section class="grid">
      <article class="card">
        <h2>Simulate</h2>
        <p>Generate a synthetic full-body motion and inspect the resulting trajectory profile.</p>
        <form id="simulate-form">
          <label>Task name<input name="task_name" value="reach_and_grasp" /></label>
          <label>Target pose JSON<textarea name="target_pose">[0.55, -0.1, 1.1]</textarea></label>
          <label>Number of steps<input name="num_steps" type="number" min="2" value="24" /></label>
          <button type="submit">Run Simulation</button>
        </form>
      </article>
      <article class="card">
        <h2>Plan</h2>
        <p>Run the household TAMP planner on a minimal symbolic manipulation scenario.</p>
        <form id="plan-form">
          <label>Objects CSV<input name="objects" value="robot,cup,home,table,counter" /></label>
          <label>Object poses JSON<textarea name="object_poses">{"robot":[0,0,0],"home":[0,0,0],"table":[1,0,0],"counter":[1.2,0.8,0.9],"cup":[1,0,0.8]}</textarea></label>
          <button type="submit">Run Planner</button>
        </form>
      </article>
      <article class="card">
        <h2>LfD</h2>
        <p>Record a demonstration, store it in memory, and create a retargeted trajectory on demand.</p>
        <form id="lfd-form">
          <label>Task name<input name="task_name" value="demo_reach" /></label>
          <label>Start joints JSON<textarea name="start">[0.0, 0.2, 0.4]</textarea></label>
          <label>End joints JSON<textarea name="end">[0.7, 0.9, 1.1]</textarea></label>
          <button type="submit">Record Demonstration</button>
        </form>
      </article>
    </section>
    <pre id="output">Ready.</pre>
  </main>
  <script>
    const output = document.getElementById("output");

    async function postJson(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const text = await response.text();
      let body = text;
      try { body = JSON.parse(text); } catch (_) {}
      output.textContent = JSON.stringify({ status: response.status, body }, null, 2);
      return body;
    }

    document.getElementById("simulate-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = new FormData(event.target);
      await postJson("/simulate", {
        task_name: form.get("task_name"),
        target_pose: JSON.parse(form.get("target_pose")),
        num_steps: Number(form.get("num_steps")),
        include_dynamics: true,
        include_safety: true
      });
    });

    document.getElementById("plan-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = new FormData(event.target);
      await postJson("/plan", {
        initial_predicates: [
          { name: "at", args: ["robot", "home"], value: true },
          { name: "object-at", args: ["cup", "table"], value: true },
          { name: "handempty", args: ["robot"], value: true }
        ],
        goal_predicates: [
          { name: "object-at", args: ["cup", "counter"], value: true }
        ],
        object_poses: JSON.parse(form.get("object_poses")),
        objects: String(form.get("objects")).split(",").map((item) => item.trim()).filter(Boolean)
      });
    });

    document.getElementById("lfd-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const form = new FormData(event.target);
      const record = await postJson("/lfd/record", {
        task_name: form.get("task_name"),
        num_steps: 20,
        start: JSON.parse(form.get("start")),
        end: JSON.parse(form.get("end"))
      });
      if (record && record.demo_id) {
        await postJson("/lfd/generate", {
          demo_id: record.demo_id,
          new_goal: [1.0, 1.2, 1.35]
        });
      }
    });
  </script>
</body>
</html>"""


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=VERSION)


@app.get("/robots", response_model=list[RobotPreset])
def robots() -> list[RobotPreset]:
    presets = [
        RobotPreset(
            id="demo_humanoid",
            name="Demo Humanoid",
            dof=31,
            description="Built-in 31-DOF humanoid preset for simulation and planning demos.",
        ),
        RobotPreset(
            id="custom",
            name="Custom Robot",
            dof=0,
            description="Placeholder preset for user-supplied robot models and task-specific configurations.",
        ),
    ]
    return presets


@app.post("/simulate", response_model=SimulationResult)
def simulate(payload: SimulationRequest) -> SimulationResult:
    return _run_synthetic_simulation(payload)


@app.post("/plan", response_model=PlanResponse)
def plan(payload: PlanRequest) -> PlanResponse:
    return _plan(payload)


@app.post("/lfd/record", response_model=LFDRecordResponse)
def lfd_record(payload: LFDRecordRequest) -> LFDRecordResponse:
    return _record_demo(payload)


@app.post("/lfd/generate", response_model=LFDGenerateResponse)
def lfd_generate(payload: LFDGenerateRequest) -> LFDGenerateResponse:
    return _generate_demo(payload)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(content=_dashboard_html())


def run_server() -> None:
    import uvicorn

    uvicorn.run("optisim.server:app", host="0.0.0.0", port=8080, reload=False)
