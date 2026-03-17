"""Web-based 3D visualization served via FastAPI and WebSockets."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from typing import Any

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without web deps installed.
    raise ModuleNotFoundError(
        "Web visualization requires the optional web dependencies. "
        "Install with `pip install optisim[web]`."
    ) from exc

from optisim.core.action_primitives import ActionPrimitive
from optisim.core.task_definition import TaskDefinition
from optisim.robot.model import RobotModel
from optisim.sim.collision import Collision
from optisim.sim.world import WorldState


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@dataclass
class WebVisualizer:
    """Three.js-backed visualizer with live websocket streaming."""

    host: str = "127.0.0.1"
    port: int | None = None
    open_browser: bool = True
    title: str = "optisim web viz"
    _app: FastAPI | None = field(default=None, init=False, repr=False)
    _server: uvicorn.Server | None = field(default=None, init=False, repr=False)
    _server_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _event_loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _clients: set[WebSocket] = field(default_factory=set, init=False, repr=False)
    _clients_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _frames: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _previous_positions: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _collision_names: set[str] = field(default_factory=set, init=False, repr=False)
    _current_action: str = field(default="idle", init=False, repr=False)
    _task_name: str = field(default="simulation", init=False, repr=False)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port or 0}"

    def start_task(self, task: TaskDefinition, world: WorldState, robot: RobotModel) -> None:
        """Boot the server and seed the initial robot metadata."""

        self._ensure_server()
        self._task_name = task.name
        self._previous_positions = dict(robot.joint_positions)
        self._frames.clear()
        self._collision_names.clear()
        self._current_action = "idle"
        self._metadata = {
            "task_name": task.name,
            "robot_name": robot.name,
            "connections": [[joint.parent, joint.child] for joint in robot.joints.values()],
            "joint_to_link": {joint.name: joint.child for joint in robot.joints.values()},
            "link_names": list(robot.links),
            "end_effectors": dict(robot.end_effectors),
        }
        self._publish({"type": "reset", "metadata": self._metadata, "frames": []})

    def start_action(self, action: ActionPrimitive, *, index: int, total_actions: int) -> None:
        """Track the active action label."""

        self._current_action = f"{index}/{total_actions}: {action.action_type.value} {action.target}"

    def update_collisions(self, collisions: list[Collision]) -> None:
        """Update names currently involved in collisions."""

        self._collision_names = {item.entity_a for item in collisions} | {item.entity_b for item in collisions}

    def render(self, world: WorldState, robot: RobotModel) -> None:
        """Convert the current runtime state into a websocket frame."""

        poses = robot.forward_kinematics()
        moving_joints = [
            name
            for name, value in robot.joint_positions.items()
            if abs(value - self._previous_positions.get(name, value)) > 1e-3
        ]
        frame = {
            "time_s": float(world.time_s),
            "active_action": self._current_action,
            "joint_positions": {name: float(value) for name, value in robot.joint_positions.items()},
            "moving_joints": moving_joints,
            "moving_links": [robot.joints[name].child for name in moving_joints if name in robot.joints],
            "link_positions": {
                name: [float(component) for component in pose.position.tolist()] for name, pose in poses.items()
            },
            "objects": {
                name: {
                    "position": [float(value) for value in obj.pose.position.tolist()],
                    "size": [float(value) for value in obj.size],
                    "held_by": obj.held_by,
                    "collision": name in self._collision_names,
                }
                for name, obj in world.objects.items()
            },
            "surfaces": {
                name: {
                    "position": [float(value) for value in surface.pose.position.tolist()],
                    "size": [float(value) for value in surface.size],
                    "collision": name in self._collision_names,
                }
                for name, surface in world.surfaces.items()
            },
            "collision_names": sorted(self._collision_names),
        }
        self._frames.append(frame)
        self._publish({"type": "frame", "metadata": self._metadata, "frame": frame})
        self._previous_positions = dict(robot.joint_positions)

    def finish(
        self,
        task: TaskDefinition,
        world: WorldState,
        robot: RobotModel,
        collisions: list[Collision],
    ) -> None:
        """Mark the stream complete and keep the page available for replay."""

        self._current_action = f"complete: {task.name}"
        self._publish(
            {
                "type": "complete",
                "metadata": self._metadata,
                "frame_count": len(self._frames),
                "collisions": [
                    {
                        "entity_a": collision.entity_a,
                        "entity_b": collision.entity_b,
                        "penetration_depth": float(collision.penetration_depth),
                    }
                    for collision in collisions
                ],
            }
        )

    def block(self) -> None:
        """Keep the server available until the user interrupts the process."""

        try:
            while True:
                time.sleep(0.25)
        except KeyboardInterrupt:
            self.close()

    def close(self) -> None:
        """Stop the background Uvicorn server."""

        if self._server is not None:
            self._server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
        self._server = None
        self._server_thread = None
        self._event_loop = None

    def _ensure_server(self) -> None:
        if self._server_thread is not None and self._server_thread.is_alive():
            return

        self.port = self.port or _find_free_port(self.host)
        app = FastAPI(title=self.title)
        self._app = app

        @app.on_event("startup")
        async def _startup() -> None:
            self._event_loop = asyncio.get_running_loop()

        @app.get("/", response_class=HTMLResponse)
        async def _index() -> HTMLResponse:
            return HTMLResponse(self._html())

        @app.websocket("/ws")
        async def _ws(websocket: WebSocket) -> None:
            await websocket.accept()
            with self._clients_lock:
                self._clients.add(websocket)
            await websocket.send_json({"type": "reset", "metadata": self._metadata, "frames": self._frames})
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                with self._clients_lock:
                    self._clients.discard(websocket)

        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server_thread = threading.Thread(target=self._server.run, name="optisim-web-viz", daemon=True)
        self._server_thread.start()

        deadline = time.time() + 5.0
        while (self._event_loop is None or not self._server.started) and time.time() < deadline:
            time.sleep(0.05)

        if self.open_browser:
            with contextlib.suppress(Exception):
                webbrowser.open(self.url)

    def _publish(self, payload: dict[str, Any]) -> None:
        if self._event_loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._event_loop)

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        with self._clients_lock:
            clients = list(self._clients)
        if not clients:
            return
        stale: list[WebSocket] = []
        for websocket in clients:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)
        if stale:
            with self._clients_lock:
                for websocket in stale:
                    self._clients.discard(websocket)

    def _html(self) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{self.title}</title>
  <style>
    :root {{
      --bg: #09111a;
      --panel: rgba(9, 17, 26, 0.82);
      --edge: rgba(131, 148, 166, 0.24);
      --text: #f1f5f9;
      --muted: #94a3b8;
      --green: #22c55e;
      --red: #ef4444;
      --yellow: #facc15;
      --blue: #38bdf8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      overflow: hidden;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(34, 197, 94, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(56, 189, 248, 0.18), transparent 22%),
        linear-gradient(180deg, #0f172a 0%, #020617 100%);
    }}
    #scene {{ width: 100vw; height: 100vh; }}
    .hud {{
      position: fixed;
      top: 18px;
      left: 18px;
      width: min(420px, calc(100vw - 36px));
      padding: 16px 18px;
      border: 1px solid var(--edge);
      border-radius: 18px;
      backdrop-filter: blur(14px);
      background: var(--panel);
      box-shadow: 0 24px 80px rgba(2, 6, 23, 0.45);
    }}
    .title {{ font-size: 18px; font-weight: 700; letter-spacing: 0.02em; }}
    .meta {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
    .controls {{
      display: flex;
      gap: 10px;
      margin-top: 14px;
      align-items: center;
      flex-wrap: wrap;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
      color: #020617;
      background: linear-gradient(135deg, #e2e8f0, #cbd5e1);
      font-weight: 700;
    }}
    .active {{ background: linear-gradient(135deg, #bbf7d0, #4ade80); }}
    input[type="range"] {{ flex: 1 1 180px; accent-color: var(--green); }}
    .legend {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-top: 14px;
      font-size: 12px;
      color: var(--muted);
    }}
    .chip {{
      display: flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }}
    .footer {{
      margin-top: 12px;
      font-size: 12px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }}
  </style>
</head>
<body>
  <div id="scene"></div>
  <div class="hud">
    <div class="title">optisim live 3D</div>
    <div class="meta" id="status">Waiting for frames...</div>
    <div class="controls">
      <button id="play" class="active">Play</button>
      <button id="pause">Pause</button>
      <button id="step">Step</button>
      <input id="scrub" type="range" min="0" max="0" value="0" />
    </div>
    <div class="legend">
      <div class="chip"><span class="swatch" style="background: var(--green)"></span>moving</div>
      <div class="chip"><span class="swatch" style="background: var(--red)"></span>collision</div>
      <div class="chip"><span class="swatch" style="background: var(--yellow)"></span>grasped</div>
      <div class="chip"><span class="swatch" style="background: var(--blue)"></span>idle</div>
    </div>
    <div class="footer">
      <span id="frameCounter">frame 0 / 0</span>
      <span id="timeLabel">t = 0.00s</span>
    </div>
  </div>
  <script type="module">
    import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
    import {{ OrbitControls }} from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07111d);

    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 100);
    camera.position.set(2.2, 1.8, 1.6);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById("scene").appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0.0, 0.0, 0.9);

    const hemi = new THREE.HemisphereLight(0xe2e8f0, 0x0f172a, 1.3);
    scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.1);
    dir.position.set(3.0, 5.0, 6.0);
    scene.add(dir);

    const grid = new THREE.GridHelper(4, 20, 0x334155, 0x1e293b);
    grid.position.z = 0;
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    const robotGroup = new THREE.Group();
    const objectGroup = new THREE.Group();
    scene.add(robotGroup);
    scene.add(objectGroup);

    const status = document.getElementById("status");
    const timeLabel = document.getElementById("timeLabel");
    const frameCounter = document.getElementById("frameCounter");
    const scrub = document.getElementById("scrub");
    const playBtn = document.getElementById("play");
    const pauseBtn = document.getElementById("pause");
    const stepBtn = document.getElementById("step");

    let metadata = {{ connections: [], joint_to_link: {{}}, link_names: [] }};
    let frames = [];
    let currentIndex = 0;
    let playing = true;
    let completed = false;
    let lastAdvanceMs = 0;
    let frameStepMs = 50;

    playBtn.addEventListener("click", () => {{
      playing = true;
      playBtn.classList.add("active");
      pauseBtn.classList.remove("active");
    }});
    pauseBtn.addEventListener("click", () => {{
      playing = false;
      pauseBtn.classList.add("active");
      playBtn.classList.remove("active");
    }});
    stepBtn.addEventListener("click", () => {{
      playing = false;
      pauseBtn.classList.add("active");
      playBtn.classList.remove("active");
      if (frames.length > 0) {{
        currentIndex = Math.min(currentIndex + 1, frames.length - 1);
        renderFrame(frames[currentIndex]);
      }}
    }});
    scrub.addEventListener("input", (event) => {{
      if (frames.length === 0) return;
      currentIndex = Number(event.target.value);
      renderFrame(frames[currentIndex]);
    }});

    function clearGroup(group) {{
      while (group.children.length > 0) {{
        const child = group.children.pop();
        group.remove(child);
      }}
    }}

    function addLine(start, end, color) {{
      const points = [
        new THREE.Vector3(start[0], start[1], start[2]),
        new THREE.Vector3(end[0], end[1], end[2]),
      ];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({{ color, linewidth: 3 }});
      robotGroup.add(new THREE.Line(geometry, material));
    }}

    function addSphere(position, color, radius) {{
      const geometry = new THREE.SphereGeometry(radius, 18, 18);
      const material = new THREE.MeshStandardMaterial({{
        color,
        emissive: color,
        emissiveIntensity: 0.16,
        roughness: 0.32,
        metalness: 0.18,
      }});
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(position[0], position[1], position[2]);
      robotGroup.add(mesh);
    }}

    function addBox(position, size, color, opacity = 0.92) {{
      const geometry = new THREE.BoxGeometry(size[0], size[1], size[2]);
      const material = new THREE.MeshStandardMaterial({{
        color,
        transparent: opacity < 1.0,
        opacity,
        roughness: 0.7,
        metalness: 0.08,
      }});
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(position[0], position[1], position[2]);
      objectGroup.add(mesh);
    }}

    function updateLabels(frame) {{
      status.textContent = `${{metadata.task_name || "simulation"}} | ${{frame.active_action || "idle"}}`;
      timeLabel.textContent = `t = ${{frame.time_s.toFixed(2)}}s`;
      frameCounter.textContent = `frame ${{currentIndex + 1}} / ${{frames.length}}`;
      scrub.max = String(Math.max(frames.length - 1, 0));
      scrub.value = String(currentIndex);
    }}

    function renderFrame(frame) {{
      if (!frame) return;
      clearGroup(robotGroup);
      clearGroup(objectGroup);

      const movingLinks = new Set(frame.moving_links || []);
      const collisionNames = new Set(frame.collision_names || []);
      const linkPositions = frame.link_positions || {{}};

      metadata.connections.forEach(([parent, child]) => {{
        if (!(parent in linkPositions) || !(child in linkPositions)) return;
        const color = movingLinks.has(child) ? 0x22c55e : 0x38bdf8;
        addLine(linkPositions[parent], linkPositions[child], color);
      }});

      Object.entries(linkPositions).forEach(([name, position]) => {{
        let color = 0xe2e8f0;
        if (movingLinks.has(name)) color = 0x22c55e;
        if (collisionNames.has(name)) color = 0xef4444;
        addSphere(position, color, name === "head" ? 0.05 : 0.032);
      }});

      Object.entries(frame.surfaces || {{}}).forEach(([name, item]) => {{
        const color = item.collision ? 0xef4444 : 0x1e293b;
        addBox(item.position, item.size, color, 0.35);
      }});

      Object.entries(frame.objects || {{}}).forEach(([name, item]) => {{
        let color = 0x38bdf8;
        if (item.held_by) color = 0xfacc15;
        if (item.collision) color = 0xef4444;
        addBox(item.position, item.size, color, 0.95);
      }});

      updateLabels(frame);
    }}

    function handleReset(message) {{
      metadata = message.metadata || metadata;
      frames = message.frames || [];
      currentIndex = Math.max(frames.length - 1, 0);
      completed = false;
      frameStepMs = 50;
      if (frames.length > 0) renderFrame(frames[currentIndex]);
      else updateLabels({{ time_s: 0, active_action: "waiting" }});
    }}

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socketUrl = `${{protocol}}://${{window.location.host}}/ws`;
    const socket = new WebSocket(socketUrl);

    socket.addEventListener("open", () => {{
      status.textContent = "Connected. Waiting for first frame...";
      socket.send("ready");
    }});

    socket.addEventListener("message", (event) => {{
      const message = JSON.parse(event.data);
      if (message.type === "reset") {{
        handleReset(message);
        if (socket.readyState === WebSocket.OPEN) socket.send("ack");
        return;
      }}
      if (message.type === "frame") {{
        metadata = message.metadata || metadata;
        frames.push(message.frame);
        frameStepMs = Math.max(10, Math.round((message.frame.time_s - (frames.at(-2)?.time_s || 0)) * 1000)) || 50;
        if (playing) {{
          currentIndex = frames.length - 1;
          renderFrame(frames[currentIndex]);
        }} else {{
          updateLabels(frames[currentIndex] || message.frame);
        }}
        if (socket.readyState === WebSocket.OPEN) socket.send("ack");
        return;
      }}
      if (message.type === "complete") {{
        completed = true;
        status.textContent = `${{metadata.task_name || "simulation"}} | complete`;
      }}
    }});

    socket.addEventListener("close", () => {{
      status.textContent = completed
        ? "Stream complete. Use play/pause/step to inspect frames."
        : "Connection closed.";
    }});

    function animate(timestamp) {{
      requestAnimationFrame(animate);
      if (playing && frames.length > 0 && timestamp - lastAdvanceMs >= frameStepMs) {{
        lastAdvanceMs = timestamp;
        currentIndex = Math.min(currentIndex + 1, frames.length - 1);
        renderFrame(frames[currentIndex]);
      }}
      controls.update();
      renderer.render(scene, camera);
    }}
    requestAnimationFrame(animate);

    window.addEventListener("resize", () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});
  </script>
</body>
</html>
"""
