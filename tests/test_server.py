from __future__ import annotations

import pytest

try:
    import httpx  # noqa: F401
    from fastapi.testclient import TestClient

    HAS_HTTPX = True
except Exception:  # pragma: no cover - depends on environment
    TestClient = None  # type: ignore[assignment]
    HAS_HTTPX = False

from optisim.server import VERSION, app


pytestmark = pytest.mark.skipif(not HAS_HTTPX, reason="httpx is required for FastAPI TestClient")


@pytest.fixture(autouse=True)
def clear_demo_store() -> None:
    from optisim.server import DEMO_STORAGE

    DEMO_STORAGE.clear()


@pytest.fixture()
def client() -> TestClient:
    assert TestClient is not None
    return TestClient(app)


def test_health_returns_ok_schema(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": VERSION}


def test_health_content_type_is_json(client: TestClient) -> None:
    response = client.get("/health")

    assert response.headers["content-type"].startswith("application/json")


def test_robots_returns_built_in_presets(client: TestClient) -> None:
    response = client.get("/robots")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) >= 2
    assert {item["id"] for item in body} >= {"demo_humanoid", "custom"}


def test_robots_entries_have_expected_schema(client: TestClient) -> None:
    response = client.get("/robots")

    first = response.json()[0]
    assert set(first) == {"id", "name", "dof", "description"}
    assert isinstance(first["dof"], int)


def test_simulate_valid_request_returns_result(client: TestClient) -> None:
    response = client.post(
        "/simulate",
        json={"task_name": "reach_and_grasp", "target_pose": [0.5, -0.1, 1.0], "num_steps": 12},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True
    assert body["num_steps"] == 12
    assert len(body["joint_trajectory"]) == 12
    assert len(body["ee_trajectory"]) == 12
    assert len(body["energy_profile"]) == 12


def test_simulate_accepts_null_optional_fields(client: TestClient) -> None:
    response = client.post(
        "/simulate",
        json={"task_name": "reach_and_grasp", "joint_positions": None, "target_pose": None},
    )

    assert response.status_code == 200
    assert response.json()["num_steps"] == 50


def test_simulate_disabling_safety_returns_empty_violations(client: TestClient) -> None:
    response = client.post(
        "/simulate",
        json={
            "task_name": "reach_and_grasp",
            "target_pose": [0.5, 0.0, 1.0],
            "include_safety": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["safety_violations"] == []


def test_simulate_disabling_dynamics_zeroes_energy_profile(client: TestClient) -> None:
    response = client.post(
        "/simulate",
        json={
            "task_name": "reach_and_grasp",
            "target_pose": [0.45, -0.05, 0.95],
            "include_dynamics": False,
            "num_steps": 8,
        },
    )

    assert response.status_code == 200
    assert response.json()["energy_profile"] == pytest.approx([0.0] * 8)


def test_simulate_rejects_missing_task_name(client: TestClient) -> None:
    response = client.post("/simulate", json={"target_pose": [0.5, 0.0, 1.0]})

    assert response.status_code == 422


def test_simulate_rejects_bad_type(client: TestClient) -> None:
    response = client.post("/simulate", json={"task_name": "reach_and_grasp", "num_steps": "bad"})

    assert response.status_code == 422


def test_simulate_rejects_bad_joint_vector_length(client: TestClient) -> None:
    response = client.post(
        "/simulate",
        json={"task_name": "reach_and_grasp", "joint_positions": [0.0, 0.1]},
    )

    assert response.status_code == 422


def test_simulate_rejects_bad_target_pose_length(client: TestClient) -> None:
    response = client.post(
        "/simulate",
        json={"task_name": "reach_and_grasp", "target_pose": [0.3, 0.4]},
    )

    assert response.status_code == 422


def test_plan_finds_feasible_plan_for_simple_transfer(client: TestClient) -> None:
    response = client.post(
        "/plan",
        json={
            "initial_predicates": [
                {"name": "at", "args": ["robot", "home"], "value": True},
                {"name": "object-at", "args": ["cup", "table"], "value": True},
                {"name": "handempty", "args": ["robot"], "value": True},
            ],
            "goal_predicates": [{"name": "object-at", "args": ["cup", "counter"], "value": True}],
            "object_poses": {
                "robot": [0.0, 0.0, 0.0],
                "home": [0.0, 0.0, 0.0],
                "table": [1.0, 0.0, 0.0],
                "counter": [1.2, 0.8, 0.9],
                "cup": [1.0, 0.0, 0.8],
            },
            "objects": ["robot", "cup", "home", "table", "counter"],
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["feasible"] is True
    assert body["num_steps"] == 4
    assert [step["operator"] for step in body["steps"]] == ["navigate-to", "pick-up", "navigate-to", "place"]


def test_plan_returns_not_feasible_when_geometry_fails(client: TestClient) -> None:
    response = client.post(
        "/plan",
        json={
            "initial_predicates": [
                {"name": "at", "args": ["robot", "home"], "value": True},
                {"name": "object-at", "args": ["cup", "table"], "value": True},
                {"name": "handempty", "args": ["robot"], "value": True},
            ],
            "goal_predicates": [{"name": "object-at", "args": ["cup", "counter"], "value": True}],
            "object_poses": {
                "robot": [0.0, 0.0, 0.0],
                "home": [0.0, 0.0, 0.0],
                "table": [12.0, 0.0, 0.0],
                "counter": [12.5, 0.0, 0.9],
                "cup": [12.0, 0.0, 0.8],
            },
            "objects": ["robot", "cup", "home", "table", "counter"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"feasible": False, "steps": [], "num_steps": 0}


def test_plan_rejects_missing_required_field(client: TestClient) -> None:
    response = client.post(
        "/plan",
        json={
            "initial_predicates": [],
            "goal_predicates": [],
            "objects": [],
        },
    )

    assert response.status_code == 422


def test_plan_rejects_bad_object_pose_length(client: TestClient) -> None:
    response = client.post(
        "/plan",
        json={
            "initial_predicates": [],
            "goal_predicates": [],
            "object_poses": {"cup": [0.0, 0.0]},
            "objects": [],
        },
    )

    assert response.status_code == 422


def test_lfd_record_creates_demo(client: TestClient) -> None:
    response = client.post(
        "/lfd/record",
        json={"task_name": "demo", "num_steps": 10, "start": [0.0, 0.1, 0.2], "end": [0.5, 0.6, 0.7]},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["task_name"] == "demo"
    assert body["num_steps"] == 10
    assert body["joint_dim"] == 3
    assert body["demo_id"]


def test_lfd_record_rejects_mismatched_dimensions(client: TestClient) -> None:
    response = client.post(
        "/lfd/record",
        json={"task_name": "demo", "start": [0.0, 0.1], "end": [0.5]},
    )

    assert response.status_code == 422


def test_lfd_record_rejects_empty_vectors(client: TestClient) -> None:
    response = client.post("/lfd/record", json={"task_name": "demo", "start": [], "end": []})

    assert response.status_code == 422


def test_lfd_generate_valid_demo_id_returns_trajectory(client: TestClient) -> None:
    record_response = client.post(
        "/lfd/record",
        json={"task_name": "demo", "num_steps": 12, "start": [0.0, 0.1, 0.2], "end": [0.5, 0.6, 0.7]},
    )
    demo_id = record_response.json()["demo_id"]

    response = client.post("/lfd/generate", json={"demo_id": demo_id, "new_goal": [0.8, 0.9, 1.0]})

    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True
    assert body["num_steps"] == 12
    assert body["trajectory"][-1] == pytest.approx([0.8, 0.9, 1.0], abs=1e-6)


def test_lfd_generate_invalid_demo_id_returns_404(client: TestClient) -> None:
    response = client.post("/lfd/generate", json={"demo_id": "missing", "new_goal": [0.8, 0.9, 1.0]})

    assert response.status_code == 404


def test_lfd_generate_rejects_bad_goal_dimension(client: TestClient) -> None:
    record_response = client.post(
        "/lfd/record",
        json={"task_name": "demo", "num_steps": 12, "start": [0.0, 0.1, 0.2], "end": [0.5, 0.6, 0.7]},
    )
    demo_id = record_response.json()["demo_id"]

    response = client.post("/lfd/generate", json={"demo_id": demo_id, "new_goal": [0.8, 0.9]})

    assert response.status_code == 422


def test_dashboard_returns_html(client: TestClient) -> None:
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_dashboard_contains_title_and_cards(client: TestClient) -> None:
    response = client.get("/dashboard")
    html = response.text

    assert "optisim — Humanoid Robot Simulator" in html
    assert "Open-source task planner &amp; simulator for humanoid robots" in html
    assert "Simulate" in html
    assert "Plan" in html
    assert "LfD" in html
    assert "fetch(" in html
