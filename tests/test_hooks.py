from __future__ import annotations

import json
from pathlib import Path

import pytest

from optisim.core import TaskDefinition
from optisim.hooks import CollisionRecorder, EventLogger, HookRegistry, SimEvent, StepSampler
from optisim.sim import ExecutionEngine, WorldState


def _pick_and_place_task() -> TaskDefinition:
    return TaskDefinition.from_file(Path("examples/pick_and_place.yaml"))


def test_hook_registry_on_decorator_registers_callback() -> None:
    registry = HookRegistry()
    events: list[SimEvent] = []

    @registry.on("task_start")
    def callback(event: SimEvent) -> None:
        events.append(event)

    registry.emit(SimEvent(kind="task_start", task_name="demo", step=0, data={}))

    assert len(registry) == 1
    assert [event.kind for event in events] == ["task_start"]


def test_hook_registry_register_adds_callback() -> None:
    registry = HookRegistry()
    events: list[str] = []

    def callback(event: SimEvent) -> None:
        events.append(event.kind)

    registry.register("task_end", callback)
    registry.emit(SimEvent(kind="task_end", task_name="demo", step=1, data={}))

    assert events == ["task_end"]


def test_hook_registry_unregister_removes_callback() -> None:
    registry = HookRegistry()
    events: list[str] = []

    def callback(event: SimEvent) -> None:
        events.append(event.kind)

    registry.register("step", callback)
    registry.unregister("step", callback)
    registry.emit(SimEvent(kind="step", task_name="demo", step=1, data={}))

    assert events == []
    assert len(registry) == 0


def test_hook_registry_clear_removes_all_callbacks_for_kind() -> None:
    registry = HookRegistry()
    registry.register("step", lambda event: None)
    registry.register("step", lambda event: None)
    registry.register("task_end", lambda event: None)

    registry.clear("step")

    assert len(registry) == 1


def test_hook_registry_clear_without_argument_removes_all_callbacks() -> None:
    registry = HookRegistry()
    registry.register("step", lambda event: None)
    registry.register("task_end", lambda event: None)

    registry.clear()

    assert len(registry) == 0


def test_hook_registry_emit_calls_only_matching_callbacks() -> None:
    registry = HookRegistry()
    calls: list[str] = []

    registry.register("step", lambda event: calls.append(f"step:{event.step}"))
    registry.register("collision", lambda event: calls.append(f"collision:{event.step}"))

    registry.emit(SimEvent(kind="step", task_name="demo", step=3, data={}))

    assert calls == ["step:3"]


def test_hook_registry_emit_catches_errors_without_raising(caplog: pytest.LogCaptureFixture) -> None:
    registry = HookRegistry()
    calls: list[str] = []

    def broken(event: SimEvent) -> None:
        del event
        raise RuntimeError("boom")

    registry.register("step", broken)
    registry.register("step", lambda event: calls.append(event.kind))

    with caplog.at_level("ERROR"):
        registry.emit(SimEvent(kind="step", task_name="demo", step=1, data={}))

    assert calls == ["step"]
    assert "hook callback failed" in caplog.text


def test_hook_registry_len_returns_total_callback_count() -> None:
    registry = HookRegistry()
    registry.register("step", lambda event: None)
    registry.register("step", lambda event: None)
    registry.register("collision", lambda event: None)

    assert len(registry) == 3


def test_event_logger_writes_valid_jsonl_when_attached_to_run(tmp_path: Path) -> None:
    task = _pick_and_place_task()
    registry = HookRegistry()
    logger = EventLogger(tmp_path / "events.jsonl")
    logger.attach(registry)

    engine = ExecutionEngine(world=WorldState.from_dict(task.world), hooks=registry)
    engine.run(task)

    payloads = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    assert payloads
    assert payloads[0]["kind"] == "task_start"
    assert payloads[-1]["kind"] == "task_end"
    assert all({"ts", "kind", "task", "step", "data"} <= payload.keys() for payload in payloads)


def test_collision_recorder_accumulates_collision_events() -> None:
    task = _pick_and_place_task()
    registry = HookRegistry()
    recorder = CollisionRecorder()
    recorder.attach(registry)

    engine = ExecutionEngine(world=WorldState.from_dict(task.world), hooks=registry)
    engine.run(task)

    assert recorder.events
    assert all(event.kind == "collision" for event in recorder.events)


def test_collision_recorder_clear_resets_event_list() -> None:
    recorder = CollisionRecorder()
    registry = HookRegistry()
    recorder.attach(registry)
    registry.emit(SimEvent(kind="collision", task_name="demo", step=1, data={"count": 1, "pairs": [("a", "b")]}))

    recorder.clear()

    assert recorder.events == []


def test_step_sampler_samples_at_correct_interval() -> None:
    sampler = StepSampler(interval=3)
    registry = HookRegistry()
    sampler.attach(registry)

    for step in range(1, 10):
        registry.emit(
            SimEvent(
                kind="step",
                task_name="demo",
                step=step,
                data={"action_type": "move", "joint_positions": {}},
            )
        )

    assert [event.step for event in sampler.samples] == [3, 6, 9]


def test_execution_engine_with_hooks_emits_task_start_and_task_end_events() -> None:
    task = _pick_and_place_task()
    registry = HookRegistry()
    seen: list[SimEvent] = []

    registry.register("task_start", seen.append)
    registry.register("task_end", seen.append)

    engine = ExecutionEngine(world=WorldState.from_dict(task.world), hooks=registry)
    engine.run(task)

    assert [event.kind for event in seen] == ["task_start", "task_end"]
    assert seen[0].data["actions_total"] == len(task.actions)
    assert seen[1].data["success"] is True


def test_execution_engine_with_hooks_none_still_runs_without_error() -> None:
    task = _pick_and_place_task()
    engine = ExecutionEngine(world=WorldState.from_dict(task.world), hooks=None)

    record = engine.run(task)

    assert record.executed_actions == ["reach", "grasp", "move", "place"]


def test_event_logger_detach_stops_writing_new_events(tmp_path: Path) -> None:
    registry = HookRegistry()
    logger = EventLogger(tmp_path / "events.jsonl", events=["step"])
    logger.attach(registry)
    registry.emit(SimEvent(kind="step", task_name="demo", step=1, data={}))
    logger.detach(registry)
    registry.emit(SimEvent(kind="step", task_name="demo", step=2, data={}))

    payloads = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    assert [payload["step"] for payload in payloads] == [1]


__all__ = [
    "test_hook_registry_on_decorator_registers_callback",
    "test_hook_registry_register_adds_callback",
    "test_hook_registry_unregister_removes_callback",
    "test_hook_registry_clear_removes_all_callbacks_for_kind",
    "test_hook_registry_clear_without_argument_removes_all_callbacks",
    "test_hook_registry_emit_calls_only_matching_callbacks",
    "test_hook_registry_emit_catches_errors_without_raising",
    "test_hook_registry_len_returns_total_callback_count",
    "test_event_logger_writes_valid_jsonl_when_attached_to_run",
    "test_collision_recorder_accumulates_collision_events",
    "test_collision_recorder_clear_resets_event_list",
    "test_step_sampler_samples_at_correct_interval",
    "test_execution_engine_with_hooks_emits_task_start_and_task_end_events",
    "test_execution_engine_with_hooks_none_still_runs_without_error",
    "test_event_logger_detach_stops_writing_new_events",
]
