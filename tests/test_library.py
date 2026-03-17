from __future__ import annotations

from pathlib import Path

import pytest

from optisim.core import TaskDefinition
from optisim.library import DifficultyLevel, TaskCatalog, TaskTemplate, TemplateParameter
from optisim.robot import build_humanoid_model
from optisim.sim import ExecutionEngine, WorldState


EXPECTED_TEMPLATES = [
    "assembly_line_pick",
    "bin_sorting",
    "dish_loading",
    "drawer_open_close",
    "laundry_folding",
    "multi_room_delivery",
    "pallet_stacking",
    "pick_and_place",
    "shelf_stocking",
    "table_clearing",
    "tool_handover",
    "walk_and_pickup",
]


def test_catalog_listing_returns_all_builtin_templates() -> None:
    catalog = TaskCatalog()

    templates = catalog.list()

    assert [template.name for template in templates] == EXPECTED_TEMPLATES
    assert all(template.description for template in templates)


def test_catalog_search_matches_name_description_and_tags() -> None:
    catalog = TaskCatalog()

    assert [match.name for match in catalog.search("laundry")] == ["laundry_folding"]
    assert [match.name for match in catalog.search("conveyor")] == ["assembly_line_pick"]
    assert [match.name for match in catalog.search("delivery")] == ["multi_room_delivery"]


def test_catalog_search_empty_keyword_returns_full_listing() -> None:
    catalog = TaskCatalog()

    assert [template.name for template in catalog.search("")] == EXPECTED_TEMPLATES


def test_catalog_get_supports_parameterized_templates() -> None:
    catalog = TaskCatalog()

    task = catalog.get("pick_and_place", object="thermos", surface="shelf")

    assert task.name == "thermos_pick_and_place"
    assert task.actions[-1].support == "shelf"
    assert task.world["objects"][0]["name"] == "thermos"


def test_shelf_stocking_level_changes_world_height() -> None:
    catalog = TaskCatalog()

    low_task = catalog.get("shelf_stocking", level="low")
    high_task = catalog.get("shelf_stocking", level="high")

    low_surface = next(surface for surface in low_task.world["surfaces"] if surface["name"] == "stock_shelf")
    high_surface = next(surface for surface in high_task.world["surfaces"] if surface["name"] == "stock_shelf")
    assert high_surface["pose"]["position"][2] > low_surface["pose"]["position"][2]


def test_catalog_can_register_custom_templates() -> None:
    catalog = TaskCatalog(templates=[])
    template = TaskTemplate(
        name="custom_demo",
        description="Custom test template",
        tags=("custom",),
        difficulty=DifficultyLevel.BEGINNER,
        parameters=(TemplateParameter("item", "Item name", "widget"),),
        build=lambda item="widget": TaskDefinition.from_dict(
            {
                "name": f"{item}_demo",
                "world": {"objects": [{"name": item, "pose": {"position": [0.4, 0.0, 0.8]}, "size": [0.1, 0.1, 0.1]}]},
                "actions": [{"type": "reach", "target": item}],
            }
        ),
    )

    catalog.register(template)

    assert catalog.info("custom_demo").parameters[0].name == "item"
    assert catalog.get("custom_demo", item="crate").name == "crate_demo"


def test_template_export_roundtrip(tmp_path: Path) -> None:
    catalog = TaskCatalog()
    task = catalog.get("tool_handover")
    output = tmp_path / "tool_handover.yaml"

    task.dump(output)
    reloaded = TaskDefinition.from_file(output)

    assert reloaded.name == task.name
    assert len(reloaded.actions) == len(task.actions)
    assert reloaded.metadata["difficulty"] == "intermediate"


@pytest.mark.parametrize("template_name", EXPECTED_TEMPLATES)
def test_each_template_generates_valid_task_definition(template_name: str) -> None:
    catalog = TaskCatalog()

    task = catalog.get(template_name)

    assert isinstance(task, TaskDefinition)
    assert task.actions
    assert task.metadata["source"] == "task_library"
    assert task.metadata["difficulty"] in {"beginner", "intermediate", "advanced"}


@pytest.mark.parametrize("template_name", EXPECTED_TEMPLATES)
def test_each_template_validates_and_runs(template_name: str) -> None:
    catalog = TaskCatalog()
    task = catalog.get(template_name)
    engine = ExecutionEngine(robot=build_humanoid_model(), world=WorldState.from_dict(task.world))

    report = engine.validate(task)

    assert report.is_valid, report.summary()
    record = engine.run(task)
    assert record.executed_actions
    assert len(record.executed_actions) == len(task.actions)
