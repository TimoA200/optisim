"""Built-in humanoid manipulation task templates."""

from __future__ import annotations

from typing import Any

from optisim.core import ActionPrimitive, TaskDefinition
from optisim.library.catalog import DifficultyLevel, TaskTemplate, TemplateParameter


def builtin_templates() -> list[TaskTemplate]:
    """Return the bundled task templates shipped with optisim."""

    return [
        TaskTemplate(
            name="pick_and_place",
            description="Pick a household object from a table and place it on a counter or shelf.",
            tags=("household", "pick", "place", "tabletop"),
            difficulty=DifficultyLevel.BEGINNER,
            build=pick_and_place,
            parameters=(
                TemplateParameter("object", "Object label used in task metadata and world state.", "mug"),
                TemplateParameter("surface", "Destination support surface name.", "counter"),
            ),
        ),
        TaskTemplate(
            name="table_clearing",
            description="Clear a dining table by moving a dish into a bus tub staging area.",
            tags=("household", "cleanup", "dining"),
            difficulty=DifficultyLevel.BEGINNER,
            build=table_clearing,
            parameters=(TemplateParameter("item", "Dish or cup to clear from the table.", "plate"),),
        ),
        TaskTemplate(
            name="dish_loading",
            description="Transfer a dish from a kitchen counter into a dish rack.",
            tags=("household", "kitchen", "loading"),
            difficulty=DifficultyLevel.BEGINNER,
            build=dish_loading,
            parameters=(TemplateParameter("item", "Dishware item to load.", "bowl"),),
        ),
        TaskTemplate(
            name="laundry_folding",
            description="Lift a towel from a laundry basket, rotate it, and place it on a folding table.",
            tags=("household", "laundry", "folding"),
            difficulty=DifficultyLevel.INTERMEDIATE,
            build=laundry_folding,
            parameters=(TemplateParameter("garment", "Garment name to place into the scene.", "towel"),),
        ),
        TaskTemplate(
            name="shelf_stocking",
            description="Restock a shelf from a prep counter at a chosen shelf level.",
            tags=("household", "shelf", "stocking"),
            difficulty=DifficultyLevel.INTERMEDIATE,
            build=shelf_stocking,
            parameters=(
                TemplateParameter("item", "Packaged good to restock.", "cereal_box"),
                TemplateParameter("level", "Shelf level: low, middle, or high.", "middle"),
            ),
        ),
        TaskTemplate(
            name="drawer_open_close",
            description="Open a kitchen drawer and close it again using the handle.",
            tags=("household", "drawer", "pull", "push"),
            difficulty=DifficultyLevel.INTERMEDIATE,
            build=drawer_open_close,
            parameters=(TemplateParameter("drawer", "Drawer name used in metadata.", "utensil_drawer"),),
        ),
        TaskTemplate(
            name="assembly_line_pick",
            description="Pick a component from a conveyor and stage it on an assembly fixture.",
            tags=("industrial", "assembly", "conveyor"),
            difficulty=DifficultyLevel.BEGINNER,
            build=assembly_line_pick,
            parameters=(TemplateParameter("item", "Component to transfer off the line.", "gear_housing"),),
        ),
        TaskTemplate(
            name="pallet_stacking",
            description="Lift a carton from an infeed stand and stack it onto a pallet.",
            tags=("industrial", "pallet", "stacking", "logistics"),
            difficulty=DifficultyLevel.INTERMEDIATE,
            build=pallet_stacking,
            parameters=(TemplateParameter("item", "Carton or tote to stack.", "carton"),),
        ),
        TaskTemplate(
            name="bin_sorting",
            description="Sort a part from a mixed bin into a labeled output bin.",
            tags=("industrial", "sorting", "bins"),
            difficulty=DifficultyLevel.BEGINNER,
            build=bin_sorting,
            parameters=(TemplateParameter("item", "Part to sort.", "fastener_box"),),
        ),
        TaskTemplate(
            name="tool_handover",
            description="Pick a tool from a bench and move it into a handoff zone.",
            tags=("industrial", "tooling", "handover"),
            difficulty=DifficultyLevel.INTERMEDIATE,
            build=tool_handover,
            parameters=(TemplateParameter("tool", "Tool name used in the scene.", "torque_wrench"),),
        ),
        TaskTemplate(
            name="walk_and_pickup",
            description="Simulate a walk-up pickup by transferring a package from a floor-side cart to a carry tray.",
            tags=("locomotion", "manipulation", "pickup"),
            difficulty=DifficultyLevel.INTERMEDIATE,
            build=walk_and_pickup,
            parameters=(TemplateParameter("item", "Package name used in the scene.", "parcel"),),
        ),
        TaskTemplate(
            name="multi_room_delivery",
            description="Deliver an item from a reception counter to a patient-room shelf with route metadata.",
            tags=("locomotion", "manipulation", "delivery", "service"),
            difficulty=DifficultyLevel.ADVANCED,
            build=multi_room_delivery,
            parameters=(TemplateParameter("item", "Delivered item name.", "med_kit"),),
        ),
    ]


def pick_and_place(object: str = "mug", surface: str = "counter") -> TaskDefinition:
    """Build a basic pick-and-place household task."""

    destination_surface = _surface_by_name(surface, {"counter": (0.78, -0.18, 0.90), "shelf": (0.70, -0.25, 1.10)})
    source_top = 0.75
    item_size = (0.09, 0.09, 0.12)
    world = _world(
        surfaces=[
            _support_surface("table", center=(0.50, 0.08), top_z=source_top, size=(0.90, 0.70, 0.05)),
            _support_surface("counter", center=(0.78, -0.18), top_z=0.90, size=(0.70, 0.45, 0.06)),
            _support_surface("shelf", center=(0.70, -0.25), top_z=1.10, size=(0.42, 0.24, 0.04)),
        ],
        objects=[
            _object(object, position=(0.44, 0.02, source_top + item_size[2] / 2.0), size=item_size, mass_kg=0.55),
        ],
    )
    destination = _surface_place_position(destination_surface, item_size, x_offset=-0.06, y_offset=0.02)
    return _task(
        name=f"{object}_pick_and_place",
        description=f"Pick {object} from the table and place it on the {surface}.",
        tags=["household", "pick", "place"],
        difficulty=DifficultyLevel.BEGINNER,
        world=world,
        actions=_pick_move_place_actions(object, destination, destination_surface["name"]),
    )


def table_clearing(item: str = "plate") -> TaskDefinition:
    """Build a table clearing task that moves one table item into a bus tub."""

    item_size = (0.26, 0.26, 0.03) if "plate" in item else (0.10, 0.10, 0.14)
    world = _world(
        surfaces=[
            _support_surface("dining_table", center=(0.54, 0.02), top_z=0.75, size=(1.00, 0.70, 0.05)),
            _support_surface("bus_tub", center=(0.76, -0.20), top_z=0.86, size=(0.42, 0.30, 0.18)),
        ],
        objects=[
            _object(item, position=(0.48, 0.10, 0.75 + item_size[2] / 2.0), size=item_size, mass_kg=0.45),
        ],
    )
    destination = _surface_place_position(_surface_by_name("bus_tub", world["surfaces"]), item_size)
    return _task(
        name=f"{item}_table_clearing",
        description=f"Clear {item} from the dining table into the bus tub.",
        tags=["household", "cleanup"],
        difficulty=DifficultyLevel.BEGINNER,
        world=world,
        actions=_pick_move_place_actions(item, destination, "bus_tub"),
    )


def dish_loading(item: str = "bowl") -> TaskDefinition:
    """Build a dish-rack loading task."""

    item_size = (0.18, 0.18, 0.09)
    world = _world(
        surfaces=[
            _support_surface("counter", center=(0.56, 0.06), top_z=0.90, size=(0.90, 0.55, 0.06)),
            _support_surface("dish_rack", center=(0.76, -0.18), top_z=0.92, size=(0.34, 0.24, 0.12)),
        ],
        objects=[
            _object(item, position=(0.48, 0.02, 0.90 + item_size[2] / 2.0), size=item_size, mass_kg=0.35),
        ],
    )
    destination = _surface_place_position(_surface_by_name("dish_rack", world["surfaces"]), item_size, x_offset=-0.04)
    return _task(
        name=f"{item}_dish_loading",
        description=f"Load {item} from the counter into the dish rack.",
        tags=["household", "kitchen"],
        difficulty=DifficultyLevel.BEGINNER,
        world=world,
        actions=_pick_move_place_actions(item, destination, "dish_rack"),
    )


def laundry_folding(garment: str = "towel") -> TaskDefinition:
    """Build a laundry folding task with a rotate step before placement."""

    item_size = (0.34, 0.28, 0.05)
    world = _world(
        surfaces=[
            _support_surface("laundry_basket", center=(0.48, 0.16), top_z=0.58, size=(0.38, 0.28, 0.26)),
            _support_surface("folding_table", center=(0.72, -0.08), top_z=0.75, size=(0.90, 0.60, 0.05)),
        ],
        objects=[
            _object(garment, position=(0.46, 0.14, 0.58 + item_size[2] / 2.0), size=item_size, mass_kg=0.25),
        ],
    )
    destination = _surface_place_position(_surface_by_name("folding_table", world["surfaces"]), item_size)
    return _task(
        name=f"{garment}_laundry_folding",
        description=f"Lift {garment} from the basket, rotate it to fold, and place it on the folding table.",
        tags=["household", "laundry"],
        difficulty=DifficultyLevel.INTERMEDIATE,
        world=world,
        actions=[
            ActionPrimitive.reach(target=garment, end_effector="right_palm"),
            ActionPrimitive.grasp(target=garment, gripper="right_gripper"),
            ActionPrimitive.move(target=garment, destination=destination, end_effector="right_palm"),
            ActionPrimitive.rotate(target=garment, axis=(0.0, 0.0, 1.0), angle_rad=1.57, end_effector="right_palm"),
            ActionPrimitive.place(target=garment, support="folding_table", end_effector="right_palm"),
        ],
    )


def shelf_stocking(item: str = "cereal_box", level: str = "middle") -> TaskDefinition:
    """Build a shelf stocking task with selectable shelf height."""

    top_z_by_level = {"low": 0.95, "middle": 1.20, "high": 1.40}
    shelf_top = top_z_by_level.get(level, top_z_by_level["middle"])
    item_size = (0.09, 0.22, 0.30)
    world = _world(
        surfaces=[
            _support_surface("prep_counter", center=(0.48, 0.16), top_z=0.90, size=(0.70, 0.45, 0.06)),
            _support_surface("stock_shelf", center=(0.76, -0.14), top_z=shelf_top, size=(0.44, 0.26, 0.04)),
        ],
        objects=[
            _object(item, position=(0.46, 0.12, 0.90 + item_size[2] / 2.0), size=item_size, mass_kg=0.8),
        ],
    )
    destination = _surface_place_position(_surface_by_name("stock_shelf", world["surfaces"]), item_size, y_offset=0.01)
    return _task(
        name=f"{item}_shelf_stocking",
        description=f"Move {item} from the prep counter to the {level} shelf.",
        tags=["household", "shelf", level],
        difficulty=DifficultyLevel.INTERMEDIATE,
        world=world,
        actions=_pick_move_place_actions(item, destination, "stock_shelf"),
    )


def drawer_open_close(drawer: str = "utensil_drawer") -> TaskDefinition:
    """Build a drawer open-close interaction using pull and push actions."""

    handle_name = f"{drawer}_handle"
    handle_size = (0.14, 0.03, 0.03)
    world = _world(
        surfaces=[
            _support_surface("counter", center=(0.68, -0.08), top_z=0.90, size=(0.90, 0.60, 0.06)),
        ],
        objects=[
            _object(handle_name, position=(0.62, -0.02, 0.82), size=handle_size, mass_kg=0.15),
        ],
    )
    return _task(
        name=f"{drawer}_drawer_open_close",
        description=f"Reach the {drawer} handle, pull the drawer open, and push it closed.",
        tags=["household", "drawer", "open", "close"],
        difficulty=DifficultyLevel.INTERMEDIATE,
        world=world,
        actions=[
            ActionPrimitive.reach(target=handle_name, end_effector="right_palm"),
            ActionPrimitive.grasp(target=handle_name, gripper="right_gripper"),
            ActionPrimitive.pull(target=handle_name, direction=(1.0, 0.0, 0.0), force_newtons=18.0, end_effector="right_palm"),
            ActionPrimitive.push(target=handle_name, direction=(1.0, 0.0, 0.0), force_newtons=18.0, end_effector="right_palm"),
        ],
    )


def assembly_line_pick(item: str = "gear_housing") -> TaskDefinition:
    """Build an assembly-line pick task."""

    item_size = (0.18, 0.14, 0.09)
    world = _world(
        surfaces=[
            _support_surface("conveyor", center=(0.52, 0.10), top_z=0.88, size=(1.10, 0.34, 0.12)),
            _support_surface("assembly_fixture", center=(0.78, -0.14), top_z=0.90, size=(0.32, 0.24, 0.08)),
        ],
        objects=[
            _object(item, position=(0.44, 0.10, 0.88 + item_size[2] / 2.0), size=item_size, mass_kg=1.2),
        ],
    )
    destination = _surface_place_position(_surface_by_name("assembly_fixture", world["surfaces"]), item_size)
    return _task(
        name=f"{item}_assembly_line_pick",
        description=f"Pick {item} from the conveyor and stage it on the assembly fixture.",
        tags=["industrial", "assembly"],
        difficulty=DifficultyLevel.BEGINNER,
        world=world,
        actions=_pick_move_place_actions(item, destination, "assembly_fixture"),
    )


def pallet_stacking(item: str = "carton") -> TaskDefinition:
    """Build a pallet stacking task."""

    item_size = (0.30, 0.22, 0.24)
    world = _world(
        surfaces=[
            _support_surface("infeed_stand", center=(0.48, 0.14), top_z=0.78, size=(0.56, 0.40, 0.08)),
            _support_surface("pallet", center=(0.82, -0.12), top_z=0.18, size=(0.80, 0.60, 0.14)),
        ],
        objects=[
            _object(item, position=(0.46, 0.14, 0.78 + item_size[2] / 2.0), size=item_size, mass_kg=4.8),
        ],
    )
    destination = _surface_place_position(_surface_by_name("pallet", world["surfaces"]), item_size, x_offset=-0.10, y_offset=0.06)
    return _task(
        name=f"{item}_pallet_stacking",
        description=f"Stack {item} from the infeed stand onto a pallet.",
        tags=["industrial", "pallet"],
        difficulty=DifficultyLevel.INTERMEDIATE,
        world=world,
        actions=_pick_move_place_actions(item, destination, "pallet"),
    )


def bin_sorting(item: str = "fastener_box") -> TaskDefinition:
    """Build a bin sorting task."""

    item_size = (0.10, 0.10, 0.08)
    world = _world(
        surfaces=[
            _support_surface("mixed_bin", center=(0.48, 0.10), top_z=0.82, size=(0.32, 0.26, 0.18)),
            _support_surface("sorted_bin", center=(0.74, -0.16), top_z=0.82, size=(0.32, 0.26, 0.18)),
        ],
        objects=[
            _object(item, position=(0.46, 0.08, 0.82 + item_size[2] / 2.0), size=item_size, mass_kg=0.6),
        ],
    )
    destination = _surface_place_position(_surface_by_name("sorted_bin", world["surfaces"]), item_size)
    return _task(
        name=f"{item}_bin_sorting",
        description=f"Sort {item} from the mixed bin into the sorted bin.",
        tags=["industrial", "sorting"],
        difficulty=DifficultyLevel.BEGINNER,
        world=world,
        actions=_pick_move_place_actions(item, destination, "sorted_bin"),
    )


def tool_handover(tool: str = "torque_wrench") -> TaskDefinition:
    """Build a tool handover task."""

    item_size = (0.30, 0.05, 0.05)
    world = _world(
        surfaces=[
            _support_surface("workbench", center=(0.52, 0.10), top_z=0.90, size=(0.90, 0.60, 0.06)),
            _support_surface("handoff_tray", center=(0.78, -0.18), top_z=0.98, size=(0.34, 0.24, 0.05)),
        ],
        objects=[
            _object(tool, position=(0.46, 0.12, 0.90 + item_size[2] / 2.0), size=item_size, mass_kg=1.1),
        ],
    )
    destination = _surface_place_position(_surface_by_name("handoff_tray", world["surfaces"]), item_size)
    return _task(
        name=f"{tool}_tool_handover",
        description=f"Move {tool} from the workbench into the handoff tray.",
        tags=["industrial", "handover"],
        difficulty=DifficultyLevel.INTERMEDIATE,
        world=world,
        actions=_pick_move_place_actions(tool, destination, "handoff_tray"),
    )


def walk_and_pickup(item: str = "parcel") -> TaskDefinition:
    """Build a locomotion-plus-pickup task with route metadata."""

    item_size = (0.24, 0.18, 0.16)
    world = _world(
        surfaces=[
            _support_surface("entry_cart", center=(0.66, 0.22), top_z=0.64, size=(0.50, 0.34, 0.05)),
            _support_surface("carry_tray", center=(1.05, -0.06), top_z=0.96, size=(0.40, 0.28, 0.04)),
        ],
        objects=[
            _object(item, position=(0.62, 0.20, 0.64 + item_size[2] / 2.0), size=item_size, mass_kg=1.6),
        ],
    )
    destination = _surface_place_position(_surface_by_name("carry_tray", world["surfaces"]), item_size, x_offset=-0.05)
    return _task(
        name=f"{item}_walk_and_pickup",
        description=f"Pick {item} from the entry cart and stage it for transport.",
        tags=["locomotion", "manipulation"],
        difficulty=DifficultyLevel.INTERMEDIATE,
        world=world,
        actions=_pick_move_place_actions(item, destination, "carry_tray"),
        extra_metadata={"navigation_waypoints": [[0.0, 0.0], [0.4, 0.1], [0.8, 0.0]]},
    )


def multi_room_delivery(item: str = "med_kit") -> TaskDefinition:
    """Build a room-to-room delivery task with route metadata."""

    item_size = (0.28, 0.16, 0.12)
    world = _world(
        surfaces=[
            _support_surface("reception_counter", center=(0.54, 0.10), top_z=0.92, size=(0.90, 0.45, 0.06)),
            _support_surface("patient_shelf", center=(1.10, -0.12), top_z=1.02, size=(0.40, 0.24, 0.04)),
        ],
        objects=[
            _object(item, position=(0.46, 0.08, 0.92 + item_size[2] / 2.0), size=item_size, mass_kg=1.2),
        ],
    )
    destination = _surface_place_position(_surface_by_name("patient_shelf", world["surfaces"]), item_size, x_offset=-0.06)
    return _task(
        name=f"{item}_multi_room_delivery",
        description=f"Pick {item} at reception and deliver it to the patient shelf.",
        tags=["locomotion", "delivery", "service"],
        difficulty=DifficultyLevel.ADVANCED,
        world=world,
        actions=_pick_move_place_actions(item, destination, "patient_shelf"),
        extra_metadata={"navigation_waypoints": [[0.0, 0.0], [0.5, 0.2], [0.9, -0.1], [1.2, -0.1]]},
    )


def _task(
    *,
    name: str,
    description: str,
    tags: list[str],
    difficulty: DifficultyLevel,
    world: dict[str, Any],
    actions: list[ActionPrimitive],
    extra_metadata: dict[str, Any] | None = None,
) -> TaskDefinition:
    """Build a task definition with common library metadata."""

    metadata: dict[str, Any] = {
        "author": "optisim",
        "source": "task_library",
        "description": description,
        "tags": tags,
        "difficulty": difficulty.value,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return TaskDefinition(
        name=name,
        actions=actions,
        world=world,
        robot={"model": "optimus_humanoid"},
        metadata=metadata,
    )


def _world(*, surfaces: list[dict[str, Any]], objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a world payload with a standard gravity setting."""

    return {"gravity": [0.0, 0.0, -9.81], "surfaces": surfaces, "objects": objects}


def _support_surface(
    name: str,
    *,
    center: tuple[float, float],
    top_z: float,
    size: tuple[float, float, float],
) -> dict[str, Any]:
    """Create a rectangular support surface payload from a top height."""

    return {
        "name": name,
        "pose": {"position": [center[0], center[1], top_z - size[2] / 2.0], "rpy": [0.0, 0.0, 0.0]},
        "size": list(size),
    }


def _object(
    name: str,
    *,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    mass_kg: float,
) -> dict[str, Any]:
    """Create an object payload."""

    return {
        "name": name,
        "pose": {"position": list(position), "rpy": [0.0, 0.0, 0.0]},
        "size": list(size),
        "mass_kg": mass_kg,
    }


def _surface_place_position(
    surface: dict[str, Any],
    object_size: tuple[float, float, float],
    *,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> tuple[float, float, float]:
    """Return a placement point resting on top of a support surface."""

    position = surface["pose"]["position"]
    size = surface["size"]
    return (
        float(position[0] + x_offset),
        float(position[1] + y_offset),
        float(position[2] + size[2] / 2.0 + object_size[2] / 2.0),
    )


def _surface_by_name(name: str, surfaces: list[dict[str, Any]] | dict[str, tuple[float, float, float]]) -> dict[str, Any]:
    """Resolve a surface payload by name or synthesize a known support surface."""

    if isinstance(surfaces, dict):
        if name not in surfaces:
            raise ValueError(f"unknown destination surface '{name}'")
        x, y, top_z = surfaces[name]
        thickness = 0.06 if name == "counter" else 0.04
        width = 0.70 if name == "counter" else 0.42
        depth = 0.45 if name == "counter" else 0.24
        return _support_surface(name, center=(x, y), top_z=top_z, size=(width, depth, thickness))
    for surface in surfaces:
        if surface["name"] == name:
            return surface
    raise ValueError(f"unknown destination surface '{name}'")


def _pick_move_place_actions(
    target: str,
    destination: tuple[float, float, float],
    support: str,
) -> list[ActionPrimitive]:
    """Create a standard pick-move-place sequence."""

    return [
        ActionPrimitive.reach(target=target, end_effector="right_palm"),
        ActionPrimitive.grasp(target=target, gripper="right_gripper"),
        ActionPrimitive.move(target=target, destination=destination, end_effector="right_palm"),
        ActionPrimitive.place(target=target, support=support, end_effector="right_palm"),
    ]
