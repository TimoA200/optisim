from __future__ import annotations

import numpy as np

import optisim
from optisim import (
    GroundedInstruction,
    Grounder,
    InstructionParser,
    InstructionTemplate,
    ParsedIntent,
    SceneGraph,
    SceneNode,
)
from optisim.language import Grounder as PackageGrounder
from optisim.language import InstructionParser as PackageInstructionParser
from optisim.language import InstructionTemplate as PackageInstructionTemplate


def make_pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def make_kitchen_scene() -> SceneGraph:
    scene = SceneGraph()
    scene.add_node(SceneNode("robot", "robot", "robot", pose=make_pose(0.0, 0.0, 0.0)))
    scene.add_node(SceneNode("cup", "cup", "container", pose=make_pose(0.5, 0.0, 0.8), properties={"graspable": True}))
    scene.add_node(SceneNode("knife", "knife", "tool", pose=make_pose(0.6, 0.1, 0.8), properties={"graspable": True}))
    scene.add_node(SceneNode("bowl", "bowl", "container", pose=make_pose(0.55, -0.1, 0.8), properties={"graspable": True}))
    scene.add_node(SceneNode("box", "box", "object", pose=make_pose(1.0, 0.0, 0.8)))
    scene.add_node(SceneNode("table", "table", "surface", pose=make_pose(0.8, 0.0, 0.75)))
    scene.add_node(SceneNode("countertop", "countertop", "surface", pose=make_pose(0.9, 0.2, 0.9)))
    scene.add_node(SceneNode("shelf", "shelf", "surface", pose=make_pose(1.5, 0.5, 1.2)))
    scene.add_node(SceneNode("door", "door", "fixture", pose=make_pose(2.0, 0.0, 0.0)))
    return scene


def test_parsed_intent_creation() -> None:
    intent = ParsedIntent("pick", "cup", None, "carefully", "pick cup", 1.0)
    assert intent.action == "pick"
    assert intent.target == "cup"
    assert intent.destination is None
    assert intent.modifier == "carefully"
    assert intent.raw_text == "pick cup"
    assert intent.confidence == 1.0


def test_language_package_exports_public_classes() -> None:
    assert PackageInstructionParser is InstructionParser
    assert PackageGrounder is Grounder
    assert PackageInstructionTemplate is InstructionTemplate


def test_root_package_exports_language_classes() -> None:
    assert optisim.InstructionParser is InstructionParser
    assert optisim.Grounder is Grounder
    assert optisim.ParsedIntent is ParsedIntent


def test_instruction_parser_parse_pick_up_cup() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("pick up the cup")
    assert intent.action == "pick"
    assert intent.target == "cup"


def test_instruction_parser_parse_grab_knife() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("grab the knife")
    assert intent.action == "pick"
    assert intent.target == "knife"


def test_instruction_parser_parse_put_cup_on_table() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("put cup on table")
    assert intent.action == "place"
    assert intent.target == "cup"
    assert intent.destination == "table"


def test_instruction_parser_parse_walk_to_shelf() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("walk to the shelf")
    assert intent.action == "go"
    assert intent.target == "shelf"


def test_instruction_parser_parse_push_box() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("push the box")
    assert intent.action == "push"
    assert intent.target == "box"


def test_instruction_parser_parse_handover() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("hand the cup to my right hand")
    assert intent.action == "handover"


def test_instruction_parser_parse_place_carefully() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("place knife carefully on table")
    assert intent.modifier == "carefully"
    assert intent.action == "place"


def test_instruction_parser_parse_garble_confidence_zero() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("xyz garble nonsense")
    assert intent.confidence == 0.0
    assert intent.action == ""


def test_instruction_parser_parse_batch_length() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intents = parser.parse_batch(["pick cup", "go to shelf", "push box"])
    assert len(intents) == 3


def test_instruction_parser_confidence_one_for_clear_action() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("pick cup")
    assert intent.confidence == 1.0


def test_instruction_parser_confidence_partial_match_positive() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("cup please")
    assert intent.confidence > 0.0
    assert intent.action == ""


def test_instruction_parser_uses_scene_context_for_target() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("go to the door")
    assert intent.target == "door"


def test_instruction_parser_move_quickly_to_door() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("move quickly to the door")
    assert intent.modifier == "quickly"
    assert intent.action == "go"
    assert intent.target == "door"


def test_instruction_parser_carefully_grab_bowl() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("carefully grab the bowl")
    assert intent.action == "pick"
    assert intent.modifier == "carefully"
    assert intent.target == "bowl"


def test_instruction_parser_empty_string() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("")
    assert intent.confidence == 0.0
    assert intent.target is None


def test_instruction_parser_destination_into_scene_object() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("put the cup into the bowl")
    assert intent.destination == "bowl"


def test_instruction_parser_recognizes_turn_action() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("turn the door")
    assert intent.action == "turn"


def test_instruction_parser_recognizes_open_action() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("open the door")
    assert intent.action == "open"


def test_instruction_parser_recognizes_close_action() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    intent = parser.parse("close the door")
    assert intent.action == "close"


def test_grounder_with_kitchen_scene_instantiates() -> None:
    grounder = Grounder(make_kitchen_scene())
    assert isinstance(grounder.parser, InstructionParser)


def test_ground_pick_cup_target_node_id() -> None:
    scene = make_kitchen_scene()
    grounder = Grounder(scene)
    grounded = grounder.ground(InstructionParser(scene=scene).parse("pick cup"))
    assert grounded.target_node_id == "cup"


def test_ground_go_to_table_includes_navigate() -> None:
    grounder = Grounder(make_kitchen_scene())
    grounded = grounder.ground_instruction("go to table")
    assert grounded.primitive_sequence[0]["primitive"] == "navigate"


def test_ground_put_cup_on_countertop_destination() -> None:
    grounder = Grounder(make_kitchen_scene())
    grounded = grounder.ground_instruction("put cup on countertop")
    assert grounded.destination_node_id == "countertop"


def test_ground_instruction_returns_grounded_instruction() -> None:
    grounded = Grounder(make_kitchen_scene()).ground_instruction("pick cup")
    assert isinstance(grounded, GroundedInstruction)


def test_ground_unknown_target_has_low_confidence() -> None:
    grounded = Grounder(make_kitchen_scene()).ground_instruction("pick pineapple")
    assert grounded.grounding_confidence <= 0.2
    assert "target not found in scene" in grounded.notes


def test_grounder_suggest_targets_returns_list() -> None:
    parser = InstructionParser(scene=make_kitchen_scene())
    suggestions = Grounder(make_kitchen_scene()).suggest_targets(parser.parse("pick cu"))
    assert isinstance(suggestions, list)


def test_grounder_primitive_sequence_contains_primitive_keys() -> None:
    grounded = Grounder(make_kitchen_scene()).ground_instruction("pick cup")
    assert isinstance(grounded.primitive_sequence, list)
    assert all("primitive" in step for step in grounded.primitive_sequence)


def test_grounder_pick_sequence_shape() -> None:
    grounded = Grounder(make_kitchen_scene()).ground_instruction("pick cup")
    assert [step["primitive"] for step in grounded.primitive_sequence] == ["navigate", "reach", "grasp"]


def test_grounder_place_sequence_shape() -> None:
    grounded = Grounder(make_kitchen_scene()).ground_instruction("put cup on table")
    assert [step["primitive"] for step in grounded.primitive_sequence] == ["navigate", "place"]


def test_grounder_push_sequence_shape() -> None:
    grounded = Grounder(make_kitchen_scene()).ground_instruction("push box")
    assert [step["primitive"] for step in grounded.primitive_sequence] == ["navigate", "push"]


def test_grounder_handover_sequence_shape() -> None:
    grounded = Grounder(make_kitchen_scene()).ground_instruction("hand cup to me")
    assert grounded.primitive_sequence[0]["primitive"] == "handover"


def test_instruction_template_generate_pick_returns_string() -> None:
    text = InstructionTemplate.generate("pick", target="cup")
    assert isinstance(text, str)


def test_instruction_template_generate_contains_target_name() -> None:
    text = InstructionTemplate.generate("pick", target="cup")
    assert "cup" in text


def test_instruction_template_all_templates_keys() -> None:
    templates = InstructionTemplate.all_templates()
    assert "pick" in templates
    assert "place" in templates
    assert "go" in templates


def test_instruction_template_pick_templates_non_empty() -> None:
    assert InstructionTemplate.PICK_TEMPLATES


def test_instruction_template_place_generation_uses_destination() -> None:
    text = InstructionTemplate.generate("place", target="cup", destination="table")
    assert "cup" in text
    assert "table" in text


def test_grounder_suggest_targets_matches_substring() -> None:
    intent = ParsedIntent("pick", "count", None, None, "pick count", 0.5)
    suggestions = Grounder(make_kitchen_scene()).suggest_targets(intent)
    assert "countertop" in suggestions


__all__ = [name for name in globals() if name.startswith("test_")]
