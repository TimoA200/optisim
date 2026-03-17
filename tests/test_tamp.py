"""Tests for the optisim.tamp module."""

from __future__ import annotations

from optisim import GeometricChecker, HouseholdDomain, Operator, PlanningState, Predicate, SymbolicPlanner, TAMPPlanner
from optisim.tamp import TAMPPlan


def make_move_operator(source: str, target: str) -> Operator:
    return Operator(
        name="move",
        parameters=["robot", source, target],
        preconditions=[Predicate("at", ["robot", source])],
        add_effects=[Predicate("at", ["robot", target])],
        del_effects=[Predicate("at", ["robot", source])],
    )


def test_predicate_defaults() -> None:
    predicate = Predicate("holding", ["robot", "cup"])
    assert predicate.name == "holding"
    assert predicate.args == ["robot", "cup"]
    assert predicate.value is True


def test_predicate_string_positive() -> None:
    assert str(Predicate("at", ["robot", "table"])) == "(at robot table)"


def test_predicate_string_negative() -> None:
    assert str(Predicate("at", ["robot", "table"], value=False)) == "(not (at robot table))"


def test_predicate_equality() -> None:
    assert Predicate("clear", ["counter"]) == Predicate("clear", ["counter"])


def test_predicate_hashing_supports_sets() -> None:
    predicates = {Predicate("clear", ["counter"]), Predicate("clear", ["counter"])}
    assert len(predicates) == 1


def test_operator_applicable_when_preconditions_hold() -> None:
    operator = make_move_operator("a", "b")
    state = {Predicate("at", ["robot", "a"])}
    assert operator.applicable(state) is True


def test_operator_not_applicable_when_preconditions_missing() -> None:
    operator = make_move_operator("a", "b")
    state = {Predicate("at", ["robot", "b"])}
    assert operator.applicable(state) is False


def test_operator_supports_negative_preconditions() -> None:
    operator = Operator(
        name="pickup",
        parameters=["robot", "cup"],
        preconditions=[Predicate("holding", ["robot", "cup"], value=False)],
        add_effects=[Predicate("holding", ["robot", "cup"])],
        del_effects=[],
    )
    assert operator.applicable(set()) is True


def test_operator_apply_adds_and_deletes_effects() -> None:
    operator = make_move_operator("a", "b")
    new_state = operator.apply({Predicate("at", ["robot", "a"])})
    assert Predicate("at", ["robot", "b"]) in new_state
    assert Predicate("at", ["robot", "a"]) not in new_state


def test_operator_apply_returns_copy_for_non_applicable_operator() -> None:
    state = {Predicate("at", ["robot", "b"])}
    operator = make_move_operator("a", "b")
    assert operator.apply(state) == state
    assert operator.apply(state) is not state


def test_planning_state_satisfies_positive_goal() -> None:
    state = PlanningState(predicates={Predicate("at", ["robot", "home"])})
    assert state.satisfies([Predicate("at", ["robot", "home"])]) is True


def test_planning_state_satisfies_negative_goal_by_absence() -> None:
    state = PlanningState(predicates={Predicate("at", ["robot", "home"])})
    assert state.satisfies([Predicate("holding", ["robot", "cup"], value=False)]) is True


def test_planning_state_add_positive_predicate() -> None:
    state = PlanningState()
    state.add(Predicate("clear", ["counter"]))
    assert Predicate("clear", ["counter"]) in state


def test_planning_state_add_negative_predicate_removes_positive() -> None:
    state = PlanningState(predicates={Predicate("clear", ["counter"])})
    state.add(Predicate("clear", ["counter"], value=False))
    assert Predicate("clear", ["counter"]) not in state.predicates


def test_planning_state_remove_predicate() -> None:
    state = PlanningState(predicates={Predicate("clear", ["counter"])})
    state.remove(Predicate("clear", ["counter"]))
    assert Predicate("clear", ["counter"]) not in state.predicates


def test_planning_state_contains_uses_semantics() -> None:
    state = PlanningState(predicates={Predicate("at", ["robot", "home"])})
    assert Predicate("at", ["robot", "home"]) in state
    assert Predicate("holding", ["robot", "cup"], value=False) in state


def test_symbolic_planner_finds_one_step_plan() -> None:
    planner = SymbolicPlanner([make_move_operator("a", "b")], max_depth=3)
    initial = PlanningState(predicates={Predicate("at", ["robot", "a"])})
    plan = planner.plan(initial, [Predicate("at", ["robot", "b"])])
    assert plan is not None
    assert [step.name for step in plan] == ["move"]


def test_symbolic_planner_finds_three_step_plan() -> None:
    planner = SymbolicPlanner(
        [make_move_operator("a", "b"), make_move_operator("b", "c"), make_move_operator("c", "d")],
        max_depth=4,
    )
    initial = PlanningState(predicates={Predicate("at", ["robot", "a"])})
    plan = planner.plan(initial, [Predicate("at", ["robot", "d"])])
    assert plan is not None
    assert [step.parameters[-1] for step in plan] == ["b", "c", "d"]


def test_symbolic_planner_avoids_cycles() -> None:
    planner = SymbolicPlanner(
        [make_move_operator("a", "b"), make_move_operator("b", "a"), make_move_operator("b", "c")],
        max_depth=5,
    )
    initial = PlanningState(predicates={Predicate("at", ["robot", "a"])})
    plan = planner.plan(initial, [Predicate("at", ["robot", "c"])])
    assert plan is not None
    assert len(plan) == 2


def test_symbolic_planner_returns_none_for_impossible_goal() -> None:
    planner = SymbolicPlanner([make_move_operator("a", "b")], max_depth=2)
    initial = PlanningState(predicates={Predicate("at", ["robot", "a"])})
    assert planner.plan(initial, [Predicate("at", ["robot", "z"])]) is None


def test_symbolic_planner_respects_max_depth() -> None:
    planner = SymbolicPlanner([make_move_operator("a", "b"), make_move_operator("b", "c")], max_depth=1)
    initial = PlanningState(predicates={Predicate("at", ["robot", "a"])})
    assert planner.plan(initial, [Predicate("at", ["robot", "c"])]) is None


def test_geometric_checker_grasp_reachable() -> None:
    checker = GeometricChecker()
    assert checker.check_grasp([0.5, 0.2, 0.8], "cup") is True


def test_geometric_checker_grasp_unreachable() -> None:
    checker = GeometricChecker()
    assert checker.check_grasp([2.0, 0.0, 0.1], "cup") is False


def test_geometric_checker_placement_valid() -> None:
    checker = GeometricChecker()
    assert checker.check_placement([0.8, 0.2, 0.9]) is True


def test_geometric_checker_placement_invalid_below_floor() -> None:
    checker = GeometricChecker()
    assert checker.check_placement([0.2, 0.1, -0.1]) is False


def test_geometric_checker_navigation_valid() -> None:
    checker = GeometricChecker()
    assert checker.check_navigation([0.0, 0.0, 0.0], [3.0, 2.0, 0.0]) is True


def test_geometric_checker_navigation_invalid() -> None:
    checker = GeometricChecker()
    assert checker.check_navigation([0.0, 0.0, 0.0], [9.0, 0.0, 0.0]) is False


def test_tamp_plan_num_steps_property() -> None:
    plan = TAMPPlan(operators=[make_move_operator("a", "b"), make_move_operator("b", "c")], geometric_params=[{}, {}], feasible=True)
    assert plan.num_steps == 2


def test_tamp_planner_end_to_end_household_transfer() -> None:
    operators, _ = HouseholdDomain.build()
    planner = TAMPPlanner(SymbolicPlanner(operators, max_depth=6), GeometricChecker())
    initial = PlanningState(
        predicates={
            Predicate("at", ["robot", "home"]),
            Predicate("object-at", ["cup", "table"]),
            Predicate("handempty", ["robot"]),
        },
        objects=["robot", "cup", "home", "table", "counter"],
    )
    goal = [Predicate("object-at", ["cup", "counter"])]
    object_poses = {
        "robot": [0.0, 0.0, 0.0],
        "home": [0.0, 0.0, 0.0],
        "table": [1.0, 0.0, 0.0],
        "counter": [1.2, 0.8, 0.9],
        "cup": [1.0, 0.0, 0.8],
    }

    plan = planner.plan(initial, goal, object_poses)

    assert plan is not None
    assert plan.feasible is True
    assert [step.name for step in plan.operators] == ["navigate-to", "pick-up", "navigate-to", "place"]
    assert plan.geometric_params[1]["type"] == "grasp"
    assert plan.geometric_params[-1]["surface"] == "counter"


def test_tamp_planner_returns_none_when_geometry_fails() -> None:
    operators, _ = HouseholdDomain.build()
    planner = TAMPPlanner(SymbolicPlanner(operators, max_depth=6), GeometricChecker())
    initial = PlanningState(
        predicates={
            Predicate("at", ["robot", "home"]),
            Predicate("object-at", ["cup", "table"]),
            Predicate("handempty", ["robot"]),
        }
    )
    goal = [Predicate("object-at", ["cup", "counter"])]
    object_poses = {
        "robot": [0.0, 0.0, 0.0],
        "home": [0.0, 0.0, 0.0],
        "table": [12.0, 0.0, 0.0],
        "counter": [12.5, 0.0, 0.9],
        "cup": [12.0, 0.0, 0.8],
    }

    assert planner.plan(initial, goal, object_poses) is None


def test_tamp_planner_backtracks_to_alternative_symbolic_plan() -> None:
    operators = [
        Operator(
            name="navigate-to",
            parameters=["robot", "goal"],
            preconditions=[Predicate("at", ["robot", "start"])],
            add_effects=[Predicate("at", ["robot", "goal"])],
            del_effects=[Predicate("at", ["robot", "start"])],
        ),
        Operator(
            name="navigate-to",
            parameters=["robot", "mid"],
            preconditions=[Predicate("at", ["robot", "start"])],
            add_effects=[Predicate("at", ["robot", "mid"])],
            del_effects=[Predicate("at", ["robot", "start"])],
        ),
        Operator(
            name="navigate-to",
            parameters=["robot", "goal"],
            preconditions=[Predicate("at", ["robot", "mid"])],
            add_effects=[Predicate("at", ["robot", "goal"])],
            del_effects=[Predicate("at", ["robot", "mid"])],
        ),
    ]
    planner = TAMPPlanner(SymbolicPlanner(operators, max_depth=3), GeometricChecker())
    initial = PlanningState(predicates={Predicate("at", ["robot", "start"])})
    goal = [Predicate("at", ["robot", "goal"])]
    object_poses = {
        "robot": [0.0, 0.0, 0.0],
        "start": [0.0, 0.0, 0.0],
        "mid": [5.0, 0.0, 0.0],
        "goal": [10.0, 0.0, 0.0],
    }

    plan = planner.plan(initial, goal, object_poses)

    assert plan is not None
    assert [step.parameters[-1] for step in plan.operators] == ["mid", "goal"]


def test_household_domain_builds_expected_operator_names() -> None:
    operators, object_types = HouseholdDomain.build()
    names = {operator.name for operator in operators}
    assert {"pick-up", "place", "open-door", "navigate-to", "pour", "wipe"} <= names
    assert "surface" in object_types
    assert len(operators) >= 6


def test_household_domain_operators_are_well_formed() -> None:
    operators, _ = HouseholdDomain.build()
    for operator in operators:
        assert operator.name
        assert isinstance(operator.parameters, list)
        assert all(isinstance(predicate, Predicate) for predicate in operator.preconditions)
        assert all(isinstance(predicate, Predicate) for predicate in operator.add_effects)
        assert all(isinstance(predicate, Predicate) for predicate in operator.del_effects)


def test_household_domain_supports_symbolic_pick_and_place_goal() -> None:
    operators, _ = HouseholdDomain.build()
    planner = SymbolicPlanner(operators, max_depth=6)
    initial = PlanningState(
        predicates={
            Predicate("at", ["robot", "home"]),
            Predicate("object-at", ["cup", "table"]),
            Predicate("handempty", ["robot"]),
        }
    )
    goal = [Predicate("object-at", ["cup", "counter"])]

    plan = planner.plan(initial, goal)

    assert plan is not None
    assert [operator.name for operator in plan] == ["navigate-to", "pick-up", "navigate-to", "place"]


__all__ = [name for name in globals() if name.startswith("test_")]
