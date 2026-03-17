"""Small TAMP demonstration for moving a cup from a table to a counter."""

from __future__ import annotations

from optisim.tamp import (
    GeometricChecker,
    HouseholdDomain,
    PlanningState,
    Predicate,
    SymbolicPlanner,
    TAMPPlanner,
)


def main() -> None:
    operators, _ = HouseholdDomain.build()
    symbolic_planner = SymbolicPlanner(operators=operators, max_depth=6)
    tamp_planner = TAMPPlanner(symbolic_planner=symbolic_planner, geometric_checker=GeometricChecker())

    initial_state = PlanningState(
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

    plan = tamp_planner.plan(initial_state=initial_state, goal=goal, object_poses=object_poses)
    if plan is None:
        print("No feasible TAMP plan found.")
        return

    print(f"Feasible: {plan.feasible}")
    print(f"Steps: {plan.num_steps}")
    for index, (operator, geometry) in enumerate(zip(plan.operators, plan.geometric_params, strict=True), start=1):
        print(f"{index}. {operator.name} {operator.parameters}")
        print(f"   geometry={geometry}")


if __name__ == "__main__":
    main()
