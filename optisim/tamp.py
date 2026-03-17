"""Task and motion planning primitives for lightweight household manipulation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class Predicate:
    """Symbolic predicate."""

    name: str
    args: list[str]
    value: bool = True

    def __str__(self) -> str:
        head = f"({self.name}{self._arg_suffix()})"
        return head if self.value else f"(not {head})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Predicate):
            return False
        return (self.name, tuple(self.args), self.value) == (other.name, tuple(other.args), other.value)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.args), self.value))

    def positive(self) -> Predicate:
        """Return the positive form of the predicate."""

        return Predicate(name=self.name, args=list(self.args), value=True)

    def _arg_suffix(self) -> str:
        if not self.args:
            return ""
        return " " + " ".join(self.args)


@dataclass(slots=True)
class Operator:
    """PDDL-style grounded planning operator."""

    name: str
    parameters: list[str]
    preconditions: list[Predicate]
    add_effects: list[Predicate]
    del_effects: list[Predicate]

    def applicable(self, state: set[Predicate]) -> bool:
        """Return whether all preconditions are satisfied in the state."""

        return all(_predicate_holds(predicate, state) for predicate in self.preconditions)

    def apply(self, state: set[Predicate]) -> set[Predicate]:
        """Return a new state with delete effects removed and add effects inserted."""

        if not self.applicable(state):
            return set(state)

        updated = {_canonical_positive(predicate) for predicate in state if predicate.value}
        for predicate in self.del_effects:
            updated.discard(_canonical_positive(predicate))
        for predicate in self.add_effects:
            if predicate.value:
                updated.add(predicate.positive())
            else:
                updated.discard(predicate.positive())
        return updated


@dataclass(slots=True)
class PlanningState:
    """Symbolic planning state."""

    predicates: set[Predicate] = field(default_factory=set)
    objects: list[str] = field(default_factory=list)

    def satisfies(self, goal: list[Predicate]) -> bool:
        """Return whether every goal predicate is satisfied."""

        return all(_predicate_holds(predicate, self.predicates) for predicate in goal)

    def add(self, pred: Predicate) -> None:
        """Insert a predicate into the state."""

        if pred.value:
            self.predicates.add(pred.positive())
        else:
            self.predicates.discard(pred.positive())

    def remove(self, pred: Predicate) -> None:
        """Remove a predicate from the state."""

        self.predicates.discard(pred.positive())

    def __contains__(self, pred: object) -> bool:
        if not isinstance(pred, Predicate):
            return False
        return _predicate_holds(pred, self.predicates)


class SymbolicPlanner:
    """Breadth-first symbolic planner over grounded operators."""

    def __init__(self, operators: list[Operator], max_depth: int = 20) -> None:
        self.operators = operators
        self.max_depth = max_depth

    def plan(self, initial: PlanningState, goal: list[Predicate]) -> list[Operator] | None:
        """Return the first BFS plan that satisfies the goal."""

        plans = self.enumerate_plans(initial, goal, limit=1)
        return plans[0] if plans else None

    def enumerate_plans(
        self,
        initial: PlanningState,
        goal: list[Predicate],
        *,
        limit: int | None = None,
    ) -> list[list[Operator]]:
        """Enumerate BFS-ordered plans up to the requested limit."""

        initial_state = frozenset(_canonical_positive(predicate) for predicate in initial.predicates if predicate.value)
        queue: deque[tuple[frozenset[Predicate], list[Operator]]] = deque([(initial_state, [])])
        visited: dict[frozenset[Predicate], int] = {initial_state: 0}
        solutions: list[list[Operator]] = []

        while queue:
            state_key, plan = queue.popleft()
            state = set(state_key)
            if _state_satisfies(state, goal):
                solutions.append(plan)
                if limit is not None and len(solutions) >= limit:
                    break
                continue
            if len(plan) >= self.max_depth:
                continue

            for operator in self.operators:
                if not operator.applicable(state):
                    continue
                next_state = frozenset(operator.apply(state))
                next_depth = len(plan) + 1
                if _state_satisfies(set(next_state), goal):
                    solutions.append([*plan, operator])
                    if limit is not None and len(solutions) >= limit:
                        return solutions
                    continue
                prior_depth = visited.get(next_state)
                if prior_depth is not None and prior_depth <= next_depth:
                    continue
                visited[next_state] = next_depth
                queue.append((next_state, [*plan, operator]))
        return solutions


class GeometricChecker:
    """Simple geometric heuristics for TAMP feasibility checks."""

    def __init__(self, robot_model=None) -> None:
        self.robot_model = robot_model
        self.max_reach = 1.5
        self.max_navigation_distance = 8.0

    def check_grasp(self, obj_pos: list[float], obj_name: str) -> bool:
        """Check whether an object is inside a reachable workspace sphere."""

        del obj_name
        position = np.asarray(obj_pos, dtype=float)
        if position.shape[0] != 3:
            return False
        distance = float(np.linalg.norm(position))
        return bool(distance <= self.max_reach and float(position[2]) >= -0.2)

    def check_placement(self, surface_pos: list[float], obj_size: float = 0.1) -> bool:
        """Check whether a surface pose looks stable for placement."""

        surface = np.asarray(surface_pos, dtype=float)
        if surface.shape[0] != 3:
            return False
        radial_distance = float(np.linalg.norm(surface[:2]))
        return bool(float(surface[2]) >= 0.0 and radial_distance <= self.max_reach + max(obj_size, 0.0))

    def check_navigation(self, from_pos: list[float], to_pos: list[float]) -> bool:
        """Check whether a navigation request is plausible in a flat indoor scene."""

        start = np.asarray(from_pos, dtype=float)
        goal = np.asarray(to_pos, dtype=float)
        if start.shape[0] != 3 or goal.shape[0] != 3:
            return False
        delta = goal - start
        horizontal_distance = float(np.linalg.norm(delta[:2]))
        height_change = abs(float(delta[2]))
        return bool(horizontal_distance <= self.max_navigation_distance and height_change <= 0.5)


@dataclass(slots=True)
class TAMPPlan:
    """Combined symbolic and geometric plan."""

    operators: list[Operator]
    geometric_params: list[dict]
    feasible: bool

    @property
    def num_steps(self) -> int:
        return len(self.operators)


class TAMPPlanner:
    """Planner that validates symbolic plans with geometric heuristics."""

    def __init__(self, symbolic_planner: SymbolicPlanner, geometric_checker: GeometricChecker) -> None:
        self.symbolic_planner = symbolic_planner
        self.geometric_checker = geometric_checker

    def plan(
        self,
        initial_state: PlanningState,
        goal: list[Predicate],
        object_poses: dict[str, list[float]],
    ) -> TAMPPlan | None:
        """Search symbolic plans, then keep the first geometrically feasible one."""

        for candidate in self.symbolic_planner.enumerate_plans(initial_state, goal, limit=32):
            result = self._check_symbolic_plan(candidate, object_poses)
            if result is not None:
                return result
        return None

    def _check_symbolic_plan(self, plan: list[Operator], object_poses: dict[str, list[float]]) -> TAMPPlan | None:
        poses = {name: list(position) for name, position in object_poses.items()}
        robot_name = self._infer_robot_name(plan, poses)
        robot_position = list(poses.get(robot_name, [0.0, 0.0, 0.0]))
        holding: str | None = None
        geometric_params: list[dict] = []

        for step in plan:
            params, robot_position, holding = self._step_geometry(
                step=step,
                poses=poses,
                robot_position=robot_position,
                holding=holding,
            )
            if params is None:
                return None
            if robot_name:
                poses[robot_name] = list(robot_position)
            geometric_params.append(params)
        return TAMPPlan(operators=plan, geometric_params=geometric_params, feasible=True)

    def _step_geometry(
        self,
        *,
        step: Operator,
        poses: dict[str, list[float]],
        robot_position: list[float],
        holding: str | None,
    ) -> tuple[dict | None, list[float], str | None]:
        name = step.name
        parameters = step.parameters

        if name == "navigate-to":
            target = parameters[-1]
            target_position = poses.get(target)
            if target_position is None:
                return None, robot_position, holding
            navigation_target = [target_position[0], target_position[1], robot_position[2]]
            if not self.geometric_checker.check_navigation(robot_position, navigation_target):
                return None, robot_position, holding
            return {
                "type": "navigation",
                "from": list(robot_position),
                "to": list(navigation_target),
                "waypoints": [list(robot_position), list(navigation_target)],
            }, list(navigation_target), holding

        if name == "pick-up":
            obj_name = parameters[1]
            obj_position = poses.get(obj_name)
            if obj_position is None:
                return None, robot_position, holding
            if not self.geometric_checker.check_grasp(_relative_pose(robot_position, obj_position), obj_name):
                return None, robot_position, holding
            return {
                "type": "grasp",
                "object": obj_name,
                "grasp_pose": list(obj_position),
            }, robot_position, obj_name

        if name == "place":
            obj_name = parameters[1]
            surface_name = parameters[2]
            surface_position = poses.get(surface_name)
            if surface_position is None or holding not in (None, obj_name):
                return None, robot_position, holding
            if not self.geometric_checker.check_placement(_relative_pose(robot_position, surface_position)):
                return None, robot_position, holding
            placed_position = [surface_position[0], surface_position[1], surface_position[2] + 0.05]
            poses[obj_name] = placed_position
            return {
                "type": "placement",
                "surface": surface_name,
                "target_pose": placed_position,
            }, robot_position, None

        if name == "open-door":
            door_name = parameters[1]
            door_position = poses.get(door_name)
            if door_position is None:
                return None, robot_position, holding
            if not self.geometric_checker.check_grasp(_relative_pose(robot_position, door_position), door_name):
                return None, robot_position, holding
            return {
                "type": "door",
                "door": door_name,
                "handle_pose": list(door_position),
            }, robot_position, holding

        if name == "pour":
            container_name = parameters[1]
            target_name = parameters[2]
            container_position = poses.get(container_name)
            target_position = poses.get(target_name)
            if container_position is None or target_position is None:
                return None, robot_position, holding
            if not self.geometric_checker.check_grasp(_relative_pose(robot_position, container_position), container_name):
                return None, robot_position, holding
            if not self.geometric_checker.check_placement(_relative_pose(robot_position, target_position), obj_size=0.15):
                return None, robot_position, holding
            return {
                "type": "pour",
                "container": container_name,
                "target": target_name,
                "pour_pose": list(target_position),
            }, robot_position, holding

        if name == "wipe":
            surface_name = parameters[1]
            surface_position = poses.get(surface_name)
            if surface_position is None:
                return None, robot_position, holding
            if not self.geometric_checker.check_placement(_relative_pose(robot_position, surface_position), obj_size=0.25):
                return None, robot_position, holding
            return {
                "type": "wipe",
                "surface": surface_name,
                "sweep_center": list(surface_position),
            }, robot_position, holding

        return {}, robot_position, holding

    def _infer_robot_name(self, plan: list[Operator], poses: dict[str, list[float]]) -> str:
        for operator in plan:
            if operator.parameters and operator.parameters[0] in poses:
                return operator.parameters[0]
        return "robot"


class HouseholdDomain:
    """Small grounded household manipulation domain."""

    @classmethod
    def build(cls) -> tuple[list[Operator], list[str]]:
        """Return a grounded household operator set and supported object types."""

        operators: list[Operator] = []
        robot = "robot"

        locations = ["home", "table", "counter", "sink", "doorway"]
        objects = ["cup", "bottle", "cloth"]
        surfaces = ["table", "counter", "sink"]

        for source in locations:
            for target in locations:
                if source == target:
                    continue
                operators.append(
                    Operator(
                        name="navigate-to",
                        parameters=[robot, target],
                        preconditions=[Predicate("at", [robot, source])],
                        add_effects=[Predicate("at", [robot, target])],
                        del_effects=[Predicate("at", [robot, source])],
                    )
                )

        for obj_name in objects:
            for location in surfaces:
                operators.append(
                    Operator(
                        name="pick-up",
                        parameters=[robot, obj_name, location],
                        preconditions=[
                            Predicate("at", [robot, location]),
                            Predicate("object-at", [obj_name, location]),
                            Predicate("handempty", [robot]),
                        ],
                        add_effects=[Predicate("holding", [robot, obj_name])],
                        del_effects=[Predicate("object-at", [obj_name, location]), Predicate("handempty", [robot])],
                    )
                )

        for obj_name in objects:
            for surface in surfaces:
                operators.append(
                    Operator(
                        name="place",
                        parameters=[robot, obj_name, surface],
                        preconditions=[Predicate("at", [robot, surface]), Predicate("holding", [robot, obj_name])],
                        add_effects=[Predicate("object-at", [obj_name, surface]), Predicate("handempty", [robot])],
                        del_effects=[Predicate("holding", [robot, obj_name])],
                    )
                )

        operators.append(
            Operator(
                name="open-door",
                parameters=[robot, "kitchen-door"],
                preconditions=[Predicate("at", [robot, "doorway"]), Predicate("closed", ["kitchen-door"])],
                add_effects=[Predicate("open", ["kitchen-door"])],
                del_effects=[Predicate("closed", ["kitchen-door"])],
            )
        )
        operators.append(
            Operator(
                name="pour",
                parameters=[robot, "bottle", "cup"],
                preconditions=[
                    Predicate("at", [robot, "counter"]),
                    Predicate("holding", [robot, "bottle"]),
                    Predicate("object-at", ["cup", "counter"]),
                ],
                add_effects=[Predicate("filled", ["cup"])],
                del_effects=[],
            )
        )
        operators.append(
            Operator(
                name="wipe",
                parameters=[robot, "counter"],
                preconditions=[
                    Predicate("at", [robot, "counter"]),
                    Predicate("holding", [robot, "cloth"]),
                    Predicate("dirty", ["counter"]),
                ],
                add_effects=[Predicate("clean", ["counter"])],
                del_effects=[Predicate("dirty", ["counter"])],
            )
        )

        object_types = ["robot", "object", "location", "surface", "door", "container", "tool"]
        return operators, object_types


def _canonical_positive(predicate: Predicate) -> Predicate:
    return predicate.positive()


def _predicate_holds(predicate: Predicate, state: Iterable[Predicate]) -> bool:
    positive = _canonical_positive(predicate)
    state_set = set(state)
    if predicate.value:
        return positive in state_set
    return positive not in state_set


def _state_satisfies(state: set[Predicate], goal: list[Predicate]) -> bool:
    return all(_predicate_holds(predicate, state) for predicate in goal)


def _relative_pose(origin: list[float], target: list[float]) -> list[float]:
    delta = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
    return delta.astype(float).tolist()


__all__ = [
    "Predicate",
    "Operator",
    "PlanningState",
    "SymbolicPlanner",
    "GeometricChecker",
    "TAMPPlan",
    "TAMPPlanner",
    "HouseholdDomain",
]
