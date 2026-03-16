"""Fluent task composition."""

from __future__ import annotations

from dataclasses import dataclass, field

from optisim.core.action_primitives import ActionPrimitive


@dataclass
class TaskComposer:
    """Builder for deterministic action sequences."""

    name: str
    actions: list[ActionPrimitive] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def append(self, action: ActionPrimitive) -> "TaskComposer":
        self.actions.append(action)
        return self

    def extend(self, actions: list[ActionPrimitive]) -> "TaskComposer":
        self.actions.extend(actions)
        return self

    def pick_and_place(
        self,
        target: str,
        pickup_effector: str,
        destination: tuple[float, float, float],
        support: str,
    ) -> "TaskComposer":
        return self.extend(
            [
                ActionPrimitive.reach(target=target, end_effector=pickup_effector),
                ActionPrimitive.grasp(target=target, gripper=pickup_effector),
                ActionPrimitive.move(target=target, destination=destination, end_effector=pickup_effector),
                ActionPrimitive.place(target=target, support=support, end_effector=pickup_effector),
            ]
        )
