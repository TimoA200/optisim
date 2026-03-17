"""Scene-graph effect application for motion primitives."""

from __future__ import annotations

from optisim.scene import SceneGraph, SceneRelation


def apply_effects(scene: SceneGraph, effects: list[dict]) -> None:
    """Apply predicate-style effects to a scene graph."""

    for effect in effects:
        name = str(effect["name"])
        args = list(effect.get("args", []))
        value = bool(effect.get("value", True))
        if not args:
            raise ValueError("effect args must contain at least one node id")
        if len(args) == 1:
            # SceneRelation is binary; encode unary predicates as a self-edge.
            subject_id = str(args[0])
            object_id = str(args[0])
        elif len(args) == 2:
            subject_id = str(args[0])
            object_id = str(args[1])
        else:
            raise ValueError("effect args must contain one or two node ids")

        if value:
            scene.add_relation(SceneRelation(subject_id=subject_id, predicate=name, object_id=object_id))
            continue

        try:
            scene.remove_relation(subject_id=subject_id, predicate=name, object_id=object_id)
        except KeyError:
            continue
